import pytest
import torch as t
import torch.nn as nn
from einops import rearrange
from dataclasses import asdict
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from collections import deque
import sys
import os

import wandb
import ruamel.yaml as yaml
from dreamerv3.Decision_Transformer.src.config import EnvironmentConfig, OfflineTrainConfig
from dreamerv3.Decision_Transformer.src.models.trajectory_transformer import (
    TrajectoryTransformer,
)

from .offline_dataset import TrajectoryDataset
from .eval import evaluate_dt_agent
from .utils import configure_optimizers, get_scheduler
from torch.utils.data import ConcatDataset

# 이 함수는 dt_train.py 또는 utils.py 같은 파일에 추가하면 좋습니다.

def dt_inference(model, trajectory_data_set, pre_task, num_actions, device="cpu"):
    """학습된 DT 모델로 태스크 전환 여부만 추론하는 함수"""
    model.eval()
    model.to(device)
    
    s, a, _, _, rtg, ti, _, _, _ = next(iter(trajectory_data_set))
    s, a, rtg, ti = s.to(device), a.to(device), rtg.to(device), ti.to(device)
    a[a == -10] = num_actions
    a = a.to(device)
    # 옵티마이저, 손실 계산, 역전파가 전혀 필요 없음
    with t.no_grad():
        # mlp_learn=True로 task_preds를 바로 얻음
        _, _, _, task_preds, _ = model(             # task_preds(1,20,3)
            states=s,                            # s.shape (20, 8, 8, 3)
            actions=a,                          # a.shape(20,1)
            rtgs=rtg,                       # rtg.shape(20,1)   
            timesteps=ti.unsqueeze(-1), 
            mlp_learn=True,
        )

        # 태스크 전환 감지 로직
        # recent_k = s.shape[0] // 2
        # recent_task_preds = task_preds[-recent_k:]
        task_probs = t.softmax(task_preds, dim=-1)
        # mean_task_probs = task_probs.mean(dim=0)
        _, current_task_tensor = t.max(task_probs, dim=-1)
        current_task_tensor = current_task_tensor.mode().values.item()

        task_shift_detected = False
        if pre_task != current_task_tensor:
            task_shift_detected = True

    # 현재 태스크와 전환 여부를 함께 반환하여 다음 스텝에서 pre_task를 업데이트
    return task_shift_detected, current_task_tensor

def dt_train(
    model: TrajectoryTransformer,
    trajectory_data_set: TrajectoryDataset,
    num_actions: int,
    offline_config: OfflineTrainConfig,
    device="cpu",
    
):

    # 전체 학습 데이터의 task_label 수집 필요
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    mode = offline_config.mode
    pre_task = 0
    task_shift_detected = False

    # get optimizer from string
    optimizer = configure_optimizers(model, offline_config)
    # TODO: Stop passing through all the args to the scheduler, shouldn't be necessary.
    scheduler_config = asdict(offline_config)
    del scheduler_config["optimizer"]

    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )

    for param in model.parameters():
        param.requires_grad = True

    # can uncomment this to get logs of gradients and pars.
    # wandb.watch(model, log="all", log_freq=train_batches_per_epoch)
    pbar = tqdm(range(offline_config.train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(trajectory_data_set):
            
            # s, a, r, d, rtg, ti, m = [x.unsqueeze(0) for x in (s, a, r, d, rtg, ti, m)]
            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = num_actions  # dummy action for padding
            optimizer.zero_grad()

            # action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            action = a[:] if a.shape[0] > 1 else None # a shape = (65,1)
            state_preds, action_preds, reward_preds= model(
                states=s,
                # remove last action
                actions=action,
                rtgs=rtg,  # remove last rtg
                timesteps=ti.unsqueeze(-1),
                mlp_learn=False,
            )

            if mode == 'state':                
                state_preds = rearrange(state_preds, "b t s -> (b t) s") # state_preds (64, 12288)
                
                s_target = s[:]  # s[1:] removes the first state
                s_exp = rearrange(s_target, "b t w c -> b (t w c)") # s_exp (65, 12288)

                # 이제 두 텐서의 모양이 동일하므로 손실 계산이 가능합니다.
                loss = nn.MSELoss()(state_preds, s_exp)

            elif mode == 'action':
                action_preds = action_preds[:, :-1]
                action_preds = rearrange(action_preds, "b t a -> (b t) a")
                a_exp = rearrange(a[:, 1:], "b t -> (b t)").to(t.int64)
                mask = a_exp != num_actions
                loss = loss_fn(action_preds[mask], a_exp[mask])

            elif mode == 'rtg':
                reward_preds = reward_preds[:, :-1]
                r = r[:, 1:] # 128, 100, 1
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s") # 12800, 1
                r_exp = rearrange(r.squeeze(-1), "b t -> (b t)").to(t.float32)
                loss = nn.MSELoss()(reward_preds.squeeze(-1), r_exp)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            if offline_config.track:
                wandb.log({
                    "train/loss": loss.item(),
                })

    # MLP training start
    for name, param in model.named_parameters():
        if "penultimate_layer" in name or "output_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 새로운 옵티마이저와 스케줄러 설정
    optimizer = configure_optimizers(model, offline_config)
    scheduler = get_scheduler(
        offline_config.scheduler,
        optimizer,
        **scheduler_config,
        training_steps=offline_config.mlp_train_epochs * len(trajectory_data_set)
    )

    # MLP만을 위한 추가 학습 루프
    pbar_mlp = tqdm(range(offline_config.mlp_train_epochs), desc="MLP Fine-Tuning")
    for epoch in pbar_mlp:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(trajectory_data_set):
            model.train()
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = num_actions
            action = a[:] if a.shape[0] > 1 else None

            state_preds, action_preds, reward_preds, task_preds, _ = model(
                states=s,
                actions=action,
                rtgs=rtg,
                timesteps=ti.unsqueeze(-1),
                mlp_learn=True,
            )

            # task classification loss만 사용
            if isinstance(task_id, int):
                task_labels = t.tensor([task_id], device=task_preds.device)
            else:
                task_labels = task_id.to(task_preds.device)
            task_loss = model.label_smoothing_loss(task_preds, task_labels)

            task_pred = t.argmax(task_preds, dim=-1)
            n_correct = (task_pred == task_labels).sum().item()
            n_total = task_labels.shape[0]
            task_accuracy = n_correct / n_total

            # Check Task Changing
            recent_k = s.shape[0] // 2 # recent_k = 8

            # (1) recent task predictions
            recent_task_preds = task_preds[-recent_k:]  # (recent_k, num_tasks)
            # softmax over task logits
            task_probs = t.softmax(recent_task_preds, dim=-1)  # (B, num_tasks)
            mean_task_probs = task_probs.mean(dim=0)  # (num_tasks,) = e.g., (3,)
            
            max_prob, current_task = t.max(mean_task_probs, dim=-1)  # current_task: int (0~2)
            

            if pre_task == current_task.item():
                task_shift_detected = False
                    
            else:
                task_shift_detected = True
                pre_task = current_task
                return task_shift_detected

            pre_task = current_task

            # task accuracy
            task_correct = {}
            task_total = {}
            for pred, true in zip(task_pred.cpu(), task_labels.cpu()):
                true = int(true)
                if true not in task_correct:
                    task_correct[true] = 0
                    task_total[true] = 0
                task_correct[true] += int(pred == true)
                task_total[true] += 1

            if offline_config.track:
                wandb.log({
                    "train/MLP_loss": task_loss.item(),
                    "train/MLP_accuracy": task_accuracy,
                })
                for tid in sorted(task_correct.keys()):
                    acc = task_correct[tid] / task_total[tid]
                    wandb.log({f"train/MLP_task{tid}_accuracy": acc})

            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar_mlp.set_description(f"MLP Fine-Tuning: task_loss = {task_loss.item():.4f}")

    return task_shift_detected
