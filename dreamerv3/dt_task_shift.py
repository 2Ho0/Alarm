from . import rssm

from dreamerv3.Decision_Transformer.src.decision_transformer.train import dt_train
from dreamerv3.Decision_Transformer.src.models.trajectory_transformer import (DecisionTransformer)
from dreamerv3.Decision_Transformer.src.config import( EnvironmentConfig, OfflineTrainConfig, TransformerModelConfig)
from dreamerv3.Decision_Transformer.src.environments.environments import make_env
from dreamerv3.Decision_Transformer.src.decision_transformer.offline_dataset import TrajectoryDataset

def _calculate_returns(self, rewards):
        """Calculate returns-to-go for each timestep.
        
        Args:
            rewards: Reward tensor of shape [B, T]
            
        Returns:
            returns_to_go: Tensor of shape [B, T] containing returns-to-go
        """
        B, T = rewards.shape
        returns = jnp.zeros_like(rewards)
        
        # Calculate returns-to-go by summing up future rewards
        for t in reversed(range(T)):
            if t == T-1:
                returns = returns.at[:, t].set(rewards[:, t])
            else:
                returns = returns.at[:, t].set(rewards[:, t] + returns[:, t+1])
                
        return returns


def detect_task_shift(obs, prevact):
    
    env = make_env(EnvironmentConfig, seed=0, idx=0, run_name="dev")()
    EnvironmentConfig.action_space = env.action_space
    EnvironmentConfig.observation_space = env.observation_space
    print("Environment created:", env)
    transformer_config_instance = TransformerModelConfig()
    model = DecisionTransformer(
        environment_config=EnvironmentConfig,
        transformer_config=transformer_config_instance,
    )
    for step, data in enumerate(dataset):
        # 1. Extract states and create DT batch
        B, T = obs['is_first'].shape
        dt_batch = {
            'states': obs['image'],               # [B, T, H, W, C]
            'actions': prevact['action'],         # [B, T, action_dim]
            'rewards': obs['reward'],             # [B, T]
            'returns_to_go': _calculate_returns(obs['reward']),  # [B, T]
            'attention_mask': ~obs['is_last'],    # [B, T]
            'timesteps': jnp.arange(T)[None, :].repeat(B, 0)  # [B, T]
        }

        
        # 2. Create TrajectoryDataset from dt_batch

        dataset = TrajectoryDataset.from_dreamer_batch(
            dt_batch=dt_batch,
            max_len=T,  # Use full sequence length
        )

        task_shift = dt_train(model = model,
                            trajectory_data_set = dataset,
                            env = env,
                            offline_config = OfflineTrainConfig,
                            )



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple


# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim: int = 10240, hidden_dim: int = 512, output_dim: int = 3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.model(x)

# # classifier = MLPClassifier(input_dim=10240, hidden_dim=512, output_dim=3)
# # torch.save(classifier.state_dict(), "mlp_task_classifier_10240.pt")



# class TaskShiftDetector:
#     def __init__(
#         self,
#         classifier_path: str,
#         input_dim: int = 10240,
#         hidden_dim: int = 512,
#         output_dim: int = 3,
#         recent_k: int = 8,
#         device: str = "cpu",
#     ):
#         """
#         MLP 기반 task shift detector

#         Args:
#             classifier_path (str): 저장된 MLP classifier 경로
#             input_dim (int): feature 차원 (Dreamer의 feat dim)
#             threshold (float): confidence threshold
#             recent_k (int): 최근 몇 개 feature로 판단할지
#         """
#         self.device = device
#         self.recent_k = recent_k

#         # 분류기 초기화 및 로딩
#         self.classifier = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)
#         # self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
#         self.classifier.eval()
#         print("✅ MLP 분류기 로딩 완료")

#         # 내부 상태
#         self.prev_task = None
#         self.current_task = None
#         self.recent_feats = []

#     def detect_task_shift(self, feat: torch.Tensor) -> Tuple[bool, int, float]:
#         """
#         Dreamer로부터 추출한 feature(feat)를 입력으로 task shift 감지 수행

#         Args:
#             feat (Tensor): [batch, feature_dim] or [feature_dim]

#         Returns:
#             (task_shift_detected, current_task_id, confidence)
#         """
#         self.classifier.eval()

#         if feat.dim() == 1:
#             feat = feat.unsqueeze(0)  # [1, D]
#         feat = feat.to(self.device)

#         self.recent_feats.append(feat)
#         if len(self.recent_feats) > self.recent_k:
#             self.recent_feats.pop(0)

#         # recent K개 feature 평균 사용
#         recent_feat_tensor = torch.cat(self.recent_feats, dim=0)  # [K, D]
#         mean_feat = recent_feat_tensor.mean(dim=0, keepdim=True)

#         with torch.no_grad():
#             logits = self.classifier(mean_feat)
#             probs = F.softmax(logits, dim=-1)
#             confidence, predicted_task = torch.max(probs, dim=-1)

#             # task transition 판단
#             self.prev_task = self.current_task
#             self.current_task = predicted_task.item()

#             task_shift_detected = False
#             if self.prev_task is not None and self.current_task != self.prev_task:
#                 if confidence.item() > self.threshold:
#                     task_shift_detected = True

#         return task_shift_detected, self.current_task, confidence.item()

#     def reset(self):
#         self.recent_feats.clear()
#         self.prev_task = None
#         self.current_task = None


