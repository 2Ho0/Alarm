import collections
from functools import partial as bind

import elements
import embodied
import numpy as np
import torch as t

try:
  from dreamerv3.Decision_Transformer.src.decision_transformer.train import dt_train, dt_inference
  from dreamerv3.Decision_Transformer.src.models.trajectory_transformer import (DecisionTransformer)
  from dreamerv3.Decision_Transformer.src.config import( EnvironmentConfig, OfflineTrainConfig, TransformerModelConfig)
  from dreamerv3.Decision_Transformer.src.environments.environments import make_env as dt_make_env
  from dreamerv3.Decision_Transformer.src.decision_transformer.offline_dataset import TrajectoryDataset
  from dreamerv3.Decision_Transformer.src.decision_transformer.utils import get_max_len_from_model_type
  DT_AVAILABLE = True
except Exception:
  DT_AVAILABLE = False
import jax
from gymnasium.spaces import Box

# JAX의 JIT 환경 밖에서 사용할 NumPy 버전의 returns_to_go 계산 함수
def _calculate_returns_np(rewards):
    """Calculate returns-to-go for each timestep using NumPy."""
    B, T = rewards.shape
    returns = np.zeros_like(rewards, dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            returns[:, t] = rewards[:, t]
        else:
            returns[:, t] = rewards[:, t] + returns[:, t + 1]
    return returns

def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  dt_replay = make_replay(mode = 'decision_transformer') if DT_AVAILABLE else None

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  dt_pre_task = 0

  if DT_AVAILABLE:
    # Decision Transformer 모델 및 관련 설정 초기화 (학습 루프 시작 전 1회 실행)
    dt_env_config = EnvironmentConfig()
    dt_model_config = TransformerModelConfig()
    dt_offline_config = OfflineTrainConfig()
    # DT용 환경은 DT 모델 내부의 observation/action space 정보를 설정하기 위해 한 번만 생성합니다.
    dt_env = make_env(0)
    act_space = dt_env.act_space
    
    # 가져온 정보로 DT 설정을 구성합니다.
    class MockDiscreteSpace:
      def __init__(self, n):
        self.n = n
    
    # embodied의 딕셔너리 형태에서 실제 Space 객체를 추출합니다.
    main_action_space = act_space['action']
    num_actions = int(main_action_space.high) # .high가 행동의 개수를 나타냅니다.
    dt_env_config.action_space = MockDiscreteSpace(num_actions)
    mock_obs_space = Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
    dt_env_config.observation_space = mock_obs_space
    
    dt_model = DecisionTransformer(
        environment_config=dt_env_config,
        transformer_config=dt_model_config,
    )
    model_path = '/home/hail/Project/dreamerv3/dreamerv3/Decision_Transformer/models/9101.pt'
    checkpoint = t.load(model_path)
    dt_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Decision Transformer initialized successfully.")
  else:
    print("Decision Transformer not available. Skipping DT-based task shift detection.")

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  if DT_AVAILABLE and dt_replay is not None:
    driver.on_step(dt_replay.add)
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    nonlocal dt_pre_task
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)

      if DT_AVAILABLE and dt_replay is not None:
        # Decision Transformer로 Task Shift 감지 (NumPy/PyTorch 영역)
        with elements.timer.section('dt_task_shift'):
          dt_batch = dt_replay.sample(args.batch_size)
          batch_cpu = jax.device_get(dt_batch)
          B, T = batch_cpu['is_first'].shape
          # `batch`는 실제 NumPy 배열이므로 안전하게 DT에 전달 가능
          dt_batch_data = {
              'states': batch_cpu['image'],
              'actions': batch_cpu['action'],
              'rewards': batch_cpu['reward'],
              'returns_to_go': _calculate_returns_np(batch_cpu['reward']),
              'attention_mask': ~batch_cpu['is_last'],
              'timesteps': np.arange(T)[None, :].repeat(B, 0)
          }
          
          dt_dataset = TrajectoryDataset.from_dreamer_batch(
              dt_batch=dt_batch_data, max_len=T
          )
          task_shift, new_task = dt_inference(
                model=dt_model,
                trajectory_data_set=dt_dataset,
                pre_task=dt_pre_task,
                num_actions=num_actions
            )
          dt_pre_task = new_task
          
          # task_shift = dt_train(
          #     model=dt_model,
          #     trajectory_data_set=dt_dataset,
          #     num_actions=num_actions,
          #     offline_config=dt_offline_config
          # )
          _, T_train = batch['is_first'].shape
          target_shape = (B, T_train, T_train)
          task_shift_batched = np.full(target_shape, task_shift, dtype=bool)
      else:
        # DT 미사용 시에는 False 매트릭스를 사용
        B, T_train = batch['is_first'].shape
        task_shift_batched = np.full((B, T_train, T_train), False, dtype=bool)

      batch['task_shift_result'] = task_shift_batched
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)

      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        batch_report = next(stream_report)
      
        target_shape = (16, 33, 65)
        task_shift_batched = np.full(target_shape, False, dtype=bool)
        batch_report['task_shift_result'] = task_shift_batched

        carry_report, mets = agent.report(carry_report, batch_report)
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
