import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml
from gymnasium.spaces import Box


def main(argv=None):
# banner 출력 code
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

# configs.yaml 파일 읽기 code
  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  
# CLI 인자 parsing code
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv) #--configs 인자를 parsed 변수에 담고, 나머지 인자는 other 변수에 담음(--configs 인자에 아무 것도 없을 시, configs의 defaults 파일을 사용)
  config = elements.Config(configs['defaults']) # config 변수에 configs.yaml 파일의 defaults 저장
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

# 분산 학습 환경 시, 노드 번호 설정 code
  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

# 로그 저장 경로 설정 code
  logdir = elements.Path(config.logdir) # logdir 변수에 config.logdir(log 저장 경로) 저장
  print('Logdir:', logdir) # logdir 출력
  print('Run script:', config.script) # 사용자가 선택한 train, test, eval, parallel, parallel_env, parallel_envs, parallel_replay 모드 출력
  # 사용자가 선택한 모드가 train, test, eval 일 때, logdir 생성 및 config.yaml 파일 저장
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir() # logdir 생성
    config.save(logdir / 'config.yaml') # config.yaml 파일 저장
  # 로그 타이머 설정 code (config.logger.timer 값에 따라 로그 타이머 활성화 여부 결정)
  def init():
    elements.timer.global_timer.enabled = config.logger.timer 

# Dreamer에서 실험 실행 환경을 설정하고,서버-클라이언트 기반 인터페이스 (예: 원격 디버깅, 시각화, 동기화 등)를 제공하는 유틸리티 프레임워크 code
  portal.setup(
      errfile=config.errfile and logdir / 'error', #config.errfile 값이 True일 때, logdir / 'error' 경로에 에러 로그 저장
      clientkw=dict(logging_color='cyan'), # 클라이언트 로그 색상 설정
      serverkw=dict(logging_color='cyan'), # 서버 로그 색상 설정
      initfns=[init], # 초기화 함수 설정  
      ipv6=config.ipv6, # IPv6 사용 여부 설정
  )

# train, test, eval등의 함수에 들어갈 인자 설정 code
  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  # TaskManager 초기화 (연속 학습)
  task_manager = None
  if config.get('task_manager', {}).get('enabled', False):
    from .task_manager import TaskManager
    task_manager = TaskManager(
        domain=config.task_manager.domain,
        tasks=config.task_manager.tasks,
        strategy=config.task_manager.strategy,
        switch_interval=config.task_manager.switch_interval,
        random_seed=config.task_manager.get('random_seed', 42),
        curriculum_order=config.task_manager.get('curriculum_order', [])
    )
    print(f"TaskManager initialized: {task_manager.get_stats()}")
    # Ensure a consistent base task so agent/env agree even if task_manager
    # is not threaded everywhere.
    initial_task = task_manager.get_current_task()
    if str(task_manager.domain).lower().startswith('minigrid'):
      config = config.update(task=f'minigrid_MiniGrid-{initial_task}')
    else:
      config = config.update(task=f'dmc_{task_manager.domain}_{initial_task}')

# 각 모드에 따라 함수 실행 code (bind 함수는 함수와 인자를 인자로 받아 해당 함수에 입력 인자를 인자로 하여 실행하는 새로운 함수를 반환하는 함수)
  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config, task_manager=task_manager),
        bind(make_replay, config, 'replay'),
        bind(make_env, config, task_manager=task_manager),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config, task_manager=task_manager),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config, task_manager=task_manager),
        bind(make_env, config, task_manager=task_manager),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config, task_manager=task_manager),
        bind(make_env, config, task_manager=task_manager),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config, task_manager=task_manager),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config, task_manager=task_manager),
        bind(make_env, config, task_manager=task_manager),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config, task_manager=task_manager), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config, task_manager=task_manager), bind(make_env, config, task_manager=task_manager), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config, task_manager=None):
  from .agent import Agent
  env = make_env(config, 0, task_manager=task_manager)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  # [수정 1] Dreamer 학습에 필요한 시퀀스 길이를 먼저 계산합니다.
  # 이 길이가 JAX가 함수를 컴파일하는 기준이 됩니다.
  # train 모드를 기준으로 길이를 계산해야 합니다.
  batlen = config.batch_length
  consec = config.consec_train
  dreamer_length = consec * batlen + config.replay_context

  if mode == 'decision_transformer':
      # DT는 최신 데이터만 필요하므로 작고 빠른 온라인 버퍼를 사용합니다.
      dt_batch_size = 32
      
      # DT가 저장할 시퀀스 길이를 늘렸으므로, capacity도 적절히 조절합니다.
      # 예: 길이 65짜리 시퀀스 5개 배치 분량 저장
      dt_capacity = dt_batch_size * dreamer_length * 5 
      
      directory = elements.Path(config.logdir) / folder
      if config.replicas > 1:
          directory /= f'{config.replica:05}'

      latest_selector = embodied.replay.selectors.Latest()
      return embodied.replay.Replay(
          length=dreamer_length,
          capacity=int(dt_capacity),
          online=True,
          chunksize=1024,
          directory=directory,
          selector=latest_selector
      )
  
  # 이전에 계산한 dreamer_length를 그대로 사용합니다.
  length = dreamer_length
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)
  
  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  # TaskManager 지원 - 연속 학습을 위한 동적 태스크 변경
  task_manager = overrides.pop('task_manager', None)
  
  if task_manager is not None:
    # TaskManager가 있으면 현재 태스크 동적 선택
    current_task = task_manager.get_current_task()
    domain = task_manager.domain
    if str(domain).lower().startswith('minigrid'):
      # MiniGrid: "MiniGrid-<Task>" 형태로 구성
      suite = 'minigrid'
      task = f'MiniGrid-{current_task}'
    else:
      # 기본: DMC(MuJoCo): "<domain>_<task>" 형태
      suite = 'dmc'
      task = f'{domain}_{current_task}'
  else:
    # 기존 방식: 고정된 태스크 사용
    task = config.task
    # miniGrid 태스크의 경우, suite를 'minigrid'로 설정
    if 'MiniGrid-' in task:
        suite = 'minigrid'
    else:
        suite, task = config.task.split('_', 1) # config.task 변수에서 '_'를 기준으로 문자열을 나누어 suite, task 변수에 저장
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  

  # ctor dict 중에서 suite 키에 해당하는 값을 ctor 변수에 저장
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
      'minigrid': lambda task, **kw: importlib.import_module(
        'embodied.envs.minigrid').Minigrid(
            task=task,
            fully_observable=False,
            hide_mission=True,
        ),
  }[suite]
  if isinstance(ctor, str): # memmaze는 ctor이 문자열이 아닌 lambda 함수이므로, 문자열인 경우에만 처리
    module, cls = ctor.split(':') # ex) 'embodied.envs.atari:Atari' -> module = 'embodied.envs.atari', cls = 'Atari'
    module = importlib.import_module(module) # ex) 'embodied.envs.atari' 모듈 임포트
    ctor = getattr(module, cls) # ex) 'embodied.envs.atari.Atari'(module)에서 Atari(cls) 클래스 가져오기
  kwargs = config.env.get(suite, {}) # config.env 변수에서 suite 키에 해당하는 값을 kwargs 변수에 저장
  kwargs.update(overrides) # overrides 변수에 저장된 값을 kwargs 변수에 추가 
  if kwargs.pop('use_seed', False): # kwargs 변수에서 use_seed 키에 해당하는 값을 제거 및 True 반환, 해당 키가 없을 시 False 반환
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1) # kwargs 딕셔너리에 seed 키에 해당하는 값을 config.seed와 index 값을 해시 함수에 전달하여 32비트 정수로 변환한 값을 2 ** 32 - 1로 나눈 나머지 값으로 설정 
  if kwargs.pop('use_logdir', False): # kwargs 변수에서 use_logdir 키에 해당하는 값을 제거 및 True 반환, 해당 키가 없을 시 False 반환
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}' # kwargs 딕셔너리에 logdir 키에 해당하는 값을 config.logdir 경로에 f'env{index}' 경로를 추가한 값으로 설정
  env = ctor(task, **kwargs) # ex) Atari(task, **kwargs) 호출(line 249에서 정의한 클래스 호출)
  return wrap_env(env, config)

# make_env 함수에서 반환된 env 객체를 처리하는 함수 code
def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
