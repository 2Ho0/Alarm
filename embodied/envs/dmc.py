import functools
import os

import elements
import embodied
import numpy as np
# Ensure headless rendering before importing dm_control
if 'MUJOCO_GL' not in os.environ:
  os.environ['MUJOCO_GL'] = 'egl'
from dm_control import manipulation
from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

from . import from_dm


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), proprio=True, image=True, camera=-1): #main.py의 make_env 함수에서 호출되고, ctor(task, **kwargs)에서 task가 env 인자로 전달 됨 
    
  # MuJoCo 환경 설정 code 
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    
  # env 값을 통해, domain, task 값 설정 code
    if isinstance(env, str):
      domain, task = env.split('_', 1) # ex) 'cup_catch' -> domain = 'cup', task = 'catch'  
      # 방어적 정리: CLI 파싱 이슈로 붙은 괄호/따옴표 제거
      domain = domain.strip().strip("[]'\"")
      task = task.strip().strip("[]'\"")
    
    # camera 지정 code
      if camera == -1: # camera 값이 -1일 때, domain 변수에 해당하는 값을 기준으로 DEFAULT_CAMERAS 딕셔너리에서 값을 가져옴
        camera = self.DEFAULT_CAMERAS.get(domain, 0) # DEFAULT_CAMERAS 딕셔너리에서 domain 키에 해당하는 값을 가져옴, 해당 키가 없을 시 0 반환 
      
    # domain 중, 특수 case 처리하여 env 환경 생성 code
      # 'cup' domain은 'ball_in_cup'로 변경 
      if domain == 'cup':  
        domain = 'ball_in_cup'
      # 'manip' domain은 'manip_vision'로 변경 후 env 생성
      if domain == 'manip': 
        env = manipulation.load(task + '_vision')
      # 'rodent' domain은 basic_rodent_2020 모듈 통해 env 생성
      elif domain == 'rodent': 
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        env = getattr(basic_rodent_2020, task)()
      # 나머지 domain은 suite 모듈 통해 env 생성
      else:
        env = suite.load(domain, task)
        
  
    self._dmenv = env # env 변수에 저장 
    self._env = from_dm.FromDM(self._dmenv) # FromDM 클래스 통해 env 객체 생성
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat) # ActionRepeat 클래스 통해 env 객체 생성(repeat 값에 따라 행동 반복)
    self._size = size # __init__ 함수에서 전달된 size 값을 저장
    self._proprio = proprio # __init__ 함수에서 전달된 proprio 값을 저장
    self._image = image # __init__ 함수에서 전달된 image 값을 저장
    self._camera = camera # __init__ 함수에서 전달된 camera 값을 저장
    self._render_failed_logged = False

  @functools.cached_property
  def obs_space(self):
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = self._env.obs_space.copy()
    if not self._proprio:
      spaces = {k: spaces[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    spaces[key] = elements.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    if not self._proprio:
      obs = {k: obs[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    if self._image:
      try:
        obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
      except Exception as e:
        if not self._render_failed_logged:
          print('Warning: MuJoCo rendering failed; falling back to zero images. ', e)
          self._render_failed_logged = True
        obs[key] = np.zeros(self._size + (3,), dtype=np.uint8)
    else:
      # Headless mode without images: provide a dummy image tensor to satisfy space checks
      obs[key] = np.zeros(self._size + (3,), dtype=np.uint8)
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs
