import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, OneHotPartialObsWrapper
from minigrid.wrappers import RGBImgPartialObsWrapper
from gymnasium.wrappers import TransformObservation

from dreamerv3.Decision_Transformer.src.config import EnvironmentConfig

from .wrappers import RenderResizeWrapper, ViewSizeWrapper


def make_env(config: EnvironmentConfig, seed: int, idx: int, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""

    # only one of fully observed or flat one hot can be true.
    assert not (
        config.fully_observed and config.one_hot_obs
    ), "Can't have both fully_observed and flat_one_hot."

    def thunk():
        kwargs = {}
        if config.render_mode:
            kwargs["render_mode"] = config.render_mode
        if config.max_steps:
            kwargs["max_steps"] = config.max_steps

        env = gym.make(config.env_id, **kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.capture_video and idx == 0:
            env = RenderResizeWrapper(env, 256, 256)
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                # Video every 50 runs for env #1
                episode_trigger=lambda x: x % config.video_frequency == 0,
                disable_logger=True,
            )

        # hard code for now!
        if config.env_id.startswith("MiniGrid"):
            
            # if config.view_size != 7:
            #     env = ViewSizeWrapper(env, agent_view_size=config.view_size)
            if config.rgb_obs:
                print("Applying RGB, Filter, and Resize wrappers.")
    
                # 1. 먼저 RGB 이미지 래퍼를 적용 (출력: Dict)
                env = RGBImgPartialObsWrapper(env)
                new_observation_space = spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                )
                env = TransformObservation(
                    env,
                    # 이 함수는 딕셔너리(obs)를 받아 'image' 키를 꺼내고 리사이즈합니다.
                    lambda obs: cv2.resize(obs['image'], (64, 64), interpolation=cv2.INTER_AREA),
                    observation_space=new_observation_space
                )

            
            elif config.one_hot_obs:
                env = OneHotPartialObsWrapper(env)
            elif config.fully_observed:
                env = FullyObsWrapper(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.run_name = run_name
        return env

    return thunk
