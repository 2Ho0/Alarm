from typing import cast
import gymnasium
from minigrid.wrappers import FullyObsWrapper, ObservationWrapper
from .from_gym import FromGymnasium
from gymnasium import spaces
import numpy as np
from PIL import Image

class HideMission(ObservationWrapper):
    """Remove the 'mission' string from the observation."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = cast(gymnasium.spaces.Dict, self.observation_space)
        obs_space.spaces.pop('mission')

    def observation(self, observation: dict):
        observation.pop('mission', None)
        return observation

class Minigrid(FromGymnasium):
    def __init__(self, task: str, fully_observable: bool, hide_mission: bool):
        env = gymnasium.make(f"{task}-v0", render_mode="rgb_array")

        if fully_observable:
            env = FullyObsWrapper(env)
        if hide_mission:
            env = HideMission(env)
        env = ResizeIntObservation(env, size=(64, 64))
        super().__init__(env=env)

class ResizeIntObservation(ObservationWrapper):
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size
        self.observation_space.spaces['image'] = spaces.Box(
            low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8
        )

    def observation(self, obs):
        image = obs['image']  # (H, W, 3)
        resized_channels = []
        for c in range(image.shape[-1]):
            channel = Image.fromarray(image[:, :, c]).resize(self.size, resample=Image.NEAREST)
            resized_channels.append(np.array(channel))
        obs['image'] = np.stack(resized_channels, axis=-1)
        return obs