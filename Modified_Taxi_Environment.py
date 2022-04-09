import numpy as np
from typing import Optional

import gym
from gym import spaces
from gym.envs.toy_text import taxi
from gym.envs.registration import register

ENV_NAME = 'Taxi-v3-modified'

class Modified_Taxi_Environment(taxi.TaxiEnv):
    def __init__(self):
        super().__init__()
        min_observation = np.array(([0] * 4), dtype=np.int32)
        max_observation = np.array(([5] * 4), dtype=np.int32)
        self.observation_space = spaces.Box(min_observation, max_observation, dtype=np.int32)
    
    def _decode_observation_space(self, observation):
        # taxi_row, taxi_col, passenger_location, destination = self.decode(observation)
        return np.array(list(self.decode(observation)))
    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        observation =  super().reset(seed=seed, return_info=return_info, options=options)
        return self._decode_observation_space(observation=observation), observation
    
    def step(self, a):
        observation, reward, done, info = super().step(a)
        return (self._decode_observation_space(observation), observation), reward, done, info

def registerEnvironment():
    register(
        id=ENV_NAME,
        entry_point=Modified_Taxi_Environment,
        max_episode_steps=500,
        reward_threshold=20.0
    )

if __name__ == "__main__":
    gym_env = gym.make(ENV_NAME)
    print(gym_env.observation_space)
