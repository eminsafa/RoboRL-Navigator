import time

import numpy as np

from roborl_navigator.environment.env_panda_ros import FrankaROSEnv


def get_ag(obs):
    print(obs.get("achieved_goal")[:3])


env = FrankaROSEnv(orientation_task=True)
env.reset()

action = np.ones(7)
observation, reward, terminated, truncated, info = env.step(action)
action = np.zeros(7)
for i in range(10_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 5 == 0:
        env.reset()
print("ok")
