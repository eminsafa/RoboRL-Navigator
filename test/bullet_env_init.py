import time

import numpy as np

from roborl_navigator.environment.env_panda_bullet import FrankaBulletEnv


def get_ag(obs):
    print(obs.get("achieved_goal")[:3])


env = FrankaBulletEnv(orientation_task=True, render_mode="human")
env.reset()

action = np.ones(7)
observation, reward, terminated, truncated, info = env.step(action)
print(f"First: {get_ag(observation)}")
action = np.zeros(7)
for i in range(10_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)
    if i % 500 == 0:
        env.reset()
print("ok")
