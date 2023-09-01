import numpy as np

from roborl_navigator.environment.env_panda_ros import FrankaROSEnv


def get_ag(obs):
    print(obs.get("achieved_goal")[:3])


env = FrankaROSEnv()
env.reset()

action = np.ones(7)
observation, reward, terminated, truncated, info = env.step(action)
print(f"First: ")
get_ag(observation)
print("------")
action = np.zeros(7)
for i in range(1_000):
    env.reset()
print("ok")
