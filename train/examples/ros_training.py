from train.trainer import Trainer
import gymnasium as gym
import roborl_navigator.environment
from stable_baselines3 import DDPG, HerReplayBuffer, TD3, SAC

env = gym.make("RoboRL-Navigator-Franka-ROS", orientation_task=True, custom_reward=True,
               distance_threshold=0.05)
model = TD3(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)
trainer = Trainer(model=model, target_step=50_000)

trainer.train()
