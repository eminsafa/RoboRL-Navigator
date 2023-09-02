from train.trainer import Trainer
import gymnasium as gym
import roborl_navigator.environment
from stable_baselines3 import DDPG, HerReplayBuffer


env = gym.make("RoboRL-Navigator-Franka-Bullet", render_mode="rgb_array", orientation_task=True, custom_reward=True)
model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)
trainer = Trainer(model=model)

trainer.train()
