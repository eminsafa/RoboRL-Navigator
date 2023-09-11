from train.trainer import Trainer
import gymnasium as gym
import roborl_navigator.environment
from stable_baselines3 import DDPG, HerReplayBuffer, TD3, SAC

env = gym.make(
    "RoboRL-Navigator-Franka-Bullet",
    render_mode="human",
    orientation_task=False,
    custom_reward=False,
    distance_threshold=0.05,
    goal_range=0.2
)

model = TD3(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

trainer = Trainer(model=model, target_step=50_000)

trainer.train()
