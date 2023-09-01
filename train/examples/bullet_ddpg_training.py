from train.trainer import Trainer
import gymnasium as gym
import roborl_navigator.environment
from stable_baselines3 import DDPG, HerReplayBuffer
from roborl_navigator.utils import get_save_path, create_directory_if_not_exists


save_path = get_save_path()
create_directory_if_not_exists(save_path)
save_path += "/DDGP_Model"
print(f"Save Path: {save_path}")


env = gym.make("RoboRL-Navigator-Franka-Bullet", render_mode="rgb_array", orientation_task=True, custom_reward=True)
model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)
trainer = Trainer(model=model, path=save_path)
trainer.train()
