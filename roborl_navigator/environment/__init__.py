from gymnasium.envs.registration import register
from .base_env import BaseEnv

register(
    id="RoboRL-Navigator-Franka-Bullet",
    entry_point="roborl_navigator.environment.env_panda_bullet:PandaBulletEnv",
    max_episode_steps=50,
)

register(
    id="RoboRL-Navigator-Franka-ROS",
    entry_point="roborl_navigator.environment.env_panda_ros:PandaROSEnv",
    max_episode_steps=50,
)
