from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)

import numpy as np

from gym.utils import seeding

from roborl_navigator.environment import BaseEnv
from roborl_navigator.simulation.ros.ros_sim import ROSSim
from roborl_navigator.robot.ros_panda_robot import ROSRobot
from roborl_navigator.task.reach_task import Reach


class FrankaROSEnv(BaseEnv):

    def __init__(self, orientation_task=False, distance_threshold=0.05, custom_reward=False) -> None:
        self.sim = ROSSim()
        self.robot = ROSRobot(self.sim)
        self.task = Reach(
            self.sim,
            self.robot,
            reward_type="dense",
            orientation_task=orientation_task,
            distance_threshold=distance_threshold,
            custom_reward=custom_reward,
        )
        super().__init__()

        self.render_width = 700
        self.render_height = 400
        self.render_target_position = (np.array([0.0, 0.0, 0.72]))
        self.render_distance = 2
        self.render_yaw = 45
        self.render_pitch = -30
        self.render_roll = 0
        with self.sim.no_rendering():
            self.sim.place_camera(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        truncated = self.robot.set_action(action)
        observation = self._get_obs()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        return self.sim.render()
