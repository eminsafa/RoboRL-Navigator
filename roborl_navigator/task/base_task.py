import math
from typing import Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from roborl_navigator.utils import distance, euler_to_quaternion


class Task(ABC):
    def __init__(
        self,
        sim,
        get_ee_position,
        get_ee_orientation,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_range=0.5,
        orientation_task=False,
        custom_reward=False,
    ) -> None:
        self.sim = sim
        self.goal = None
        self.reward_type = reward_type
        self.orientation_task = orientation_task
        self.custom_reward = custom_reward
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.get_ee_orientation = get_ee_orientation
        self.goal_range_low = np.array([-goal_range / 2 + 0.7, -goal_range / 2, 0.73])
        self.goal_range_high = np.array([goal_range / 2 + 0.7, goal_range / 2, 0.73 + goal_range / 2])
        self.orientation_range_low = np.array([-3.0, -0.8, -1.75])
        self.orientation_range_high = np.array([-2.0, 0.4, 0.0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.3, width=2, height=0.71, x_offset=0.5)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.85]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        if self.orientation_task:
            ee_orientation = np.array(self.get_ee_orientation())
        else:
            ee_orientation = np.zeros(3)
        return np.concatenate([
            ee_position,
            ee_orientation,
        ])

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        if self.orientation_task:
            self.sim.set_base_pose("target_box", self.goal[:3], euler_to_quaternion(self.goal[3:]))

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        result = np.array(d < self.distance_threshold, dtype=bool)
        return result

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal, self.custom_reward)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()
