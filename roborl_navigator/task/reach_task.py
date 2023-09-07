from typing import (
    Dict,
    Any,
)

import numpy as np
from roborl_navigator.utils import distance, euler_to_quaternion


class Reach:
    def __init__(
        self,
        sim,
        robot,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_range=0.3,
        orientation_task=False,
        custom_reward=False,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.goal = None

        self.reward_type = reward_type
        self.orientation_task = orientation_task
        self.custom_reward = custom_reward
        self.distance_threshold = distance_threshold

        # min X can be 0.07
        self.goal_range_low = np.array([0.5 - (goal_range / 2), -goal_range / 2, 0.05])
        self.goal_range_high = np.array([0.5 + (goal_range / 2), goal_range / 2, goal_range / 2])
        self.orientation_range_low = np.array([-3, -0.8])
        self.orientation_range_high = np.array([-2, 0.4])

        with self.sim.no_rendering():
            self.create_scene()

    def create_scene(self) -> None:
        self.sim.create_scene()

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        if self.orientation_task:
            goal_orientation = euler_to_quaternion([self.goal[3], self.goal[4], 0])
            self.sim.set_base_pose("target_orientation_mark", self.goal[:3], goal_orientation)

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        if self.orientation_task:
            ee_orientation = np.array(self.robot.get_ee_orientation())[:2]
            return np.concatenate([
                ee_position,
                ee_orientation,
            ])
        return ee_position

    def _sample_goal(self) -> np.ndarray:
        position = np.random.uniform(self.goal_range_low, self.goal_range_high)
        if self.orientation_task:
            orientation = np.random.uniform(self.orientation_range_low, self.orientation_range_high)
            return np.concatenate((
                position,
                orientation,
            )).astype(np.float32)
        return position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal, self.orientation_task)
        result = np.array(d < self.distance_threshold, dtype=bool)
        return result

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal, self.orientation_task)
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
