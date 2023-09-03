from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.utils import seeding

from roborl_navigator.environment import BaseEnv
from roborl_navigator.simulation.bullet import BulletSim
from roborl_navigator.robot.bullet_panda_robot import BulletPanda
from roborl_navigator.task.reach_task import Reach
from roborl_navigator.utils import euler_to_quaternion


class FrankaBulletEnv(BaseEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode="human", orientation_task=False, distance_threshold=0.05, custom_reward=False) -> None:
        self.sim = BulletSim(render_mode=render_mode, renderer="Tiny", n_substeps=30, orientation_task=orientation_task)
        self.robot = BulletPanda(self.sim, orientation_task=orientation_task)
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
        self.robot.set_action(action)
        self.sim.step()
        if np.sum(self.robot.get_ee_velocity()) > 0.1:
            self.sim.step()
            if np.sum(self.robot.get_ee_velocity()) > 0.1:
                self.sim.step()

        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        # self.task.goal[3:] = observation["observation"][3:]
        # self.sim.set_base_pose("target_box", self.task.goal[:3], orientation=self.task.goal[3:])
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Render."""
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
