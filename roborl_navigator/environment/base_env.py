from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    robot = None
    sim = None
    task = None

    def __init__(self) -> None:

        observation, _ = self.reset()

        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["achieved_goal"].shape

        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            )
        )

        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward

        self._saved_goal = dict()


    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32)
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        desired_goal = self.task.get_goal().astype(np.float32)
        return {
            "observation": robot_obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return NotImplemented

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        return NotImplemented

    def close(self):
        return NotImplemented

    def render(self) -> Optional[np.ndarray]:
        return NotImplemented

    # STATE METHODS

    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`."""
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier."""
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state."""
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

