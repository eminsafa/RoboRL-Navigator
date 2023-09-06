from typing import TypeVar, Optional

import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod

from roborl_navigator.simulation import Simulation
Sim = TypeVar('Sim', bound=Simulation)


class Robot(ABC):

    def __init__(self, sim: Sim, orientation_task=False) -> None:
        n_action = 7  # DOF
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.sim = sim
        # self.neutral_joint_values = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78])
        self.neutral_joint_values = np.array([0.0, 0.4, 0.0, -1.78, 0.0, 2.24, 0.77])
        self.orientation_task = orientation_task

    def get_obs(self) -> np.ndarray:
        if self.orientation_task:
            return np.concatenate(
                [
                    np.array(self.get_ee_position()),
                    np.array(self.get_ee_orientation())[:2],
                ]
            )
        else:
            return np.array(self.get_ee_position())

    def reset(self) -> None:
        self.set_joint_neutral()

    @abstractmethod
    def set_action(self, action: np.ndarray) -> Optional[bool]:
        pass

    @abstractmethod
    def set_joint_neutral(self) -> None:
        pass

    @abstractmethod
    def get_ee_position(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_ee_orientation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_ee_velocity(self) -> np.ndarray:
        pass
