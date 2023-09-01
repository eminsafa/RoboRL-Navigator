from collections import deque
from typing import Optional

import numpy as np

from roborl_navigator.robot.base_robot import Robot
from roborl_navigator.simulation.ros.ros_sim import ROSSim
from roborl_navigator.utils import *

try:
    import moveit_commander
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except ImportError:
    print("ROS Packages are not initialized!")


class ROSRobot(Robot):

    def __init__(self, sim: ROSSim) -> None:
        super().__init__(sim)

        self.group = moveit_commander.MoveGroupCommander("panda_manipulator")
        self.status_queue = deque(maxlen=5)

    def set_action(self, action: np.ndarray) -> Optional[bool]:
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        arm_joint_ctrl = action[:7]
        target_arm_angles = self.get_target_arm_angles(arm_joint_ctrl)

        stuck = None
        result = self.go_to_joint_state(target_arm_angles)
        self.status_queue.append(result)

        if result in [PlannerResult.MOVEIT_ERROR, PlannerResult.COLLISION]:
            stuck = self.stuck_check()
            if not stuck:
                target_arm_angles = self.get_target_arm_angles(arm_joint_ctrl/2)
                result = self.go_to_joint_state(target_arm_angles)

        if result != PlannerResult.SUCCESS:
            if stuck:
                truncated = True
            else:
                truncated = False
        else:
            truncated = False
        return truncated

    def set_joint_neutral(self) -> None:
        self.group.plan(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        pass

    def get_ee_position(self) -> np.ndarray:
        position = self.group.get_current_pose().pose.position
        return np.array(
            [
                position.x,
                position.y,
                position.z,
            ]
        ).astype(np.float32)

    def get_ee_orientation(self) -> np.ndarray:
        orientation = self.group.get_current_pose().pose.orientation
        return np.array(
            euler_from_quaternion(
                [
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ]
            )
        ).astype(np.float32)

    def get_ee_velocity(self) -> np.ndarray:
        return np.zeros(3)

    def get_target_arm_angles(self, joint_actions: np.ndarray) -> np.ndarray:
        joint_actions = joint_actions * 0.05  # @todo limit maximum change in position
        current_arm_joint_angles = np.array(self.get_current_joint_state())
        target_arm_angles = current_arm_joint_angles + joint_actions
        return target_arm_angles

    def get_current_joint_state(self):
        return self.group.get_current_joint_values()

    def go_to_joint_state(self, joint_goal):
        try:
            success, plan, _, _ = self.group.plan(joint_goal)
        except MoveItCommanderException:
            print(f"{ANSI_PURPLE}>>>>> MoveitCommanderException during planning!{ANSI_RESET}")
            return PlannerResult.MOVEIT_ERROR
        if not success:
            print(f"{ANSI_RED}>>>>> Collision Detected!{ANSI_RESET}")
            return PlannerResult.COLLISION
        try:
            self.group.go(joint_goal, True)
        except MoveItCommanderException:
            print(f">>>>> MoveItCommanderException")
            return PlannerResult.MOVEIT_ERROR
        self.group.stop()
        return PlannerResult.SUCCESS

    def stuck_check(self):
        if not self.status_queue:
            return False
        first_value = self.status_queue[0]
        if first_value == PlannerResult.SUCCESS:
            return False
        if len(self.status_queue) < self.status_queue.maxlen:
            return False
        for value in self.status_queue:
            if value != first_value:
                return False
        return True
