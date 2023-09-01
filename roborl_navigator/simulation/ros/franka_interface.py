from math import pi
import numpy as np
import rospy
import sys

try:
    import moveit_commander
    from moveit_commander import MoveItCommanderException
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except Exception as e:
    print(f"FATAL: {e}")

from .franka_wrapper import FrankaWrapper
from .object_interface import ObjectInterface
from est.utils.ansi_colors import (
    ANSI_PURPLE,
    ANSI_RED,
    ANSI_RESET,
)
from est.utils.enums import PlannerResult

# @Todo stuck location checker, compare last 3 location
# @Todo check if realtime factor update by python is possible


class FrankaInterface:

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_interface', anonymous=True)
        self.group = moveit_commander.MoveGroupCommander("panda_manipulator")
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()

        self.joint_limits = (
            [-2.87, 2.87],
            [-1.74, 1.74],
            [-2.87, 2.87],
            [-3.05, 0.04],
            [-2.87, 2.87],
            [0.00, 3.73],
            [-2.87, 2.87],
        )  # 0.02 tolerance added
        self.joint_defaults = (
            0.0,
            -0.78,
            0.0,
            -2.35,
            0.0,
            1.57,
            0.78,
        )
        self.action_coef = 10.0

        self.goal_location = np.array([0, 0, 0])
        self.goal_orientation = np.array([-pi, -pi, -pi])

        # Helpers
        self.wrapper = FrankaWrapper()
        self.object_interface = ObjectInterface(self.robot, self.scene)
        self.object_interface.rviz_spawn_table()

        self.gripper_position = None  # np.array([2, 2, 2])
        self.gripper_orientation = None  # np.array([pi, pi, pi])
        self.joint_state = None

    def reset_robot(self, goal_location, goal_orientation):
        self.goal_location = goal_location
        self.goal_orientation = goal_orientation
        self.unset_values()
        self._go_to_joint_state(self.joint_defaults)
        self.object_interface.renew_target(goal_location, goal_orientation)
        self.update_values()
        # self.spawn_obstacle(1)
        # self.spawn_obstacle(2)
        # self.spawn_obstacle(3)
        return self.get_observation()

    def move(self, action):
        self.unset_values()  # to ensure
        current_joint_state = self.get_current_joint_state()
        joint_goal = self.get_next_joint_state(current_joint_state, action)
        result = self._go_to_joint_state(joint_goal)
        self.update_values()
        return result

    def move_bullet(self, action):
        self.unset_values()  # to ensure
        current_joint_state = self.get_current_joint_state()
        mapped_joints = self.wrapper.real_to_bullet(current_joint_state)
        joint_goal = self.get_next_joint_state_bullet(mapped_joints, action)
        result = self._go_to_joint_state(joint_goal)
        self.update_values()
        return result

    def get_observation(self):
        return np.concatenate((
            # self.wrapper.joint_zip(self.joint_state),  # 7 Joint
            self.gripper_position + [0, 0, 0.05],  # 3 Position
            self.gripper_orientation,  # 3 Orientation
        ))

    def get_position_distance(self):
        return float(
            np.linalg.norm(
                self.goal_location - self.gripper_position
            )
        )

    def get_orientation_distance(self):
        return float(
            np.linalg.norm(
                self.goal_orientation - self.gripper_orientation
            )
        )

    def get_gripper_position(self):
        position = self.group.get_current_pose().pose.position
        return np.array(
            [
                round(position.x, 5),
                round(position.y, 5),
                round(position.z + 0.72, 5),
            ]
        )

    def get_gripper_orientation(self):
        orientation = self.group.get_current_pose().pose.orientation
        return np.array(
            euler_from_quaternion(
                [
                    round(orientation.x, 5),
                    round(orientation.y, 5),
                    round(orientation.z, 5),
                    round(orientation.w, 5),
                ]
            )
        )

    def get_next_joint_state(self, current, action):
        def rate(jid: int, a: float, c: float) -> float:
            j_min, j_max = self.joint_limits[jid]
            t = abs(j_min) + abs(j_max) / 2.0  # computation mistake
            m = t / self.action_coef
            diff = m * a
            return round(max(j_min, min(j_max, c + diff)), 5)

        if type(action) in (tuple, list, set):
            size = len(action)
        else:
            size = action.size

        return [
            rate(jid, action[jid], current[jid])
            for jid in range(size)
        ]

    def get_next_joint_state_bullet(self, current, action):
        def rate(jid: int, a: float, c: float) -> float:
            j_min, j_max = self.wrapper.bullet_panda_limits[jid]
            result = c + a * 0.05
            return max(j_min, min(j_max, result))

        if type(action) in (tuple, list, set):
            size = len(action)
        else:
            size = action.size

        bullet_joints = [
            rate(jid, action[jid], current[jid])
            for jid in range(size)
        ]
        # print(f"Ros (bullet calculated:)\n{bullet_joints}")
        return self.wrapper.bullet_to_real(
            bullet_joints
        )

    def get_current_joint_state(self):
        return self.group.get_current_joint_values()

    def _go_to_joint_state(self, joint_goal):
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
        # self.group.go(joint_goal, wait=True)
        self.group.stop()
        return PlannerResult.SUCCESS
        # @todo check this function: all_close(joint_goal, current_joints, 0.01)

    @staticmethod
    def _get_obstacle_distance():
        # Not Implemented Yet
        return np.array([-2, -2, -2])

    def spawn_obstacle(self, i):
        desk_height = 0.72
        px = np.random.uniform(-0.359, 0.118, size=1)[0]
        py = np.random.uniform(0.118, 0.81, size=1)[0]
        size = np.random.uniform(0.05, 0.2, size=3)
        pz = size[2] / 2.0 + desk_height
        self.object_interface.renew_obstacle([px, py, pz], size, i)

    def unset_values(self):
        self.gripper_position = None
        self.gripper_orientation = None
        self.joint_state = None

    def update_values(self):
        self.gripper_position = self.get_gripper_position()
        self.gripper_orientation = self.get_gripper_orientation()
        self.joint_state = self.get_current_joint_state()