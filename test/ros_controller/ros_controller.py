import time
from collections import deque
from typing import Optional

import numpy as np

from roborl_navigator.utils import distance, PlannerResult
from scipy.spatial.transform import Rotation

try:
    import moveit_commander
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
    from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
except ImportError:
    print("ROS Packages are not initialized!")


class ROSController:
    def __init__(self):
        self.move_group = moveit_commander.MoveGroupCommander("panda_manipulator")
        self.status_queue = deque(maxlen=5)

    def create_pose(self, position: np.ndarray, orientation: Optional[np.ndarray] = None) -> Pose:
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        if orientation is not None:
            pose.orientation.x = orientation[0]
            pose.orientation.y = orientation[1]
            pose.orientation.z = orientation[2]
            pose.orientation.w = orientation[3]
        return pose

    def get_pose_goal_plan_with_duration(self, pose: Pose, planner):
        if planner == 'RRT':
            self.move_group.set_planner_id("RRTConnectkConfigDefault")
        elif planner == 'PRM':
            self.move_group.set_planner_id("PRMstarkConfigDefault")
        self.move_group.set_pose_target(pose)
        start_time = time.time()
        plan = self.move_group.plan()
        end_time = time.time()
        planning_time = round((end_time - start_time)*1000)
        return plan, planning_time

    def execute_plan(self, plan):
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def create_random_pose(self):
        position = np.random.uniform(0.1, 0.3, size=(3,))
        orientation = Rotation.random().as_quat()
        return self.create_pose(position, orientation)
