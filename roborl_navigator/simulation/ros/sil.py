import random

import rospy
import time
import moveit_commander
from roborl_navigator.utils import euler_to_quaternion, quaternion_to_euler
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState

class Getting:

    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def move(self, body, position):
        rospy.wait_for_service('/gazebo/set_model_state')
        state_msg = ModelState()
        state_msg.model_name = body
        state_msg.pose.position.x = position[0]
        state_msg.pose.position.y = position[1]
        state_msg.pose.position.z = position[2]
        state_msg.pose.orientation.x = orientation[0]
        state_msg.pose.orientation.y = orientation[1]
        state_msg.pose.orientation.z = orientation[2]
        state_msg.pose.orientation.w = 0.0
        self.set_model_state_proxy(state_msg)
        rospy.wait_for_service('/gazebo/set_model_state')

gt = Getting()
random_location = [0.8, - random.randint(0, 10) / 10.0, random.randint(2, 10) / 10.0]
for i in range(50):
    gt.move("target", random_location)


