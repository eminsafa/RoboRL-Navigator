from collections import OrderedDict

import numpy as np

from production.ros_controller import ROSController
import time

from stable_baselines3 import TD3, HerReplayBuffer
import gymnasium as gym
from roborl_navigator.robot.ros_panda_robot import ROSRobot
from roborl_navigator.simulation.ros import ROSSim
from roborl_navigator.utils import distance
from roborl_navigator.environment.env_panda_ros import FrankaROSEnv


env = FrankaROSEnv(
    orientation_task=False,
    custom_reward=False,
    distance_threshold=0.025,
    experiment=True,
)
model = TD3.load(
    '/home/juanhernandezvega/dev/RoboRL-Navigator/models/roborl-navigator/TD3_Bullet_0.05_Threshold_200K/model.zip',
    env=env,
    replay_buffer_class=HerReplayBuffer,
)
sim = ROSSim(orientation_task=False)
robot = ROSRobot(sim=sim, orientation_task=False)
ros_controller = ROSController()
remote_ip = "http://localhost:5000/run"

# Send Request to Contact Graspnet Server
ros_controller.request_graspnet_result(
    path="/home/juanhernandezvega/dev/RoboRL-Navigator/assets/image_captures/data.npy",
    remote_ip=remote_ip,
)
ros_controller.hand_open()
ros_controller.hand_grasp()
exit()
ros_controller.add_collision_object()
# Parse Responded File
# processed_pose = ros_controller.process_grasping_results()
processed_pose = None
if not processed_pose:
    exit()

# Transform Frame to Panda Base
target_pose = ros_controller.transform_camera_to_world(processed_pose)
# Convert Pose to Array
target_pose_array = ros_controller.pose_to_array(target_pose)

print(f"Desired Goal: {target_pose_array[:3]}")

ros_controller.go_to_home_position()
# Go To Trained Starting Point
observation = env.reset(options={"goal": np.array(target_pose_array[:3]).astype(np.float32)})[0]

for _ in range(50):
    action = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(np.array(action[0]).astype(np.float32))
    if terminated or info.get('is_success', False):
        print("Reached destination!")
        break

# Close Gripper
ros_controller.hand_close()



