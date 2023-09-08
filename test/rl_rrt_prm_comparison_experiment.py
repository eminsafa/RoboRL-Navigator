import time

from stable_baselines3 import TD3, HerReplayBuffer
import gymnasium as gym
from production.ros_controller import ROSController


ros_controller = ROSController()
env = gym.make(
    "RoboRL-Navigator-Franka-ROS",
    orientation_task=False,
    custom_reward=False,
    distance_threshold=0.05)

model = TD3.load(
    '/home/juanhernandezvega/dev/RoboRL-Navigator/models/roborl-navigator/TD3_Bullet_0.05_Threshold_200K/model.zip',
    env=env,
    replay_buffer_class=HerReplayBuffer,
)

observation = model.env.reset()

model.predict(observation)  # to initialize

results = {}

for episode in range(1_000):
    results[episode] = {}
    target_position = observation["desired_goal"][:3][0]
    pose = ros_controller.create_pose(target_position)

    results[episode]['rrt'] = ros_controller.get_pose_goal_plan_with_duration(pose, "RRT")[1]
    results[episode]['prm'] = ros_controller.get_pose_goal_plan_with_duration(pose, "PRM")[1]

    rl_episode_total = 0.0
    for _ in range(50):
        start_time = time.time()
        action = model.predict(observation)
        end_time = time.time()
        planning_time = round((end_time - start_time)*1000)
        rl_episode_total += planning_time
        action = action[0]
        observation, reward, terminated, info = model.env.step(action)
        model.env.render()
        success = info[0].get('is_success', False)
        if terminated or success:
            print(f"RL EPISODE TOTAL: {rl_episode_total}")
            results[episode]['rl'] = rl_episode_total
            time.sleep(3)
            break
    observation = model.env.reset()
    print(results[episode])

print("\n\n\n")
print(results)
print("\n\n\n")
