from production.ros_controller import ROSController

controller = ROSController()

for i in range(10):
    pose = controller.create_random_pose()
    controller.get_pose_goal_plan(pose, 'RRT')
    controller.get_pose_goal_plan(pose, 'PRM')

