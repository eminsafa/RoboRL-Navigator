import time
from ros_controller.ros_controller import ROSController


ros_controller = ROSController(real_robot=True)
time.sleep(3)
remote_ip = "http://172.20.10.10:6161/run"

ros_controller.capture_image_and_save_info()
ros_controller.view_image()

# Send Request to Contact Graspnet Server
saved_file_path = ros_controller.request_graspnet_result(remote_ip=remote_ip)
# Parse Responded File
target_pose_by_camera = ros_controller.process_grasping_results(path=saved_file_path)

print(target_pose_by_camera)

