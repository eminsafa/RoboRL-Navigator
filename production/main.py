from .pipeline import PandaPipeline

fpl = PandaPipeline()

# Open the gripper
fpl.hand_open()
# Go to Image Capturing Location
fpl.go_to_capture_location()
# Save image, depth data and camera info
fpl.capture_image_and_save_info()
# View image
# fpl.view_image()

# Send Request to Contact Graspnet Server
fpl.request_graspnet_result()
# Parse Responded File
raw_pose = fpl.transform_grasp_results()
# Transform Frame to Panda Base
pose = fpl.transform_camera_to_world(raw_pose)
pose = fpl.clear_pose(pose)
# Go to target (grasping) pose
fpl.go_to_pose_goal(pose)

checkpoint = input("Ready to grasp? y: yes")

if checkpoint == "y":
    fpl.hand_close()
