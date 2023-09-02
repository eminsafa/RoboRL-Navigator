import math
import time

import moveit_commander
import numpy as np
import rospy
import requests

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from PIL import Image as PILImage
from sensor_msgs.msg import CameraInfo, Image
from tf import TransformListener
from roborl_navigator.utils import get_assets_path


# @todo manual pipeline will be converted to prediction based pipeline!
class PandaPipeline:

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.rgb_array = None
        self.depth_array = None
        self.camera_info = None
        self.save_dir = get_assets_path(['image_captures'])

        rospy.init_node("panda_pipeline", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.tf_listener = TransformListener()
        self.move_group = moveit_commander.MoveGroupCommander("panda_manipulator")
        self.hand_group = moveit_commander.MoveGroupCommander("panda_hand")
        self.ee_link = self.move_group.get_end_effector_link()
        self.box_name = "yum_yum_link_0"
        self.latest_capture_path = None
        self.latest_grasp_result_path = None
        self.graspnet_url = "http://localhost:5000/read_file?path={path}"

    def rgb_callback(self, msg):
        self.rgb_array = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        time.sleep(1)

    def depth_callback(self, msg):
        depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_array = np.array(depth_img, dtype=np.dtype("f8"))
        time.sleep(1)

    def camera_info_callback(self, msg):
        cam_info = msg.K
        self.camera_info = np.array(
            [
                [cam_info[0], 0.0, cam_info[2]],
                [0.0, cam_info[4], cam_info[5]],
                [0.0, 0.0, 0.0],
            ]
        )
        time.sleep(1)

    def go_to_capture_location(self):
        self.move_group.go(
            [0, -1.571, 0, -2.688, 0, 1.728, 0.7854]
            , wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def hand_open(self):
        self.hand_group.set_joint_value_target(self.hand_group.get_named_target_values("open"))
        self.hand_group.go(wait=True)

    def hand_close(self):
        self.hand_group.set_joint_value_target(self.hand_group.get_named_target_values("close"))
        self.hand_group.go(wait=True)

    def capture_image_and_save_info(self):
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.sleep(5)
        data_dict = {
            "rgb": np.array(self.rgb_array),
            "depth": np.array(self.depth_array) / 1000.0,
            "label": np.zeros((720, 1280), dtype=np.uint8),
            "K": self.camera_info
        }
        main_save_path = self.save_dir + "est"
        np.save(main_save_path, data_dict)
        np.save(self.save_dir + "rgb.npy", np.array(self.rgb_array))
        np.save(self.save_dir + "depth.npy", np.array(self.depth_array) / 1000.0)
        self.latest_capture_path = main_save_path + '.npy'
        print("Data saved on", self.latest_capture_path)
        return self.latest_capture_path

    def view_image(self):
        file_path = self.save_dir + 'rgb.npy'
        data = np.load(file_path)
        image = PILImage.fromarray(data)
        image.show()

    def request_graspnet_result(self, path=None):
        if path is None:
            if self.latest_capture_path is None:
                return None
            path = self.latest_capture_path

        response = requests.get(self.graspnet_url.format(path=path))
        print(f"Response Text: {response.text}")
        self.latest_grasp_result_path = response.text
        return response.text

    def transform_grasp_results(self, path=None):
        if path is None:
            if self.latest_grasp_result_path is None:
                return None
            path = self.latest_grasp_result_path

        data = np.load(path, allow_pickle=True)
        maxy = 0
        argmax = 0
        # for i in range(len(data['contact_pts'].item()[-1])):
        #    if data['contact_pts'].item()[-1][i][2] > maxy:
        #        argmax = i
        argmax = data['scores'].item()[-1].argmax()
        # argmax = np.random.uniform(0, len(data['contact_pts'].item()[-1]))
        # argmax = int(argmax)
        print(f"ARGMAX: {argmax}")
        pred_grasp = data['pred_grasps_cam'].item()[-1][argmax]
        contact_pts = data['contact_pts'].item()[-1][argmax]

        orientation = np.array((
            math.atan2(pred_grasp[2][1], pred_grasp[2][2]),
            math.asin(-pred_grasp[2][0]),
            math.atan2(pred_grasp[1][0], pred_grasp[0][0]),
        ))
        position = np.array((
            pred_grasp[0][3],
            pred_grasp[1][3],
            pred_grasp[2][3],
        ))
        result = np.concatenate((
            position,
            orientation,
        ))
        return result

    @staticmethod
    def rotation_matrix_to_quaternion(transformation_matrix):
        matrix = transformation_matrix[:3, :3]
        w = math.sqrt(1.0 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2.0
        x = (matrix[2, 1] - matrix[1, 2]) / (4.0 * w)
        y = (matrix[0, 2] - matrix[2, 0]) / (4.0 * w)
        z = (matrix[1, 0] - matrix[0, 1]) / (4.0 * w)
        return np.array((x, y, z, w))

    def go_to_pose_goal(self, pose):
        self.move_group.set_pose_target(pose)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def transform_camera_to_world(self, cv_pose):
        from_link = "camera_depth_optical_frame"
        to_link = "world"

        base_pose = PoseStamped()
        quaternion = self.get_quaternion_from_euler(
            np.double(cv_pose[3]), np.double(cv_pose[4]), np.double(cv_pose[5])
        )
        base_pose.pose.position.x = cv_pose[0]
        base_pose.pose.position.y = cv_pose[1]
        base_pose.pose.position.z = cv_pose[2]
        base_pose.pose.orientation.x = quaternion[0]
        base_pose.pose.orientation.y = quaternion[1]
        base_pose.pose.orientation.z = quaternion[2]
        base_pose.pose.orientation.w = quaternion[3]
        base_pose.header.frame_id = from_link

        result = self.tf_listener.transformPose(to_link, base_pose)
        return result

    @staticmethod
    def clear_pose(pose):
        new_pose = Pose()
        new_pose.position.x = pose.pose.position.x
        new_pose.position.y = pose.pose.position.y
        new_pose.position.z = pose.pose.position.z
        new_pose.orientation.w = pose.pose.orientation.w
        new_pose.orientation.x = pose.pose.orientation.x
        new_pose.orientation.y = pose.pose.orientation.y
        new_pose.orientation.z = pose.pose.orientation.z
        return new_pose

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return x, y, z, w
