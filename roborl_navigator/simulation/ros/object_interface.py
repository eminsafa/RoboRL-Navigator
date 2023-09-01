import os
import time

import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, GetModelProperties
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker


class ObjectInterface:
    def __init__(self, robot, scene):
        self.robot = robot
        self.scene = scene
        self.model_paths = {
            "target_object": "target_object.xml",
            "target_normal": "target_normal.xml",
            "obstacle_object": "obstacle_object_base.xml",
            "aim_sphere": "small_aim_sphere.xml",
        }
        self.models = {}
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.object_ids = set()
        self.collision_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=10)
        self.target_id_counter = 0

    def renew_target(self, location, orientation, recall=0):
        recall += 1
        if recall > 3:
            return
        model_name = "target_object"
        prev_object_id = f'target_object_{self.target_id_counter}'
        self.target_id_counter += 1
        object_id = f'target_object_{self.target_id_counter}'
        if prev_object_id in self.object_ids:
            self.gazebo_remove_object(prev_object_id)
            self.object_ids.remove(prev_object_id)
        self.gazebo_create_object(model_name, location, orientation, object_id=object_id)
        self.object_ids.add(object_id)

    def renew_pointer(self, location, pointer_id):
        object_id = f'pointer_{pointer_id}'
        if object_id in self.object_ids:
            self.gazebo_remove_object(object_id)
            self.object_ids.remove(object_id)
        result = self.gazebo_create_object("aim_sphere", location, object_id=object_id)
        if result is not False:
            self.object_ids.add(object_id)

    def renew_obstacle(self, location, size, i: int = 1):
        object_id = f'obstacle_object_{i}'
        if object_id in self.object_ids:
            self.gazebo_remove_object(object_id, object_id)
            self.object_ids.remove(object_id)
        result = self.gazebo_create_object("obstacle_object", location, size=size)
        if result is not False:
            self.object_ids.add(object_id)

    def gazebo_create_object(self, model_name, location, orientation=None, size=None, object_id=None):
        # @todo instead of removing object, check if re-locating possible
        self.retrieve_model(model_name)
        if model_name not in self.models:
            print(f"Model name {model_name} not in self.models cache.")
            return False
        model = self.models[model_name]

        model_state = Pose()
        model_state.position.x = location[0]
        model_state.position.y = location[1]
        model_state.position.z = location[2]

        if orientation is not None:
            quaternion = quaternion_from_euler(
                orientation[0],
                orientation[1],
                orientation[2]
            )
            model_state.orientation.x = quaternion[0]
            model_state.orientation.y = quaternion[1]
            model_state.orientation.z = quaternion[2]
            model_state.orientation.w = quaternion[3]

        if size is not None:
            model = model.format(
                model_name=object_id or model_name,
                origin_x=location[0],
                origin_y=location[1],
                origin_z=location[2],
                size_x=size[0],
                size_y=size[1],
                size_z=size[2],
            )

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model(object_id or model_name, str(model), "", model_state, "world")

    def gazebo_replace_object(self, model_name, location, orientation=None, object_id=None):
        if object_id is not None:
            model_name = object_id
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.pose.position.x = location[0]
        model_state.pose.position.y = location[1]
        model_state.pose.position.z = location[2]
        model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        model_state_pub.publish(model_state)
        input(">>> replaced... OK")

    def gazebo_remove_object(self, model_name, object_id=None):
        if object_id is not None:
            model_name = object_id
        rospy.wait_for_service('/gazebo/delete_model')
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model(model_name)
        time.sleep(0.1)

    def rviz_spawn_table(self):
        p = PoseStamped()
        p.header.frame_id = "panda_link0"
        p.pose.position.x = 0.
        p.pose.position.y = 0.
        p.pose.position.z = -0.05
        self.scene.add_box("table", p, (2.0, 2.0, 0.1))
        return True

    def retrieve_model(self, model_name):
        if model_name not in self.model_paths:
            print(f"Model name ({model_name}) not in self.model_paths")
            return False
        if model_name not in self.models:
            model_path = os.path.join(os.path.dirname(__file__), "assets/" + self.model_paths[model_name])
            self.models[model_name] = open(model_path, "r+").read()

    def gazebo_check_model(self, model_name):
        get_model_properties = rospy.ServiceProxy('/gazebo/get_model_properties', GetModelProperties)
        try:
            response = get_model_properties(model_name)
            if response.success:
                self.object_ids.add(model_name)
                return True
            else:
                return False
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return False
