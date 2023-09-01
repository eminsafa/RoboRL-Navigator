from roborl_navigator.simulation import Simulation


class ROSSim(Simulation):
    """Convenient class to use PyBullet physics engine."""

    def __init__(
            self,
    ) -> None:
        super().__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_interface', anonymous=True)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()

    def create_sphere(self):
        pass

    def create_plane(self):
        pass

    def create_scene(self):
        pass

    def set_base_pose(self, *args):
        pass

    def place_camera(self, target_position, distance, yaw, pitch):
        pass
