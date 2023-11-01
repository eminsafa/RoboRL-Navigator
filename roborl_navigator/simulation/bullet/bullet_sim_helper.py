import numpy as np

from .bullet_sim import p


class BulletSimHelper:
    
    def __init__(self, bullet_sim):
        self.sim = bullet_sim
        # self.sim.set_base_pose("shelf", np.array([self.shelf_distance, 0, 0]), euler_to_quaternion([0, 0, math.pi/2]))
        self.shelf_width = .3
        self.shelf_depth = .2
        self.shelf_height = .5
        self.shelf_thickness = .01
        self.shelf_bodies = {"shelf_bottom", "shelf_middle", "shelf_top", "shelf_left", "shelf_right"}

    def create_shelf(self, x, y, orientation=None):
        color = np.array([.75, .6, .41, 1])
        # horizontal: bottom, middle, top
        horizontal_size = np.array([self.shelf_depth / 2, self.shelf_width / 2, self.shelf_thickness / 2])
        self.sim.create_box(
            body_name="shelf_bottom",
            half_extents=horizontal_size,
            position=np.array([x, 0, y]),
            rgba_color=color,
        )
        self.sim.create_box(
            body_name="shelf_middle",
            half_extents=horizontal_size,
            position=np.array([x, 0, y + self.shelf_height/2]),
            rgba_color=color,
        )
        self.sim.create_box(
            body_name="shelf_top",
            half_extents=horizontal_size,
            position=np.array([x, 0, y + self.shelf_height]),
            rgba_color=color,
        )
        # vertical: right, left
        horizontal_size = np.array([self.shelf_depth / 2, self.shelf_thickness / 2, self.shelf_height / 2])
        self.sim.create_box(
            body_name="shelf_left",
            half_extents=horizontal_size,
            position=np.array([x, -self.shelf_width/2, y + self.shelf_height / 2]),
            rgba_color=color,
        )
        self.sim.create_box(
            body_name="shelf_right",
            half_extents=horizontal_size,
            position=np.array([x, self.shelf_width / 2, y + self.shelf_height / 2]),
            rgba_color=color,
        )

    def check_collisions(self):
        for body in self.shelf_bodies:
            contacts = p.getContactPoints(bodyA=self.sim.bodies_idx['panda'], bodyB=self.sim.bodies_idx[body])
            if contacts:  # If the list is not empty, there's a collision
                print(f">>> Panda and {body} are in contact!")
        