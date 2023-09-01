import numpy as np
from math import pi


class FrankaWrapper:

    def __init__(self):
        # Joint Degree Wrapper
        self.joint_limits = (
            [-2.89, 2.89],
            [-1.76, 1.76],
            [-2.89, 2.89],
            [-3.07, 0.06],
            [-2.89, 2.89],
            [-0.02, 3.75],
            [-2.89, 2.89],
        )
        self.joint_observation_range = (-10.0, 10.0)

        # Location Wrapper
        self.location_limits = (
            [-1.5, 1.5],
            [-1.5, 1.5],
            [0.5, 2.0],
        )
        self.location_observation_range = (-10.0, 10.0)

        # Orientation Wrapper
        self.orientation_limits = (
            [-pi, pi],
            [-pi, pi],
            [-pi, pi],
        )
        self.orientation_observation_range = (-10.0, 10.0)

        # Obstacle Distance Wrapper
        self.obs_distance_limit = (
            [.0, 1.0],
        )
        self.obs_distance_observation_range = (-10.0, 10.0)

        # Target Position Distance Wrapper
        self.target_position_distance_limit = (
            [.0, 2.0],
        )
        self.target_position_distance_observation_range = (.0, 1.0)

        # Target Position Distance Wrapper
        self.target_orientation_distance_limit = (
            [.0, 2.0],
        )
        self.target_position_distance_observation_range = (.0, 1.0)

        # Bullet Simulation Franka Limits
        self.bullet_panda_limits = (
            [-2.967, 2.967],
            [-1.83, 1.83],
            [-2.967, 2.967],
            [-3.14, 0.0],
            [-2.967, 2.967],
            [-0.087, 3.822],
            [-2.967, 2.967],
        )

    def map(self, value, from_min, from_max, to_min, to_max, round_decimal=2):
        clamped_value = max(from_min, min(value, from_max))
        return round(((clamped_value - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min, round_decimal)

    def zip(self, values, limits, obs_range):
        return np.array([
            max(obs_range[0], min(obs_range[1], self.map(v, limits[i][0], limits[i][1], obs_range[0], obs_range[1])))
            for i, v in enumerate(values)
        ])

    def unzip(self, values, limits, obs_range):
        return np.array([
            self.map(v, obs_range[0], obs_range[1], limits[i][0], limits[i][1])
            for i, v in enumerate(values)
        ])

    def joint_zip(self, joint_degrees):
        return self.zip(joint_degrees, self.joint_limits, self.joint_observation_range)

    def joint_unzip(self, zipped_degrees):
        return self.unzip(zipped_degrees, self.joint_limits, self.joint_observation_range)

    def location_zip(self, location):
        return self.zip(location, self.location_limits, self.location_observation_range)

    def location_unzip(self, zipped_locations):
        return self.unzip(zipped_locations, self.location_limits, self.location_observation_range)

    def orientation_zip(self, orientations):
        return self.zip(orientations, self.orientation_limits, self.orientation_observation_range)

    def orientation_unzip(self, zipped_orientations):
        return self.unzip(zipped_orientations, self.orientation_limits, self.orientation_observation_range)

    def distance_zip(self, distance: float) -> float:
        return self.zip([distance], self.obs_distance_limit, self.obs_distance_observation_range)[0]

    def distance_unzip(self, zipped_distance: float) -> float:
        return self.unzip([zipped_distance], self.obs_distance_limit, self.obs_distance_observation_range)[0]

    def obstacle_distances_zip(self, distances: np.ndarray) -> np.ndarray:
        """
        For Array with multiple distances
        """
        limits = [self.obs_distance_limit[0] for _ in range(distances.size)]
        return self.zip(distances, limits, self.obs_distance_observation_range)

    def obstacle_distances_unzip(self, zipped_distances: np.ndarray) -> np.ndarray:
        """
        For Array with multiple distances
        """
        limits = [self.obs_distance_limit[0] for _ in range(zipped_distances.size)]
        return self.unzip(zipped_distances, limits, self.obs_distance_observation_range)

    def target_distance_zip(self, target_distance: float) -> float:
        return self.zip(
            [target_distance],
            self.target_position_distance_limit,
            self.target_position_distance_observation_range
        )[0]

    def target_distance_unzip(self, zipped_distance: float) -> float:
        return self.unzip(
            [zipped_distance],
            self.target_position_distance_limit,
            self.target_position_distance_observation_range
        )[0]

    @staticmethod
    def map_value(value, from_range, to_range):
        from_min, from_max = from_range
        to_min, to_max = to_range

        # Ensure the value is within the from_range
        value = max(from_min, min(from_max, value))

        # Map the value to the new range
        mapped_value = ((value - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
        return mapped_value

    def bullet_to_real(self, joint_values):
        mapped_values = []
        for input_value, from_range, to_range in zip(joint_values, self.bullet_panda_limits, self.joint_limits):
            mapped_value = self.map_value(input_value, from_range, to_range)
            mapped_values.append(mapped_value)
        return np.array(mapped_values)

    def real_to_bullet(self, joint_values):
        mapped_values = []
        for input_value, from_range, to_range in zip(joint_values, self.joint_limits, self.bullet_panda_limits):
            mapped_value = self.map_value(input_value, from_range, to_range)
            mapped_values.append(mapped_value)
        return np.array(mapped_values)
