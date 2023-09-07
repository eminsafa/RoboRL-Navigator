import numpy as np
from math import cos, sin, sqrt, atan2, asin
from typing import List, Union


def euler_to_quaternion(orientation: Union[np.ndarray, List[float]]) -> np.ndarray:
    roll = orientation[0]
    pitch = orientation[1]
    yaw = orientation[2]

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return np.array((qx, qy, qz, qw))


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = quaternion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return np.array((roll_x, pitch_y, yaw_z))


def spherical_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    x1, y1 = point_a
    x2, y2 = point_b

    dlon = x2 - x1
    dlat = y2 - y1

    # Haversine formula
    a = sin(dlat/2)**2 + cos(y1) * cos(y2) * sin(dlon/2)**2
    return 2 * atan2(sqrt(a), sqrt(1 - a))
