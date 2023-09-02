import numpy as np
from scipy.spatial.transform import Rotation

def get_radial_distance(euler1, euler2):
    rotation1 = Rotation.from_euler('XYZ', euler1, degrees=True)
    rotation2 = Rotation.from_euler('XYZ', euler2, degrees=True)
    quaternion1 = rotation1.as_quat()
    quaternion2 = rotation2.as_quat()
    dot_product = np.dot(quaternion1, quaternion2)
    geodesic_distance = 2 * np.arccos(np.abs(dot_product))

    return geodesic_distance


mind = 1000
maxd = 0
total = 0
count = 100_000
for i in range(count):
    min_angle = -180
    max_angle = 180
    # Generate random Euler angle arrays
    euler1 = np.random.uniform(min_angle, max_angle, 3)
    euler2 = np.random.uniform(min_angle, max_angle, 3)

    d = get_radial_distance(euler1, euler2)

    if d > maxd:
        maxd = d
    if d < mind:
        mind = d
    total += d

mean = total / count

print(f"mean = {mean}, max: {maxd}, min: {mind}")

