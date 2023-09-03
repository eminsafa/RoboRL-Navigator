import numpy as np
import pybullet as p
from roborl_navigator.utils import quaternion_to_euler

# roll, pitch, yaw
b = p.getQuaternionFromEuler((0.5, -.7, 1))
print(b)
a = p.getEulerFromQuaternion(b)
print(np.round(a, 2))
a2 = quaternion_to_euler(b)
print(np.round(a2, 2))
