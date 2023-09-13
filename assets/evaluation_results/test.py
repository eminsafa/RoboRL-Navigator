import numpy as np
from math import pi
import matplotlib.pyplot as plt
from roborl_navigator.utils import euler_to_quaternion


def quaternion_difference(q1, q2):
    dot_product = np.dot(q1, q2)
    angular_diff = 2 * np.arccos(abs(dot_product))
    return angular_diff


def random_euler_angles():
    roll = np.random.uniform(-pi, pi)
    pitch = np.random.uniform(-pi/2, pi/2)
    yaw = np.random.uniform(-pi, pi)
    return [roll, pitch, yaw]


results = []

for i in range(100_000):
    q1 = euler_to_quaternion(random_euler_angles())
    q2 = euler_to_quaternion(random_euler_angles())

    difference = quaternion_difference(q1, q2)
    results.append(difference)

# Example array of "difference" results (replace this with your actual data)
# difference_array = np.random.normal(0, 1, 1000)  # Replace with your data or generate data
results = np.array(results)
mean = np.mean(results)
print(f"MEAN: {mean}")
# Create a histogram
plt.hist(results, bins=20, color='b', alpha=0.6, edgecolor='k')

# Create a bar plot

# Add labels and title
plt.xlabel('Difference')
plt.ylabel('Count')
plt.title('Frequency of Orientation Difference')
plt.axvline(x=mean, color='r', linestyle='--', label='Mean')

# Show the plot
plt.legend()
plt.show()
