import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Heading angle in degrees (replace with your desired heading)
heading_degrees = 30.0

# Convert heading angle to radians
heading_rad = np.radians(heading_degrees)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Origin of the vector (0, 0, 0)
origin = np.array([0, 0, 0])

# Vector direction
vector_direction = np.array([np.cos(heading_rad), np.sin(heading_rad), 0])

# Plot the heading vector
ax.quiver(*origin, *vector_direction, color='b', label=f'Heading {heading_degrees}°')

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot title
ax.set_title('Heading Vector Visualization')

# Add a legend
ax.legend()

# Show the plot
plt.show()
