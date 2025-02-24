import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "F1tenthTest/inputs/foxgloveTest.csv"  # Replace with your CSV file path
trajectory_file = "F1tenthTest/outputs/traj_race_cl.csv"
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
data1 = np.loadtxt(trajectory_file, delimiter=',', skiprows=1)


x_center_opt = data1[:, 1]  # Centerline X
y_center_opt = data1[:, 2]  # Centerline Y

# Extract data columns
x_center = data[:, 0]  # Centerline X
y_center = data[:, 1]  # Centerline Y
width_left = data[:, 2]  # Width to the left
width_right = data[:, 3]  # Width to the right

# Compute tangent vectors and perpendicular directions
tangents = np.diff(np.column_stack((x_center, y_center)), axis=0)
tangents = np.vstack((tangents, tangents[-1]))  # Repeat the last tangent for same length
tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)  # Normalize

# Perpendicular vectors (normal to the tangent)
perpendiculars = np.column_stack((-tangents[:, 1], tangents[:, 0]))

# Calculate inner and outer boundaries
x_inner = x_center - perpendiculars[:, 0] * width_left
y_inner = y_center - perpendiculars[:, 1] * width_left

x_outer = x_center + perpendiculars[:, 0] * width_right
y_outer = y_center + perpendiculars[:, 1] * width_right

# Plotting without connecting points
plt.figure(figsize=(10, 10))

# Plot centerline points
plt.scatter(x_center, y_center, s=1, color="blue")

# Plot inner boundary points
plt.scatter(x_inner, y_inner, s=1, color="green")

# Plot outer boundary points
plt.scatter(x_outer, y_outer, s=1, color="green")

# plt.scatter(x_center_opt, y_center_opt, s=1, color="red", label="Optimized Racing Line")

plt.plot(x_center_opt, y_center_opt, color="red", linewidth=1.5)

# Add legend and labels
plt.legend()
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("Track Visualization: Centerline, Inner, and Outer Boundaries")
plt.axis("equal")  # Maintain aspect ratio
plt.grid(True)

# Save or display the plot
plt.savefig("track_visualization_points.png")  # Save as an image
plt.show()
