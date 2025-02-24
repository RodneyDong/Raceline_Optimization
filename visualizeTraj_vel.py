import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Load files
track_file = "F1tenthTest/inputs/foxgloveTest.csv"          # Original track
traj_file = "F1tenthTest/outputs/traj_race_with_vel.csv"    # Trajectory WITH velocity

# Load data
track_data = np.loadtxt(track_file, delimiter=',', skiprows=1)
traj_data = np.loadtxt(traj_file, delimiter=',', skiprows=1)

# Extract racing line data WITH VELOCITY
x_race = traj_data[:, 1]    # x coordinates
y_race = traj_data[:, 2]    # y coordinates
velocities = traj_data[:, 5]  # Velocity values (vx_mps)

# Extract track boundaries
x_center = track_data[:, 0]
y_center = track_data[:, 1]
width_left = track_data[:, 2]
width_right = track_data[:, 3]

# Compute track boundaries (same as before)
tangents = np.diff(np.column_stack((x_center, y_center)), axis=0)
tangents = np.vstack((tangents, tangents[-1])) 
tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
perpendiculars = np.column_stack((-tangents[:, 1], tangents[:, 0]))

x_inner = x_center - perpendiculars[:, 0] * width_left
y_inner = y_center - perpendiculars[:, 1] * width_left
x_outer = x_center + perpendiculars[:, 0] * width_right
y_outer = y_center + perpendiculars[:, 1] * width_right

# Plot
plt.figure(figsize=(10, 10))

# Plot track boundaries
# plt.scatter(x_center, y_center, s=1, color="blue", label="Centerline")
plt.scatter(x_inner, y_inner, s=1, color="green", label="Track Edges")
plt.scatter(x_outer, y_outer, s=1, color="green")

# Color-code racing line by velocity
norm = Normalize(vmin=np.min(velocities), vmax=np.max(velocities))
plt.scatter(
    x_race, y_race, 
    c=velocities,          # Color values
    cmap='plasma',         # Colormap (try 'viridis' or 'inferno')
    s=8,                   # Point size
    norm=norm, 
    edgecolors='none', 
    label="Velocity Profile"
)

# Add colorbar
cbar = plt.colorbar(label='Velocity (m/s)')
cbar.set_label('Velocity (m/s)', fontsize=12)

# Finalize plot
plt.xlabel("X (meters)", fontsize=12)
plt.ylabel("Y (meters)", fontsize=12)
plt.title("Racing Line with Velocity Coloring", fontsize=14)
plt.axis("equal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()