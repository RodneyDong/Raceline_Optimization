import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
from scipy.ndimage import gaussian_filter1d

'''
Uses OpenCV to extract contours from the map
Then uses resampling algorithm to make outer and inner boundaries have evenly spaced waypoints and same number of waypoints
Finds the midpoint by iterating through each outer boundary and finds the closest inner boundary waypoint to pair with
Finds the centerline waypoint of these two paired inner and outer boundary waypoints 
Uses Gaussian filtering and resampling algorithm again to smooth out and even out waypoints
'''
MAP_NAME = "Montreal_map"
TRACK_WIDTH_MARGIN = 0.0  # Extra Safety margin, in meters, max 0.5

map_img_path = f"F1tenthTest/maps/{MAP_NAME}.png"
map_yaml_path = f"F1tenthTest/maps/{MAP_NAME}.yaml"
output_csv_path = f"F1tenthTest/outputs/centerlineTest21.csv"
num_interpolated_points = 3000

# Load Map
map_img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)

# Ensure map_img is a valid numeric array
if map_img is None:
    raise FileNotFoundError(f"Map image not found")

# Convert Map to Black and White (Binary)
_, binary_map = cv2.threshold(map_img, 250, 255, cv2.THRESH_BINARY)

# Extract Boundaries using OpenCV
contours, _ = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Ensure we have at least two contours (connected waypoints representing boundaries)
if len(contours) < 2:
    raise ValueError("Not enough contours found to determine both outer and inner boundaries.")

# Sort contours by length (descending)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
outer_contour = sorted_contours[0]  # Largest contour (outer boundary)
inner_contour = sorted_contours[1]  # Second largest contour (inner boundary)

print(f"Found {len(contours)} contours.")  # Debugging: Check number of contours

# Convert contour points into a list of (x, y) waypoints
outer_boundary_waypoints = np.array(outer_contour.squeeze())  # Remove extra dimension
inner_boundary_waypoints = np.array(inner_contour.squeeze())  # Remove extra dimension

# Close the contours by adding the first point to the end
if len(outer_boundary_waypoints) > 0:
    outer_boundary_waypoints = np.vstack([outer_boundary_waypoints, outer_boundary_waypoints[0]])

if len(inner_boundary_waypoints) > 0:
    inner_boundary_waypoints = np.vstack([inner_boundary_waypoints, inner_boundary_waypoints[0]])


# Function to determine track direction (Clockwise or Counterclockwise)
def check_track_direction(midpoints: np.ndarray) -> str:
    """
    Determine if the track direction is clockwise or counterclockwise using the shoelace formula.
    
    :param midpoints: np.ndarray (N, 2) - Track centerline waypoints (x, y)
    :return: "CW" if clockwise, "CCW" if counterclockwise
    """
    x = midpoints[:, 0]
    y = midpoints[:, 1]
    
    # Compute the signed area (shoelace formula)
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    
    return "CCW" if signed_area > 0 else "CW"


# Makes the waypoints evenly spaced with given number of waypoints (Resampling)
def resample_contour(contour, num_points):
    contour = contour.squeeze()
    valid_indices = ~np.isnan(contour).any(axis=1)
    contour = contour[valid_indices]

    if len(contour) < 2:
        raise ValueError("Contour has too few valid points to resample.")

    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cum_dist = np.insert(np.cumsum(distances), 0, 0)
    total_dist = cum_dist[-1]

    if total_dist == 0:
        return np.repeat(contour[0:1], num_points, axis=0)

    new_distances = np.linspace(0, total_dist, num_points)
    new_contour = np.zeros((num_points, 2))

    current_idx = 0
    for i in range(num_points):
        while (current_idx < len(cum_dist) - 1 and cum_dist[current_idx + 1] < new_distances[i]):
            current_idx += 1
        if current_idx >= len(cum_dist) - 1:
            new_contour[i:] = contour[-1]
            break
        t = (new_distances[i] - cum_dist[current_idx]) / distances[current_idx]
        new_contour[i] = contour[current_idx] * (1 - t) + contour[current_idx + 1] * t

    return new_contour


# Resample Inner and Outer Boundaries
interpolated_inner = resample_contour(inner_boundary_waypoints, num_interpolated_points)
interpolated_outer = resample_contour(outer_boundary_waypoints, num_interpolated_points)

# Arrays to store aligned points
aligned_inner, aligned_outer, aligned_midpoints, aligned_width_to_inner, aligned_width_to_outer = [], [], [], [], []

# Calculate midpoints
for outer_point in interpolated_outer:
    distances = np.linalg.norm(interpolated_inner - outer_point, axis=1)
    closest_idx = np.argmin(distances)
    inner_point = interpolated_inner[closest_idx]
    midpoint = (inner_point + outer_point) / 2

    aligned_inner.append(inner_point)
    aligned_outer.append(outer_point)
    aligned_midpoints.append(midpoint)

# Convert lists to NumPy arrays
aligned_midpoints = np.array(aligned_midpoints)
aligned_inner = np.array(aligned_inner)
aligned_outer = np.array(aligned_outer)

# Check track direction
track_direction = check_track_direction(aligned_midpoints)
print(f"Track Direction: {track_direction}")

# Compute width to inner and outer boundaries
aligned_width_to_inner = np.linalg.norm(aligned_midpoints - aligned_inner, axis=1).reshape(-1, 1)
aligned_width_to_outer = np.linalg.norm(aligned_midpoints - aligned_outer, axis=1).reshape(-1, 1)

# If track is clockwise, swap inner/outer widths
if track_direction == "CW":
    print("Track is clockwise, swapping inner/outer boundaries.")
    aligned_width_to_inner, aligned_width_to_outer = aligned_width_to_outer, aligned_width_to_inner

# Merge all into a single array with shape (N, 4)
data = np.concatenate((aligned_midpoints, aligned_width_to_outer, aligned_width_to_inner), axis=1)

# Save CSV Output
np.savetxt(output_csv_path, data, fmt='%0.4f', delimiter=',', header='x_m,y_m,w_tr_right,w_tr_left')

# Visualize the Result
plt.figure(figsize=(10, 10))
plt.title("Aligned Boundaries and Midpoints")
plt.scatter(aligned_inner[:, 0], aligned_inner[:, 1], c='g', s=5, label="Inner Boundary")
plt.scatter(aligned_outer[:, 0], aligned_outer[:, 1], c='r', s=5, label="Outer Boundary")
plt.scatter(aligned_midpoints[:, 0], aligned_midpoints[:, 1], c='b', s=5, label="Aligned Midpoints")  
plt.legend()
plt.axis("equal")
plt.show()
