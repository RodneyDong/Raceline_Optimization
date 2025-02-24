import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
from scipy.ndimage import gaussian_filter1d
'''
Uses OpenCV to extract contours from the map
Then uses resampling algorithm to make outer and inner boundaries have evenly spaced waypoints and same number of waypoints
Finds the midpoint by iterating through each outer boundary and finding the closest inner boundary waypoint to pair with
Finds the centerpoint of these two paired inner and outer boundary waypoints and append to centerline waypoint array
Uses gaussion filtering and resampling algorithm again to smooth out and even out waypoints
'''
MAP_NAME = "Spielberg_map"
TRACK_WIDTH_MARGIN = 0.0 # Extra Safety margin, in meters, max 0.5

map_img_path = f"F1tenthTest/maps/{MAP_NAME}.png"
map_yaml_path = f"F1tenthTest/maps/{MAP_NAME}.yaml"
output_csv_path = f"F1tenthTest/outputs/foxgloveTest.csv"
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

print(len(outer_boundary_waypoints))
print(len(inner_boundary_waypoints))

# Close the contours by adding the first point to the end
if len(outer_boundary_waypoints) > 0:
    outer_boundary_waypoints = np.vstack([outer_boundary_waypoints, outer_boundary_waypoints[0]])

if len(inner_boundary_waypoints) > 0:
    inner_boundary_waypoints = np.vstack([inner_boundary_waypoints, inner_boundary_waypoints[0]])

# Function to determine track direction (Clockwise or Counterclockwise)
def check_track_direction(midpoints: np.ndarray) -> str:
    x = midpoints[:, 0]
    y = midpoints[:, 1]
    
    # Compute the signed area (shoelace formula)
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    
    return "CCW" if signed_area > 0 else "CW"

# Makes the waypoints evenly spaced with given number of waypoints (Resampling)
def resample_contour(contour, num_points):
    contour = contour.squeeze()

    # Remove NaN values from input contour
    valid_indices = ~np.isnan(contour).any(axis=1)
    contour = contour[valid_indices]

    if len(contour) < 2:
        raise ValueError("Contour has too few valid points to resample.")

    # Calculate cumulative distance along the contour
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cum_dist = np.insert(np.cumsum(distances), 0, 0)
    total_dist = cum_dist[-1]

    # Handle edge case (stationary contour)
    if total_dist == 0:
        return np.repeat(contour[0:1], num_points, axis=0)

    # Linear interpolation to resample
    new_distances = np.linspace(0, total_dist, num_points)
    new_contour = np.zeros((num_points, 2))

    current_idx = 0
    for i in range(num_points):
        while (current_idx < len(cum_dist) - 1 and
               cum_dist[current_idx + 1] < new_distances[i]):
            current_idx += 1
        if current_idx >= len(cum_dist) - 1:
            new_contour[i:] = contour[-1]
            break
        t = (new_distances[i] - cum_dist[current_idx]) / distances[current_idx]
        new_contour[i] = contour[current_idx] * (1 - t) + contour[current_idx + 1] * t

    # If NaN values appear, replace them with interpolated values
    if np.isnan(new_contour).any():
        print(f"Warning: NaN values detected in resampling. Replacing them.")
        nan_indices = np.where(np.isnan(new_contour).any(axis=1))[0]

        for idx in nan_indices:
            if 0 < idx < len(new_contour) - 1:
                new_contour[idx] = (new_contour[idx - 1] + new_contour[idx + 1]) / 2
            elif idx == 0:
                new_contour[idx] = new_contour[idx + 1]
            elif idx == len(new_contour) - 1:
                new_contour[idx] = new_contour[idx - 1]

    return new_contour

# ==== Resampled Inner and Outer Boundaries ====
num_interpolated_points = 3000  # Adjust the number of points as needed
interpolated_inner = resample_contour(inner_boundary_waypoints, num_interpolated_points)
interpolated_outer = resample_contour(outer_boundary_waypoints, num_interpolated_points)

# 5 Arrays to store aligned points
aligned_inner, aligned_outer, aligned_midpoints, aligned_width_to_inner, aligned_width_to_outer = [], [], [], [], []

# Calculate each pair of midpoint, using outer and inner boundary (Aligned)
for outer_point in interpolated_outer:
    # Calculate distances to all inner boundary points
    distances = np.linalg.norm(interpolated_inner - outer_point, axis=1)
    
    # Find the index of the closest inner boundary point
    closest_idx = np.argmin(distances)
    
    # Get the corresponding inner boundary point
    inner_point = interpolated_inner[closest_idx]
    
    # Compute the midpoint
    midpoint = (inner_point + outer_point) / 2

    # # Compute the width from midpoint to inner and outer boundaries
    # width_to_inner = np.linalg.norm(midpoint - inner_point)
    # width_to_outer = np.linalg.norm(midpoint - outer_point)

    # # Store the computed widths
    # aligned_width_to_inner.append(width_to_inner)
    # aligned_width_to_outer.append(width_to_outer)

    aligned_inner.append(inner_point)
    aligned_outer.append(outer_point)
    aligned_midpoints.append(midpoint)

def smooth_closed_contour(contour, sigma):
    padded_contour = np.vstack([contour[-sigma:], contour, contour[:sigma]])
    smoothed_contour = gaussian_filter1d(padded_contour, sigma=sigma, axis=0)
    smoothed_contour = smoothed_contour[sigma:-sigma]  # Remove padding
    smoothed_contour[-1] = smoothed_contour[0]  # Ensure closure
    return smoothed_contour


sigma_value = 10
aligned_inner = smooth_closed_contour(aligned_inner, sigma=sigma_value)
aligned_outer = smooth_closed_contour(aligned_outer, sigma=sigma_value)
aligned_midpoints = smooth_closed_contour(aligned_midpoints, sigma=sigma_value)

# Resample and Convert aligned lists to NumPy arrays
aligned_inner = resample_contour(np.array(aligned_inner), num_interpolated_points)
aligned_outer = resample_contour(np.array(aligned_outer), num_interpolated_points)
aligned_midpoints = resample_contour(np.array(aligned_midpoints), num_interpolated_points)

np.nan_to_num(aligned_inner)

print(f"NaN in aligned_inner: {np.isnan(aligned_inner).sum()}")
print(f"NaN in aligned_outer: {np.isnan(aligned_outer).sum()}")
print(f"NaN in aligned_midpoints: {np.isnan(aligned_midpoints).sum()}")

# Compute width separately after midpoints are generated
for mid_point in aligned_midpoints:
    # Compute distances to all inner boundary points
    distances_inner = np.linalg.norm(aligned_inner - mid_point, axis=1)
    
    # Compute distances to all outer boundary points
    distances_outer = np.linalg.norm(aligned_outer - mid_point, axis=1)

    # Find the index of the closest inner and outer boundary points
    closest_inner_idx = np.argmin(distances_inner)
    closest_outer_idx = np.argmin(distances_outer)

    # Retrieve the closest inner and outer boundary points
    inner_point = aligned_inner[closest_inner_idx]
    outer_point = aligned_outer[closest_outer_idx]

    # Compute the width from midpoint to inner and outer boundaries
    width_to_inner = np.linalg.norm(mid_point - inner_point)
    width_to_outer = np.linalg.norm(mid_point - outer_point)

    # Append the computed widths
    aligned_width_to_inner.append(width_to_inner)
    aligned_width_to_outer.append(width_to_outer)

# Convert to NumPy arrays
aligned_width_to_inner = np.array(aligned_width_to_inner).reshape(-1, 1)
aligned_width_to_outer = np.array(aligned_width_to_outer).reshape(-1, 1)

# Flip the y-coordinates to match the correct orientation
aligned_inner[:, 1] *= -1
aligned_outer[:, 1] *= -1
aligned_midpoints[:, 1] *= -1


print(aligned_inner.shape)
print(aligned_outer.shape)
print(aligned_midpoints.shape)
print(aligned_width_to_inner.shape)
print(aligned_width_to_outer.shape)

# Visualize the result
plt.figure(figsize=(10, 10))
plt.title("Aligned Boundaries and Midpoints")
plt.scatter(aligned_inner[:, 0], aligned_inner[:, 1], c='g', s=5, label="Interpolated Inner Boundary")
plt.scatter(aligned_outer[:, 0], aligned_outer[:, 1], c='r', s=5, label="Interpolated Outer Boundary")
plt.scatter(aligned_midpoints[:, 0], aligned_midpoints[:, 1], c='b', s=5, label="Aligned Midpoints")  # Small blue dots
plt.legend()
plt.axis("equal")
plt.show()

data =[]

if check_track_direction(aligned_midpoints) == "CW":
    # Merge all into a single array with shape (N, 4)
    data = np.concatenate((aligned_midpoints, aligned_width_to_outer, aligned_width_to_inner), axis=1)
else:
    data = np.concatenate((aligned_midpoints, aligned_width_to_inner, aligned_width_to_outer), axis=1)

# load map yaml
with open(map_yaml_path, 'r') as yaml_stream:
    try:
        map_metadata = yaml.safe_load(yaml_stream)
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']
    except yaml.YAMLError as ex:
        print(ex)

# calculate map parameters
orig_x = origin[0]
orig_y = origin[1]
# ??? Should be 0
orig_s = np.sin(origin[2])
orig_c = np.cos(origin[2])

# get the distance transform
transformed_data = data
transformed_data *= map_resolution
transformed_data += np.array([orig_x, orig_y, 0, 0])

# Safety margin
transformed_data -= np.array([0, 0, TRACK_WIDTH_MARGIN, TRACK_WIDTH_MARGIN])

with open(output_csv_path, 'wb') as fh:
    np.savetxt(fh, transformed_data, fmt='%0.4f', delimiter=',', header='x_m,y_m,width_inner,width_outer')