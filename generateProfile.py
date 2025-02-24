"""
Generates velocity profile from existing traj_race_cl.csv
"""
import numpy as np
import configparser
import os
import ast
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import vel_acc_profile

# PROJECT STRUCTURE CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# FILE PATHS (hardcoded based on your structure)
PATHS = {
    "trajectory_input": os.path.join(OUTPUTS_DIR, "traj_race_cl.csv"),
    "veh_params": os.path.join(INPUTS_DIR, "f110.ini"),
    "ggv": os.path.join(INPUTS_DIR, "veh_dyn_info", "ggv.csv"),
    "ax_max": os.path.join(INPUTS_DIR, "veh_dyn_info", "ax_max_machines.csv"),
    "output": os.path.join(OUTPUTS_DIR, "traj_race_with_vel.csv")
}

# VEHICLE PARAMETERS SETUP
print("Loading vehicle parameters...")
config = configparser.ConfigParser()
config.read(PATHS["veh_params"])

try:
    veh_params = ast.literal_eval(config.get('GENERAL_OPTIONS', 'veh_params'))
    vel_calc_opts = ast.literal_eval(config.get('GENERAL_OPTIONS', 'vel_calc_opts'))
except (SyntaxError, ValueError) as e:
    raise ValueError("Invalid format in INI file parameters. Ensure Python dictionary syntax is used.") from e

# LOAD TRAJECTORY DATA
print(f"Loading trajectory from {PATHS['trajectory_input']}...")
traj_data = np.genfromtxt(PATHS["trajectory_input"], delimiter=',', skip_header=1)
s_points = traj_data[:, 0]        # s coordinate
raceline = traj_data[:, 1:3]      # x,y coordinates
psi = traj_data[:, 3]             # heading
kappa = traj_data[:, 4]           # curvature

# CALCULATE ELEMENT LENGTHS (WITH ZERO-LENGTH PREVENTION)
print("Calculating path elements...")
el_lengths = np.diff(s_points, prepend=0)
el_lengths[el_lengths == 0] = 1e-6  # Add epsilon to zero-length segments

# LOAD VEHICLE DYNAMICS
print("Importing vehicle dynamics...")
ggv, ax_max_machines = vel_acc_profile.import_veh_dyn_info(
    ggv_import_path=PATHS["ggv"],
    ax_max_machines_import_path=PATHS["ax_max"]
)

# CALCULATE VELOCITY PROFILE
print("Computing velocity profile...")
vx_profile = vel_acc_profile.calc_vel_profile(
    ggv=ggv,
    ax_max_machines=ax_max_machines,
    v_max=veh_params["v_max"],
    kappa=kappa,
    el_lengths=el_lengths,
    closed=True,
    filt_window=None,
    dyn_model_exp=vel_calc_opts["dyn_model_exp"],
    drag_coeff=veh_params["dragcoeff"],
    m_veh=veh_params["mass"]
)

# CALCULATE ACCELERATION PROFILE (FIXED ARRAY LENGTH)
print("Computing acceleration profile...")
vx_profile_cl = np.append(vx_profile, vx_profile[0])
ax_profile = vel_acc_profile.calc_ax_profile(
    vx_profile=vx_profile_cl,
    el_lengths=el_lengths,
    eq_length_output=False
)  # Removed [:-1] slice

# CREATE OUTPUT DATA
print("Assembling final trajectory...")
full_trajectory = np.column_stack((
    s_points,
    raceline,
    psi,
    kappa,
    vx_profile,
    ax_profile
))

# SAVE RESULTS
print(f"Saving results to {PATHS['output']}...")
header = "s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2"
np.savetxt(PATHS["output"], full_trajectory, delimiter=',', header=header, comments='')

# VISUALIZATION
print("Generating velocity profile plot...")
plt.figure(figsize=(12, 6))
norm = Normalize(vmin=vx_profile.min(), vmax=vx_profile.max())
sc = plt.scatter(s_points, vx_profile, c=vx_profile, cmap='plasma', norm=norm, s=10)
plt.colorbar(sc, label='Velocity (m/s)')
plt.plot(s_points, vx_profile, 'w--', alpha=0.5, linewidth=1)
plt.xlabel("Track Distance (m)", fontsize=12)
plt.ylabel("Velocity (m/s)", fontsize=12)
plt.title("Velocity Profile Visualization", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Process completed successfully!")