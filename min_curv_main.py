import numpy as np
import configparser
import os
import ast
import matplotlib.pyplot as plt
import racelineOptimization

# USER INPUT
file_paths = {
    "module" : os.path.dirname(os.path.abspath(__file__)),
    "veh_params_file": "f110.ini",
    "track_name": "foxgloveTest"  # Example track
}

opt_type = 'mincurv'  # or 'mincurv_iqp'

plot_opts = {
    "mincurv_curv_lin": True,
    "raceline": True,
    "raceline_curv": True,
    "spline_normals": False
}
imp_opts = {
    "flip_imp_track": False,
    "set_new_start": False,
    "min_track_width": None,
    "num_laps": 1
}

# PATHS
inputs_dir = os.path.join(file_paths["module"], "inputs")

file_paths["track_file"] = os.path.join(inputs_dir, file_paths["track_name"] + ".csv")
file_paths["veh_params_file"] = os.path.join(inputs_dir, file_paths["veh_params_file"])

outputs_dir = os.path.join(file_paths["module"], "outputs")
os.makedirs(outputs_dir, exist_ok=True)
file_paths["traj_race_export"] = os.path.join(outputs_dir, "casadiTest.csv")

# VEHICLE PARAMETERS
parser = configparser.ConfigParser()
parser.read(file_paths["veh_params_file"])

veh_params = ast.literal_eval(parser.get('GENERAL_OPTIONS', 'veh_params'))
optim_opts = ast.literal_eval(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))

print("Finish path declaration")

# IMPORT TRACK
reftrack_imp = racelineOptimization.import_track(
    imp_opts=imp_opts, file_path=file_paths["track_file"], width_veh=veh_params["width"])

print("Finish importing track")

# PREPARE TRACK
reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = racelineOptimization.prep_track(
    reftrack_imp=reftrack_imp,
    reg_smooth_opts= ast.literal_eval(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts')),
    stepsize_opts= ast.literal_eval(parser.get('GENERAL_OPTIONS', 'stepsize_opts')),
    debug=True,
    min_width=imp_opts["min_track_width"]
)

print("Finish preparing track")

# RUN OPTIMIZATION
if opt_type == 'mincurv':
    alpha_opt = racelineOptimization.opt_min_curv_casadi(
        reftrack=reftrack_interp,
        normvectors=normvec_normalized_interp,
        A=a_interp,
        kappa_bound=veh_params["curvlim"],
        w_veh=optim_opts["width_opt"],
        print_debug=True
    )[0]
# elif opt_type == 'mincurv_iqp':
#     alpha_opt, reftrack_interp, normvec_normalized_interp = tph.iqp_handler.iqp_handler(
#         reftrack=reftrack_interp,
#         normvectors=normvec_normalized_interp,
#         kappa_bound=veh_params["curvlim"],
#         w_veh=optim_opts["width_opt"],
#         print_debug=True
#     )

print("Finish optimizing")

# INTERPOLATE RACELINE
raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds, t_vals, s_points, spline_lengths_opt, el_lengths = racelineOptimization.create_raceline(
    refline=reftrack_interp[:, :2],
    normvectors=normvec_normalized_interp,
    alpha=alpha_opt,
    stepsize_interp=ast.literal_eval(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))["stepsize_interp_after_opt"]
)

print("Finish interpolating")

# CALCULATE HEADING AND CURVATURE
psi, kappa = racelineOptimization.calc_head_curv_an(
    coeffs_x=coeffs_x_opt,
    coeffs_y=coeffs_y_opt,
    ind_spls=spline_inds,
    t_spls=t_vals
)

# EXPORT TRAJECTORY
trajectory = np.column_stack((
    s_points, 
    raceline_interp, 
    psi, 
    kappa,
    np.zeros_like(s_points),  # vx_profile (zeros for now)
    np.zeros_like(s_points)   # ax_profile (zeros for now)
))

traj_race_cl = np.vstack((trajectory, trajectory[0, :]))
traj_race_cl[-1, 0] = np.sum(spline_lengths_opt)
np.savetxt(file_paths["traj_race_export"], traj_race_cl, delimiter=',', header='s_m,x_m,y_m,psi_rad,kappa_radpm')

print("Finish export")