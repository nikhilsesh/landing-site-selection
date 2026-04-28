import numpy as np
import matplotlib.pyplot as plt

from old_stuff.project_scaffold.terrain import generate_height_field_fractal
from old_stuff.project_scaffold.pointcloud import heightmap_to_points
from old_stuff.project_scaffold.sim_traj import make_planar_trajectory
from old_stuff.project_scaffold.sim_imu import simulate_imu
from old_stuff.project_scaffold.sim_lidar import simulate_lidar_scan_from_map, icp_covariance_placeholder_6x6
from old_stuff.project_scaffold.ekf import ESEKF_IMU_LiDAR_Bias
from old_stuff.project_scaffold.se3 import make_T
from old_stuff.project_scaffold.icp import icp_point_to_point

# =========================
# Parameters
# =========================
T_TOTAL = 40.0
DT_IMU = 0.01
DT_LIDAR = 0.5

# Map / DEM
NX, NY = 260, 260
SIZE_X, SIZE_Y = 120.0, 120.0
FRACTAL_ALPHA = 2.4
FRACTAL_AMP = 2.5
MAP_STRIDE = 2

# IMU noise/bias (simulation truth)
ACCEL_NOISE_SIGMA = 0.20
GYRO_NOISE_SIGMA = np.deg2rad(0.5)  # rad/s
ACCEL_BIAS_TRUE = np.array([0.02, -0.015, 0.0])
GYRO_BIAS_TRUE  = np.array([0.0, 0.0, np.deg2rad(0.1)])

# EKF noise params (tuning knobs)
ACCEL_MEAS_VAR = (0.35)**2
GYRO_MEAS_VAR  = (np.deg2rad(1.0))**2
ACCEL_BIAS_RW_VAR = (0.002)**2
GYRO_BIAS_RW_VAR  = (np.deg2rad(0.02))**2

# Initial bias guesses (can be zero or intentionally wrong)
BA0 = np.zeros(3)
BG0 = np.zeros(3)

# LiDAR scan sim
LIDAR_RADIUS = 18.0
LIDAR_SCAN_STRIDE = 2
LIDAR_POINT_NOISE_SIGMA = 0.03

# ICP
ICP_MAX_ITERS = 40
ICP_TOL = 1e-6
ICP_MAX_CORR_DIST = 1.5
ICP_USE_KDTREE = True
ICP_VERBOSE = False

# Initial covariance (15x15): [p v theta ba bg]
P0 = np.diag([
    5.0**2, 5.0**2, 3.0**2,
    2.0**2, 2.0**2, 1.0**2,
    np.deg2rad(15.0)**2, np.deg2rad(15.0)**2, np.deg2rad(20.0)**2,
    0.1**2, 0.1**2, 0.1**2,
    np.deg2rad(1.0)**2, np.deg2rad(1.0)**2, np.deg2rad(1.0)**2
])

def plot_xy(gt_p, est_p, lidar_p=None):
    plt.figure(figsize=(6,6))
    plt.plot(gt_p[:,0], gt_p[:,1], 'k-', label="GT")
    plt.plot(est_p[:,0], est_p[:,1], 'b-', label="ESEKF (bias)")
    if lidar_p is not None and len(lidar_p) > 0:
        lp = np.asarray(lidar_p)
        plt.scatter(lp[:,0], lp[:,1], c='r', s=15, label="ICP pose meas")
    plt.axis("equal"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
    plt.show()

def main():
    # Map points
    X, Y, Z = generate_height_field_fractal(NX, NY, SIZE_X, SIZE_Y, alpha=FRACTAL_ALPHA, amp=FRACTAL_AMP, seed=2)
    map_pts = heightmap_to_points(X, Y, Z, stride=MAP_STRIDE)

    # Truth
    traj = make_planar_trajectory(T_TOTAL, DT_IMU, kind="lemniscate")

    # IMU measurements
    f_b_meas, w_b_meas = simulate_imu(
        traj,
        accel_noise_sigma=ACCEL_NOISE_SIGMA,
        gyro_noise_sigma=GYRO_NOISE_SIGMA,
        accel_bias=ACCEL_BIAS_TRUE,
        gyro_bias=GYRO_BIAS_TRUE,
        seed=1
    )

    # EKF init (pose/vel from truth; biases guessed)
    p0 = traj["p"][0]
    v0 = traj["v"][0]
    R0 = traj["R_wb"][0]

    # EKF that will get ICP updates
    ekf = ESEKF_IMU_LiDAR_Bias(
        p0=p0, v0=v0, R0=R0, ba0=BA0, bg0=BG0, P0=P0,
        accel_noise_var=ACCEL_MEAS_VAR,
        gyro_noise_var=GYRO_MEAS_VAR,
        accel_bias_rw_var=ACCEL_BIAS_RW_VAR,
        gyro_bias_rw_var=GYRO_BIAS_RW_VAR
    )

    # EKF that will be IMU-only (no ICP). Use a copy of P0 so they start identical.
    ekf_imu_only = ESEKF_IMU_LiDAR_Bias(
        p0=p0, v0=v0, R0=R0, ba0=BA0, bg0=BG0, P0=P0.copy(),
        accel_noise_var=ACCEL_MEAS_VAR,
        gyro_noise_var=GYRO_MEAS_VAR,
        accel_bias_rw_var=ACCEL_BIAS_RW_VAR,
        gyro_bias_rw_var=GYRO_BIAS_RW_VAR
    )

    t = traj["t"]
    est_p_icp = np.zeros((len(t), 3))
    est_p_imu_only = np.zeros((len(t), 3))
    lidar_meas_xy = []

    est_R_icp = np.zeros((len(t), 3, 3))
    est_R_imu_only = np.zeros((len(t), 3, 3))

    est_P_diag_icp = np.zeros((len(t), 15))
    est_P_diag_imu_only = np.zeros((len(t), 15))

    next_lidar_t = 0.0

    for k in range(len(t)):
        # Predict step for both filters
        ekf.predict(f_b_meas=f_b_meas[k], w_b_meas=w_b_meas[k], dt=DT_IMU)
        ekf_imu_only.predict(f_b_meas=f_b_meas[k], w_b_meas=w_b_meas[k], dt=DT_IMU)

        # store poses
        est_p_icp[k] = ekf.p
        est_R_icp[k] = ekf.R
        est_p_imu_only[k] = ekf_imu_only.p
        est_R_imu_only[k] = ekf_imu_only.R

        est_P_diag_icp[k] = np.diag(ekf.P)
        est_P_diag_imu_only[k] = np.diag(ekf_imu_only.P)

        # LiDAR/ICP happens only for the ICP-enabled filter
        if t[k] + 1e-12 >= next_lidar_t:
            # simulate scan from the true vehicle pose (same as before)
            T_wb_true = make_T(traj["R_wb"][k], traj["p"][k])

            scan_b, _local_w = simulate_lidar_scan_from_map(
                map_pts_w=map_pts,
                T_wb_true=T_wb_true,
                radius=LIDAR_RADIUS,
                stride=LIDAR_SCAN_STRIDE,
                noise_sigma=LIDAR_POINT_NOISE_SIGMA,
                seed=k
            )

            # use the current ekf estimate as ICP init (same behavior as before)
            T_init = make_T(ekf.R, ekf.p)

            T_wb_est_icp, _hist, R_icp = icp_point_to_point(
                source=scan_b,
                target=map_pts,
                init_T=T_init,
                max_iters=ICP_MAX_ITERS,
                tol=ICP_TOL,
                max_corr_dist=ICP_MAX_CORR_DIST,
                use_kdtree=ICP_USE_KDTREE,
                verbose=ICP_VERBOSE
            )

            # placeholder covariance and update only the ICP-enabled EKF
            #R_icp = icp_covariance_placeholder_6x6(T_wb_est_icp)
            ekf.update_pose_se3(T_wb_meas=T_wb_est_icp, R_meas_6x6=R_icp)

            lidar_meas_xy.append(T_wb_est_icp[:2, 3].copy())
            next_lidar_t += DT_LIDAR

    # Plot GT, EKF with ICP, and EKF imu-only
    plt.figure(figsize=(6,6))
    plt.plot(traj["p"][:,0], traj["p"][:,1], 'k-', label="GT")
    plt.plot(est_p_icp[:,0], est_p_icp[:,1], 'b-', label="ESEKF (with ICP)")
    plt.plot(est_p_imu_only[:,0], est_p_imu_only[:,1], 'g--', label="ESEKF (IMU-only)")
    if len(lidar_meas_xy) > 0:
        lp = np.asarray(lidar_meas_xy)
        plt.scatter(lp[:,0], lp[:,1], c='r', s=15, label="ICP pose meas")
    plt.axis("equal"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
    plt.savefig('ekf_track.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()

    # ========================
    # --- ERROR PLOTTING (2x2 Grid) ---
    # ========================

    # Create a 2x2 subplot grid. `sharex=True` links the x-axes of plots in the same column.
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), sharex=True)
    fig.suptitle("Filter Error with 95% Confidence Bounds")

    # --- (Top Row) Position Error Calculations ---
    pos_err_icp = np.linalg.norm(traj["p"] - est_p_icp, axis=1)
    pos_err_imu_only = np.linalg.norm(traj["p"] - est_p_imu_only, axis=1)

    pos_var_icp = est_P_diag_icp[:, 0:3]
    pos_std_icp = np.sqrt(np.sum(pos_var_icp, axis=1))
    pos_bound_icp = 2.0 * pos_std_icp

    pos_var_imu_only = est_P_diag_imu_only[:, 0:3]
    pos_std_imu_only = np.sqrt(np.sum(pos_var_imu_only, axis=1))
    pos_bound_imu_only = 2.0 * pos_std_imu_only

    # --- Plot 1 (Top-Left): Position Error (with ICP) ---
    ax = axs[0, 0]
    ax.plot(t, pos_err_icp, 'b-', label="Position Error")
    ax.fill_between(t, 0, pos_bound_icp, color='b', alpha=0.2, label='2-sigma bound')
    ax.set_title("Position Error (with ICP)")
    ax.set_ylabel("Error (m)")
    ax.set_ylim(0, 10)
    ax.legend()
    ax.grid(True)

    # --- Plot 2 (Top-Right): Position Error (IMU-only) ---
    ax = axs[0, 1]
    ax.plot(t, pos_err_imu_only, 'g--', label="Position Error")
    ax.fill_between(t, 0, pos_bound_imu_only, color='g', alpha=0.2, label='2-sigma bound')
    ax.set_title("Position Error (IMU-only)")
    ax.legend()
    ax.grid(True)


    # --- (Bottom Row) Rotation Error Calculations ---
    R_err_icp = traj["R_wb"] @ est_R_icp.transpose(0, 2, 1)
    angle_err_icp_rad = np.arccos(np.clip((np.trace(R_err_icp, axis1=1, axis2=2) - 1) / 2.0, -1.0, 1.0))

    R_err_imu = traj["R_wb"] @ est_R_imu_only.transpose(0, 2, 1)
    angle_err_imu_rad = np.arccos(np.clip((np.trace(R_err_imu, axis1=1, axis2=2) - 1) / 2.0, -1.0, 1.0))

    rot_var_icp = est_P_diag_icp[:, 6:9]
    rot_std_icp = np.sqrt(np.sum(rot_var_icp, axis=1))
    rot_bound_icp = 2.0 * rot_std_icp

    rot_var_imu_only = est_P_diag_imu_only[:, 6:9]
    rot_std_imu_only = np.sqrt(np.sum(rot_var_imu_only, axis=1))
    rot_bound_imu_only = 2.0 * rot_std_imu_only

    # --- Plot 3 (Bottom-Left): Rotation Error (with ICP) ---
    ax = axs[1, 0]
    ax.plot(t, np.rad2deg(angle_err_icp_rad), 'b-', label="Rotation Error")
    ax.fill_between(t, 0, np.rad2deg(rot_bound_icp), color='b', alpha=0.2, label='2-sigma bound')
    ax.set_title("Rotation Error (with ICP)")
    ax.set_ylabel("Error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 6)
    ax.legend()
    ax.grid(True)

    # --- Plot 4 (Bottom-Right): Rotation Error (IMU-only) ---
    ax = axs[1, 1]
    ax.plot(t, np.rad2deg(angle_err_imu_rad), 'g--', label="Rotation Error")
    ax.fill_between(t, 0, np.rad2deg(rot_bound_imu_only), color='g', alpha=0.2, label='2-sigma bound')
    ax.set_title("Rotation Error (IMU-only)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True)


    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('filter_error.png', transparent=True, dpi=300, bbox_inches='tight')
    plt.show()

    # Print bias estimates for both
    print("Estimated accel bias (with ICP):", ekf.ba)
    print("Estimated gyro bias (with ICP):", ekf.bg)
    print("Estimated accel bias (IMU-only):", ekf_imu_only.ba)
    print("Estimated gyro bias (IMU-only):", ekf_imu_only.bg)
    print("True accel bias ba:", ACCEL_BIAS_TRUE)
    print("True gyro bias bg:", GYRO_BIAS_TRUE)

if __name__ == "__main__":
    main()