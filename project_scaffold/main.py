import numpy as np
import matplotlib.pyplot as plt

from terrain import generate_height_field, generate_height_field_fractal
from pointcloud import heightmap_to_points, crop_points_xy, add_gaussian_noise
from transforms import so3_from_euler, make_T, invert_T, apply_T, rotation_angle_deg
from icp import icp_point_to_point
from viz import plot_heightmap, plot_xy_overlay, plot_safety_map
from safety_score import compute_safety_map


# =========================
# User-set parameters
# =========================

# --- Terrain / DEM generation ---
TERRAIN_MODE = "fractal"   # "fractal" or "sinusoid"
NX, NY = 220, 220
SIZE_X, SIZE_Y = 80.0, 80.0

# sinusoid terrain params (used if TERRAIN_MODE == "sinusoid")
COMPONENTS = [
    {"amp": 2.0, "scale": 55.0},
    {"amp": 0.9, "scale": 20.0},
    {"amp": 0.3, "scale": 7.0},
]
TERRAIN_NOISE_SIGMA = 0.0
TERRAIN_SEED = 4

# fractal terrain params (used if TERRAIN_MODE == "fractal")
FRACTAL_ALPHA = 2.4
FRACTAL_AMP = 2.5
FRACTAL_SEED = 4

# point cloud sampling
TARGET_STRIDE = 2

# --- "LiDAR scan" construction ---
CROP_XMIN, CROP_XMAX = 15.0, 65.0
CROP_YMIN, CROP_YMAX = 10.0, 70.0
SOURCE_NOISE_SIGMA = 0.2
SOURCE_NOISE_SEED = 0

# ground-truth transform (world <- lidar)
GT_ROLL_DEG = 0.5
GT_PITCH_DEG = -0.5
GT_YAW_DEG = 2.0
GT_T = np.array([0.8, -0.6, 0.1])

# --- ICP settings ---
USE_INITIAL_GUESS = False

# initial guess (world <- lidar) if enabled
INIT_ROLL_DEG = 0.0
INIT_PITCH_DEG = 0.0
INIT_YAW_DEG = 8.0
INIT_T = np.array([2.0, -1.0, 0.0])

ICP_MAX_ITERS = 60
ICP_TOL = 1e-7
ICP_MAX_CORR_DIST = 1.5
ICP_USE_KDTREE = True
ICP_VERBOSE = True

# --- Plotting ---
PLOT_3D = False  # if you later add a 3D plot util
DOWNSAMPLE_VIS_TARGET = 1  # e.g., 5 to downsample for faster plotting
DOWNSAMPLE_VIS_SOURCE = 1


def build_terrain():
    if TERRAIN_MODE.lower() == "fractal":
        return generate_height_field_fractal(
            NX, NY, SIZE_X, SIZE_Y,
            alpha=FRACTAL_ALPHA,
            amp=FRACTAL_AMP,
            seed=FRACTAL_SEED,
            noise_sigma=TERRAIN_NOISE_SIGMA,
        )
    elif TERRAIN_MODE.lower() == "sinusoid":
        return generate_height_field(
            NX, NY, SIZE_X, SIZE_Y,
            components=COMPONENTS,
            noise_sigma=TERRAIN_NOISE_SIGMA,
            seed=TERRAIN_SEED,
        )
    else:
        raise ValueError(f"Unknown TERRAIN_MODE: {TERRAIN_MODE}")


def build_ground_truth_transform():
    R_gt = so3_from_euler(
        roll=np.deg2rad(GT_ROLL_DEG),
        pitch=np.deg2rad(GT_PITCH_DEG),
        yaw=np.deg2rad(GT_YAW_DEG),
    )
    return make_T(R_gt, GT_T)


def build_initial_guess():
    if not USE_INITIAL_GUESS:
        return None
    R0 = so3_from_euler(
        roll=np.deg2rad(INIT_ROLL_DEG),
        pitch=np.deg2rad(INIT_PITCH_DEG),
        yaw=np.deg2rad(INIT_YAW_DEG),
    )
    return make_T(R0, INIT_T)


def main():
    # --- Offline DEM-like map ---
    X, Y, Z = build_terrain()
    safety_map = compute_safety_map(X, Y, Z)
    target = heightmap_to_points(X, Y, Z, stride=TARGET_STRIDE)

    # --- Build an "online LiDAR scan": crop subset, transform into LiDAR frame, add noise ---
    src_world = crop_points_xy(
        target,
        xmin=CROP_XMIN, xmax=CROP_XMAX,
        ymin=CROP_YMIN, ymax=CROP_YMAX
    )

    T_gt = build_ground_truth_transform()      # world <- lidar
    T_gt_inv = invert_T(T_gt)                  # lidar <- world

    source = apply_T(src_world, T_gt_inv)      # points now in LiDAR frame

    # simulate LiDAR sensor noise
    source = add_gaussian_noise(source, sigma=SOURCE_NOISE_SIGMA, seed=SOURCE_NOISE_SEED)

    # --- ICP: estimate world <- lidar ---
    init_T = build_initial_guess()

    T_est, hist, uncertainty = icp_point_to_point(
        source=source,
        target=target,
        init_T=init_T,
        max_iters=ICP_MAX_ITERS,
        tol=ICP_TOL,
        max_corr_dist=ICP_MAX_CORR_DIST,
        use_kdtree=ICP_USE_KDTREE,
        verbose=ICP_VERBOSE,
    )

    print("\n=== Ground truth T (world <- lidar) ===")
    print(T_gt)
    print("\n=== Estimated T (world <- lidar) ===")
    print(T_est)

    dT = T_est @ invert_T(T_gt)
    rot_err = rotation_angle_deg(dT[:3, :3])
    trans_err = np.linalg.norm(dT[:3, 3])
    print(f"\nRotation error (deg): {rot_err:.4f}")
    print(f"Translation error (m): {trans_err:.4f}")
    print(f"Uncertainty: {uncertainty:.6e}")

    # --- Plotting ---
    plot_heightmap(X, Y, Z, title="Target / offline DEM (synthetic)")
    plt.savefig('./project_scaffold/figures/heightmap.png', dpi=300)

    offline_safety_map = compute_safety_map(X, Y, Z)
    plot_safety_map(X, Y, offline_safety_map, title="Offline safety score map")
    plt.savefig('./project_scaffold/figures/offline_safety_map.png', dpi=300)

    tgt_vis = target[::DOWNSAMPLE_VIS_TARGET]
    src_vis = source[::DOWNSAMPLE_VIS_SOURCE]
    plot_xy_overlay(tgt_vis, src_vis, title="Before ICP (different frames)")
    plt.savefig('./project_scaffold/figures/before_icp.png', dpi=300)

    source_aligned = apply_T(source, T_est)
    src_aligned_vis = source_aligned[::DOWNSAMPLE_VIS_SOURCE]
    plot_xy_overlay(tgt_vis, src_aligned_vis, title="After ICP (aligned)")
    plt.savefig('./project_scaffold/figures/after_icp.png', dpi=300)

    plt.show()  # blocks only once, at the end


if __name__ == "__main__":
    main()