import numpy as np
from se3 import make_T, invert_T, apply_T

def simulate_lidar_scan_from_map(map_pts_w, T_wb_true, radius=18.0, stride=2, noise_sigma=0.03, seed=None):
    """
    map_pts_w: (N,3) points in world.
    T_wb_true: world<-body transform (body is LiDAR frame here).
    """
    rng = np.random.default_rng(seed)

    p = T_wb_true[:3, 3]
    dx = map_pts_w[:, 0] - p[0]
    dy = map_pts_w[:, 1] - p[1]
    mask = (dx*dx + dy*dy) <= radius*radius
    local_w = map_pts_w[mask][::stride]

    T_bw = invert_T(T_wb_true)
    scan_b = apply_T(local_w, T_bw)

    if noise_sigma > 0:
        scan_b = scan_b + rng.normal(0, noise_sigma, size=scan_b.shape)

    return scan_b, local_w

def icp_covariance_placeholder_6x6(T_est):
    """
    Placeholder 6x6 covariance in (translation[m], rotation[rad]) coordinates.
    Your friend will replace.
    """
    sig_p = 0.5   # m
    sig_r = np.deg2rad(2.0)  # rad
    return np.diag([sig_p**2, sig_p**2, (2.0*sig_p)**2, sig_r**2, sig_r**2, sig_r**2])