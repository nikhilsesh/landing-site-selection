import numpy as np
from scipy.interpolate import griddata



def precision_weighted_average(S_off, S_on, sigma_off, sigma_on):
    """
    Precision-weighted average of offline and online safety maps.

    S_fuse(p) = (S_off(p)/σ²_off(p) + S_on(p)/σ²_on(p))
                / (1/σ²_off(p) + 1/σ²_on(p))

    Map with higher certainty contributes more to the result.
    """
    prec_off = 1.0 / np.square(sigma_off)
    prec_on  = 1.0 / np.square(sigma_on)
    return (S_off * prec_off + S_on * prec_on) / (prec_off + prec_on)


def interpolate_online_map(pts: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Convert LiDAR point clouds (combined in global frame using ICP) --> rasterized/grid height map.

    Scattered XYZ points are interpolated onto the offline DEM grid (X, Y) using
    linear interpolation, with nearest-neighbor fallback to fill boundary gaps.

    Args:
        pts: (N, 3) point cloud already transformed into the world frame via ICP.
        X:   (H, W) meshgrid of x-coordinates matching the offline DEM.
        Y:   (H, W) meshgrid of y-coordinates matching the offline DEM.

    Returns:
        Z_on: (H, W) height map on the same grid as X, Y.
    """
    xy = pts[:, :2]
    z  = pts[:, 2]
    grid_pts = (X, Y)

    Z_linear  = griddata(xy, z, grid_pts, method="linear")
    Z_nearest = griddata(xy, z, grid_pts, method="nearest")

    # fill any NaNs at the boundary (outside convex hull of scan) with nearest
    Z_on = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)
    return Z_on