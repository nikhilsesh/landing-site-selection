import rasterio
import numpy as np
import cv2
from loguru import logger
#import matplotlib.pyplot as plt

def compute_strike_dip(elevation, transform):
    """
    Compute per-pixel strike, dip, and dip direction from a DEM. Assumes north-up imagery.

    - strike: azimuth (degrees, 0-360) of the horizontal line on the plane
    - dip: angle from horizontal (degrees, 0-90)
    - dip_direction: azimuth (degrees, 0-360) of steepest descent
    
    """
    z = np.asarray(elevation)

    dx = transform.a
    dy = abs(transform.e)

    # Gradient along row (y, downward) and column (x, eastward)
    grad_row, grad_col = np.gradient(z, dy, dx)

    # Convert to derivatives in cardinal directions (north/east)
    dz_dx_east = grad_col 
    dz_dy_north = -grad_row

    rows = z.shape[0]
    center_row = (rows - 1) * 0.5
    
    # y (latitude) coordinate of the center row
    lat_center_deg = transform.f + (center_row + 0.5) * transform.e
    phi = np.deg2rad(lat_center_deg)
    
    # High-accuracy approximations for meters per degree
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi) - 0.0023 * np.cos(6 * phi)
    m_per_deg_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi) + 0.118 * np.cos(5 * phi)
    
    # Convert derivatives to per-meter
    dz_dx_east  /= m_per_deg_lon
    dz_dy_north /= m_per_deg_lat

    # Dip direction (azimuth of steepest descent) using per-meter derivatives
    downhill_east = -dz_dx_east
    downhill_north = -dz_dy_north
    dip_dir_deg = (np.degrees(np.arctan2(downhill_east, downhill_north)) + 360.0) % 360.0

    # Strike is perpendicular to dip direction
    strike_deg = (dip_dir_deg + 90.0) % 360.0

    # Dip (slope angle from horizontal)
    slope_rad = np.arctan(np.hypot(dz_dx_east, dz_dy_north))
    dip_deg = np.degrees(slope_rad)

    return strike_deg, dip_deg, dip_dir_deg

def compute_variance(elevation, window_size=3, min_valid_fraction=0.5, pad_mode="reflect"):
    """
    Compute local elevation variance over a square moving window. Returns an array of shape
    equal to elevation with per-pixel variance.

    - window_size: odd integer >= 1 defining the side length of the window
    - min_valid_fraction: required fraction of finite pixels in the window to emit a value
      (otherwise the output is NaN). Set to 0.0 to always compute if any valid pixels exist.
    - pad_mode: numpy pad mode for borders (e.g., "reflect", "edge", "constant")

    """
    z = np.asarray(elevation)

    if window_size < 1 or (window_size % 2) == 0:
        raise ValueError("window_size must be an odd integer >= 1")

    # replace NaNs with 0s
    z = z.astype(np.float64, copy=False)
    valid_mask = np.isfinite(z)
    z = np.where(valid_mask, z, 0.0)

    pad = window_size // 2

    def _box_sum(arr):
        arr_pad = np.pad(arr, ((pad, pad), (pad, pad)), mode=pad_mode)
        # Integral image with leading zero row/col for simpler region sums
        S = np.pad(arr_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0).cumsum(0).cumsum(1)
        # Vectorized window sum for each top-left position over the padded array
        return S[window_size:, window_size:] - S[:-window_size, window_size:] - S[window_size:, :-window_size] + S[:-window_size, :-window_size]

    sum_x  = _box_sum(z)
    sum_x2 = _box_sum(z*z)
    count  = _box_sum(valid_mask.astype(np.float64))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_x / count
        mean_sq = sum_x2 / count
        var = mean_sq - (mean * mean)

    # clamp numerical imprecision to zero
    var = np.where(var < 0.0, 0.0, var)
    valid_fraction = count / float(window_size * window_size)
    var = np.where(valid_fraction >= float(min_valid_fraction), var, np.nan)

    return var.astype(np.float32)

def compute_safety_score(dip_deg, var, a=5, b=1):
    """
    Compute safety score from dip and variance.
    TODO: Weights to be tuned.
    """
    return  - (a*dip_deg + b*var)

def compute_safety_map(dem_path, output_path='results/davis_safety_score.png'):
    logger.info("=" * 60)
    logger.info("Computing safety score from DEM data")
    logger.info("=" * 60)
    
    with rasterio.open(dem_path) as src:
        subset = src.read(1)
        strike_deg, dip_deg, dip_dir_deg = compute_strike_dip(subset, src.transform)
        var = compute_variance(subset)
        safety_score = compute_safety_score(dip_deg, var)
    
    # normalize safety score to 0-255 range
    safety_score_min = np.nanmin(safety_score)
    safety_score_max = np.nanmax(safety_score)
    safety_score_normalized = ((safety_score - safety_score_min) / (safety_score_max - safety_score_min + 1e-8) * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, safety_score_normalized)
    logger.info(f"Saved safety score map to: {output_path}")
    
    return safety_score

if __name__ == "__main__":
    dem_path = 'dem_maps/davis_dem.tif'
    output_path = 'results/davis_safety_score.png'
    compute_safety_map(dem_path, output_path)

# plt.figure(figsize=(10, 6))
# plt.imshow(dip_deg, cmap="viridis", extent=(left, right, bottom, top))
# plt.colorbar(label="Dip (degrees)")
# plt.title("Dip (slope angle)")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.savefig("davis_dip.png", dpi=300, bbox_inches="tight")

# plt.figure(figsize=(10, 6))
# plt.imshow(var, cmap="viridis", extent=(left, right, bottom, top))
# plt.colorbar(label="Variance (m^2)")
# plt.title("Variance")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.savefig("davis_var.png", dpi=300, bbox_inches="tight")

# plt.figure(figsize=(10, 6))
# plt.imshow(safety_score, cmap="viridis", extent=(left, right, bottom, top))
# plt.colorbar(label="Safety Score")
# plt.title("Safety Score")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.savefig("davis_safety_score.png", dpi=300, bbox_inches="tight")

