import numpy as np
import cv2

def compute_dip(Z, dx, dy):
    """
    Compute per-pixel dip (slope angle from horizontal, degrees) from a
    meter-space height grid. dx and dy are the grid spacings in meters.
    """
    z = np.asarray(Z)

    # Gradient along row (y) and column (x) directions, already in meters
    grad_row, grad_col = np.gradient(z, dy, dx)

    # Dip (slope angle from horizontal)
    slope_rad = np.arctan(np.hypot(grad_col, -grad_row))
    return np.degrees(slope_rad)

def compute_variance(elevation, window_size=11, min_valid_fraction=0.5, pad_mode="reflect"):
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

def compute_safety_score(dip_deg, var, a=1, b=5):
    """
    Compute safety score from dip and variance.
    TODO: Weights to be tuned.
    """
    return  - (a*dip_deg + b*var)

def compute_safety_map(X, Y, Z):
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # blur map for dip angle is its less noisy
    Z_smooth = cv2.GaussianBlur(Z.astype(np.float32), (0, 0), sigmaX=3.0)
    dip_deg = compute_dip(Z_smooth, dx, dy)

    var = compute_variance(Z)
    
    safety_score = compute_safety_score(dip_deg, var)

    # normalize safety score to 0-1 range
    safety_score_min = np.nanmin(safety_score)
    safety_score_max = np.nanmax(safety_score)
    safety_score_normalized = (safety_score - safety_score_min) / (safety_score_max - safety_score_min + 1e-8)

    return safety_score_normalized
