import numpy as np

def heightmap_to_points(X, Y, Z, stride: int = 1):
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    Zs = Z[::stride, ::stride]
    return np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], axis=1)

def crop_points_xy(pts: np.ndarray, xmin, xmax, ymin, ymax):
    m = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    return pts[m]

def add_gaussian_noise(pts: np.ndarray, sigma: float, seed: int | None = None):
    if sigma <= 0:
        return pts
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0.0, sigma, size=pts.shape)