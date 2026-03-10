import numpy as np

def best_fit_transform_kabsch(A: np.ndarray, B: np.ndarray):
    """
    Solve: minimize || (R A + t) - B || over R in SO(3), t in R^3
    A, B: (N,3) corresponding points
    Returns: R, t
    """
    assert A.shape == B.shape
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t

def _nn_bruteforce(src: np.ndarray, dst: np.ndarray):
    """
    For each point in src, find nearest neighbor in dst.
    Returns indices (len(src),) and squared distances (len(src),)
    """
    # O(N*M) but simple and dependency-free
    diffs = src[:, None, :] - dst[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    idx = np.argmin(d2, axis=1)
    return idx, d2[np.arange(src.shape[0]), idx]

def _nn_kdtree(src: np.ndarray, dst: np.ndarray):
    """
    Uses SciPy KDTree if available, otherwise falls back.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        d, idx = tree.query(src, k=1)
        return idx, d*d
    except Exception:
        return _nn_bruteforce(src, dst)

def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    init_T: np.ndarray | None = None,
    max_iters: int = 50,
    tol: float = 1e-6,
    max_corr_dist: float | None = None,
    use_kdtree: bool = True,
    verbose: bool = True,
):
    """
    ICP aligning source -> target.
    Returns:
      T (4x4): estimated transform such that source_aligned ~ target
      history: list of dicts with rmse, num_corr
    """
    assert source.shape[1] == 3 and target.shape[1] == 3

    if init_T is None:
        T = np.eye(4)
        src = source.copy()
    else:
        T = init_T.copy()
        src = _apply_T(source, T)

    nn = _nn_kdtree if use_kdtree else _nn_bruteforce

    prev_rmse = None
    history = []

    Sigma = np.eye(6) * 1e6          # default: very uncertain (ICP failure)
    last_src_corr = None
    last_tgt_corr = None

    for it in range(max_iters):
        idx, d2 = nn(src, target)

        if max_corr_dist is not None:
            m = d2 <= (max_corr_dist * max_corr_dist)
            src_corr = src[m]
            tgt_corr = target[idx[m]]
        else:
            src_corr = src
            tgt_corr = target[idx]

        if src_corr.shape[0] < 6:
            if verbose:
                print(f"ICP iter {it}: too few correspondences ({src_corr.shape[0]}), stopping.")
            break

        R, t = best_fit_transform_kabsch(src_corr, tgt_corr)

        dT = np.eye(4)
        dT[:3,:3] = R
        dT[:3, 3] = t

        # Update total transform: new_src = dT * src; src already = T * source
        T = dT @ T
        src = (R @ src.T).T + t

        rmse = float(np.sqrt(np.mean(np.sum((src_corr @ R.T + t - tgt_corr)**2, axis=1))))
        history.append({"iter": it, "rmse": rmse, "num_corr": int(src_corr.shape[0])})

        if verbose:
            print(f"ICP iter {it:02d}: rmse={rmse:.6f}, corr={src_corr.shape[0]}")

        if prev_rmse is not None and abs(prev_rmse - rmse) < tol:
            if verbose:
                print(f"Converged: |Δrmse| < {tol}")
            break
        prev_rmse = rmse

        # compute covariance matrix at each ICP iteration
        Sigma = estimate_covariance(src_corr, tgt_corr, T)  
        uncertainty = np.trace(Sigma) # sum of variance in all 6 DOF as a simple scalar uncertainty metric

    # compute covariance once, using final correspondences if we have them
    if last_src_corr is not None:
        Sigma = estimate_covariance(last_src_corr, last_tgt_corr, T)
        
    return T, history, Sigma

def estimate_covariance(
    reading: np.ndarray,
    reference: np.ndarray,
    transformation: np.ndarray,
    sensor_std_dev: float = 0.01,
) -> np.ndarray:
    """
    Estimate the 6x6 pose covariance of an ICP result using the Hessian method.
    Adapted from C++ implementation in libpointmatcher.

        Sigma = sigma^2 * H^-1 * (d2J/dZdX * d2J/dZdX^T) * H^-1

    where H is the Gauss-Newton Hessian of the alignment cost J (sum of squared residuals)
    and d2J/dZdX is the mixed partial w.r.t. sensor noise and pose.

    The 6-DOF pose is parameterised as [t_x, t_y, t_z, alpha, beta, gamma]
    (ZYX Euler angles extracted from `transformation`).

    Args:
        reading:        (N,3) source points already transformed into the target frame.
                        [p_x1, p_y1, p_z1]
                        [p_x2, p_y2, p_z2]
                        [...,  ...,  ... ]
                        [p_xN, p_yN, p_zN]
        reference:      (N,3) corresponding target points (same ordering as reading).
                        [q_x1, q_y1, q_z1]
                        [q_x2, q_y2, q_z2]
                        [...,  ...,  ... ]
                        [q_xN, q_yN, q_zN]
        transformation: (4,4) homogeneous transform that maps the original source to
                        the target frame (i.e. the ICP result T).
                        [ _  _  _  t_x ]
                        [ _  R  _  t_y ]
                        [ _  _  _  t_z ]
                        [ 0  0  0  1   ]

        sensor_std_dev: 1-sigma range noise of the sensor [metres].

    Returns:
        covariance: (6,6) pose covariance matrix.
    """
    # --- Extract Euler angles from ICP rotation matrix ---
    beta  = -np.arcsin(transformation[2, 0])
    cb    = np.cos(beta)
    alpha = np.arctan2(transformation[2, 1] / cb, transformation[2, 2] / cb)
    gamma = np.arctan2(transformation[1, 0] / cb, transformation[0, 0] / cb)

    t_x, t_y, t_z = transformation[0, 3], transformation[1, 3], transformation[2, 3]

    # --- Per-point geometry (all vectorised) ---
    r_range = np.linalg.norm(reading,    axis=1)           # (N,)
    ref_range = np.linalg.norm(reference, axis=1)          # (N,)
    r_dir   = reading    / r_range[:, None]                # (N,3)  reading unit vectors
    ref_dir = reference  / ref_range[:, None]              # (N,3)  reference unit vectors

    # normal = [1,1,1] for point-to-point ICP
    # n_{alpha,beta,gamma} = normal x reading_direction  (cross-product components)
    n_alpha = r_dir[:, 1] - r_dir[:, 2]   # normal[2]*rd[1] - normal[1]*rd[2]
    n_beta  = r_dir[:, 2] - r_dir[:, 0]   # normal[0]*rd[2] - normal[2]*rd[0]
    n_gamma = r_dir[:, 0] - r_dir[:, 1]   # normal[1]*rd[0] - normal[0]*rd[1]

    # Linearised residual E = normal . (R_lin * reading + t - reference)
    E = (  (reading[:, 0] - gamma * reading[:, 1] + beta  * reading[:, 2] + t_x - reference[:, 0])
         + (gamma * reading[:, 0] + reading[:, 1] - alpha * reading[:, 2] + t_y - reference[:, 1])
         + (-beta * reading[:, 0] + alpha * reading[:, 1] + reading[:, 2] + t_z - reference[:, 2])
    )  # (N,)

    # dE/d(reading_range) projected onto the normal
    N_reading = (
          (r_dir[:, 0] - gamma * r_dir[:, 1] + beta  * r_dir[:, 2])
        + (gamma * r_dir[:, 0] + r_dir[:, 1] - alpha * r_dir[:, 2])
        + (-beta * r_dir[:, 0] + alpha * r_dir[:, 1] + r_dir[:, 2])
    )  # (N,)

    # dE/d(reference_range) projected onto the normal (negative)
    N_reference = -(ref_dir[:, 0] + ref_dir[:, 1] + ref_dir[:, 2])  # (N,)

    # --- Hessian accumulation: H += b * b^T ---
    # b = [nx, ny, nz, range*n_alpha, range*n_beta, range*n_gamma]  with normal=[1,1,1]
    ones = np.ones(len(reading))
    B = np.column_stack([
        ones,
        ones,
        ones,
        r_range * n_alpha,
        r_range * n_beta,
        r_range * n_gamma,
    ])  # (N,6)
    J_hessian = B.T @ B  # (6,6)

    # --- Mixed partial d2J/dReadingdX ---
    rN = r_range * N_reading
    B_reading = np.column_stack([
        N_reading,
        N_reading,
        N_reading,
        n_alpha * (E + rN),
        n_beta  * (E + rN),
        n_gamma * (E + rN),
    ])  # (N,6)

    # --- Mixed partial d2J/dReferencedX ---
    B_reference = np.column_stack([
        N_reference,
        N_reference,
        N_reference,
        ref_range * n_alpha * N_reference,
        ref_range * n_beta  * N_reference,
        ref_range * n_gamma * N_reference,
    ])  # (N,6)

    # d2J/dZdX is (6, 2N); build as (2N, 6) then transpose
    d2J_dZdX = np.vstack([B_reading, B_reference]).T  # (6, 2N)

    inv_H = np.linalg.pinv(J_hessian) # pseudo-inverse to avoid singular matrix 
    cov = inv_H @ (d2J_dZdX @ d2J_dZdX.T) @ inv_H

    return (sensor_std_dev ** 2) * cov


def _apply_T(pts: np.ndarray, T: np.ndarray):
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    out = (T @ pts_h.T).T
    return out[:, :3]