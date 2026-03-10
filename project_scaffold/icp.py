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

    return T, history

def _apply_T(pts: np.ndarray, T: np.ndarray):
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    out = (T @ pts_h.T).T
    return out[:, :3]