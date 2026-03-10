import numpy as np

def skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def so3_exp(phi):
    """
    Exponential map from so(3) vector to SO(3).
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.eye(3) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def so3_log(R):
    """
    Log map from SO(3) to so(3) vector.
    """
    cosang = (np.trace(R) - 1) / 2
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    if ang < 1e-12:
        return np.zeros(3)
    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]]) / (2*np.sin(ang))
    return ang * w

def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def make_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def invert_T(T):
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def apply_T(pts, T):
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    out = (T @ pts_h.T).T
    return out[:, :3]