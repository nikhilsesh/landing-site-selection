import numpy as np

def so3_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def make_T(R: np.ndarray, t: np.ndarray):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert_T(T: np.ndarray):
    R = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def apply_T(pts: np.ndarray, T: np.ndarray):
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    out = (T @ pts_h.T).T
    return out[:, :3]

def rotation_angle_deg(R: np.ndarray):
    # numerical safety
    tr = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(tr))