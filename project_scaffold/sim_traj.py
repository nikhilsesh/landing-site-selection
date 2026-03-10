import numpy as np
from se3 import Rz

def make_planar_trajectory(T_total: float, dt: float, kind="lemniscate"):
    t = np.arange(0.0, T_total + 1e-12, dt)

    if kind == "circle":
        R = 25.0
        w = 2*np.pi / T_total
        x = R * np.cos(w*t)
        y = R * np.sin(w*t)
    else:
        a = 25.0
        w = 2*np.pi / T_total
        x = a * np.sin(w*t)
        y = a * np.sin(w*t) * np.cos(w*t)

    z = np.zeros_like(x)
    p = np.stack([x, y, z], axis=1)
    v = np.gradient(p, dt, axis=0)
    a_w = np.gradient(v, dt, axis=0)

    # Yaw points along velocity direction
    yaw = np.unwrap(np.arctan2(v[:,1], v[:,0]))
    yaw_rate = np.gradient(yaw, dt)

    # Full 3D orientation (truth): yaw-only
    R_wb = np.stack([Rz(y) for y in yaw], axis=0)

    # Angular velocity in body frame: for yaw-only, omega_b = [0,0,yaw_rate]
    omega_b = np.stack([np.zeros_like(yaw_rate),
                        np.zeros_like(yaw_rate),
                        yaw_rate], axis=1)

    return {
        "t": t,
        "p": p,
        "v": v,
        "a_w": a_w,
        "yaw": yaw,
        "R_wb": R_wb,
        "omega_b": omega_b,
    }