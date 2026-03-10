import numpy as np

def simulate_imu(traj, accel_noise_sigma=0.15, gyro_noise_sigma=0.01,
                 accel_bias=np.zeros(3), gyro_bias=np.zeros(3),
                 g=9.81, seed=0):
    """
    accel measures specific force in body: f_b = R^T (a_w - g_w)
    gyro measures body ang vel: omega_b
    """
    rng = np.random.default_rng(seed)
    t = traj["t"]
    R_wb = traj["R_wb"]
    a_w = traj["a_w"]
    omega_b_true = traj["omega_b"]

    g_w = np.array([0, 0, -g], dtype=float)

    f_b_meas = np.zeros_like(a_w)
    w_meas = np.zeros_like(omega_b_true)

    for k in range(len(t)):
        f_b_true = R_wb[k].T @ (a_w[k] - g_w)
        f_b_meas[k] = f_b_true + accel_bias + rng.normal(0, accel_noise_sigma, size=3)

        w_meas[k] = omega_b_true[k] + gyro_bias + rng.normal(0, gyro_noise_sigma, size=3)

    return f_b_meas, w_meas