import numpy as np
from se3 import skew, so3_exp, so3_log

class ESEKF_IMU_LiDAR_Bias:
    """
    Nominal:
      p, v, R, b_a, b_g
    Error-state (15):
      dx = [dp, dv, dtheta, dba, dbg]
    """

    def __init__(self, p0, v0, R0, ba0, bg0, P0,
                 accel_noise_var=(0.3**2),
                 gyro_noise_var=(0.02**2),
                 accel_bias_rw_var=(0.001**2),
                 gyro_bias_rw_var=(0.0005**2),
                 g=9.81):
        self.p = p0.astype(float).copy()
        self.v = v0.astype(float).copy()
        self.R = R0.astype(float).copy()
        self.ba = ba0.astype(float).copy()
        self.bg = bg0.astype(float).copy()

        self.P = P0.astype(float).copy()   # 15x15

        self.q_a = float(accel_noise_var)
        self.q_w = float(gyro_noise_var)
        self.q_ba = float(accel_bias_rw_var)
        self.q_bg = float(gyro_bias_rw_var)

        self.g_w = np.array([0, 0, -g], dtype=float)

    def predict(self, f_b_meas, w_b_meas, dt):
        """
        Bias-correct, propagate nominal; propagate covariance with linearized error dynamics.
        """
        # Bias-correct
        w = w_b_meas - self.bg
        f = f_b_meas - self.ba

        Rk = self.R

        # Nominal propagation
        self.R = Rk @ so3_exp(w * dt)
        a_w = Rk @ f + self.g_w
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        # biases are random walk -> nominal unchanged here

        # Error-state dynamics (discrete approx)
        # dx = [dp dv dth dba dbg]
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt

        # dv, dp dependence on attitude error
        F[3:6, 6:9] = -Rk @ skew(f) * dt
        F[0:3, 6:9] = -0.5 * Rk @ skew(f) * dt * dt

        # dv, dp dependence on accel bias error (since f = f_meas - ba)
        F[3:6, 9:12] = -Rk * dt
        F[0:3, 9:12] = -0.5 * Rk * dt * dt

        # dtheta dependence on gyro bias error
        F[6:9, 12:15] = -np.eye(3) * dt

        # Noise Jacobian G for [na, nw, nba, nbg] each 3D => 12 dims
        G = np.zeros((15, 12))
        # accel measurement noise na affects dv, dp
        G[3:6, 0:3] = Rk * dt
        G[0:3, 0:3] = 0.5 * Rk * dt * dt
        # gyro measurement noise nw affects dtheta
        G[6:9, 3:6] = np.eye(3) * dt
        # accel bias random walk nba affects dba
        G[9:12, 6:9] = np.eye(3) * dt
        # gyro bias random walk nbg affects dbg
        G[12:15, 9:12] = np.eye(3) * dt

        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = np.eye(3) * self.q_a
        Qc[3:6, 3:6] = np.eye(3) * self.q_w
        Qc[6:9, 6:9] = np.eye(3) * self.q_ba
        Qc[9:12, 9:12] = np.eye(3) * self.q_bg

        Q = G @ Qc @ G.T
        self.P = F @ self.P @ F.T + Q

    def update_pose_se3(self, T_wb_meas, R_meas_6x6):
        """
        Pose measurement update with residual:
          r_p = p_meas - p
          r_theta = Log(R_meas R^T)
        Measurement is 6D, affects dp and dtheta.
        """
        p_meas = T_wb_meas[:3, 3]
        R_meas = T_wb_meas[:3, :3]

        r_p = p_meas - self.p
        r_theta = so3_log(R_meas @ self.R.T)
        r = np.hstack([r_p, r_theta])  # (6,)

        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)

        S = H @ self.P @ H.T + R_meas_6x6
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ r

        dp = dx[0:3]
        dv = dx[3:6]
        dth = dx[6:9]
        dba = dx[9:12]
        dbg = dx[12:15]

        # Inject
        self.p = self.p + dp
        self.v = self.v + dv
        self.R = so3_exp(dth) @ self.R
        self.ba = self.ba + dba
        self.bg = self.bg + dbg

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P