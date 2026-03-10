import numpy as np
import matplotlib.pyplot as plt

def plot_tracks(t, gt_pos, est_pos, lidar_pos=None):
    plt.figure(figsize=(6,6))
    plt.plot(gt_pos[:,0], gt_pos[:,1], 'k-', label="GT")
    plt.plot(est_pos[:,0], est_pos[:,1], 'b-', label="EKF")
    if lidar_pos is not None and len(lidar_pos) > 0:
        lp = np.array(lidar_pos)
        plt.scatter(lp[:,0], lp[:,1], c='r', s=15, label="LiDAR/ICP updates")
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("2D track")
    plt.legend()
    plt.tight_layout()
    plt.show()