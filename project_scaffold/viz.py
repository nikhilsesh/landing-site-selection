import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "savefig.transparent": True,
})

def show_nonblocking(pause=0.001):
    plt.show(block=False)
    plt.pause(pause)  # lets the GUI event loop update

def plot_heightmap(X, Y, Z, title="Height map", cmap="terrain"):
    x0, x1 = float(X.min()), float(X.max())
    y0, y1 = float(Y.min()), float(Y.max())

    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap=cmap,
        aspect="equal",
    )
    plt.colorbar(label="elevation (m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.tight_layout()
    show_nonblocking()

def plot_xy_overlay(target_pts: np.ndarray, source_pts: np.ndarray, title="XY overlay"):
    """
    2D overlay in XY. Good quick sanity check for alignment.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(target_pts[:, 0], target_pts[:, 1], s=1, c="k", alpha=0.25, label="target (map)")
    plt.scatter(source_pts[:, 0], source_pts[:, 1], s=3, c="r", alpha=0.6, label="source (scan)")
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    show_nonblocking()

def plot_3d_overlay(target_pts: np.ndarray, source_pts: np.ndarray, title="3D overlay", elev=25, azim=-60):
    """
    3D scatter overlay. Slower for large clouds; consider downsampling.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
               s=1, c="k", alpha=0.15, label="target (map)")
    ax.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2],
               s=4, c="r", alpha=0.6, label="source (scan)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    plt.tight_layout()
    show_nonblocking()

def plot_safety_map(X, Y, safety_map, title="Safety score map"):
    """
    Plot normalized safety score map as heatmap.
    """

    plt.figure(figsize=(6, 5))
    plt.imshow(
        safety_map,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap=colormaps["RdYlGn_r"],
        aspect="equal",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.tight_layout()
    show_nonblocking()