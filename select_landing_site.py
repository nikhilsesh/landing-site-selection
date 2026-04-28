import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from loguru import logger
from satlas import compute_building_map
from safety_score_old import compute_safety_map

# Generate segmentation and safety maps
rgb_path = 'rgb_maps/davis_rgb.tif'
dem_path = 'dem_maps/davis_dem.tif'
segmentation_path = 'results/davis_rgb_segmented.png'
safety_path = 'results/davis_safety_score.png'

compute_building_map(rgb_path, segmentation_path, n_clusters=2)
compute_safety_map(dem_path, safety_path)

rgb          = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
segmented    = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
safety_score = cv2.imread(safety_path, cv2.IMREAD_GRAYSCALE)

# Ensure all images are the same size
if segmented.shape != safety_score.shape:
    logger.warning(f"Warning: Resizing safety_score from {safety_score.shape} to {segmented.shape}")
    safety_score = cv2.resize(safety_score, (segmented.shape[1], segmented.shape[0]))
if rgb.shape[:2] != segmented.shape:
    logger.warning(f"Warning: Resizing RGB from {rgb.shape} to {segmented.shape}")
    rgb = cv2.resize(rgb, (segmented.shape[1], segmented.shape[0]))

logger.info(f"Segmented image shape: {segmented.shape}")
logger.info(f"Safety score shape: {safety_score.shape}")
logger.info(f"RGB image shape: {rgb.shape}")

cluster_vals = np.unique(segmented)
cluster_vals = cluster_vals[cluster_vals > 0] if np.any(cluster_vals > 0) else cluster_vals

# Get two cluster values or use default threshold
cluster1, cluster2 = cluster_vals[:2] if len(cluster_vals) >= 2 else (0, 255)
if len(cluster_vals) < 2:
    logger.warning("Expected 2 clusters but found fewer. Using default threshold (0, 255).")

# Create masks for each cluster
mask_cluster1 = (segmented == cluster1).astype(np.uint8)
mask_cluster2 = (segmented == cluster2).astype(np.uint8)

#TODO: Determine which cluster is buildings (typically the smaller one?)
mask_buildings = mask_cluster2
mask_safe_areas = mask_cluster1

# Apply mask to safety score: only consider non-building areas
safety_score_masked = safety_score.copy().astype(np.float32)
safety_score_masked[mask_buildings == 1] = 0

# Find the point with maximum safety score in non-building areas
max_safety_idx = np.argmax(safety_score_masked)
max_y, max_x = np.unravel_index(max_safety_idx, safety_score_masked.shape)
max_safety_value = safety_score_masked[max_y, max_x]

logger.info(f"Landing site selected at: ({max_x}, {max_y})")
logger.info(f"Safety score at landing site: {max_safety_value}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Plot 0: Original RGB
if rgb is not None:
    axes[0].imshow(rgb)
    axes[0].plot(max_x, max_y, 'g*', markersize=20)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')

# Plot 1: Segmentation
axes[1].imshow(segmented, cmap='viridis')
axes[1].plot(max_x, max_y, 'g*', markersize=20)
axes[1].set_title('Building Segmentation')
axes[1].axis('off')

# Plot 2: Safety score
im2 = axes[2].imshow(safety_score, cmap='hot')
axes[2].plot(max_x, max_y, 'g*', markersize=20)
axes[2].set_title('Safety Score')
axes[2].axis('off')

# Plot 3: Combined overlay with colorized scheme
segmented_colored = cv2.applyColorMap((segmented / segmented.max() * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
safety_score_rgb = np.stack([safety_score, safety_score, safety_score], axis=2)
overlay = (segmented_colored * 0.6 + safety_score_rgb * 0.4).astype(np.uint8)
axes[3].imshow(overlay)
axes[3].plot(max_x, max_y, 'g*', markersize=20)
axes[3].set_title('Combined: Segmentation + Safety Score')
axes[3].axis('off')
plt.tight_layout()
plt.savefig('results/landing_site_selection.png', dpi=150, bbox_inches='tight')
logger.info(f"Saved 2x2 visualization to: results/landing_site_selection.png")
plt.show()

# this is just the last combined plot
fig_combined = plt.figure(figsize=(12, 10))
ax_combined = fig_combined.add_axes([0, 0, 1, 1])  # Full figure area
ax_combined.imshow(overlay)
ax_combined.plot(max_x, max_y, 'g*', markersize=30)
ax_combined.axis('off')
plt.savefig('results/landing_site_combined.png', dpi=150, bbox_inches='tight', pad_inches=0)
logger.info(f"Saved combined plot to: results/landing_site_combined.png")
plt.close()