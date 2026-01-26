import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from loguru import logger

rgb          = cv2.cvtColor(cv2.imread('rgb_maps/davis_rgb.tif'), cv2.COLOR_BGR2RGB)
segmented    = cv2.imread('results/davis_rgb_segmented.png', cv2.IMREAD_GRAYSCALE)
safety_score = cv2.imread('results/davis_safety_score.png', cv2.IMREAD_GRAYSCALE)

# Ensure all images are the same size
if segmented.shape != safety_score.shape:
    print(f"Warning: Resizing safety_score from {safety_score.shape} to {segmented.shape}")
    safety_score = cv2.resize(safety_score, (segmented.shape[1], segmented.shape[0]))

if rgb.shape[:2] != segmented.shape:
    print(f"Warning: Resizing RGB from {rgb.shape} to {segmented.shape}")
    rgb = cv2.resize(rgb, (segmented.shape[1], segmented.shape[0]))

print(f"Segmented image shape: {segmented.shape}")
print(f"Safety score shape: {safety_score.shape}")
print(f"RGB image shape: {rgb.shape}")

# Identify the two clusters
unique_values = np.unique(segmented)
print(f"Unique cluster values: {unique_values}")

# Assume cluster 0 or 255 is buildings, the other is non-building
# We'll determine this by assuming the larger area or try both
cluster_vals = unique_values[unique_values > 0]  # Skip 0 if present

if len(cluster_vals) == 0:
    cluster_vals = unique_values

if len(cluster_vals) >= 2:
    cluster1 = cluster_vals[0]
    cluster2 = cluster_vals[1]
else:
    print("Warning: Expected 2 clusters but found fewer. Using threshold at 128.")
    cluster1 = 0
    cluster2 = 255

print(f"Cluster 1: {cluster1}, Cluster 2: {cluster2}")

# Create masks for each cluster
mask_cluster1 = (segmented == cluster1).astype(np.uint8)
mask_cluster2 = (segmented == cluster2).astype(np.uint8)

# Determine which cluster is buildings (typically the smaller one or based on prior knowledge)
# For now, assume cluster2 is buildings (you can swap if needed)
mask_buildings = mask_cluster2
mask_safe_areas = mask_cluster1

print(f"Buildings pixels: {np.sum(mask_buildings)}")
print(f"Safe area pixels: {np.sum(mask_safe_areas)}")

# Apply mask to safety score: only consider non-building areas
safety_score_masked = safety_score.copy().astype(np.float32)
safety_score_masked[mask_buildings == 1] = 0

# Find the point with maximum safety score in non-building areas
max_safety_idx = np.argmax(safety_score_masked)
max_y, max_x = np.unravel_index(max_safety_idx, safety_score_masked.shape)
max_safety_value = safety_score_masked[max_y, max_x]

print(f"\nLanding site selected at: ({max_x}, {max_y})")
print(f"Safety score at landing site: {max_safety_value}")

# Create 2x2 visualization
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
# Create a combined visualization using colormapped segmentation
segmented_colored = cv2.applyColorMap((segmented / segmented.max() * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
# Blend with safety score for combined effect
safety_score_rgb = np.stack([safety_score, safety_score, safety_score], axis=2)
overlay = (segmented_colored * 0.6 + safety_score_rgb * 0.4).astype(np.uint8)
axes[3].imshow(overlay)
axes[3].plot(max_x, max_y, 'g*', markersize=20)
axes[3].set_title('Combined: Segmentation + Safety Score')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('results/landing_site_selection.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: results/landing_site_selection.png")
plt.show()

# Save combined plot without borders
fig_combined = plt.figure(figsize=(12, 10))
ax_combined = fig_combined.add_axes([0, 0, 1, 1])  # Full figure area
ax_combined.imshow(overlay)
ax_combined.plot(max_x, max_y, 'g*', markersize=30)
ax_combined.axis('off')
plt.savefig('results/landing_site_combined.png', dpi=150, bbox_inches='tight', pad_inches=0)
print(f"Saved combined plot to: results/landing_site_combined.png")
plt.close()