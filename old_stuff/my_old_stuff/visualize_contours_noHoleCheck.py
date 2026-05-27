"""
visualize_valid_contours.py

Visualizes each individual valid contour after applying:
1. Minimum area filter (6000 m²) from overlay_osm_on_binary_map.py
2. Minimum rectangle filter (15m × 400m) from runway_fit_new.py

Creates individual plots for each valid contour to examine shape and potential holes.
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import xy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def filter_by_min_rectangle(binary_image, min_width=15, min_height=400):
    """
    Keep only regions that can fit a rectangle of at least min_width × min_height.
    The rectangle can be rotated to best fit the region.
    
    Parameters:
    - binary_image: boolean or uint8 binary image
    - min_width: minimum smaller dimension (default 15)
    - min_height: minimum larger dimension (default 400)
    
    Returns:
    - filtered binary image
    """
    # Ensure binary image is uint8
    if binary_image.dtype == bool:
        binary_uint8 = binary_image.astype(np.uint8) * 255
    else:
        binary_uint8 = binary_image.astype(np.uint8)
    
    # Find all contours (connected components)
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    filtered_image = np.zeros_like(binary_uint8)
    
    for contour in contours:
        # Get minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        # rect = ((center_x, center_y), (width, height), angle)
        
        (center), (width, height), angle = rect
        
        # Get the smaller and larger dimensions
        smaller_dim = min(width, height)
        larger_dim = max(width, height)
        
        # Check if rectangle meets minimum size requirements
        if smaller_dim >= min_width and larger_dim >= min_height:
            # Draw the contour on the filtered image
            cv2.drawContours(filtered_image, [contour], -1, 255, -1)
    
    return filtered_image.astype(bool) if binary_image.dtype == bool else filtered_image


def visualize_individual_contours(region='norcoast8', 
                                  runway_width=15, 
                                  runway_length=400):
    """
    Load filtered landable areas and visualize each valid contour individually.
    
    Parameters:
    -----------
    region : str
        Region identifier
    runway_width : float
        Runway width in meters (for min rectangle filter)
    runway_length : float
        Runway length in meters (for min rectangle filter)
    """
    print("=" * 60)
    print(f"VISUALIZING VALID CONTOURS FOR REGION: {region}")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'valid_contour_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load filtered landable areas (already has 6000 m² area filter applied)
    filtered_path = f'results/{region}_filtered_landable_areas.tif'
    
    if not os.path.exists(filtered_path):
        print(f"ERROR: {filtered_path} not found!")
        print("Please run overlay_osm_on_binary_map.py first.")
        return None
    
    with rasterio.open(filtered_path) as src:
        landable_map = src.read(1).astype(bool)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    pixel_size = abs(transform.a)
    print(f"\nLoaded landable map: {landable_map.shape}")
    print(f"Pixel size: {pixel_size:.2f} meters")
    print(f"CRS: {crs}")
    
    # Count regions before min rectangle filter
    num_regions_before = cv2.connectedComponents(landable_map.astype(np.uint8))[0] - 1
    print(f"\nRegions after area filter (6000 m²): {num_regions_before}")
    
    # Apply additional min rectangle filter
    print(f"Applying min rectangle filter ({runway_width}m × {runway_length}m)...")
    landable_filtered = filter_by_min_rectangle(landable_map, 
                                                 min_width=runway_width/pixel_size,
                                                 min_height=runway_length/pixel_size)
    
    # Get connected components after filtering
    num_labels, labels = cv2.connectedComponents(landable_filtered.astype(np.uint8))
    num_valid_regions = num_labels - 1  # Subtract background
    
    print(f"Regions after min rectangle filter: {num_valid_regions}")
    print(f"\n{'=' * 60}")
    print(f"Creating {num_valid_regions} individual contour plots...")
    print(f"{'=' * 60}\n")
    
    # Load DEM for background elevation
    dem_path = f'dem_maps/{region}_dem_utm.tif'
    if not os.path.exists(dem_path):
        dem_path = f'dem_maps/{region.replace("norcoast8", "norcoast_b23")}_dem_utm.tif'
        if not os.path.exists(dem_path):
            print(f"Warning: DEM not found, will plot without elevation background")
            elevation = None
    
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            elevation = src.read(1, masked=True)
    else:
        elevation = None
    
    # Process each contour
    for region_label in range(1, num_labels):
        # Create binary mask for this contour only
        contour_mask = (labels == region_label)
        
        # Get bounding box to zoom in on contour
        rows, cols = np.where(contour_mask)
        if len(rows) == 0:
            continue
        
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Add padding
        padding = int(50 / pixel_size)  # 50 meter padding
        row_min = max(0, row_min - padding)
        row_max = min(contour_mask.shape[0], row_max + padding)
        col_min = max(0, col_min - padding)
        col_max = min(contour_mask.shape[1], col_max + padding)
        
        # Extract zoomed region
        contour_zoom = contour_mask[row_min:row_max, col_min:col_max]
        
        # Get geographic extent of this zoomed region
        top_left = xy(transform, row_min, col_min)
        bottom_right = xy(transform, row_max, col_max)
        extent_zoom = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]
        
        # Calculate contour statistics
        area_pixels = np.sum(contour_mask)
        area_m2 = area_pixels * (pixel_size ** 2)
        
        # Get minimum area rectangle for this contour
        contour_uint8 = contour_zoom.astype(np.uint8) * 255
        contours_cv, _ = cv2.findContours(contour_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_cv) > 0:
            rect = cv2.minAreaRect(contours_cv[0])
            (center), (width, height), angle = rect
            width_m = width * pixel_size
            height_m = height * pixel_size
            rect_area_m2 = width_m * height_m
        else:
            width_m, height_m, angle, rect_area_m2 = 0, 0, 0, 0
        
        # Check for holes (internal contours)
        contours_all, hierarchy = cv2.findContours(contour_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        num_holes = 0
        if hierarchy is not None:
            # hierarchy[0][i] = [next, previous, first_child, parent]
            # If parent >= 0, it's a hole
            num_holes = np.sum(hierarchy[0][:, 3] >= 0)
        
        print(f"Contour {region_label}/{num_valid_regions}:")
        print(f"  Area: {area_m2:.0f} m² ({area_pixels} pixels)")
        print(f"  Min rect: {min(width_m, height_m):.1f}m × {max(width_m, height_m):.1f}m (angle: {angle:.1f}°)")
        print(f"  Holes detected: {num_holes}")
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left plot: Contour with elevation background
        if elevation is not None:
            elevation_zoom = elevation[row_min:row_max, col_min:col_max]
            im = axes[0].imshow(elevation_zoom, cmap='terrain', extent=extent_zoom, aspect='auto', alpha=0.6)
        
        # Overlay contour in green
        contour_overlay = np.ma.masked_where(~contour_zoom, contour_zoom)
        axes[0].imshow(contour_overlay, cmap='Greens', extent=extent_zoom, aspect='auto', alpha=0.7)
        
        # Draw minimum area rectangle
        if len(contours_cv) > 0:
            box = cv2.boxPoints(rect)
            box_geo = []
            for point in box:
                # Convert from zoomed coordinates to original coordinates
                col_orig = col_min + point[0]
                row_orig = row_min + point[1]
                geo_point = xy(transform, row_orig, col_orig)
                box_geo.append(geo_point)
            box_geo = np.array(box_geo)
            
            # Close the rectangle
            box_geo_closed = np.vstack([box_geo, box_geo[0]])
            axes[0].plot(box_geo_closed[:, 0], box_geo_closed[:, 1], 'r--', linewidth=2, 
                        label=f'Min rect: {min(width_m, height_m):.0f}×{max(width_m, height_m):.0f}m')
        
        # Add colorbar for elevation
        if elevation is not None:
            cbar = plt.colorbar(im, ax=axes[0], label='Elevation (m)', fraction=0.046, pad=0.04)

        axes[0].set_title(f'Contour {region_label} - Geographic View')
        axes[0].set_xlabel('Easting (m)')
        axes[0].set_ylabel('Northing (m)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: Binary mask (better for seeing holes)
        axes[1].imshow(contour_zoom, cmap='gray', aspect='auto')
        axes[1].set_title(f'Contour {region_label} - Binary Mask\n(White = landable, Black = obstacle/hole)')
        axes[1].set_xlabel('Column (pixels)')
        axes[1].set_ylabel('Row (pixels)')
        
        # Add overall statistics as text
        stats_text = f"Area: {area_m2:.0f} m² ({area_pixels} px)\n"
        stats_text += f"Min Rect: {min(width_m, height_m):.1f}×{max(width_m, height_m):.1f} m\n"
        stats_text += f"Rect Angle: {angle:.1f}°\n"
        stats_text += f"Holes: {num_holes}"
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'Region: {region} | Contour {region_label} of {num_valid_regions}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = f'{output_dir}/{region}_contour_{region_label:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to: {output_path}\n")
    
    print("=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nAll {num_valid_regions} contour plots saved to: {output_dir}/")
    print(f"\nSummary:")
    print(f"  Total valid contours: {num_valid_regions}")
    print(f"  Output directory: {output_dir}/")
    print(f"  Files: {region}_contour_001.png to {region}_contour_{num_valid_regions:03d}.png")
    
    return num_valid_regions


if __name__ == "__main__":
    # Configuration
    region = 'norcoast8'
    runway_length = 400  # meters
    runway_width = 15    # meters
    
    # Run visualization
    num_contours = visualize_individual_contours(
        region=region,
        runway_width=runway_width,
        runway_length=runway_length
    )
    
    if num_contours is not None and num_contours > 0:
        print(f"\n✓ Successfully created {num_contours} contour visualizations")
    else:
        print("\n✗ No valid contours found or error occurred")