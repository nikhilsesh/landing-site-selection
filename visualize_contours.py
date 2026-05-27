"""
visualize_valid_contours_with_hole_check.py

Visualizes each individual valid contour and checks for holes BEFORE and AFTER
the minimum rectangle filter is applied.

Shows:
1. Original contours from overlay_osm_on_binary_map.py (with holes preserved)
2. Contours after minimum rectangle filter (holes may be filled)
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import xy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def detect_holes(contour_binary):
    """
    Detect holes in a binary contour image.
    
    Parameters:
    -----------
    contour_binary : np.ndarray
        Binary image of a single contour (uint8, 0 or 255)
    
    Returns:
    --------
    num_holes : int
        Number of holes detected
    hole_areas : list
        Area of each hole in pixels
    """
    contours_all, hierarchy = cv2.findContours(contour_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    num_holes = 0
    hole_areas = []
    
    if hierarchy is not None and len(contours_all) > 0:
        # hierarchy[0][i] = [next, previous, first_child, parent]
        # If parent >= 0, it's a hole (child of an external contour)
        for i in range(len(contours_all)):
            if hierarchy[0][i][3] >= 0:  # Has a parent = it's a hole
                num_holes += 1
                hole_areas.append(cv2.contourArea(contours_all[i]))
    
    return num_holes, hole_areas

# CORRECTED function to only analyze parent holes
def filter_by_min_rectangle(binary_image, min_width=15, min_height=400):
    """
    Correctly filters by minimum rectangle, treating holes as part of their parent region.
    """
    if binary_image.dtype == bool:
        binary_uint8 = binary_image.astype(np.uint8) * 255
    else:
        binary_uint8 = binary_image.astype(np.uint8)
    
    # Use RETR_CCOMP to get hierarchy information
    contours, hierarchy = cv2.findContours(binary_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_image = np.zeros_like(binary_uint8)
    
    if hierarchy is None or len(contours) == 0:
        return filtered_image.astype(bool) if binary_image.dtype == bool else filtered_image
    
    # Process only external contours (parent == -1), ignore holes
    for i, contour in enumerate(contours):
        # hierarchy[0][i] = [next, previous, first_child, parent]
        parent = hierarchy[0][i][3]
        
        # Skip holes (they have a parent >= 0)
        if parent >= 0:
            continue
        
        # This is an external contour
        rect = cv2.minAreaRect(contour)
        (center), (width, height), angle = rect
        
        smaller_dim = min(width, height)
        larger_dim = max(width, height)
        
        if smaller_dim >= min_width and larger_dim >= min_height:
            # Draw the external contour (fills holes)
            cv2.drawContours(filtered_image, [contour], -1, 255, -1)
    
    return filtered_image.astype(bool) if binary_image.dtype == bool else filtered_image

def visualize_contours_with_hole_analysis(region='norcoast8', 
                                          runway_width=15, 
                                          runway_length=400):
    """
    Load filtered landable areas and analyze holes before and after min rectangle filter.
    """
    print("=" * 60)
    print(f"HOLE ANALYSIS FOR REGION: {region}")
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
        landable_map_original = src.read(1).astype(bool)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    pixel_size = abs(transform.a)
    print(f"\nLoaded landable map: {landable_map_original.shape}")
    print(f"Pixel size: {pixel_size:.2f} meters")
    print(f"CRS: {crs}")
    
    # Get connected components BEFORE min rectangle filter
    num_labels_original, labels_original = cv2.connectedComponents(landable_map_original.astype(np.uint8))
    num_regions_original = num_labels_original - 1
    print(f"\nRegions after area filter (6000 m²): {num_regions_original}")
    
    # Apply min rectangle filter
    print(f"Applying min rectangle filter ({runway_width}m × {runway_length}m)...")
    landable_map_filtered = filter_by_min_rectangle(landable_map_original, 
                                                     min_width=runway_width/pixel_size,
                                                     min_height=runway_length/pixel_size)
    
    # Get connected components AFTER min rectangle filter
    num_labels_filtered, labels_filtered = cv2.connectedComponents(landable_map_filtered.astype(np.uint8))
    num_regions_filtered = num_labels_filtered - 1
    print(f"Regions after min rectangle filter: {num_regions_filtered}")
    
    # Load DEM for background elevation
    dem_path = f'dem_maps/{region}_dem_utm.tif'
    if not os.path.exists(dem_path):
        dem_path = f'dem_maps/{region.replace("norcoast8", "norcoast_b23")}_dem_utm.tif'
    
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            elevation = src.read(1, masked=True)
    else:
        print(f"Warning: DEM not found, will plot without elevation background")
        elevation = None
    
    print(f"\n{'=' * 60}")
    print(f"ANALYZING HOLES IN EACH CONTOUR")
    print(f"{'=' * 60}\n")
    
    # Track statistics
    total_holes_before = 0
    total_holes_after = 0
    regions_with_holes_before = 0
    regions_with_holes_after = 0
    
    # NEW APPROACH: Instead of matching filtered→original, match original→filtered
    # Process each ORIGINAL region and see if it survived filtering
    regions_processed = 0
    
    for region_label_original in range(1, num_labels_original):
        # Get original contour mask
        contour_mask_original = (labels_original == region_label_original)
        
        # Check if this region still exists after filtering (find overlap)
        overlap = contour_mask_original & landable_map_filtered
        
        if not np.any(overlap):
            # This region was completely eliminated by the filter
            print(f"Region {region_label_original} (original) was eliminated by filter - skipping")
            continue
        
        # Find which filtered region(s) this corresponds to
        # Get the labels of filtered regions that overlap with this original region
        filtered_labels_in_overlap = np.unique(labels_filtered[overlap])
        filtered_labels_in_overlap = filtered_labels_in_overlap[filtered_labels_in_overlap > 0]  # Remove background
        
        if len(filtered_labels_in_overlap) == 0:
            print(f"Region {region_label_original} (original) has no valid filtered overlap - skipping")
            continue
        
        # Use the filtered label with the most overlap
        max_overlap_count = 0
        region_label_filtered = 0
        for filt_label in filtered_labels_in_overlap:
            overlap_count = np.sum((labels_filtered == filt_label) & contour_mask_original)
            if overlap_count > max_overlap_count:
                max_overlap_count = overlap_count
                region_label_filtered = filt_label
        
        regions_processed += 1
        contour_mask_filtered = (labels_filtered == region_label_filtered)
        
        # Get bounding box from ORIGINAL region (to show the full context)
        rows_orig, cols_orig = np.where(contour_mask_original)
        if len(rows_orig) == 0:
            continue
        
        row_min, row_max = rows_orig.min(), rows_orig.max()
        col_min, col_max = cols_orig.min(), cols_orig.max()
        
        # Add padding
        padding = int(50 / pixel_size)  # 50 meter padding
        row_min = max(0, row_min - padding)
        row_max = min(contour_mask_original.shape[0], row_max + padding)
        col_min = max(0, col_min - padding)
        col_max = min(contour_mask_original.shape[1], col_max + padding)
        
        # Extract zoomed regions
        contour_zoom_original = contour_mask_original[row_min:row_max, col_min:col_max]
        contour_zoom_filtered = contour_mask_filtered[row_min:row_max, col_min:col_max]
        
        # Get geographic extent
        top_left = xy(transform, row_min, col_min)
        bottom_right = xy(transform, row_max, col_max)
        extent_zoom = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]
        
        # Calculate contour statistics
        area_pixels_original = np.sum(contour_mask_original)
        area_pixels_filtered = np.sum(contour_mask_filtered)
        area_m2_original = area_pixels_original * (pixel_size ** 2)
        area_m2_filtered = area_pixels_filtered * (pixel_size ** 2)
        
        # Detect holes BEFORE filter
        contour_uint8_original = contour_zoom_original.astype(np.uint8) * 255
        num_holes_before, hole_areas_before = detect_holes(contour_uint8_original)
        
        # Detect holes AFTER filter
        contour_uint8_filtered = contour_zoom_filtered.astype(np.uint8) * 255
        num_holes_after, hole_areas_after = detect_holes(contour_uint8_filtered)
        
        # Update statistics
        total_holes_before += num_holes_before
        total_holes_after += num_holes_after
        if num_holes_before > 0:
            regions_with_holes_before += 1
        if num_holes_after > 0:
            regions_with_holes_after += 1
        
        # Get minimum area rectangle (from filtered version)
        contours_cv, _ = cv2.findContours(contour_uint8_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_cv) > 0:
            rect = cv2.minAreaRect(contours_cv[0])
            (center), (width, height), angle = rect
            width_m = width * pixel_size
            height_m = height * pixel_size
        else:
            width_m, height_m, angle = 0, 0, 0
        
        # Print statistics
        print(f"Region {regions_processed} (original #{region_label_original} → filtered #{region_label_filtered}):")
        print(f"  BEFORE min rect filter:")
        print(f"    Area: {area_m2_original:.0f} m² ({area_pixels_original} pixels)")
        print(f"    Holes: {num_holes_before}", end="")
        if num_holes_before > 0:
            total_hole_area = sum(hole_areas_before) * (pixel_size ** 2)
            print(f" (total area: {total_hole_area:.0f} m²)")
        else:
            print()
        print(f"  AFTER min rect filter:")
        print(f"    Area: {area_m2_filtered:.0f} m² ({area_pixels_filtered} pixels)")
        print(f"    Holes: {num_holes_after}")
        print(f"    Min rect: {min(width_m, height_m):.1f}m × {max(width_m, height_m):.1f}m (angle: {angle:.1f}°)")
        
        if num_holes_before > 0 and num_holes_after == 0:
            print(f"  ✓ HOLES WERE FILLED by min rectangle filter!")
        if area_m2_filtered < area_m2_original * 0.5:
            print(f"  ⚠️  WARNING: Region lost {100*(1-area_m2_filtered/area_m2_original):.1f}% of its area!")
        
        # Create figure with 3 columns: original, filtered, comparison
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # ===== LEFT: Original contour (BEFORE filter) =====
        if elevation is not None:
            elevation_zoom = elevation[row_min:row_max, col_min:col_max]
            axes[0].imshow(elevation_zoom, cmap='terrain', extent=extent_zoom, aspect='auto', alpha=0.6)
        
        contour_overlay_original = np.ma.masked_where(~contour_zoom_original, contour_zoom_original)
        axes[0].imshow(contour_overlay_original, cmap='Greens', extent=extent_zoom, aspect='auto', alpha=0.7)
        
        axes[0].set_title(f'BEFORE Min Rect Filter\nHoles: {num_holes_before}', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Easting (m)')
        axes[0].set_ylabel('Northing (m)')
        axes[0].grid(True, alpha=0.3)
        
        # ===== MIDDLE: Filtered contour (AFTER filter) =====
        if elevation is not None:
            im = axes[1].imshow(elevation_zoom, cmap='terrain', extent=extent_zoom, aspect='auto', alpha=0.6)
        
        contour_overlay_filtered = np.ma.masked_where(~contour_zoom_filtered, contour_zoom_filtered)
        axes[1].imshow(contour_overlay_filtered, cmap='Greens', extent=extent_zoom, aspect='auto', alpha=0.7)
        
        # Draw minimum area rectangle
        if len(contours_cv) > 0:
            box = cv2.boxPoints(rect)
            box_geo = []
            for point in box:
                col_orig = col_min + point[0]
                row_orig = row_min + point[1]
                geo_point = xy(transform, row_orig, col_orig)
                box_geo.append(geo_point)
            box_geo = np.array(box_geo)
            box_geo_closed = np.vstack([box_geo, box_geo[0]])
            axes[1].plot(box_geo_closed[:, 0], box_geo_closed[:, 1], 'r--', linewidth=2, 
                        label=f'Min rect: {min(width_m, height_m):.0f}×{max(width_m, height_m):.0f}m')
        
        # Add colorbar for elevation
        if elevation is not None:
            cbar = plt.colorbar(im, ax=axes[1], label='Elevation (m)', fraction=0.046, pad=0.04)
        
        axes[1].set_title(f'AFTER Min Rect Filter\nHoles: {num_holes_after}', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Easting (m)')
        axes[1].set_ylabel('Northing (m)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # ===== RIGHT: Binary comparison =====
        # Create RGB overlay showing differences
        comparison = np.zeros((*contour_zoom_original.shape, 3))
        
        # Green = present in both (unchanged)
        both = contour_zoom_original & contour_zoom_filtered
        comparison[both] = [0, 1, 0]
        
        # Yellow = only in original (removed by filter)
        only_original = contour_zoom_original & (~contour_zoom_filtered)
        comparison[only_original] = [1, 1, 0]
        
        # Blue = only in filtered (filled holes)
        only_filtered = (~contour_zoom_original) & contour_zoom_filtered
        comparison[only_filtered] = [0.3, 0.5, 1]
        
        axes[2].imshow(comparison, aspect='auto')
        axes[2].set_title('Comparison\n(Blue = Filled, Yellow = Removed)', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('Column (pixels)')
        axes[2].set_ylabel('Row (pixels)')
        
        # Add legend for comparison
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0, 1, 0], label='Unchanged'),
            Patch(facecolor=[0.3, 0.5, 1], label='Filled holes'),
            Patch(facecolor=[1, 1, 0], label='Removed area')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right')
        
        # Add overall statistics as text
        stats_text = f"Original Area: {area_m2_original:.0f} m²\n"
        stats_text += f"Filtered Area: {area_m2_filtered:.0f} m²\n"
        stats_text += f"Difference: {area_m2_filtered - area_m2_original:.0f} m²\n"
        stats_text += f"\nHoles Before: {num_holes_before}\n"
        stats_text += f"Holes After: {num_holes_after}"
        
        if num_holes_before > 0 and num_holes_after == 0:
            stats_text += f"\n\n✓ HOLES FILLED!"
        if area_m2_filtered < area_m2_original * 0.5:
            stats_text += f"\n\n⚠️ AREA REDUCED!"
        
        axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.suptitle(f'Region: {region} | Original #{region_label_original} → Filtered #{region_label_filtered}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = f'{output_dir}/{region}_orig{region_label_original:03d}_filt{region_label_filtered:03d}_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to: {output_path}\n")
    
    # Print summary statistics
    print("=" * 60)
    print("HOLE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nBEFORE min rectangle filter:")
    print(f"  Total regions: {num_regions_original}")
    print(f"  Regions with holes: {regions_with_holes_before}")
    print(f"  Total holes: {total_holes_before}")
    
    print(f"\nAFTER min rectangle filter:")
    print(f"  Total regions: {num_regions_filtered}")
    print(f"  Regions processed: {regions_processed}")
    print(f"  Regions with holes: {regions_with_holes_after}")
    print(f"  Total holes: {total_holes_after}")
    
    print(f"\nCHANGES:")
    print(f"  Regions eliminated: {num_regions_original - regions_processed}")
    print(f"  Holes filled: {total_holes_before - total_holes_after}")
    
    if total_holes_before > 0:
        print(f"\n⚠️  WARNING: {total_holes_before} holes were present in original data!")
        print(f"   The min rectangle filter filled {total_holes_before - total_holes_after} of them.")
        print(f"   Your runway optimization may place runways over obstacles that were holes.")
    else:
        print(f"\n✓ Good news: No holes detected in original filtered landable areas.")
        print(f"  Your OSM obstacle filtering created separate regions rather than holes.")
    
    if num_regions_original - regions_processed > 0:
        print(f"\n⚠️  NOTE: {num_regions_original - regions_processed} regions were completely eliminated by the filter.")
        print(f"   They likely didn't meet the minimum rectangle size requirement.")
    
    print(f"\n{'=' * 60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nAll {regions_processed} contour plots saved to: {output_dir}/")
    
    return {
        'num_regions_original': num_regions_original,
        'num_regions_filtered': num_regions_filtered,
        'regions_processed': regions_processed,
        'total_holes_before': total_holes_before,
        'total_holes_after': total_holes_after,
        'regions_with_holes_before': regions_with_holes_before,
        'regions_with_holes_after': regions_with_holes_after
    }


if __name__ == "__main__":
    # Configuration
    region = 'norcoast8'
    runway_length = 400  # meters
    runway_width = 15    # meters
    
    # Run hole analysis
    results = visualize_contours_with_hole_analysis(
        region=region,
        runway_width=runway_width,
        runway_length=runway_length
    )
    
    if results is not None:
        print(f"\n{'=' * 60}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(f"Original regions: {results['num_regions_original']}")
        print(f"Filtered regions: {results['num_regions_filtered']}")
        print(f"Regions processed: {results['regions_processed']}")
        print(f"Holes found before filter: {results['total_holes_before']}")
        print(f"Holes found after filter: {results['total_holes_after']}")
        
        if results['total_holes_before'] > 0:
            print(f"\n⚠️  ACTION REQUIRED:")
            print(f"   Your data has holes that are being filled by the rectangle filter.")
            print(f"   Consider modifying filter_by_min_rectangle() to preserve holes,")
            print(f"   or ensure your runway optimization avoids filled regions.")
        else:
            print(f"\n✓ No action needed: Your data naturally has no holes.")
        
        if results['regions_processed'] < results['num_regions_original']:
            print(f"\n⚠️  {results['num_regions_original'] - results['regions_processed']} regions were eliminated.")
            print(f"   They didn't meet the 15m × 400m minimum rectangle requirement.")