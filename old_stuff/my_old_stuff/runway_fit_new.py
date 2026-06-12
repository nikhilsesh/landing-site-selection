"""
runway_fit_new.py

Fits optimal 400m runway lines within filtered landable areas.
Evaluates gradient metrics and checks for 15m width buffers.
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import rowcol, xy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from scipy.ndimage import binary_dilation
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


def compute_gradient_along_line(elevation, transform, pixel_coords):
    """
    Compute gradient metrics along a line defined by pixel coordinates.
    
    Parameters:
    -----------
    elevation : ndarray
        DEM elevation data (meters)
    transform : Affine
        Geotransform from rasterio
    pixel_coords : list of tuples
        List of (row, col) pixel coordinates along the line
    
    Returns:
    --------
    dict with keys:
        - 'gradients': array of gradients between consecutive points (degrees)
        - 'mean_abs_gradient': mean absolute gradient (degrees)
        - 'max_abs_gradient': maximum absolute gradient (degrees)
        - 'std_gradient': standard deviation of gradients (degrees)
        - 'elevations': array of elevations along the line
    """
    # Get pixel size in meters
    pixel_size = abs(transform.a)  # Assume square pixels in UTM
    
    # Extract elevations along the line
    elevations = []
    for row, col in pixel_coords:
        if 0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1]:
            elevations.append(elevation[row, col])
        else:
            return None  # Line goes out of bounds
    
    elevations = np.array(elevations)
    
    # Check for masked/invalid values
    if np.ma.is_masked(elevations) and np.any(elevations.mask):
        return None  # Line contains invalid data
    
    # Compute gradients between consecutive points
    distances = [pixel_size * np.hypot(pixel_coords[i+1][0] - pixel_coords[i][0],
                                        pixel_coords[i+1][1] - pixel_coords[i][1])
                 for i in range(len(pixel_coords) - 1)]
    
    elevation_diffs = np.diff(elevations)
    gradients_rad = np.arctan2(elevation_diffs, distances)
    gradients_deg = np.degrees(gradients_rad)
    
    return {
        'gradients': gradients_deg,
        'mean_abs_gradient': np.mean(np.abs(gradients_deg)),
        'max_abs_gradient': np.max(np.abs(gradients_deg)),
        'std_gradient': np.std(gradients_deg),
        'elevations': elevations,
        'valid': True
    }


def generate_line_pixels(center_row, center_col, angle_deg, length_meters, pixel_size):
    """
    Generate pixel coordinates for a line centered at (center_row, center_col).
    
    Parameters:
    -----------
    center_row, center_col : int
        Center pixel coordinates
    angle_deg : float
        Angle in degrees (0 = East, 90 = North)
    length_meters : float
        Total length of line in meters
    pixel_size : float
        Pixel size in meters
    
    Returns:
    --------
    list of (row, col) tuples
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Half-length in pixels
    half_length_pixels = (length_meters / 2.0) / pixel_size
    
    # Number of sample points (sample every pixel)
    num_points = int(2 * half_length_pixels) + 1
    
    # Generate points along the line
    t = np.linspace(-half_length_pixels, half_length_pixels, num_points)
    
    # Line direction (angle measured from East, CCW)
    # In image coordinates: col increases East, row increases South
    cols = center_col + t * np.cos(angle_rad)
    rows = center_row - t * np.sin(angle_rad)  # Negative because row increases downward
    
    pixel_coords = [(int(round(r)), int(round(c))) for r, c in zip(rows, cols)]
    
    return pixel_coords


def check_width_buffer(landable_map, pixel_coords, buffer_width_meters, pixel_size):
    """
    Check if a buffer of specified width exists on both sides of the line.
    
    Parameters:
    -----------
    landable_map : ndarray (bool)
        Binary map of landable areas
    pixel_coords : list of tuples
        Line pixel coordinates
    buffer_width_meters : float
        Required buffer width (meters) on each side
    pixel_size : float
        Pixel size in meters
    
    Returns:
    --------
    bool : True if buffer requirement is satisfied
    """
    buffer_pixels = int(np.ceil(buffer_width_meters / pixel_size))
    
    # Create a structuring element for dilation
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*buffer_pixels+1, 2*buffer_pixels+1))
    
    # Create a temporary image with just this line
    line_image = np.zeros_like(landable_map, dtype=np.uint8)
    for row, col in pixel_coords:
        if 0 <= row < line_image.shape[0] and 0 <= col < line_image.shape[1]:
            line_image[row, col] = 1
    
    # Dilate to get the required buffer zone
    buffered = cv2.dilate(line_image, struct, iterations=1)
    
    # Check if all buffered pixels are landable
    required_pixels = buffered > 0
    available_pixels = landable_map & required_pixels
    
    # Return True if all required pixels are available
    return np.sum(available_pixels) == np.sum(required_pixels)

def find_optimal_runways(region='norcoast8', 
                         runway_length=400,
                         runway_width=15,
                         angle_step=5,
                         spacing_meters=50):
    """
    Find optimal runway placements in filtered landable areas.
    
    Parameters:
    -----------
    region : str
        Region identifier
    runway_length : float
        Runway length in meters
    runway_width : float
        Runway width in meters (for buffer check)
    angle_step : float
        Angular resolution for rotation search (degrees)
    spacing_meters : float
        Spacing between candidate center points (meters)
    
    Returns:
    --------
    dict with results and visualization data
    """
    print("=" * 60)
    print(f"RUNWAY FITTING FOR REGION: {region}")
    print("=" * 60)
    
    # Load filtered landable areas
    filtered_path = f'results/{region}_filtered_landable_areas.tif'
    
    if not os.path.exists(filtered_path):
        print(f"ERROR: {filtered_path} not found!")
        print("Please run overlay_osm_on_binary_map.py first.")
        return None
    
    with rasterio.open(filtered_path) as src:
        landable_map = src.read(1).astype(bool)
        transform = src.transform
        profile = src.profile
        crs = src.crs
    
    pixel_size = abs(transform.a)
    print(f"\nLoaded landable map: {landable_map.shape}")
    print(f"Pixel size: {pixel_size:.2f} meters")
    print(f"CRS: {crs}")
    
    # Apply additional min rectangle filter
    print(f"\nApplying min rectangle filter ({runway_width}m × {runway_length}m)...")
    landable_filtered = filter_by_min_rectangle(landable_map, 
                                                 min_width=runway_width/pixel_size,
                                                 min_height=runway_length/pixel_size)
    
    num_regions_before = cv2.connectedComponents(landable_map.astype(np.uint8))[0] - 1
    num_regions_after = cv2.connectedComponents(landable_filtered.astype(np.uint8))[0] - 1
    print(f"Regions before filter: {num_regions_before}")
    print(f"Regions after filter: {num_regions_after}")
    
    landable_map = landable_filtered
    
    # Get connected components (contours) for per-region analysis
    num_labels, labels = cv2.connectedComponents(landable_map.astype(np.uint8))
    print(f"\nIdentifying {num_labels - 1} separate landable regions...")
    
    # Load DEM for gradient computation
    dem_path = f'dem_maps/{region}_dem_utm.tif'
    if not os.path.exists(dem_path):
        # Try without _utm suffix
        dem_path = f'dem_maps/{region.replace("norcoast8", "norcoast_b23")}_dem_utm.tif'
        if not os.path.exists(dem_path):
            print(f"ERROR: DEM file not found at {dem_path}")
            return None
    
    with rasterio.open(dem_path) as src:
        elevation = src.read(1, masked=True)
        dem_transform = src.transform
    
    print(f"\nLoaded DEM: {elevation.shape}")
    print(f"Elevation range: {elevation.min():.1f} to {elevation.max():.1f} meters")
    
    # Verify transforms match
    if transform != dem_transform:
        print("WARNING: Landable map and DEM transforms don't match!")
        print(f"Landable: {transform}")
        print(f"DEM: {dem_transform}")
    
    # Find candidate center points (on a grid within landable areas)
    spacing_pixels = int(spacing_meters / pixel_size)
    rows, cols = np.where(landable_map)
    
    # Subsample to create grid
    candidate_points = []
    for r in range(0, landable_map.shape[0], spacing_pixels):
        for c in range(0, landable_map.shape[1], spacing_pixels):
            if landable_map[r, c]:
                # Find which region (label) this point belongs to
                region_label = labels[r, c]
                candidate_points.append((r, c, region_label))
    
    print(f"\nSearching {len(candidate_points)} candidate positions × {int(180/angle_step)} angles...")
    print(f"Total evaluations: {len(candidate_points) * int(180/angle_step)}")
    
    # Search for valid runway lines
    # Store runways organized by region
    runways_by_region = {}  # {region_label: [list of runways]}
    
    angles = np.arange(0, 180, angle_step)
    
    for idx, (center_row, center_col, region_label) in enumerate(candidate_points):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(candidate_points)} positions...")
        
        for angle in angles:
            # Generate line pixels
            pixel_coords = generate_line_pixels(center_row, center_col, angle, 
                                                runway_length, pixel_size)
            
            # Check if all pixels are within landable area AND in the same region
            valid_line = True
            for row, col in pixel_coords:
                if (row < 0 or row >= landable_map.shape[0] or 
                    col < 0 or col >= landable_map.shape[1] or 
                    not landable_map[row, col] or
                    labels[row, col] != region_label):  # Must stay in same region
                    valid_line = False
                    break
            
            if not valid_line:
                continue
            
            # Compute gradient metrics
            gradient_metrics = compute_gradient_along_line(elevation, transform, pixel_coords)
            
            if gradient_metrics is None:
                continue
            
            # Check width buffer
            has_width = check_width_buffer(landable_map, pixel_coords, 
                                          runway_width/2, pixel_size)
            
            # Store valid runway
            runway = {
                'center': (center_row, center_col),
                'angle': angle,
                'pixel_coords': pixel_coords,
                'gradient_metrics': gradient_metrics,
                'has_width_buffer': has_width,
                'score': gradient_metrics['max_abs_gradient'],  # Lower is better
                'region_label': region_label
            }
            
            # Add to region-specific list
            if region_label not in runways_by_region:
                runways_by_region[region_label] = []
            runways_by_region[region_label].append(runway)
    
    # Flatten to get all valid runways
    valid_runways = []
    for runways in runways_by_region.values():
        valid_runways.extend(runways)
    
    print(f"\n" + "=" * 60)
    print(f"FOUND {len(valid_runways)} VALID RUNWAY LINES")
    print(f"Across {len(runways_by_region)} regions")
    print("=" * 60)
    
    if len(valid_runways) == 0:
        print("No valid runways found!")
        return None
    
    # Find best runway in each region
    best_per_region = {}
    for region_label, runways in runways_by_region.items():
        if len(runways) > 0:
            best = min(runways, key=lambda r: r['score'])
            best_per_region[region_label] = best
            print(f"\nRegion {region_label}: {len(runways)} runways, best max gradient = {best['score']:.2f}°")
    
    # Count how many satisfy width requirement
    with_width = sum(1 for r in valid_runways if r['has_width_buffer'])
    print(f"\nTotal runways with {runway_width}m width buffer: {with_width} ({100*with_width/len(valid_runways):.1f}%)")
    
    # Find overall best runway (minimum max gradient across all regions)
    overall_best = min(valid_runways, key=lambda r: r['score'])
    print(f"\n" + "=" * 60)
    print(f"OVERALL BEST RUNWAY (across all regions):")
    print("=" * 60)
    print(f"  Region label: {overall_best['region_label']}")
    print(f"  Center: row={overall_best['center'][0]}, col={overall_best['center'][1]}")
    print(f"  Angle: {overall_best['angle']:.1f}°")
    print(f"  Max gradient: {overall_best['gradient_metrics']['max_abs_gradient']:.2f}°")
    print(f"  Mean gradient: {overall_best['gradient_metrics']['mean_abs_gradient']:.2f}°")
    print(f"  Std gradient: {overall_best['gradient_metrics']['std_gradient']:.2f}°")
    print(f"  Has width buffer: {overall_best['has_width_buffer']}")
    
    # Create visualization
    visualize_runways(region, landable_map, elevation, transform, 
                     valid_runways, overall_best, best_per_region, runway_width, pixel_size)
    
    return {
        'valid_runways': valid_runways,
        'best_runway': overall_best,
        'best_per_region': best_per_region,
        'runways_by_region': runways_by_region,
        'landable_map': landable_map,
        'elevation': elevation,
        'transform': transform
    }

def visualize_runways(region, landable_map, elevation, transform, 
                     valid_runways, overall_best, best_per_region, runway_width, pixel_size):
    """
    Create visualization showing all valid runways, best runway per region, and overall best.
    """
    print(f"\nCreating visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Get geographic extent for plotting
    height, width = landable_map.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    extent = [left, right, bottom, top]
    
    # 1. Landable areas with all valid runway lines
    ax = axes[0, 0]
    ax.imshow(landable_map, cmap='Greens', extent=extent, aspect='auto', alpha=0.6)
    
    # Draw all valid runways (cyan for lines without width, yellow for lines with width)
    for runway in valid_runways:
        coords = runway['pixel_coords']
        # Convert pixel coords to geographic coords
        geo_coords = [transform * (col, row) for row, col in coords]
        xs = [c[0] for c in geo_coords]
        ys = [c[1] for c in geo_coords]
        
        color = 'yellow' if runway['has_width_buffer'] else 'cyan'
        alpha = 0.3
        ax.plot(xs, ys, color=color, linewidth=1, alpha=alpha)
    
    ax.set_title(f'All Valid Runway Lines (n={len(valid_runways)})\nCyan=line only, Yellow=with {runway_width}m buffer')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')
    
    # 2. Best runway per region (blue) + overall best (red)
    ax = axes[0, 1]
    ax.imshow(landable_map, cmap='Greens', extent=extent, aspect='auto', alpha=0.6)
    
    # Draw all valid runways in light gray
    for runway in valid_runways:
        coords = runway['pixel_coords']
        geo_coords = [transform * (col, row) for row, col in coords]
        xs = [c[0] for c in geo_coords]
        ys = [c[1] for c in geo_coords]
        ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=0.2)
    
    # Draw best runway per region in blue
    for region_label, best_runway in best_per_region.items():
        coords = best_runway['pixel_coords']
        geo_coords = [transform * (col, row) for row, col in coords]
        xs = [c[0] for c in geo_coords]
        ys = [c[1] for c in geo_coords]
        
        # Only label the first one for legend
        if region_label == list(best_per_region.keys())[0]:
            ax.plot(xs, ys, color='blue', linewidth=2.5, alpha=0.8, label=f'Best per region (n={len(best_per_region)})')
        else:
            ax.plot(xs, ys, color='blue', linewidth=2.5, alpha=0.8)
    
    # Draw overall best runway in red (on top)
    best_coords = overall_best['pixel_coords']
    best_geo_coords = [transform * (col, row) for row, col in best_coords]
    best_xs = [c[0] for c in best_geo_coords]
    best_ys = [c[1] for c in best_geo_coords]
    ax.plot(best_xs, best_ys, color='red', linewidth=4, alpha=0.9, label='Overall best')
    ax.plot(best_xs, best_ys, color='white', linewidth=2, alpha=1.0)
    
    ax.set_title(f'Best Runways\nOverall best: {overall_best["gradient_metrics"]["max_abs_gradient"]:.2f}° max gradient')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.legend()
    
    # 3. Elevation profile of overall best runway
    ax = axes[1, 0]
    elevations = overall_best['gradient_metrics']['elevations']
    distance = np.linspace(0, 400, len(elevations))
    ax.plot(distance, elevations, 'b-', linewidth=2)
    ax.fill_between(distance, elevations, alpha=0.3)
    ax.set_xlabel('Distance along runway (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Profile of Overall Best Runway')
    ax.grid(True, alpha=0.3)
    
    # Add gradient info as text
    metrics = overall_best['gradient_metrics']
    info_text = (f"Max gradient: {metrics['max_abs_gradient']:.2f}°\n"
                 f"Mean gradient: {metrics['mean_abs_gradient']:.2f}°\n"
                 f"Std gradient: {metrics['std_gradient']:.2f}°\n"
                 f"Elevation change: {elevations.max() - elevations.min():.1f}m\n"
                 f"Region: {overall_best['region_label']}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    # 4. Gradient histogram for all valid runways
    ax = axes[1, 1]
    all_max_gradients = [r['gradient_metrics']['max_abs_gradient'] for r in valid_runways]
    all_mean_gradients = [r['gradient_metrics']['mean_abs_gradient'] for r in valid_runways]
    
    ax.hist(all_max_gradients, bins=30, alpha=0.6, label='Max gradient', color='red', edgecolor='black')
    ax.hist(all_mean_gradients, bins=30, alpha=0.6, label='Mean gradient', color='blue', edgecolor='black')
    
    # Mark overall best runway
    ax.axvline(overall_best['gradient_metrics']['max_abs_gradient'], 
               color='red', linestyle='--', linewidth=2, label='Best (max)')
    ax.axvline(overall_best['gradient_metrics']['mean_abs_gradient'], 
               color='blue', linestyle='--', linewidth=2, label='Best (mean)')
    
    ax.set_xlabel('Gradient (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Runway Gradients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'results/{region}_runway_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Create a second figure showing runway lines colored by gradient quality
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Background: elevation
    im = ax2.imshow(elevation, cmap='terrain', extent=extent, aspect='auto', alpha=0.4)
    plt.colorbar(im, ax=ax2, label='Elevation (m)')
    
    # Overlay landable areas
    landable_overlay = np.ma.masked_where(~landable_map, landable_map)
    ax2.imshow(landable_overlay, cmap='Greens', extent=extent, aspect='auto', alpha=0.3)
    
    # Color runway lines by their max gradient score
    max_gradients = [r['gradient_metrics']['max_abs_gradient'] for r in valid_runways]
    vmin = min(max_gradients)
    vmax = max(max_gradients)
    
    # Create line collection colored by gradient
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # Red = bad (high gradient), Green = good (low gradient)
    
    for runway in valid_runways:
        coords = runway['pixel_coords']
        geo_coords = [transform * (col, row) for row, col in coords]
        xs = [c[0] for c in geo_coords]
        ys = [c[1] for c in geo_coords]
        
        color = cmap(norm(runway['gradient_metrics']['max_abs_gradient']))
        alpha = 0.6 if runway['has_width_buffer'] else 0.3
        linewidth = 2 if runway['has_width_buffer'] else 1
        ax2.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    
    # Highlight best per region in blue
    for region_label, best_runway in best_per_region.items():
        coords = best_runway['pixel_coords']
        geo_coords = [transform * (col, row) for row, col in coords]
        xs = [c[0] for c in geo_coords]
        ys = [c[1] for c in geo_coords]
        
        # Only label the first one for legend
        if region_label == list(best_per_region.keys())[0]:
            ax2.plot(xs, ys, color='blue', linewidth=4, alpha=0.8, 
                    label=f'Best per region (n={len(best_per_region)})')
        else:
            ax2.plot(xs, ys, color='blue', linewidth=4, alpha=0.8)
    
    # Highlight overall best runway in red with white outline
    best_coords = overall_best['pixel_coords']
    best_geo_coords = [transform * (col, row) for row, col in best_coords]
    best_xs = [c[0] for c in best_geo_coords]
    best_ys = [c[1] for c in best_geo_coords]
    ax2.plot(best_xs, best_ys, color='black', linewidth=6, alpha=0.8)
    ax2.plot(best_xs, best_ys, color='red', linewidth=4, alpha=1.0, label='Overall best')
    ax2.plot(best_xs, best_ys, color='white', linewidth=2, alpha=1.0)
    
    # Add colorbar for gradient
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, label='Max Gradient (degrees)', pad=0.02)
    
    ax2.set_title(f'Runway Lines Colored by Gradient Quality\n(Thick lines = with {runway_width}m buffer)')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_path2 = f'results/{region}_runway_gradient_map.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved gradient map to: {output_path2}")
    
    plt.close('all')


def save_best_runway_geotiff(region, best_runway, landable_map, transform, profile):
    """
    Save the best runway as a GeoTIFF for GIS use.
    """
    runway_raster = np.zeros_like(landable_map, dtype=np.uint8)
    
    for row, col in best_runway['pixel_coords']:
        if 0 <= row < runway_raster.shape[0] and 0 <= col < runway_raster.shape[1]:
            runway_raster[row, col] = 1
    
    output_path = f'results/{region}_best_runway.tif'
    
    profile_copy = profile.copy()
    profile_copy.update(dtype=rasterio.uint8, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile_copy) as dst:
        dst.write(runway_raster, 1)
    
    print(f"Saved best runway GeoTIFF to: {output_path}")


def save_all_best_runways_geotiff(region, best_per_region, landable_map, transform, profile):
    """
    Save all best-per-region runways as a GeoTIFF for GIS use.
    Different regions get different pixel values.
    """
    runway_raster = np.zeros_like(landable_map, dtype=np.uint8)
    
    for region_label, best_runway in best_per_region.items():
        for row, col in best_runway['pixel_coords']:
            if 0 <= row < runway_raster.shape[0] and 0 <= col < runway_raster.shape[1]:
                # Use region_label as pixel value (may need to cap at 255)
                runway_raster[row, col] = min(region_label, 255)
    
    output_path = f'results/{region}_all_best_runways.tif'
    
    profile_copy = profile.copy()
    profile_copy.update(dtype=rasterio.uint8, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile_copy) as dst:
        dst.write(runway_raster, 1)
    
    print(f"Saved all best runways GeoTIFF to: {output_path}")

def print_summary_statistics(valid_runways, best_per_region):
    """
    Print comprehensive statistics about all valid runways.
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    max_grads = [r['gradient_metrics']['max_abs_gradient'] for r in valid_runways]
    mean_grads = [r['gradient_metrics']['mean_abs_gradient'] for r in valid_runways]
    std_grads = [r['gradient_metrics']['std_gradient'] for r in valid_runways]
    with_width = sum(1 for r in valid_runways if r['has_width_buffer'])
    
    print(f"\nTotal valid runway lines: {len(valid_runways)}")
    print(f"Number of separate regions: {len(best_per_region)}")
    print(f"Lines with width buffer: {with_width} ({100*with_width/len(valid_runways):.1f}%)")
    
    print(f"\nMax Gradient Statistics (all runways):")
    print(f"  Min: {min(max_grads):.2f}°")
    print(f"  Max: {max(max_grads):.2f}°")
    print(f"  Mean: {np.mean(max_grads):.2f}°")
    print(f"  Median: {np.median(max_grads):.2f}°")
    print(f"  Std: {np.std(max_grads):.2f}°")
    
    print(f"\nMean Gradient Statistics (all runways):")
    print(f"  Min: {min(mean_grads):.2f}°")
    print(f"  Max: {max(mean_grads):.2f}°")
    print(f"  Mean: {np.mean(mean_grads):.2f}°")
    print(f"  Median: {np.median(mean_grads):.2f}°")
    
    # Runways meeting certain thresholds
    excellent = sum(1 for g in max_grads if g < 1.0)
    good = sum(1 for g in max_grads if 1.0 <= g < 2.0)
    acceptable = sum(1 for g in max_grads if 2.0 <= g < 3.0)
    marginal = sum(1 for g in max_grads if g >= 3.0)
    
    print(f"\nRunway Quality (by max gradient):")
    print(f"  Excellent (<1°): {excellent} ({100*excellent/len(valid_runways):.1f}%)")
    print(f"  Good (1-2°): {good} ({100*good/len(valid_runways):.1f}%)")
    print(f"  Acceptable (2-3°): {acceptable} ({100*acceptable/len(valid_runways):.1f}%)")
    print(f"  Marginal (≥3°): {marginal} ({100*marginal/len(valid_runways):.1f}%)")
    
    # Statistics for best per region
    print(f"\n" + "=" * 60)
    print(f"BEST RUNWAY PER REGION STATISTICS")
    print("=" * 60)
    
    best_max_grads = [r['gradient_metrics']['max_abs_gradient'] for r in best_per_region.values()]
    best_mean_grads = [r['gradient_metrics']['mean_abs_gradient'] for r in best_per_region.values()]
    best_with_width = sum(1 for r in best_per_region.values() if r['has_width_buffer'])
    
    print(f"\nBest runways with width buffer: {best_with_width}/{len(best_per_region)} ({100*best_with_width/len(best_per_region):.1f}%)")
    
    print(f"\nMax Gradient (best per region):")
    print(f"  Min: {min(best_max_grads):.2f}°")
    print(f"  Max: {max(best_max_grads):.2f}°")
    print(f"  Mean: {np.mean(best_max_grads):.2f}°")
    print(f"  Median: {np.median(best_max_grads):.2f}°")


if __name__ == "__main__":
    """
    MAIN RUNNER - Execute this to perform runway fitting analysis
    """
    
    print("\n" + "=" * 60)
    print("RUNWAY FITTING ANALYSIS")
    print("=" * 60)
    
    # ========================================================================
    # CONFIGURATION - Adjust these parameters as needed
    # ========================================================================
    
    region = 'norcoast8'          # Region identifier (must match your file names)
    runway_length = 400           # Runway length in meters
    runway_width = 15             # Runway width in meters (for buffer check)
    angle_step = 5                # Rotation search step in degrees (smaller = more thorough but slower)
    spacing_meters = 50           # Spacing between candidate center points (smaller = more thorough but slower)
    
    print(f"\nConfiguration:")
    print(f"  Region: {region}")
    print(f"  Runway dimensions: {runway_length}m × {runway_width}m")
    print(f"  Angle step: {angle_step}°")
    print(f"  Candidate spacing: {spacing_meters}m")
    
    # ========================================================================
    # STEP 1: Find optimal runways
    # ========================================================================
    # This is the main function that does everything:
    # - Loads filtered landable areas TIF
    # - Loads DEM for gradient computation
    # - Applies min rectangle filter
    # - Searches for valid runway lines at different positions and angles
    # - Computes gradient metrics for each line
    # - Checks width buffer requirements
    # - Finds best runway per region AND overall best
    # - Creates visualizations
    
    results = find_optimal_runways(
        region=region,
        runway_length=runway_length,
        runway_width=runway_width,
        angle_step=angle_step,
        spacing_meters=spacing_meters
    )
    
    # ========================================================================
    # STEP 2: Print summary statistics (if runways were found)
    # ========================================================================
    
    if results is not None:
        print_summary_statistics(results['valid_runways'], results['best_per_region'])
        
        # ========================================================================
        # STEP 3: Save runways as GeoTIFFs for GIS use
        # ========================================================================
        
        with rasterio.open(f'results/{region}_filtered_landable_areas.tif') as src:
            profile = src.profile
        
        # Save overall best runway
        save_best_runway_geotiff(
            region=region,
            best_runway=results['best_runway'],
            landable_map=results['landable_map'],
            transform=results['transform'],
            profile=profile
        )
        
        # Save all best-per-region runways
        save_all_best_runways_geotiff(
            region=region,
            best_per_region=results['best_per_region'],
            landable_map=results['landable_map'],
            transform=results['transform'],
            profile=profile
        )
        
        # ========================================================================
        # FINAL OUTPUT SUMMARY
        # ========================================================================
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\nOutputs saved to results/ folder:")
        print(f"  1. {region}_runway_analysis.png")
        print(f"     → 4-panel visualization showing:")
        print(f"       - All valid runway lines")
        print(f"       - Best runway per region (blue) + overall best (red)")
        print(f"       - Elevation profile of overall best runway")
        print(f"       - Gradient distribution histogram")
        print(f"  2. {region}_runway_gradient_map.png")
        print(f"     → Map with runways colored by gradient quality")
        print(f"     → Best per region shown in blue")
        print(f"     → Overall best shown in red")
        print(f"  3. {region}_best_runway.tif")
        print(f"     → Georeferenced raster of the overall best runway")
        print(f"  4. {region}_all_best_runways.tif")
        print(f"     → Georeferenced raster of all best-per-region runways")
        
        print(f"\nOverall best runway location:")
        best = results['best_runway']
        print(f"  Region label: {best['region_label']}")
        print(f"  Center pixel: (row={best['center'][0]}, col={best['center'][1]})")
        print(f"  Angle: {best['angle']:.1f}°")
        print(f"  Max gradient: {best['gradient_metrics']['max_abs_gradient']:.2f}°")
        print(f"  Mean gradient: {best['gradient_metrics']['mean_abs_gradient']:.2f}°")
        print(f"  Has {runway_width}m width buffer: {best['has_width_buffer']}")
        
        print(f"\nFound {len(results['best_per_region'])} regions with valid runways:")
        for region_label, runway in sorted(results['best_per_region'].items(), 
                                          key=lambda x: x[1]['gradient_metrics']['max_abs_gradient'])[:5]:
            print(f"  Region {region_label}: max grad = {runway['gradient_metrics']['max_abs_gradient']:.2f}°, "
                  f"has buffer = {runway['has_width_buffer']}")
        if len(results['best_per_region']) > 5:
            print(f"  ... and {len(results['best_per_region']) - 5} more")
        
    else:
        print("\n" + "=" * 60)
        print("ANALYSIS FAILED")
        print("=" * 60)
        print("\nNo valid runways found. Possible reasons:")
        print("  - Filtered landable areas TIF file not found")
        print("  - DEM file not found")
        print("  - No regions large enough after min rectangle filter")
        print("  - No valid runway lines could be placed")
        print("\nPlease check:")
        print(f"  1. results/{region}_filtered_landable_areas.tif exists")
        print(f"  2. dem_maps/{region}_dem_utm.tif exists (or adjust DEM path in code)")
        print("  3. Run overlay_osm_on_binary_map.py first if needed")