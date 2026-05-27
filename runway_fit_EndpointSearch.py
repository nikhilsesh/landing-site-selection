"""
runway_fit_EndpointSearch.py

Optimizes runway placement using parametric contour approach.
Design variables: two endpoints (t_i, t_j) on parametrized outer contour.
Objective: Maximize runway length subject to line staying inside contour.
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import xy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
import os

def filter_by_min_rectangle(binary_image, min_width=15, min_height=400):
    """
    Keep only regions that can fit a rectangle of at least min_width × min_height.
    Uses hierarchy to ignore holes (only processes external contours).
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


class ParametricContour:
    """
    Represents a contour parametrized by arc length t ∈ [0, 1).
    """
    def __init__(self, contour_points):
        """
        Parameters:
        -----------
        contour_points : np.ndarray
            Contour points from cv2.findContours, shape (N, 1, 2) or (N, 2)
        """
        # Reshape if needed
        if contour_points.ndim == 3:
            contour_points = contour_points.reshape(-1, 2)
        
        self.points = contour_points.astype(np.float64)
        self.n_points = len(self.points)
        
        # Compute cumulative arc length
        # Distance between consecutive points
        distances = np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1))
        
        # Distance from last point back to first (closing the contour)
        closing_distance = np.sqrt(np.sum((self.points[-1] - self.points[0])**2))
        
        # Cumulative distances (starting at 0)
        self.cumulative_length = np.zeros(self.n_points)
        self.cumulative_length[1:] = np.cumsum(distances)
        
        # Total perimeter including the closing segment
        self.total_length = self.cumulative_length[-1] + closing_distance
        
        # Normalize to [0, 1)
        self.t_values = self.cumulative_length / self.total_length
        
        # Create interpolators for x and y coordinates (periodic)
        # Add wraparound point for periodicity: point at t=1 is same as t=0
        points_extended = np.vstack([self.points, self.points[0:1]])  # Add first point at end
        t_extended = np.append(self.t_values, 1.0)  # Add t=1.0
        
        # Verify lengths match
        assert len(t_extended) == len(points_extended), \
            f"Length mismatch: t_extended={len(t_extended)}, points_extended={len(points_extended)}"
        
        self.interp_x = interp1d(t_extended, points_extended[:, 0], kind='linear')
        self.interp_y = interp1d(t_extended, points_extended[:, 1], kind='linear')
        
        # Store contour for point-in-polygon tests
        self.contour_cv = self.points.astype(np.int32)
    
    def get_point(self, t):
        """
        Get (x, y) coordinates at parameter value t ∈ [0, 1).
        """
        t = np.clip(t % 1.0, 0, 0.9999999)  # Ensure t < 1
        x = float(self.interp_x(t))
        y = float(self.interp_y(t))
        return np.array([x, y])
    
    def get_points(self, t_array):
        """
        Get multiple points for an array of t values.
        """
        t_array = np.clip(t_array % 1.0, 0, 0.9999999)
        x = self.interp_x(t_array)
        y = self.interp_y(t_array)
        return np.column_stack([x, y])
    
    def is_line_inside(self, t1, t2, n_samples=100):
        """
        Check if the line segment from t1 to t2 stays inside the contour.
        
        Parameters:
        -----------
        t1, t2 : float
            Parameter values for the two endpoints
        n_samples : int
            Number of points to sample along the line for checking
        
        Returns:
        --------
        bool : True if line is fully inside, False otherwise
        """
        p1 = self.get_point(t1)
        p2 = self.get_point(t2)
        
        # Sample points along the line
        alphas = np.linspace(0, 1, n_samples)
        line_points = p1[np.newaxis, :] + alphas[:, np.newaxis] * (p2 - p1)[np.newaxis, :]
        
        # Check if all points are inside the contour
        for point in line_points:
            dist = cv2.pointPolygonTest(self.contour_cv, tuple(point), False)
            if dist < 0:  # Point is outside
                return False
        
        return True
    
    def line_length(self, t1, t2):
        """
        Compute Euclidean distance between points at t1 and t2.
        """
        p1 = self.get_point(t1)
        p2 = self.get_point(t2)
        return np.linalg.norm(p2 - p1)


def optimize_runway_for_contour(param_contour, min_length=400, pixel_size=1.0):
    """
    Find the longest valid runway for a given parametric contour.
    
    Parameters:
    -----------
    param_contour : ParametricContour
        Parametrized contour object
    min_length : float
        Minimum required runway length in pixels
    pixel_size : float
        Pixel size in meters (for converting to real-world length)
    
    Returns:
    --------
    dict with:
        - t1, t2: optimal parameter values
        - length: runway length in meters
        - p1, p2: endpoint coordinates (row, col)
        - valid: whether a valid runway was found
    """
    print(f"  Optimizing runway (contour has {param_contour.n_points} points, perimeter={param_contour.total_length*pixel_size:.1f}m)...")
    
    # Objective function: negative length (for minimization)
    def objective(x):
        t1, t2 = x
        
        # Enforce t1 < t2 constraint via penalty
        if t2 <= t1:
            return 1e10
        
        # Check if line is inside
        if not param_contour.is_line_inside(t1, t2, n_samples=50):
            return 1e10  # Large penalty for invalid lines
        
        # Return negative length (we want to maximize length)
        length = param_contour.line_length(t1, t2)
        return -length
    
    # Search space: 0 <= t1 < 1, 0 <= t2 < 1
    # We'll enforce t1 < t2 via penalty in objective
    bounds = [(0.0, 0.999), (0.001, 1.0)]
    
    # Try differential evolution (global optimizer)
    # Note: differential_evolution doesn't support constraints, so we use penalty method
    print("    Running global optimization (differential_evolution)...")
    
    try:
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        if result_de.success and result_de.fun < 1e9:
            t1_opt, t2_opt = result_de.x
            
            # Ensure t1 < t2 (swap if needed)
            if t1_opt > t2_opt:
                t1_opt, t2_opt = t2_opt, t1_opt
            
            length_pixels = -result_de.fun
            length_meters = length_pixels * pixel_size
            
            p1 = param_contour.get_point(t1_opt)
            p2 = param_contour.get_point(t2_opt)
            
            print(f"    ✓ Found runway: {length_meters:.1f}m (t1={t1_opt:.4f}, t2={t2_opt:.4f})")
            
            return {
                't1': t1_opt,
                't2': t2_opt,
                'length': length_meters,
                'length_pixels': length_pixels,
                'p1': p1,  # (col, row) in image coordinates
                'p2': p2,
                'valid': length_meters >= min_length,
                'param_contour': param_contour
            }
        else:
            print(f"    ✗ Optimization failed (status: {result_de.message})")
            return {
                'valid': False,
                'length': 0,
                'param_contour': param_contour
            }
            
    except Exception as e:
        print(f"    ✗ Optimization error: {e}")
        return {
            'valid': False,
            'length': 0,
            'param_contour': param_contour
        }


def compute_gradient_along_line(elevation, transform, p1, p2, n_samples=100):
    """
    Compute elevation gradient metrics along a line from p1 to p2.
    
    Parameters:
    -----------
    elevation : np.ndarray
        DEM elevation data
    transform : Affine
        Rasterio transform
    p1, p2 : np.ndarray
        Endpoints in image coordinates (col, row)
    n_samples : int
        Number of points to sample along the line
    
    Returns:
    --------
    dict with gradient statistics
    """
    # Sample points along the line
    alphas = np.linspace(0, 1, n_samples)
    line_points = p1[np.newaxis, :] + alphas[:, np.newaxis] * (p2 - p1)[np.newaxis, :]
    
    # Get elevations (p1 and p2 are in (col, row) format)
    elevations = []
    valid_mask = []
    
    for point in line_points:
        col, row = point
        row_int, col_int = int(round(row)), int(round(col))
        
        if 0 <= row_int < elevation.shape[0] and 0 <= col_int < elevation.shape[1]:
            elev = elevation[row_int, col_int]
            if not np.ma.is_masked(elev) and np.isfinite(elev):
                elevations.append(float(elev))
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        else:
            valid_mask.append(False)
    
    if len(elevations) < 10:
        return None
    
    elevations = np.array(elevations)
    
    # Compute gradients (change in elevation between consecutive points)
    dists = np.linalg.norm(np.diff(line_points[valid_mask], axis=0), axis=1)
    d_elev = np.diff(elevations)
    
    # Avoid division by zero
    dists = np.maximum(dists, 1e-6)
    
    # Gradient in degrees
    gradients_rad = np.arctan(d_elev / dists)
    gradients_deg = np.degrees(gradients_rad)
    
    return {
        'max_abs_gradient': np.max(np.abs(gradients_deg)),
        'mean_abs_gradient': np.mean(np.abs(gradients_deg)),
        'std_gradient': np.std(gradients_deg),
        'elevations': elevations,
        'distance': np.sum(dists)
    }


def find_optimal_runways_parametric(region='norcoast8',
                                     runway_length=400,
                                     runway_width=15):
    """
    Find optimal runway placements using parametric contour optimization.
    
    Parameters:
    -----------
    region : str
        Region identifier
    runway_length : float
        Minimum runway length in meters
    runway_width : float
        Runway width in meters (for min rectangle filter)
    
    Returns:
    --------
    dict with results
    """
    print("=" * 60)
    print(f"PARAMETRIC RUNWAY OPTIMIZATION FOR REGION: {region}")
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
    
    # Apply min rectangle filter
    print(f"\nApplying min rectangle filter ({runway_width}m × {runway_length}m)...")
    landable_filtered = filter_by_min_rectangle(landable_map,
                                                 min_width=runway_width/pixel_size,
                                                 min_height=runway_length/pixel_size)
    
    # Get contours (external only, ignoring holes)
    binary_uint8 = landable_filtered.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(binary_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter to get only external contours
    external_contours = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            parent = hierarchy[0][i][3]
            if parent == -1:  # External contour
                external_contours.append(contour)
    
    num_contours = len(external_contours)
    print(f"\nFound {num_contours} valid contours after filtering")
    
    if num_contours == 0:
        print("No valid contours found!")
        return None
    
    # Load DEM for gradient computation
    dem_path = f'dem_maps/{region}_dem_utm.tif'
    if not os.path.exists(dem_path):
        dem_path = f'dem_maps/{region.replace("norcoast8", "norcoast_b23")}_dem_utm.tif'
        if not os.path.exists(dem_path):
            print(f"ERROR: DEM file not found at {dem_path}")
            return None
    
    with rasterio.open(dem_path) as src:
        elevation = src.read(1, masked=True)
    
    print(f"Loaded DEM: {elevation.shape}")
    print(f"Elevation range: {elevation.min():.1f} to {elevation.max():.1f} meters")
    
    # Optimize runway for each contour
    print(f"\n{'=' * 60}")
    print("OPTIMIZING RUNWAYS FOR EACH CONTOUR")
    print(f"{'=' * 60}\n")
    
    valid_runways = []
    best_per_contour = {}
    
    for contour_idx, contour in enumerate(external_contours):
        print(f"Contour {contour_idx + 1}/{num_contours}:")
        
        # Create parametric contour
        param_contour = ParametricContour(contour)
        
        # Optimize runway placement
        runway = optimize_runway_for_contour(
            param_contour,
            min_length=runway_length,
            pixel_size=pixel_size
        )
        
        if not runway['valid']:
            print(f"  Skipping - no valid runway found\n")
            continue
        
        # Compute gradient metrics
        # Convert from (col, row) to actual coordinates
        p1_col, p1_row = runway['p1']
        p2_col, p2_row = runway['p2']
        
        gradient_metrics = compute_gradient_along_line(
            elevation, transform,
            runway['p1'], runway['p2'],
            n_samples=100
        )
        
        if gradient_metrics is None:
            print(f"  Warning: Could not compute gradients\n")
            continue
        
        # Store runway with all info
        runway_info = {
            'contour_idx': contour_idx,
            't1': runway['t1'],
            't2': runway['t2'],
            'length': runway['length'],
            'p1': runway['p1'],  # (col, row)
            'p2': runway['p2'],  # (col, row)
            'gradient_metrics': gradient_metrics,
            'param_contour': param_contour
        }
        
        valid_runways.append(runway_info)
        best_per_contour[contour_idx] = runway_info
        
        print(f"  Gradient: max={gradient_metrics['max_abs_gradient']:.2f}°, "
              f"mean={gradient_metrics['mean_abs_gradient']:.2f}°\n")
    
    if len(valid_runways) == 0:
        print("No valid runways found!")
        return None
    
    print(f"{'=' * 60}")
    print(f"FOUND {len(valid_runways)} VALID RUNWAYS")
    print(f"{'=' * 60}\n")
    
    # Find overall best runway (minimum max gradient)
    best_runway = min(valid_runways, key=lambda r: r['gradient_metrics']['max_abs_gradient'])
    
    print(f"Best runway:")
    print(f"  Contour: {best_runway['contour_idx'] + 1}")
    print(f"  Length: {best_runway['length']:.1f}m")
    print(f"  Max gradient: {best_runway['gradient_metrics']['max_abs_gradient']:.2f}°")
    print(f"  Mean gradient: {best_runway['gradient_metrics']['mean_abs_gradient']:.2f}°")
    
    # Create visualizations
    print(f"\n{'=' * 60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 60}\n")
    
    # create_visualizations(
    #     region=region,
    #     landable_map=landable_filtered,
    #     elevation=elevation,
    #     transform=transform,
    #     valid_runways=valid_runways,
    #     best_per_contour=best_per_contour,
    #     overall_best=best_runway,
    #     pixel_size=pixel_size
    # )
    # Visualizations: region, landable_map, elevation, transform, valid_runways, best_per_contour, overall_best, pixel_size

    create_runway_analysis_plots(
        region=region,
        landable_map=landable_filtered,
        elevation=elevation,
        transform=transform,
        best_per_contour=best_per_contour,
        best_runway=best_runway,
        pixel_size=pixel_size
    )
    # Runway Analysis Plots: region, landable_map, elevation, transform, best_per_contour, best_runway, pixel_size

    # Create individual contour plots
    plot_individual_contour_runways(
        region=region,
        valid_runways=valid_runways,
        elevation=elevation,
        transform=transform,
        landable_map=landable_filtered,
        pixel_size=pixel_size
    )
    
    return {
        'valid_runways': valid_runways,
        'best_per_contour': best_per_contour,
        'best_runway': best_runway,
        'landable_map': landable_filtered,
        'elevation': elevation,
        'transform': transform
    }


def create_visualizations(region, landable_map, elevation, transform, 
                         valid_runways, best_per_contour, overall_best, pixel_size):
    """
    Create visualization plots similar to runway_fit_new.py.
    """
    # Get geographic extent
    height, width = landable_map.shape
    top_left = transform * (0, 0)
    bottom_right = transform * (width, height)
    extent = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]
    
    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ========================================================================
    # Panel 1: All valid runways
    # ========================================================================
    ax = axes[0, 0]
    ax.imshow(landable_map, cmap='Greens', extent=extent, aspect='auto', alpha=0.6)
    
    for runway in valid_runways:
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        
        gradient = runway['gradient_metrics']['max_abs_gradient']
        color = 'cyan' if gradient < 5 else 'yellow'
        
        ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
               color=color, linewidth=1.5, alpha=0.7)
    
    ax.set_title(f'All Valid Runways (n={len(valid_runways)})', fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 2: Best per contour + overall best
    # ========================================================================
    ax = axes[0, 1]
    ax.imshow(landable_map, cmap='Greens', extent=extent, aspect='auto', alpha=0.6)
    
    # Draw all in light gray
    for runway in valid_runways:
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
               color='gray', linewidth=0.5, alpha=0.3)
    
    # Draw best per contour in blue
    for idx, (contour_idx, runway) in enumerate(best_per_contour.items()):
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        
        if idx == 0:
            ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                   color='blue', linewidth=2.5, alpha=0.8, 
                   label=f'Best per contour (n={len(best_per_contour)})')
        else:
            ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                   color='blue', linewidth=2.5, alpha=0.8)
    
    # Draw overall best in red
    p1_best_geo = transform * (overall_best['p1'][0], overall_best['p1'][1])
    p2_best_geo = transform * (overall_best['p2'][0], overall_best['p2'][1])
    ax.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
           color='red', linewidth=3, alpha=1.0, label='Overall best')
    
    ax.set_title('Best Runway per Contour + Overall Best', fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 3: Elevation profile of best runway
    # ========================================================================
    ax = axes[1, 0]
    
    elevations = overall_best['gradient_metrics']['elevations']
    distances = np.linspace(0, overall_best['length'], len(elevations))
    
    ax.plot(distances, elevations, 'b-', linewidth=2)
    ax.fill_between(distances, elevations, alpha=0.3)
    ax.set_xlabel('Distance along runway (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Elevation Profile - Best Runway\n'
                f'Length: {overall_best["length"]:.1f}m, '
                f'Max Gradient: {overall_best["gradient_metrics"]["max_abs_gradient"]:.2f}°',
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add elevation change annotation
    elev_change = elevations.max() - elevations.min()
    ax.text(0.02, 0.98, f'Elevation change: {elev_change:.1f}m', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # Panel 4: Gradient distribution
    # ========================================================================
    ax = axes[1, 1]
    
    all_max_gradients = [r['gradient_metrics']['max_abs_gradient'] for r in valid_runways]
    all_mean_gradients = [r['gradient_metrics']['mean_abs_gradient'] for r in valid_runways]
    
    ax.hist(all_max_gradients, bins=20, alpha=0.6, label='Max gradient', color='red', edgecolor='black')
    ax.hist(all_mean_gradients, bins=20, alpha=0.6, label='Mean gradient', color='blue', edgecolor='black')
    
    # Mark best runway
    ax.axvline(overall_best['gradient_metrics']['max_abs_gradient'], 
              color='red', linestyle='--', linewidth=2, label='Best max gradient')
    
    ax.set_xlabel('Gradient (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Gradient Distribution Across All Runways', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'results/endpoint_search/{region}_parametric_runway_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 4-panel analysis to: {output_path}")
    plt.close()
    
    # ========================================================================
    # Create gradient map (similar to runway_fit_new.py)
    # ========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Background: elevation
    im = ax2.imshow(elevation, cmap='terrain', extent=extent, aspect='auto', alpha=0.4)
    plt.colorbar(im, ax=ax2, label='Elevation (m)')
    
    # Overlay landable areas
    landable_overlay = np.ma.masked_where(~landable_map, landable_map)
    ax2.imshow(landable_overlay, cmap='Greens', extent=extent, aspect='auto', alpha=0.3)
    
    # Color runway lines by their max gradient
    max_gradients = [r['gradient_metrics']['max_abs_gradient'] for r in valid_runways]
    vmin = min(max_gradients)
    vmax = max(max_gradients)
    
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # Red = bad (high gradient), Green = good (low gradient)
    
    for runway in valid_runways:
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        
        color = cmap(norm(runway['gradient_metrics']['max_abs_gradient']))
        ax2.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                color=color, linewidth=2, alpha=0.6)
    
    # Highlight best per contour in blue
    for idx, runway in enumerate(best_per_contour.values()):
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        
        if idx == 0:
            ax2.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                    color='blue', linewidth=4, alpha=0.8, 
                    label=f'Best per contour (n={len(best_per_contour)})')
        else:
            ax2.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                    color='blue', linewidth=4, alpha=0.8)
    
    # Highlight overall best runway in red
    p1_best_geo = transform * (overall_best['p1'][0], overall_best['p1'][1])
    p2_best_geo = transform * (overall_best['p2'][0], overall_best['p2'][1])
    ax2.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
            color='black', linewidth=6, alpha=0.8)
    ax2.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
            color='red', linewidth=4, alpha=1.0, label='Overall best')
    ax2.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
            color='white', linewidth=2, alpha=1.0)
    
    # Add colorbar for gradient
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, label='Max Gradient (degrees)', pad=0.02)
    
    ax2.set_title(f'Runway Lines Colored by Gradient Quality\n'
                  f'Best: {overall_best["gradient_metrics"]["max_abs_gradient"]:.2f}° max gradient, '
                  f'{overall_best["length"]:.1f}m length')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_path2 = f'results/endpoint_search/{region}_parametric_gradient_map.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved gradient map to: {output_path2}")
    
    plt.close('all')
    
    print("\nVisualization complete!")

def create_runway_analysis_plots(region, landable_map, elevation, transform, best_per_contour, best_runway, pixel_size):
    """
    Create comprehensive visualization of runway analysis results.
    Modified to show:
    - Top left: Best runway per contour + overall best with DEM background
    - Top right: Best runway on satellite imagery
    - Bottom left: Elevation profile
    - Bottom right: Gradient distribution
    """
    
    # Import the satellite plotting function
    try:
        from best_runway_satellite_overlay import plot_runway_on_satellite_ax
        satellite_available = True
    except ImportError:
        print("Warning: Could not import satellite overlay function. Top-right panel will be blank.")
        satellite_available = False
    
    valid_runways = list(best_per_contour.values())
    overall_best = best_runway
    
    # Get geographic extent for the entire region
    height, width = landable_map.shape
    top_left = transform * (0, 0)
    bottom_right = transform * (width, height)
    extent = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]
    
    # ========================================================================
    # Create 2x2 panel figure
    # ========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ========================================================================
    # Panel 1 (Top Left): Best runway per contour + overall best WITH DEM BACKGROUND
    # (Similar style to parametric_gradient_map.png but with contours added)
    # ========================================================================
    ax = axes[0, 0]
    
    # Background: elevation (faded)
    im = ax.imshow(elevation, cmap='terrain', extent=extent, aspect='auto', alpha=0.4)
    
    # Overlay landable areas
    landable_overlay = np.ma.masked_where(~landable_map, landable_map)
    ax.imshow(landable_overlay, cmap='Greens', extent=extent, aspect='auto', alpha=0.3)
    
    # Plot contours for all best-per-contour runways (solid color, visible but subtle)
    for idx, (contour_idx, runway) in enumerate(best_per_contour.items()):
        param_contour = runway['param_contour']
        
        # Draw the contour boundary - FIX THE COORDINATE CONVERSION
        contour_geo_coords = [transform * (p[0], p[1]) for p in param_contour.points]
        contour_xs = [c[0] for c in contour_geo_coords]
        contour_ys = [c[1] for c in contour_geo_coords]
        
        # Only add label for the FIRST contour
        if idx == 0:
            ax.plot(contour_xs + [contour_xs[0]], contour_ys + [contour_ys[0]], 
                   color='gray', linewidth=1.2, alpha=0.5, label='Valid contours')
        else:
            ax.plot(contour_xs + [contour_xs[0]], contour_ys + [contour_ys[0]], 
                   color='gray', linewidth=1.2, alpha=0.5)
    
    # Plot all best-per-contour runways (blue)
    for idx, (contour_idx, runway) in enumerate(best_per_contour.items()):
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        
        # Blue for best-per-contour (skip if it's the overall best)
        if contour_idx != overall_best['contour_idx']:
            # Only add label for the FIRST blue runway
            if idx == 0:
                ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                       color='blue', linewidth=4, alpha=0.8, 
                       label=f'Best per contour (n={len(best_per_contour)})')
            else:
                ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                       color='blue', linewidth=4, alpha=0.8)
    
    # Highlight overall best runway (red with black outline and white center)
    p1_best_geo = transform * (overall_best['p1'][0], overall_best['p1'][1])
    p2_best_geo = transform * (overall_best['p2'][0], overall_best['p2'][1])
    ax.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
           color='black', linewidth=6, alpha=0.8)
    ax.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
           color='red', linewidth=4, alpha=1.0, label='Overall best')
    ax.plot([p1_best_geo[0], p2_best_geo[0]], [p1_best_geo[1], p2_best_geo[1]], 
           color='white', linewidth=2, alpha=1.0)
    
    # Add colorbar for elevation reference
    cbar = plt.colorbar(im, ax=ax, label='Elevation (m)', fraction=0.046, pad=0.04)
    
    ax.set_title(f'Best Runways with DEM Background\n'
                f'Overall best: {overall_best["gradient_metrics"]["max_abs_gradient"]:.2f}° max gradient, '
                f'{overall_best["length"]:.1f}m length',
                fontweight='bold', fontsize=11)
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 2 (Top Right): Best runway on SATELLITE IMAGERY
    # ========================================================================
    ax = axes[0, 1]
    
    if satellite_available:
        try:
            print("\nFetching satellite imagery for top-right panel...")
            print("(This may take 10-30 seconds)")
            
            # Get CRS from the transform
            with rasterio.open(f'results/{region}_filtered_landable_areas.tif') as src:
                crs = src.crs
            
            # Call the satellite plotting function
            plot_runway_on_satellite_ax(
                ax=ax,
                runway=overall_best,
                transform=transform,
                crs=crs,
                padding_meters=500,
                zoom_level='auto',
                satellite_source='Esri.WorldImagery',
                show_labels=True
            )
            
            # Modify title to fit the panel layout
            ax.set_title(f'Best Runway - Satellite View\n'
                        f'Length: {overall_best["length"]:.1f}m, '
                        f'Max Gradient: {overall_best["gradient_metrics"]["max_abs_gradient"]:.2f}°',
                        fontweight='bold', fontsize=11)
            
            print("✓ Satellite imagery loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not create satellite view: {e}")
            print("Falling back to simple DEM view...")
            
            # Fallback: show zoomed DEM view
            p1 = overall_best['p1']
            p2 = overall_best['p2']
            
            # Calculate zoom extent (500m padding)
            padding_pixels = int(500 / pixel_size)
            row_min = max(0, min(p1[1], p2[1]) - padding_pixels)
            row_max = min(elevation.shape[0], max(p1[1], p2[1]) + padding_pixels)
            col_min = max(0, min(p1[0], p2[0]) - padding_pixels)
            col_max = min(elevation.shape[1], max(p1[0], p2[0]) + padding_pixels)
            
            elevation_zoom = elevation[row_min:row_max, col_min:col_max]
            
            top_left_zoom = transform * (col_min, row_min)
            bottom_right_zoom = transform * (col_max, row_max)
            extent_zoom = [top_left_zoom[0], bottom_right_zoom[0], 
                          bottom_right_zoom[1], top_left_zoom[1]]
            
            im_zoom = ax.imshow(elevation_zoom, cmap='terrain', extent=extent_zoom, 
                               aspect='auto', alpha=0.8)
            
            p1_geo = transform * (p1[0], p1[1])
            p2_geo = transform * (p2[0], p2[1])
            ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                   'r-', linewidth=4, label=f'Runway ({overall_best["length"]:.0f}m)')
            ax.plot(p1_geo[0], p1_geo[1], 'yo', markersize=10, 
                   markeredgecolor='black', markeredgewidth=2, label='Start')
            ax.plot(p2_geo[0], p2_geo[1], 'ys', markersize=10, 
                   markeredgecolor='black', markeredgewidth=2, label='End')
            
            ax.set_title('Best Runway - Close-up View (DEM)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Easting (m)', fontsize=10)
            ax.set_ylabel('Northing (m)', fontsize=10)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(useOffset=False, style='plain')
    else:
        # No satellite function available - show placeholder
        ax.text(0.5, 0.5, 'Satellite imagery unavailable\n(best_runway_satellite_overlay.py not found)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Satellite View (Unavailable)', fontweight='bold', fontsize=11)
    
    # ========================================================================
    # Panel 3 (Bottom Left): Elevation profile of best runway
    # ========================================================================
    ax = axes[1, 0]
    
    elevations = overall_best['gradient_metrics']['elevations']
    distances = np.linspace(0, overall_best['length'], len(elevations))
    
    ax.plot(distances, elevations, 'b-', linewidth=2)
    ax.fill_between(distances, elevations, alpha=0.3)
    ax.set_xlabel('Distance along runway (m)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title(f'Elevation Profile - Best Runway\n'
                f'Length: {overall_best["length"]:.1f}m, '
                f'Max Gradient: {overall_best["gradient_metrics"]["max_abs_gradient"]:.2f}°',
                fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add elevation change annotation
    elev_change = elevations.max() - elevations.min()
    ax.text(0.02, 0.98, f'Elevation change: {elev_change:.1f}m', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
    
    # ========================================================================
    # Panel 4 (Bottom Right): Gradient distribution
    # ========================================================================
    ax = axes[1, 1]
    
    all_max_gradients = [r['gradient_metrics']['max_abs_gradient'] for r in valid_runways]
    all_mean_gradients = [r['gradient_metrics']['mean_abs_gradient'] for r in valid_runways]
    
    ax.hist(all_max_gradients, bins=20, alpha=0.6, label='Max gradient', color='red', edgecolor='black')
    ax.hist(all_mean_gradients, bins=20, alpha=0.6, label='Mean gradient', color='blue', edgecolor='black')
    
    # Mark best runway
    ax.axvline(overall_best['gradient_metrics']['max_abs_gradient'], 
              color='red', linestyle='--', linewidth=2, label='Best max gradient')
    
    ax.set_xlabel('Gradient (degrees)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Gradient Distribution Across All Runways', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Save the figure
    # ========================================================================
    
    plt.tight_layout()
    output_path = f'results/endpoint_search/{region}_parametric_runway_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 4-panel analysis to: {output_path}")
    plt.close()

def plot_individual_contour_runways(region, valid_runways, elevation, transform, 
                                   landable_map, pixel_size):
    """
    Create individual plots for each contour showing its optimal runway.
    """
    output_dir = 'results/endpoint_search/optimal_runway_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"CREATING INDIVIDUAL CONTOUR PLOTS")
    print(f"{'=' * 60}\n")
    
    for runway in valid_runways:
        contour_idx = runway['contour_idx']
        param_contour = runway['param_contour']
        
        # Get bounding box for this contour (with padding)
        contour_points = param_contour.points
        row_min = int(np.floor(contour_points[:, 1].min()))
        row_max = int(np.ceil(contour_points[:, 1].max()))
        col_min = int(np.floor(contour_points[:, 0].min()))
        col_max = int(np.ceil(contour_points[:, 0].max()))
        
        # Add padding
        padding = int(100 / pixel_size)  # 100 meter padding
        row_min = max(0, row_min - padding)
        row_max = min(landable_map.shape[0], row_max + padding)
        col_min = max(0, col_min - padding)
        col_max = min(landable_map.shape[1], col_max + padding)
        
        # Extract zoomed region
        landable_zoom = landable_map[row_min:row_max, col_min:col_max]
        elevation_zoom = elevation[row_min:row_max, col_min:col_max]
        
        # Get geographic extent
        top_left = transform * (col_min, row_min)
        bottom_right = transform * (col_max, row_max)
        extent_zoom = [top_left[0], bottom_right[0], bottom_right[1], top_left[1]]
        
        # Create figure with 2 panels
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # ===== LEFT PANEL: Contour with runway =====
        im = axes[0].imshow(elevation_zoom, cmap='terrain', extent=extent_zoom, 
                           aspect='auto', alpha=0.6)
        
        # Overlay landable area
        landable_overlay = np.ma.masked_where(~landable_zoom, landable_zoom)
        axes[0].imshow(landable_overlay, cmap='Greens', extent=extent_zoom, 
                      aspect='auto', alpha=0.7)
        
        # Draw the optimal runway
        p1_geo = transform * (runway['p1'][0], runway['p1'][1])
        p2_geo = transform * (runway['p2'][0], runway['p2'][1])
        axes[0].plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
                    'r-', linewidth=3, label=f'Optimal runway: {runway["length"]:.1f}m')
        
        # Mark endpoints
        axes[0].plot(p1_geo[0], p1_geo[1], 'ro', markersize=8, label=f't₁={runway["t1"]:.3f}')
        axes[0].plot(p2_geo[0], p2_geo[1], 'rs', markersize=8, label=f't₂={runway["t2"]:.3f}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0], label='Elevation (m)', fraction=0.046, pad=0.04)
        
        axes[0].set_title(f'Contour {contour_idx + 1} - Optimal Runway\n'
                         f'Max Gradient: {runway["gradient_metrics"]["max_abs_gradient"]:.2f}°',
                         fontweight='bold')
        axes[0].set_xlabel('Easting (m)')
        axes[0].set_ylabel('Northing (m)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # ===== RIGHT PANEL: Elevation profile =====
        elevations = runway['gradient_metrics']['elevations']
        distances = np.linspace(0, runway['length'], len(elevations))
        
        axes[1].plot(distances, elevations, 'b-', linewidth=2)
        axes[1].fill_between(distances, elevations, alpha=0.3)
        
        axes[1].set_xlabel('Distance along runway (m)')
        axes[1].set_ylabel('Elevation (m)')
        axes[1].set_title('Elevation Profile', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text box
        elev_change = elevations.max() - elevations.min()
        stats_text = f"Length: {runway['length']:.1f} m\n"
        stats_text += f"Elev change: {elev_change:.1f} m\n"
        stats_text += f"Max gradient: {runway['gradient_metrics']['max_abs_gradient']:.2f}°\n"
        stats_text += f"Mean gradient: {runway['gradient_metrics']['mean_abs_gradient']:.2f}°\n"
        stats_text += f"Std gradient: {runway['gradient_metrics']['std_gradient']:.2f}°"
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.suptitle(f'Region: {region} | Contour {contour_idx + 1}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = f'{output_dir}/{region}_contour_{contour_idx + 1:03d}_optimal_runway.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Contour {contour_idx + 1}/{len(valid_runways)}: Saved to {output_path}")
    
    print(f"\n✓ All {len(valid_runways)} individual contour plots saved to {output_dir}/")

def save_best_runway_geotiff(region, best_runway, landable_map, transform, profile, pixel_size):
    """
    Save the best runway as a GeoTIFF for GIS use.
    """
    runway_raster = np.zeros_like(landable_map, dtype=np.uint8)
    
    # Draw line on raster
    p1 = best_runway['p1']  # (col, row)
    p2 = best_runway['p2']
    
    # Convert to integer pixel coordinates
    p1_int = (int(round(p1[0])), int(round(p1[1])))  # (col, row)
    p2_int = (int(round(p2[0])), int(round(p2[1])))
    
    # Draw line using Bresenham's algorithm (via cv2)
    cv2.line(runway_raster, p1_int, p2_int, 1, thickness=1)
    
    output_path = f'results/endpoint_search/{region}_best_runway_parametric.tif'
    
    profile_copy = profile.copy()
    profile_copy.update(dtype=rasterio.uint8, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile_copy) as dst:
        dst.write(runway_raster, 1)
    
    print(f"Saved best runway GeoTIFF to: {output_path}")


def save_all_best_runways_geotiff(region, best_per_contour, landable_map, transform, profile, pixel_size):
    """
    Save all best-per-contour runways as a GeoTIFF for GIS use.
    """
    runway_raster = np.zeros_like(landable_map, dtype=np.uint8)
    
    # for contour_idx, runway in results['best_per_contour'].items():
    for contour_idx, runway in best_per_contour.items():
        p1 = runway['p1']  # (col, row)
        p2 = runway['p2']
        
        # Convert to integer pixel coordinates
        p1_int = (int(round(p1[0])), int(round(p1[1])))
        p2_int = (int(round(p2[0])), int(round(p2[1])))
        
        # Draw line with contour_idx as value (cap at 255)
        value = min(contour_idx + 1, 255)
        cv2.line(runway_raster, p1_int, p2_int, value, thickness=1)
    
    output_path = f'results/endpoint_search/{region}_all_best_runways_parametric.tif'
    
    profile_copy = profile.copy()
    profile_copy.update(dtype=rasterio.uint8, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile_copy) as dst:
        dst.write(runway_raster, 1)
    
    print(f"Saved all best runways GeoTIFF to: {output_path}")


if __name__ == "__main__":
    # Configuration
    region = 'alameda_b21_x59y418'
    search_type = 'endpoint_search'
    runway_length = 400  # meters
    runway_width = 15    # meters

    output_dir = f'results/{search_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run parametric optimization
    results = find_optimal_runways_parametric(
        region=region,
        runway_length=runway_length,
        runway_width=runway_width
    )
    
    if results is not None:
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 60)
        
        # Save GeoTIFFs
        with rasterio.open(f'results/{region}_filtered_landable_areas.tif') as src:
            profile = src.profile
        
        pixel_size = abs(results['transform'].a)
        
        save_best_runway_geotiff(
            region=region,
            best_runway=results['best_runway'],
            landable_map=results['landable_map'],
            transform=results['transform'],
            profile=profile,
            pixel_size=pixel_size
        )
        
        save_all_best_runways_geotiff(
            region=region,
            best_per_contour=results['best_per_contour'],
            landable_map=results['landable_map'],
            transform=results['transform'],
            profile=profile,
            pixel_size=pixel_size
        )
        
        print(f"\nOutputs saved to results/endpoint_search/ folder:")
        print(f"  1. {region}_parametric_runway_analysis.png")
        print(f"     → 4-panel visualization")
        print(f"  2. {region}_parametric_gradient_map.png")
        print(f"     → Map with runways colored by gradient quality")
        print(f"  3. {region}_best_runway_parametric.tif")
        print(f"     → Georeferenced raster of the overall best runway")
        print(f"  4. {region}_all_best_runways_parametric.tif")
        print(f"     → Georeferenced raster of all best-per-contour runways")
        print(f"  5. optimal_runway_plots/{region}_contour_XXX_optimal_runway.png")
        print(f"     → Individual plots for each of the {len(results['best_per_contour'])} valid contours")
        
        print(f"\nOverall best runway:")
        best = results['best_runway']
        print(f"  Contour: {best['contour_idx'] + 1}")
        print(f"  Length: {best['length']:.1f}m")
        print(f"  Parameters: t1={best['t1']:.4f}, t2={best['t2']:.4f}")
        print(f"  Max gradient: {best['gradient_metrics']['max_abs_gradient']:.2f}°")
        print(f"  Mean gradient: {best['gradient_metrics']['mean_abs_gradient']:.2f}°")
        
        print(f"\nFound {len(results['best_per_contour'])} contours with valid runways")
        
        # Show top 5 by gradient quality
        sorted_runways = sorted(results['best_per_contour'].values(), 
                               key=lambda r: r['gradient_metrics']['max_abs_gradient'])
        
        print("\nTop 5 runways by gradient quality:")
        for i, runway in enumerate(sorted_runways[:5]):
            print(f"  {i+1}. Contour {runway['contour_idx']+1}: "
                  f"max grad = {runway['gradient_metrics']['max_abs_gradient']:.2f}°, "
                  f"length = {runway['length']:.1f}m")
    
    else:
        print("\n" + "=" * 60)
        print("OPTIMIZATION FAILED")
        print("=" * 60)
        print("\nNo valid runways found. Possible reasons:")
        print("  - Filtered landable areas TIF file not found")
        print("  - DEM file not found")
        print("  - No regions large enough after min rectangle filter")
        print("  - No valid runway lines could be placed")