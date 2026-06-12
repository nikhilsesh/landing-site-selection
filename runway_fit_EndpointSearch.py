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
import sys
import time 

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
    def __init__(self, contour_points, contour_idx=None, landable_map=None):
        """
        Parameters:
        -----------
        contour_points : np.ndarray
            Contour points from cv2.findContours, shape (N, 1, 2) or (N, 2)
        """
        # Assign contour idx to self
        self.contour_idx = contour_idx
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
        
        # Store landable map for rasterization checks
        self.landable_map = landable_map

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

    # VERSION 3 To work with Nelder Mead and Diff Evolution
    def is_line_inside(self, t1, t2, debug_plot=False, save_path=None):
        """
        Check if the line segment from t1 to t2 stays inside the contour.
        Uses rasterization on a LOCAL bounding box (much faster).
        """
        p1 = self.get_point(t1)
        p2 = self.get_point(t2)
        
        # Get integer pixel coordinates
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        
        # ===================================================================
        # KEY OPTIMIZATION: Create small local bounding box instead of full image
        # ===================================================================
        padding = 5  # Small padding around the line
        x_min = max(0, min(x1, x2) - padding)
        x_max = max(x1, x2) + padding
        y_min = max(0, min(y1, y2) - padding)
        y_max = max(y1, y2) + padding
        
        # FIXED: Clip to valid bounds BEFORE calculating dimensions
        if self.landable_map is not None:
            height_full, width_full = self.landable_map.shape
            y_max = min(y_max, height_full - 1)
            x_max = min(x_max, width_full - 1)
        
        # Get dimensions of local box (AFTER clipping!)
        local_width = x_max - x_min + 1
        local_height = y_max - y_min + 1
        
        # Create small local rasters
        local_runway = np.zeros((local_height, local_width), dtype=np.uint8)
        
        # Translate line coordinates to local box coordinates
        x1_local = x1 - x_min
        y1_local = y1 - y_min
        x2_local = x2 - x_min
        y2_local = y2 - y_min
        
        # Draw line in local coordinates (FAST - only ~400 pixels, not millions!)
        cv2.line(local_runway, (x1_local, y1_local), (x2_local, y2_local), 1, thickness=1)
        
        # Extract corresponding region from contour mask
        if self.landable_map is not None:
            # Extract the same region from the landable map
            local_contour = self.landable_map[y_min:y_max+1, x_min:x_max+1].astype(np.uint8)
        else:
            # Create local contour mask (translate contour coordinates)
            local_contour = np.zeros((local_height, local_width), dtype=np.uint8)
            contour_translated = self.contour_cv - np.array([x_min, y_min])
            cv2.fillPoly(local_contour, [contour_translated], 1)
        
        # Count runway pixels (FAST - only checking ~400 pixels!)
        num_runway_pixels = np.sum(local_runway > 0)
        
        # Count runway pixels inside contour (FAST)
        num_inside_pixels = np.sum((local_runway > 0) & (local_contour > 0))
        
        # Determine validity
        is_valid = (num_runway_pixels == num_inside_pixels) and (num_runway_pixels > 0)
        
        # ========================================================================
        # DEBUG VISUALIZATION
        # ========================================================================
        if debug_plot:
            import matplotlib.pyplot as plt
            
            # Create RGB composite for visualization
            rgb_composite = np.zeros((local_height, local_width, 3), dtype=np.uint8)
            
            # Contour only (gray background)
            rgb_composite[:, :, 0] = local_contour * 100
            rgb_composite[:, :, 1] = local_contour * 100
            rgb_composite[:, :, 2] = local_contour * 100
            
            # Runway pixels that are INSIDE contour (GREEN)
            inside_mask = (local_runway > 0) & (local_contour > 0)
            rgb_composite[inside_mask, 0] = 0
            rgb_composite[inside_mask, 1] = 255
            rgb_composite[inside_mask, 2] = 0
            
            # Runway pixels that are OUTSIDE contour (RED - BAD!)
            outside_mask = (local_runway > 0) & (local_contour == 0)
            rgb_composite[outside_mask, 0] = 255
            rgb_composite[outside_mask, 1] = 0
            rgb_composite[outside_mask, 2] = 0
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Show composite
            ax.imshow(rgb_composite)
            ax.set_title(f'Runway Validity Check\n'
                        f'Green = Inside ({num_inside_pixels} px), '
                        f'Red = Outside ({num_runway_pixels - num_inside_pixels} px)\n'
                        f'Valid = {is_valid}',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Saved debug plot to: {save_path}")
            else:
                plt.savefig('debug/runway_intersect.png', dpi=150, bbox_inches='tight')
                print(f"  Saved debug plot to: debug/runway_intersect.png")
            
            plt.close()
        
        return is_valid
        
    def line_length(self, t1, t2):
        """
        Compute Euclidean distance between points at t1 and t2.
        """
        p1 = self.get_point(t1)
        p2 = self.get_point(t2)
        return np.linalg.norm(p2 - p1)

def optimize_runway_for_contour(param_contour, min_length=400, pixel_size=1.0, method='differential_evolution'):
    """
    Find the longest valid runway for a given parametric contour.
    
    Parameters:
    -----------
    param_contour : ParametricContour
        Parametrized contour object
    min_length : float
        Minimum required runway length in meters (not pixels!)
    pixel_size : float
        Pixel size in meters (for converting to real-world length)
    method : str
        Optimization method: 'differential_evolution' or 'nelder_mead'
    
    Returns:
    --------
    dict with:
        - t1, t2: optimal parameter values
        - length: runway length in meters
        - p1, p2: endpoint coordinates (col, row)
        - valid: whether a valid runway was found
        - method: optimization method used
        - iterations: number of function evaluations
    """
    print(f"  Optimizing runway (method={method}, contour has {param_contour.n_points} points, perimeter={param_contour.total_length*pixel_size:.1f}m)...")
    
    # Objective function: negative length (for minimization)
    def objective(x):
        t1, t2 = x
        
        # Enforce t1 < t2 constraint via penalty
        if t2 <= t1:
            return 1e10
        
        # DEBUGGING - Track evaluation count
        if not hasattr(objective, 'eval_count'):
            objective.eval_count = 0
        objective.eval_count += 1
        
        # Debug first evaluation for specific contour if needed
        if objective.eval_count == 1 and param_contour.contour_idx == 11:
            debug_runway_validity = False  # Set to True for debugging
        else:
            debug_runway_validity = False
        
        # CORRECTED: Use is_line_inside method
        is_valid = param_contour.is_line_inside(t1, t2, debug_plot=debug_runway_validity)
        
        if not is_valid:
            return 1e10  # Invalid - return huge penalty
        
        # Valid - return negative length (we want to maximize length)
        length = param_contour.line_length(t1, t2)
        return -length
    
    # Bounds: search over t1 ∈ [0, 1), t2 ∈ [0, 1) with t2 > t1
    bounds = [(0, 0.9999), (0, 0.9999)]
    
    if method == 'differential_evolution':
        # Use differential evolution (population-based method)
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev  # Number of function evaluations
        success = result.success
        
    elif method == 'nelder_mead':
        # Use Nelder-Mead (simplex method)
        # Need a good initial guess - try center of the contour
        # Start with points roughly 1/4 and 3/4 around the contour
        x0 = [0.25, 0.75]
        
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 0.001,  # Tolerance in parameter space
                'fatol': 1.0,    # Tolerance in function value (1 pixel)
                'adaptive': True
            }
        )
        
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev  # Number of function evaluations
        success = result.success
        
    elif method == 'multi_start_nm':
        # Multi-start Nelder-Mead (RECOMMENDED for non-population)
        n_starts = 20
        best_result = None
        best_objective = 1e10
        
        for start_idx in range(n_starts):
            t1_init = np.random.uniform(0, 0.7)
            t2_init = np.random.uniform(t1_init + 0.1, 0.99)
            x0 = [t1_init, t2_init]
            
            result_local = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 0.005,
                    'fatol': 5.0,
                    'adaptive': True
                }
            )
            
            if result_local.fun < best_objective:
                best_objective = result_local.fun
                best_result = result_local
        
        result = best_result
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev
        
    elif method == 'basin_hopping':
        from scipy.optimize import basinhopping
        
        class RandomDisplacementBounds:
            def __init__(self, xmin, xmax, stepsize=0.1):
                self.xmin = xmin
                self.xmax = xmax
                self.stepsize = stepsize
            
            def __call__(self, x):
                x_new = x + np.random.uniform(-self.stepsize, self.stepsize, x.shape)
                x_new = np.clip(x_new, self.xmin, self.xmax)
                if x_new[1] <= x_new[0]:
                    x_new[1] = x_new[0] + 0.05
                return x_new
        
        x0 = [0.25, 0.75]
        bounds_obj = RandomDisplacementBounds(
            xmin=np.array([0.0, 0.0]), 
            xmax=np.array([0.9999, 0.9999]),
            stepsize=0.15
        )
        
        result = basinhopping(
            objective,
            x0,
            niter=50,
            T=1.0,
            stepsize=0.15,
            take_step=bounds_obj,
            minimizer_kwargs={
                'method': 'Nelder-Mead',
                'options': {'maxiter': 100, 'adaptive': True}
            },
            seed=42
        )
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev
        
    elif method == 'direct':
        from scipy.optimize import direct
        
        result = direct(
            objective,
            bounds,
            eps=0.01,
            maxfun=2000,
            locally_biased=True
        )
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev
        
    elif method == 'dual_annealing':
        from scipy.optimize import dual_annealing
        
        result = dual_annealing(
            objective,
            bounds,
            maxiter=200,
            no_local_search=False,
            seed=42
        )
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev
        
    elif method == 'shgo':
        from scipy.optimize import shgo
        
        result = shgo(
            objective,
            bounds,
            n=200,
            iters=3,
            sampling_method='sobol'
        )
        t1_opt, t2_opt = result.x
        n_iterations = result.nfev

    else:
        raise ValueError(f"Unknown optimization method: {method}. Use 'differential_evolution' or 'nelder_mead'")
    
    # Check if optimization succeeded
    if result.fun >= 1e9:  # If objective is still penalty value
        print(f"  ✗ No valid runway found (all candidates violated constraints)")
        return {
            'valid': False,
            't1': None,
            't2': None,
            'length': 0,
            'p1': None,
            'p2': None,
            'method': method,
            'iterations': n_iterations
        }
    
    # Get endpoint coordinates
    p1 = param_contour.get_point(t1_opt)
    p2 = param_contour.get_point(t2_opt)
    
    # Compute actual length in meters
    length_pixels = param_contour.line_length(t1_opt, t2_opt)
    length_meters = length_pixels * pixel_size
    
    # Verify minimum length constraint
    if length_meters < min_length:
        print(f"  ✗ Runway too short: {length_meters:.1f}m < {min_length}m")
        return {
            'valid': False,
            't1': t1_opt,
            't2': t2_opt,
            'length': length_meters,
            'p1': p1,
            'p2': p2,
            'method': method,
            'iterations': n_iterations
        }
    
    print(f"  ✓ Found runway: {length_meters:.1f}m at t1={t1_opt:.3f}, t2={t2_opt:.3f} ({n_iterations} evals)")
    
    return {
        'valid': True,
        't1': t1_opt,
        't2': t2_opt,
        'length': length_meters,
        'p1': p1,  # (col, row)
        'p2': p2,  # (col, row)
        'method': method,
        'iterations': n_iterations
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
                                     runway_width=15,
                                     optimization_method='differential_evolution'):
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
    optimization_method : str
        Optimization algorithm: 'differential_evolution' or 'nelder_mead'
    
    Returns:
    --------
    dict with results
    """
    import time  # Add at top of function
    
    print("=" * 60)
    print(f"PARAMETRIC RUNWAY OPTIMIZATION FOR REGION: {region}")
    print(f"OPTIMIZATION METHOD: {optimization_method.upper()}")
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
    
    # START TIMING
    optimization_start_time = time.time()
    
    # Optimize runway for each contour
    print(f"\n{'=' * 60}")
    print("OPTIMIZING RUNWAYS FOR EACH CONTOUR")
    print(f"{'=' * 60}\n")
    
    valid_runways = []
    best_per_contour = {}
    total_iterations = 0
    
    for contour_idx, contour in enumerate(external_contours):
        print(f"Contour {contour_idx + 1}/{num_contours}:")
        
        # Create parametric contour WITH landable_map
        param_contour = ParametricContour(contour, contour_idx=contour_idx, landable_map=landable_filtered)
        
        # Optimize runway placement
        runway = optimize_runway_for_contour(
            param_contour,
            min_length=runway_length,
            pixel_size=pixel_size,
            method=optimization_method
        )
        
        if not runway['valid']:
            print(f"  Skipping - no valid runway found\n")
            continue
        
        # Track iterations
        total_iterations += runway['iterations']
        
        # Compute gradient metrics
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
            'param_contour': param_contour,
            'iterations': runway['iterations']
        }
        
        valid_runways.append(runway_info)
        best_per_contour[contour_idx] = runway_info
        
        print(f"  Gradient: max={gradient_metrics['max_abs_gradient']:.2f}°, "
              f"mean={gradient_metrics['mean_abs_gradient']:.2f}°\n")
    
    # END TIMING
    optimization_end_time = time.time()
    total_time = optimization_end_time - optimization_start_time
    
    if len(valid_runways) == 0:
        print("No valid runways found!")
        return None
    
    print(f"{'=' * 60}")
    print(f"FOUND {len(valid_runways)} VALID RUNWAYS")
    print(f"{'=' * 60}\n")
    
    # Compute timing statistics
    avg_time_per_contour = total_time / len(external_contours)
    avg_iterations_per_contour = total_iterations / len(valid_runways)
    
    print(f"{'=' * 60}")
    print(f"OPTIMIZATION PERFORMANCE")
    print(f"{'=' * 60}")
    print(f"Method: {optimization_method}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per contour: {avg_time_per_contour:.2f} seconds")
    print(f"Total function evaluations: {total_iterations}")
    print(f"Average evaluations per valid runway: {avg_iterations_per_contour:.1f}")
    print(f"{'=' * 60}\n")
    
    # Find overall best runway (minimum max gradient)
    best_runway = min(valid_runways, key=lambda r: r['gradient_metrics']['max_abs_gradient'])
    
    print(f"Best runway:")
    print(f"  Contour: {best_runway['contour_idx'] + 1}")
    print(f"  Length: {best_runway['length']:.1f}m")
    print(f"  Max gradient: {best_runway['gradient_metrics']['max_abs_gradient']:.2f}°")
    print(f"  Mean gradient: {best_runway['gradient_metrics']['mean_abs_gradient']:.2f}°")
    print(f"  Function evaluations: {best_runway['iterations']}")
    
# CREATE OBJECTIVE FUNCTION CONTOUR PLOTS (NEW!)
    # plot_objective_contours(
    #     region=region,
    #     best_per_contour=best_per_contour,
    #     pixel_size=pixel_size,
    #     resolution=100,  # Adjust for speed vs detail tradeoff
    #     min_runway_length=runway_length
    # )

    oracle_results, total_oracle_time = plot_objective_contours(
        region=region,
        best_per_contour=best_per_contour,
        pixel_size=pixel_size,
        resolution=1000,
        min_runway_length=runway_length,
        skip_plot=False  # Set to True if you only want oracle search without plots
    )

    # Extract oracle statistics
    oracle_total_time = sum(r['time'] for r in oracle_results.values() if r['valid'])
    oracle_best_length = max(r['length'] for r in oracle_results.values() if r['valid'])
    oracle_avg_gap = np.mean([
        ((r['length'] - best_per_contour[idx]['length']) / r['length'] * 100)
        for idx, r in oracle_results.items() if r['valid']
    ])

    # Create visualizations
    print(f"\n{'=' * 60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 60}\n")

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
        'best_runway': best_runway,
        'all_runways': valid_runways,
        'best_per_contour': best_per_contour,
        'landable_map': landable_filtered,
        'transform': transform,
        'optimization_method': optimization_method,
        'total_time_seconds': total_time,
        'total_iterations': total_iterations,
        'avg_time_per_contour': avg_time_per_contour,
        'oracle_results': oracle_results,
        'total_oracle_time': total_oracle_time
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
    os.makedirs(os.path.dirname('results/endpoint_search/'), exist_ok=True)
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

def plot_objective_contours(region, best_per_contour, pixel_size, resolution=100, min_runway_length=400, skip_plot=False):
    """
    Plot objective function contours over (t1, t2) design space for each valid contour.
    Also performs oracle exhaustive search to find global optimum.
    
    Parameters:
    -----------
    region : str
        Region identifier
    best_per_contour : dict
        Dictionary of best runway per contour
    pixel_size : float
        Pixel size in meters
    resolution : int
        Number of grid points along each axis (default 100)
    min_runway_length : float
        Minimum runway length to display on colorbar
    skip_plot : bool
        If True, skip plotting (only do oracle search)
    
    Returns:
    --------
    oracle_results : dict
        Dictionary with oracle search results for each contour
    """
    import os
    import time
    os.makedirs('results/objective_contours', exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print("CREATING OBJECTIVE FUNCTION CONTOUR PLOTS")
    print(f"{'=' * 60}\n")
    
    # Determine global colorbar range from all optimal runway lengths
    all_lengths = [runway['length'] for runway in best_per_contour.values()]
    vmax = max(all_lengths)  # Longest optimal runway
    
    print(f"Colorbar range: {min_runway_length:.1f}m to {vmax:.1f}m")
    print(f"Grid resolution: {resolution} × {resolution} = {resolution**2} evaluations per contour\n")
    
    # Store oracle results for all contours
    oracle_results = {}
    total_oracle_time = 0
    
    # Create contour plots for each valid contour
    for contour_idx, runway_data in best_per_contour.items():
        print(f"Processing contour {contour_idx + 1}/{len(best_per_contour)}...")
        
        param_contour = runway_data['param_contour']
        
        # Create grid over (t1, t2) space
        t1_grid = np.linspace(0.0, 0.999, resolution)
        t2_grid = np.linspace(0.001, 1.0, resolution)
        T1, T2 = np.meshgrid(t1_grid, t2_grid)
        
        # Evaluate objective function at each grid point
        obj_values = np.zeros_like(T1)
        
        # Start timer for oracle search
        oracle_start_time = time.time()

        for i in range(resolution):
            if i % 20 == 0 and not skip_plot:
                print(f"  Progress: {i}/{resolution} rows...")
            for j in range(resolution):
                t1 = T1[i, j]
                t2 = T2[i, j]
                
                # Check constraints
                if t2 <= t1:
                    obj_values[i, j] = np.nan  # Invalid region (white)
                else:
                    # Check if line is inside
                    if param_contour.is_line_inside(t1, t2, debug_plot=False):
                        # Valid: compute runway length
                        length = param_contour.line_length(t1, t2) * pixel_size
                        obj_values[i, j] = length  # Positive length
                    else:
                        obj_values[i, j] = np.nan  # Outside contour (white)
        
        oracle_end_time = time.time()
        oracle_time = oracle_end_time - oracle_start_time
        total_oracle_time += oracle_time
        
        # Find MAXIMUM (not minimum) - we want longest runway
        # Use nanmax to ignore NaN values
        valid_values = obj_values[~np.isnan(obj_values)]
        
        if len(valid_values) == 0:
            print(f"  ✗ Oracle found no valid runway")
            oracle_results[contour_idx] = {
                'contour_idx': contour_idx,
                'length': 0,
                't1': None,
                't2': None,
                'time': oracle_time,
                'evaluations': resolution ** 2,
                'valid': False
            }
            continue
        
        oracle_max_length = np.nanmax(obj_values)
        
        # Find the indices of the maximum value
        # Use np.unravel_index to convert flat index to 2D indices
        max_flat_idx = np.nanargmax(obj_values)
        max_i, max_j = np.unravel_index(max_flat_idx, obj_values.shape)
        oracle_t1_opt = T1[max_i, max_j]
        oracle_t2_opt = T2[max_i, max_j]
        
        # Store oracle results
        oracle_results[contour_idx] = {
            'contour_idx': contour_idx,
            'length': oracle_max_length,
            't1': oracle_t1_opt,
            't2': oracle_t2_opt,
            'time': oracle_time,
            'evaluations': resolution ** 2,
            'valid': True
        }
        
        # Print comparison
        alg_length = runway_data['length']
        gap = ((oracle_max_length - alg_length) / oracle_max_length) * 100 if oracle_max_length > 0 else 0
        
        print(f"  Oracle: {oracle_max_length:.1f}m at (t1={oracle_t1_opt:.4f}, t2={oracle_t2_opt:.4f}) in {oracle_time:.2f}s")
        print(f"  Algorithm: {alg_length:.1f}m")
        print(f"  Optimality gap: {gap:.2f}%")
        
        # Create the plot
        if not skip_plot:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Define explicit contour levels using the global range
            levels = np.linspace(min_runway_length, vmax, 21)  # 20 intervals
            
            # Plot contours with fixed colorbar range and explicit levels
            contour_plot = ax.contourf(T1, T2, obj_values, 
                                    levels=levels,
                                    cmap='viridis',
                                    vmin=min_runway_length, 
                                    vmax=vmax,
                                    extend='neither')
            
            # Mark invalid regions (NaN) as white
            invalid_mask = np.isnan(obj_values)
            ax.contourf(T1, T2, invalid_mask.astype(float), 
                    levels=[0.5, 1.5], 
                    colors='white', 
                    alpha=1.0)
            
            # Mark the constraint boundary t1 = t2
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='$t_1 = t_2$ boundary')
            
            # Mark the optimal solution from optimizer
            t1_opt = runway_data['t1']
            t2_opt = runway_data['t2']
            ax.plot(t1_opt, t2_opt, 'r*', markersize=20, 
                markeredgecolor='white', markeredgewidth=2,
                label=f'Alg. Optimal: $L$ = {alg_length:.1f}m')
            
            # Mark optimal solution from oracle
            ax.plot(oracle_t1_opt, oracle_t2_opt, 'cX', markersize=15, 
                markeredgecolor='black', markeredgewidth=2,
                label=f'Oracle Optimal: $L$ = {oracle_max_length:.1f}m')
            
            # Colorbar
            cbar = plt.colorbar(contour_plot, ax=ax, label='Runway Length (m)')
            
            # Labels and title
            ax.set_xlabel('$t_1$ (start parameter)', fontsize=12)
            ax.set_ylabel('$t_2$ (end parameter)', fontsize=12)
            ax.set_title(f'Objective Function: Contour {contour_idx + 1}\n'
                        f'Algorithm: {alg_length:.1f}m | Oracle: {oracle_max_length:.1f}m | Gap: {gap:.2f}%',
                        fontsize=13, fontweight='bold')
            
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Save
            output_path = f'results/objective_contours/{region}_contour_{contour_idx+1:03d}_objective.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved to: {output_path}")
    
    if not skip_plot:
        print(f"\n✓ All objective contour plots saved to results/objective_contours/")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("ORACLE SEARCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total oracle time: {total_oracle_time:.2f} seconds")
    print(f"Average time per contour: {total_oracle_time/len(best_per_contour):.2f} seconds")
    print(f"Total evaluations: {len(best_per_contour) * resolution**2}")
    
    return oracle_results, total_oracle_time

if __name__ == '__main__':    
    # Configuration
    region = 'norcoast_b23'  # or 'alameda_b21_x59y418'
    runway_length = 400  # meters
    runway_width = 15    # meters
    
    # Choose optimization method
    # Options: 'differential_evolution' or 'nelder_mead', 'multi_start_nm', 'basin_hopping', 'direct', 'dual_annealing', 'shgo'
    optimization_method = 'differential_evolution'  
    
    # You can also pass it as a command-line argument
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['differential_evolution', 'nelder_mead', 'de', 'nm']:
            if arg == 'de':
                optimization_method = 'differential_evolution'
            elif arg == 'nm':
                optimization_method = 'nelder_mead'
            else:
                optimization_method = arg
            print(f"\n>>> Using optimization method from command line: {optimization_method}\n")
    
    print(f"\n{'=' * 60}")
    print(f"RUNWAY OPTIMIZATION CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Region: {region}")
    print(f"Optimization method: {optimization_method}")
    print(f"Minimum runway length: {runway_length} m")
    print(f"Runway width: {runway_width} m")
    print(f"{'=' * 60}\n")
    
    # Run optimization
    results = find_optimal_runways_parametric(
        region=region,
        runway_length=runway_length,
        runway_width=runway_width,
        optimization_method=optimization_method
    )

    if results:
        # Create output directory
        os.makedirs('results/optimization_logs', exist_ok=True)
        
        # Create filename with timestamp and method
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'results/optimization_logs/{region}_{optimization_method}_{timestamp}.txt'
        
        # Function to print and write simultaneously
        def print_and_write(text, file_handle):
            print(text)
            file_handle.write(text + '\n')
        
        with open(output_file, 'w') as f:
            print_and_write("\n" + "=" * 60, f)
            print_and_write("OPTIMIZATION COMPLETE!", f)
            print_and_write("=" * 60, f)
            
            # Print summary statistics
            print_and_write(f"\nOptimization Summary:", f)
            print_and_write(f"  Region: {region}", f)
            print_and_write(f"  Method: {results['optimization_method']}", f)
            print_and_write(f"  Total runtime: {results['total_time_seconds']:.2f} seconds", f)
            print_and_write(f"  Total function evaluations: {results['total_iterations']}", f)
            print_and_write(f"  Average time per contour: {results['avg_time_per_contour']:.2f} seconds", f)
            print_and_write(f"  Valid runways found: {len(results['all_runways'])}", f)
            
            print_and_write(f"\nBest Runway:", f)
            best = results['best_runway']
            print_and_write(f"  Contour: {best['contour_idx'] + 1}", f)
            print_and_write(f"  Length: {best['length']:.1f} m", f)
            print_and_write(f"  Max gradient: {best['gradient_metrics']['max_abs_gradient']:.2f}°", f)
            print_and_write(f"  Mean gradient: {best['gradient_metrics']['mean_abs_gradient']:.2f}°", f)

            print_and_write(f"\nLongest Runway:", f)
            longest = max(results['all_runways'], key=lambda r: r['length'])
            print_and_write(f"  Contour: {longest['contour_idx'] + 1}", f)
            print_and_write(f"  Length: {longest['length']:.1f} m", f)
            print_and_write(f"  Max gradient: {longest['gradient_metrics']['max_abs_gradient']:.2f}°", f)
            print_and_write(f"  Mean gradient: {longest['gradient_metrics']['mean_abs_gradient']:.2f}°", f)

            # Print oracle search result for longest runway
            print_and_write(f"\nBest Oracle Search Results:", f)
            oracle_longest = max(results['oracle_results'].values(), key=lambda r: r['length'] if r['valid'] else 0)
            print_and_write(f"  Contour: {oracle_longest['contour_idx'] + 1}", f)
            print_and_write(f"  Length: {oracle_longest['length']:.1f} m", f)
            print_and_write(f"  t1: {oracle_longest['t1']:.4f}", f)
            print_and_write(f"  t2: {oracle_longest['t2']:.4f}", f)
            print_and_write(f"  Total oracle search time: {results['total_oracle_time']:.2f} seconds", f)

            # Sort by gradient and show top 5
            sorted_runways = sorted(results['all_runways'], 
                                key=lambda r: r['gradient_metrics']['max_abs_gradient'])
            
            print_and_write(f"\nTop 5 runways by gradient:", f)
            for i, runway in enumerate(sorted_runways[:5]):
                print_and_write(f"  {i+1}. Contour {runway['contour_idx']+1}: "
                    f"max grad = {runway['gradient_metrics']['max_abs_gradient']:.2f}°, "
                    f"length = {runway['length']:.1f}m", f)
            
            # Optional: Print all valid runways
            print_and_write(f"\n" + "=" * 60, f)
            print_and_write(f"ALL VALID RUNWAYS ({len(results['all_runways'])} total)", f)
            print_and_write("=" * 60, f)
            
            # Sort by contour index for easy reference
            sorted_by_idx = sorted(results['all_runways'], key=lambda r: r['contour_idx'])
            for runway in sorted_by_idx:
                print_and_write(f"\nContour {runway['contour_idx'] + 1}:", f)
                print_and_write(f"  Length: {runway['length']:.1f} m", f)
                print_and_write(f"  Max gradient: {runway['gradient_metrics']['max_abs_gradient']:.2f}°", f)
                print_and_write(f"  Mean gradient: {runway['gradient_metrics']['mean_abs_gradient']:.2f}°", f)
                print_and_write(f"  Std gradient: {runway['gradient_metrics']['std_gradient']:.2f}°", f)
                print_and_write(f"  Parameters: t1={runway['t1']:.4f}, t2={runway['t2']:.4f}", f)
                print_and_write(f"  Function evaluations: {runway['iterations']}", f)
        
        print(f"\n✓ Results saved to: {output_file}")

    else:
        # Create output directory
        os.makedirs('results/optimization_logs', exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'results/optimization_logs/{region}_{optimization_method}_{timestamp}_FAILED.txt'
        
        def print_and_write(text, file_handle):
            print(text)
            file_handle.write(text + '\n')
        
        with open(output_file, 'w') as f:
            print_and_write("\n" + "=" * 60, f)
            print_and_write("OPTIMIZATION FAILED", f)
            print_and_write("=" * 60, f)
            print_and_write(f"\nRegion: {region}", f)
            print_and_write(f"Method: {optimization_method}", f)
            print_and_write("\nNo valid runways found. Possible reasons:", f)
            print_and_write("  - Filtered landable areas TIF file not found", f)
            print_and_write("  - DEM file not found", f)
            print_and_write("  - No regions large enough after min rectangle filter", f)
            print_and_write("  - No valid runway lines could be placed", f)
        
        print(f"\n✓ Failure log saved to: {output_file}")