"""
runway_fit.py - Optimization-Based Runway Line Fitting

Uses derivative-free optimization to find optimal runway line placements
within landable contours by maximizing runway length.
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import rowcol, xy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import minimize, differential_evolution
import os
from shapely.geometry import LineString, Polygon, Point
from shapely.validation import explain_validity
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def pixel_to_geo(row, col, transform):
    """Convert pixel coordinates to geographic coordinates."""
    return transform * (col, row)


def geo_to_pixel(x, y, transform):
    """Convert geographic coordinates to pixel coordinates."""
    return rowcol(transform, x, y)


# def extract_contour_polygon(contour, transform):
#     """
#     Convert OpenCV contour to Shapely polygon in geographic coordinates.
    
#     Parameters:
#     -----------
#     contour : ndarray
#         OpenCV contour array
#     transform : Affine
#         Rasterio transform
    
#     Returns:
#     --------
#     Shapely Polygon in geographic coordinates
#     """
#     # Contour is shaped (N, 1, 2) with (x=col, y=row)
#     points = []
#     for pt in contour:
#         col, row = pt[0]
#         geo_x, geo_y = pixel_to_geo(row, col, transform)
#         points.append((geo_x, geo_y))
    
#     if len(points) < 3:
#         return None
    
#     return Polygon(points)

def extract_contour_polygon(contour, transform):
    """
    Convert OpenCV contour to Shapely polygon in geographic coordinates.
    
    Parameters:
    -----------
    contour : ndarray
        OpenCV contour array
    transform : Affine
        Rasterio transform
    
    Returns:
    --------
    Shapely Polygon in geographic coordinates
    """
    # Contour is shaped (N, 1, 2) with (x=col, y=row)
    points = []
    
    print(f"    Contour shape: {contour.shape}, dtype: {contour.dtype}")
    
    for pt in contour:
        col, row = pt[0]
        geo_x, geo_y = pixel_to_geo(row, col, transform)
        points.append((geo_x, geo_y))
    
    print(f"    Extracted {len(points)} points")
    
    if len(points) < 3:
        print(f"    ERROR: Less than 3 points")
        return None
    
    # Check for duplicate points (common Shapely invalidity issue)
    unique_points = []
    for i, pt in enumerate(points):
        if i == 0 or pt != points[i-1]:
            unique_points.append(pt)
    
    print(f"    Unique points: {len(unique_points)}")
    
    if len(unique_points) < 3:
        print(f"    ERROR: Less than 3 unique points")
        return None
    
    try:
        polygon = Polygon(unique_points)
        print(f"    Polygon created: valid={polygon.is_valid}, area={polygon.area:.2f}")
        
        if not polygon.is_valid:
            print(f"    Invalid reason: {explain_validity(polygon)}")
            # Try to fix it
            polygon = polygon.buffer(0)
            print(f"    After buffer(0) fix: valid={polygon.is_valid}")
        
        return polygon if polygon.is_valid else None
        
    except Exception as e:
        print(f"    ERROR creating polygon: {e}")
        return None


def erode_contour(polygon, erosion_distance_meters):
    """
    Erode a polygon by a specified distance (negative buffer).
    
    Parameters:
    -----------
    polygon : Shapely Polygon
        Input polygon in meters (UTM)
    erosion_distance_meters : float
        Erosion distance
    
    Returns:
    --------
    Eroded Shapely Polygon (or None if erosion eliminates the polygon)
    """
    eroded = polygon.buffer(-erosion_distance_meters)
    
    # Buffer can return MultiPolygon if the shape splits
    if eroded.is_empty:
        return None
    
    # If multiple polygons result, take the largest
    if eroded.geom_type == 'MultiPolygon':
        eroded = max(eroded.geoms, key=lambda p: p.area)
    
    return eroded if eroded.area > 0 else None


# def compute_runway_length(center_x, center_y, theta, contour_polygon):
#     """
#     Compute maximum runway length for a given center and direction.
    
#     The runway extends from the center in both directions along theta
#     until it hits the contour boundary.
    
#     Parameters:
#     -----------
#     center_x, center_y : float
#         Center point in geographic coordinates (meters)
#     theta : float
#         Direction angle in radians (0 = East, π/2 = North)
#     contour_polygon : Shapely Polygon
#         Outer contour boundary
    
#     Returns:
#     --------
#     length : float
#         Maximum runway length (meters), or 0 if invalid
#     """
#     # Create a very long line through the center in direction theta
#     max_extent = 10000  # 10 km should be more than enough
    
#     dx = np.cos(theta)
#     dy = np.sin(theta)
    
#     # Line endpoints
#     x1 = center_x - max_extent * dx
#     y1 = center_y - max_extent * dy
#     x2 = center_x + max_extent * dx
#     y2 = center_y + max_extent * dy
    
#     line = LineString([(x1, y1), (x2, y2)])
    
#     # Intersect with polygon boundary
#     try:
#         intersection = line.intersection(contour_polygon.boundary)
#     except:
#         return 0.0
    
#     # Extract intersection points
#     if intersection.is_empty:
#         return 0.0
    
#     # Get all intersection points
#     points = []
#     if intersection.geom_type == 'Point':
#         return 0.0
#     elif intersection.geom_type == 'MultiPoint':
#         points = list(intersection.geoms)
#     elif intersection.geom_type == 'LineString':
#         coords = list(intersection.coords)
#         points = [Point(coords[0]), Point(coords[-1])]
#     elif intersection.geom_type == 'MultiLineString':
#         for line_seg in intersection.geoms:
#             coords = list(line_seg.coords)
#             points.extend([Point(coords[0]), Point(coords[-1])])
#     elif intersection.geom_type == 'GeometryCollection':
#         for geom in intersection.geoms:
#             if geom.geom_type == 'Point':
#                 points.append(geom)
#             elif geom.geom_type == 'LineString':
#                 coords = list(geom.coords)
#                 points.extend([Point(coords[0]), Point(coords[-1])])
#     else:
#         return 0.0
    
#     if len(points) < 2:
#         return 0.0
    
#     # Find the two points farthest apart (entry/exit points)
#     max_dist = 0.0
#     for i, p1 in enumerate(points):
#         for p2 in points[i+1:]:
#             dist = p1.distance(p2)
#             if dist > max_dist:
#                 max_dist = dist
    
#     return max_dist


# def get_runway_endpoints(center_x, center_y, theta, length):
#     """
#     Get the two endpoints of a runway line.
    
#     Parameters:
#     -----------
#     center_x, center_y : float
#         Center point
#     theta : float
#         Direction angle (radians)
#     length : float
#         Runway length
    
#     Returns:
#     --------
#     (x1, y1), (x2, y2) : tuples
#         Two endpoints
#     """
#     dx = np.cos(theta)
#     dy = np.sin(theta)
#     half_length = length / 2.0
    
#     x1 = center_x - half_length * dx
#     y1 = center_y - half_length * dy
#     x2 = center_x + half_length * dx
#     y2 = center_y + half_length * dy
    
#     return (x1, y1), (x2, y2)

# Combines runway length and endpoints function
def compute_runway_length_and_endpoints(center_x, center_y, theta, contour_polygon):
    """
    Compute maximum runway length AND the actual endpoints.
    
    Returns:
    --------
    length : float
        Maximum runway length (meters), or 0 if invalid
    endpoints : tuple or None
        ((x1, y1), (x2, y2)) or None if invalid
    """
    # Create a very long line through the center in direction theta
    max_extent = 10000  # 10 km should be more than enough
    
    dx = np.cos(theta)
    dy = np.sin(theta)
    
    # Line endpoints
    x1 = center_x - max_extent * dx
    y1 = center_y - max_extent * dy
    x2 = center_x + max_extent * dx
    y2 = center_y + max_extent * dy
    
    line = LineString([(x1, y1), (x2, y2)])
    
    # Intersect with polygon boundary
    try:
        intersection = line.intersection(contour_polygon.boundary)
    except:
        return 0.0, None
    
    # Extract intersection points
    if intersection.is_empty:
        return 0.0, None
    
    # Get all intersection points
    points = []
    if intersection.geom_type == 'Point':
        return 0.0, None
    elif intersection.geom_type == 'MultiPoint':
        points = list(intersection.geoms)
    elif intersection.geom_type == 'LineString':
        coords = list(intersection.coords)
        points = [Point(coords[0]), Point(coords[-1])]
    elif intersection.geom_type == 'MultiLineString':
        for line_seg in intersection.geoms:
            coords = list(line_seg.coords)
            points.extend([Point(coords[0]), Point(coords[-1])])
    elif intersection.geom_type == 'GeometryCollection':
        for geom in intersection.geoms:
            if geom.geom_type == 'Point':
                points.append(geom)
            elif geom.geom_type == 'LineString':
                coords = list(geom.coords)
                points.extend([Point(coords[0]), Point(coords[-1])])
    else:
        return 0.0, None
    
    if len(points) < 2:
        return 0.0, None
    
    # Find the two points farthest apart AND return them
    max_dist = 0.0
    best_p1 = None
    best_p2 = None
    
    for i, p1 in enumerate(points):
        for p2 in points[i+1:]:
            dist = p1.distance(p2)
            if dist > max_dist:
                max_dist = dist
                best_p1 = p1
                best_p2 = p2
    
    if best_p1 is None or best_p2 is None:
        return 0.0, None
    
    endpoints = ((best_p1.x, best_p1.y), (best_p2.x, best_p2.y))
    
    return max_dist, endpoints


def sample_elevation_along_line(center_x, center_y, theta, length, elevation, transform, num_samples=100):
    """
    Sample elevation values along a runway line.
    
    Parameters:
    -----------
    center_x, center_y : float
        Center point (geographic coords)
    theta : float
        Direction angle (radians)
    length : float
        Line length (meters)
    elevation : ndarray
        DEM elevation data
    transform : Affine
        Rasterio transform
    num_samples : int
        Number of sample points
    
    Returns:
    --------
    elevations : ndarray or None
        Array of elevation values, or None if out of bounds
    """
    # Sample points along line
    t = np.linspace(-length/2, length/2, num_samples)
    dx = np.cos(theta)
    dy = np.sin(theta)
    
    xs = center_x + t * dx
    ys = center_y + t * dy
    
    # Convert to pixel coordinates
    elevations = []
    for x, y in zip(xs, ys):
        row, col = geo_to_pixel(x, y, transform)
        
        # Check bounds
        if (row < 0 or row >= elevation.shape[0] or 
            col < 0 or col >= elevation.shape[1]):
            return None
        
        elev = elevation[row, col]
        
        # Check for masked/invalid data
        if np.ma.is_masked(elev) or np.isnan(elev):
            return None
        
        elevations.append(elev)
    
    return np.array(elevations)


def compute_gradient_metrics(elevations, length):
    """
    Compute gradient statistics from elevation samples.
    
    Parameters:
    -----------
    elevations : ndarray
        Elevation values along line
    length : float
        Total length (meters)
    
    Returns:
    --------
    dict with gradient metrics
    """
    if len(elevations) < 2:
        return None
    
    # Compute gradients between consecutive points
    dists = length / (len(elevations) - 1)
    dz = np.diff(elevations)
    gradients_rad = np.arctan2(dz, dists)
    gradients_deg = np.degrees(gradients_rad)
    
    return {
        'max_abs_gradient': np.max(np.abs(gradients_deg)),
        'mean_abs_gradient': np.mean(np.abs(gradients_deg)),
        'std_gradient': np.std(gradients_deg),
        'elevation_change': elevations.max() - elevations.min(),
        'elevations': elevations
    }


# ============================================================================
# OPTIMIZATION
# ============================================================================

class RunwayOptimizer:
    """Encapsulates the optimization problem for a single contour."""
    
    def __init__(self, contour_polygon, eroded_polygon, elevation, transform):
        self.contour_polygon = contour_polygon
        self.eroded_polygon = eroded_polygon
        self.elevation = elevation
        self.transform = transform
        self.best_length = 0
        self.eval_count = 0
    
    # def objective(self, x):
    #     """
    #     Objective function: minimize negative length.
    #     x = [center_x, center_y, theta]
    #     """
    #     self.eval_count += 1
    #     center_x, center_y, theta = x
        
    #     # Check if center is in eroded polygon
    #     center_point = Point(center_x, center_y)
    #     if not self.eroded_polygon.contains(center_point):
    #         return 1e9
        
    #     # Compute runway length
    #     length = compute_runway_length(center_x, center_y, theta, self.contour_polygon)
        
    #     if length == 0:
    #         return 1e9
        
    #     # Track best for reporting
    #     if length > self.best_length:
    #         self.best_length = length
        
    #     return -length  # Minimize negative length = maximize length

    def objective(self, x):
        self.eval_count += 1
        center_x, center_y, theta = x
        
        center_point = Point(center_x, center_y)
        if not self.eroded_polygon.contains(center_point):
            return 1e9
        
        # Compute runway length (ignore endpoints for now)
        length, _ = compute_runway_length_and_endpoints(center_x, center_y, theta, self.contour_polygon)
        
        if length == 0:
            return 1e9
        
        if length > self.best_length:
            self.best_length = length
        
        return -length
    
    def get_bounds(self):
        """Get optimization bounds from eroded polygon."""
        bounds = self.eroded_polygon.bounds  # (minx, miny, maxx, maxy)
        
        return [
            (bounds[0], bounds[2]),  # x bounds
            (bounds[1], bounds[3]),  # y bounds
            (0, np.pi/2)                # theta bounds (0 to 90 degrees)
        ]
    
    def get_initial_guess(self):
        """Get initial guess from eroded polygon centroid."""
        centroid = self.eroded_polygon.centroid
        return [centroid.x, centroid.y, np.pi/4]  # Start at 45 degrees


def optimize_runway_for_contour(contour_data, elevation, transform, method='Nelder-Mead', max_iter=500):
    """
    Find optimal runway for a single contour using optimization.
    
    Parameters:
    -----------
    contour_data : dict
        Contains 'polygon', 'eroded_polygon', 'id', etc.
    elevation : ndarray
        DEM data
    transform : Affine
        Rasterio transform
    method : str
        Optimization method ('Nelder-Mead', 'Powell', or 'differential_evolution')
    max_iter : int
        Maximum iterations
    
    Returns:
    --------
    dict with optimization results, or None if failed
    """
    optimizer = RunwayOptimizer(
        contour_data['polygon'],
        contour_data['eroded_polygon'],
        elevation,
        transform
    )
    
    print(f"\n  Optimizing contour {contour_data['id']} (area={contour_data['area']:.0f} m²)...")
    
    if method == 'differential_evolution':
        # Global optimization
        result = differential_evolution(
            optimizer.objective,
            bounds=optimizer.get_bounds(),
            maxiter=max_iter,
            seed=42,
            atol=1.0,  # Tolerance in meters
            tol=0.01
        )
    else:
        # Local optimization
        x0 = optimizer.get_initial_guess()
        result = minimize(
            optimizer.objective,
            x0,
            method=method,
            bounds=optimizer.get_bounds(),
            options={'maxiter': max_iter}
        )
    
    if not result.success or result.fun >= 1e8:
        print(f"    Optimization failed for contour {contour_data['id']}")
        return None
    
    # # Extract results
    # center_x, center_y, theta = result.x
    # length = -result.fun
    
    # # Sample elevation along optimal runway
    # elevations = sample_elevation_along_line(center_x, center_y, theta, length, 
    #                                         elevation, transform)

    # Extract results
    center_x, center_y, theta = result.x
    
    # Get the ACTUAL endpoints from the intersection
    length, endpoints = compute_runway_length_and_endpoints(center_x, center_y, theta, 
                                                             contour_data['polygon'])
    
    if endpoints is None:
        print(f"    Could not compute endpoints for contour {contour_data['id']}")
        return None
    
    (x1, y1), (x2, y2) = endpoints  # USE THESE, not get_runway_endpoints()
    
    # Sample elevation along optimal runway
    elevations = sample_elevation_along_line(center_x, center_y, theta, length, 
                                            elevation, transform)
    
    if elevations is None:
        print(f"    Invalid elevation data for contour {contour_data['id']}")
        return None
    
    # Compute gradient metrics
    gradient_metrics = compute_gradient_metrics(elevations, length)
    
    if gradient_metrics is None:
        return None
    
    # Get endpoints
    # (x1, y1), (x2, y2) = get_runway_endpoints(center_x, center_y, theta, length)
    
    print(f"    Found runway: length={length:.1f}m, max_grad={gradient_metrics['max_abs_gradient']:.2f}°, "
          f"evals={optimizer.eval_count}")
    
    return {
        'contour_id': contour_data['id'],
        'center': (center_x, center_y),
        'theta': theta,
        'theta_deg': np.degrees(theta),
        'length': length,
        'endpoints': ((x1, y1), (x2, y2)),
        'gradient_metrics': gradient_metrics,
        'optimization_result': result,
        'eval_count': optimizer.eval_count
    }


# ============================================================================
# CONTOUR PROCESSING
# ============================================================================

def filter_by_min_rectangle(binary_image, min_width=15, min_height=400):
    """
    Keep only regions that can fit a rectangle of at least min_width × min_height.
    """
    if binary_image.dtype == bool:
        binary_uint8 = binary_image.astype(np.uint8) * 255
    else:
        binary_uint8 = binary_image.astype(np.uint8)

    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros_like(binary_uint8)

    pixel_size = 1  # Will be scaled properly when called

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (center), (width, height), angle = rect
        smaller_dim = min(width, height)
        larger_dim = max(width, height)

        if smaller_dim >= min_width and larger_dim >= min_height:
            cv2.drawContours(filtered_image, [contour], -1, 255, -1)

    return filtered_image.astype(bool) if binary_image.dtype == bool else filtered_image

def process_contours(landable_map, transform, min_area=6000, erosion_distance=200):
    """
    Extract and process contours from landable map.
    
    Parameters:
    -----------
    landable_map : ndarray (bool)
        Binary landable areas
    transform : Affine
        Rasterio transform
    min_area : float
        Minimum contour area (m²)
    erosion_distance : float
        Erosion distance for inner contour (m)
    
    Returns:
    --------
    list of dicts, each containing:
        - 'polygon': outer contour polygon
        - 'eroded_polygon': eroded inner polygon
        - 'cv_contour': original OpenCV contour
        - 'area': contour area (m²)
        - 'id': contour ID
    """
    # Convert to uint8 for OpenCV
    binary_uint8 = landable_map.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\nProcessing {len(contours)} contours...")

    # DEBUG: Check first few contours
    print("\nDEBUG: First 3 contours:")
    for i, cnt in enumerate(contours[:3]):
        print(f"  Contour {i}: shape={cnt.shape}, points={len(cnt)}")
        # Print first few points
        print(f"    First 5 points: {cnt[:5].reshape(-1, 2)}")
    
    viable_contours = []
    pixel_size = abs(transform.a)
    
    for idx, contour in enumerate(contours):
        # Convert to polygon
        polygon = extract_contour_polygon(contour, transform)
        
        if polygon is None or not polygon.is_valid:
            # print(f"  Contour {idx}: invalid polygon, skipped")
            continue
        
        area = polygon.area
        
        # Filter by minimum area
        if area < min_area:
            # print(f"  Contour {idx}: area={area:.0f} m² - too small, skipped")
            continue

        # After the area check but before erosion:
        print(f"  Contour {idx}: area={area:.0f} m², trying {erosion_distance}m erosion...")

        # Erode to get inner contour
        eroded = erode_contour(polygon, erosion_distance)
        
        if eroded is None:
            print(f"  Contour {idx}: area={area:.0f} m² - too small after erosion, skipped")
            continue
        
        viable_contours.append({
            'id': idx,
            'polygon': polygon,
            'eroded_polygon': eroded,
            'cv_contour': contour,
            'area': area,
            'eroded_area': eroded.area
        })
        
        print(f"  Contour {idx}: area={area:.0f} m², eroded_area={eroded.area:.0f} m² - viable")
    
    print(f"\nFound {len(viable_contours)} viable contours after filtering")
    
    return viable_contours


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(region, landable_map, elevation, transform, 
                     contour_data_list, optimal_runways):
    """
    Create visualization of optimization results.
    
    Parameters:
    -----------
    region : str
        Region name
    landable_map : ndarray
        Binary landable areas
    elevation : ndarray
        DEM data
    transform : Affine
        Rasterio transform
    contour_data_list : list
        List of contour data dicts
    optimal_runways : list
        List of optimal runway results
    """
    print(f"\nCreating visualizations...")
    
    # Get geographic extent
    height, width = landable_map.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    extent = [left, right, bottom, top]
    
    # Create main figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ========================================================================
    # Panel 1: All contours with optimal runways
    # ========================================================================
    ax = axes[0, 0]
    ax.imshow(landable_map, cmap='Greens', extent=extent, aspect='auto', alpha=0.6)
    
    # Draw all optimal runways
    for runway in optimal_runways:
        (x1, y1), (x2, y2) = runway['endpoints']
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2.5, alpha=0.8)
    
    ax.set_title(f'Optimal Runways per Contour (n={len(optimal_runways)})')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')
    
    # ========================================================================
    # Panel 2: Best runway highlighted
    # ========================================================================
    ax = axes[0, 1]
    ax.imshow(landable_map, cmap='Greens', extent=extent, aspect='auto', alpha=0.6)
    
    # Draw all runways in blue
    for runway in optimal_runways:
        (x1, y1), (x2, y2) = runway['endpoints']
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)
    
    # Find and highlight overall best (longest or best gradient)
    best_runway = min(optimal_runways, key=lambda r: r['gradient_metrics']['max_abs_gradient'])
    (x1, y1), (x2, y2) = best_runway['endpoints']
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=5, alpha=0.8, label='Overall best (outline)')
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=1.0, label='Overall best')
    
    ax.set_title(f'Overall Best Runway\n(Length: {best_runway["length"]:.1f}m, '
                f'Max Grad: {best_runway["gradient_metrics"]["max_abs_gradient"]:.2f}°)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.legend()
    
    # ========================================================================
    # Panel 3: Elevation profile of best runway
    # ========================================================================
    ax = axes[1, 0]
    elevations = best_runway['gradient_metrics']['elevations']
    distance = np.linspace(0, best_runway['length'], len(elevations))
    
    ax.plot(distance, elevations, 'b-', linewidth=2)
    ax.fill_between(distance, elevations, alpha=0.3)
    ax.set_xlabel('Distance along runway (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Profile of Best Runway')
    ax.grid(True, alpha=0.3)
    
    # Add gradient info
    metrics = best_runway['gradient_metrics']
    info_text = (f"Length: {best_runway['length']:.1f}m\n"
                 f"Max gradient: {metrics['max_abs_gradient']:.2f}°\n"
                 f"Mean gradient: {metrics['mean_abs_gradient']:.2f}°\n"
                 f"Std gradient: {metrics['std_gradient']:.2f}°\n"
                 f"Elevation change: {metrics['elevation_change']:.1f}m\n"
                 f"Contour: {best_runway['contour_id']}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    # ========================================================================
    # Panel 4: Statistics
    # ========================================================================
    ax = axes[1, 1]
    
    # Gradient histogram
    max_grads = [r['gradient_metrics']['max_abs_gradient'] for r in optimal_runways]
    mean_grads = [r['gradient_metrics']['mean_abs_gradient'] for r in optimal_runways]
    lengths = [r['length'] for r in optimal_runways]
    
    ax.hist(max_grads, bins=20, alpha=0.6, label='Max gradient', color='red', edgecolor='black')
    ax.hist(mean_grads, bins=20, alpha=0.6, label='Mean gradient', color='blue', edgecolor='black')
    
    # Mark best
    ax.axvline(best_runway['gradient_metrics']['max_abs_gradient'], 
               color='red', linestyle='--', linewidth=2, label='Best (max)')
    ax.axvline(best_runway['gradient_metrics']['mean_abs_gradient'], 
               color='blue', linestyle='--', linewidth=2, label='Best (mean)')
    
    ax.set_xlabel('Gradient (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Runway Gradients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'results/{region}_runway_optimization_Fix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved main visualization to: {output_path}")
    
    # ========================================================================
    # Second figure: Runways colored by gradient on elevation map
    # ========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Background: elevation
    im = ax2.imshow(elevation, cmap='terrain', extent=extent, aspect='auto', alpha=0.5)
    plt.colorbar(im, ax=ax2, label='Elevation (m)')
    
    # Overlay landable areas
    landable_overlay = np.ma.masked_where(~landable_map, landable_map)
    ax2.imshow(landable_overlay, cmap='Greens', extent=extent, aspect='auto', alpha=0.3)
    
    # Color runways by gradient
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    vmin = min(max_grads)
    vmax = max(max_grads)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # Red = bad, Green = good
    
    for runway in optimal_runways:
        (x1, y1), (x2, y2) = runway['endpoints']
        color = cmap(norm(runway['gradient_metrics']['max_abs_gradient']))
        ax2.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)
    
    # Highlight best
    (x1, y1), (x2, y2) = best_runway['endpoints']
    ax2.plot([x1, x2], [y1, y2], 'k-', linewidth=6, alpha=0.8)
    ax2.plot([x1, x2], [y1, y2], 'white', linewidth=3, alpha=1.0, label='Best runway')
    
    # Colorbar for gradient
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, label='Max Gradient (degrees)', pad=0.02)
    
    ax2.set_title(f'Optimal Runways Colored by Gradient Quality\n'
                  f'(One per contour, n={len(optimal_runways)})')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_path2 = f'results/{region}_runway_gradient_map_opt_Fix.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved gradient map to: {output_path2}")
    
    plt.close('all')


def save_results_geotiff(region, optimal_runways, landable_map, transform, profile):
    """
    Save optimal runways as GeoTIFF.
    """
    runway_raster = np.zeros_like(landable_map, dtype=np.uint8)
    
    pixel_size = abs(transform.a)
    
    for runway in optimal_runways:
        (x1, y1), (x2, y2) = runway['endpoints']
        
        # Convert to pixel coords
        row1, col1 = geo_to_pixel(x1, y1, transform)
        row2, col2 = geo_to_pixel(x2, y2, transform)
        
        # Draw line using Bresenham
        cv2.line(runway_raster, (col1, row1), (col2, row2), runway['contour_id'] % 255 + 1, 1)
    
    output_path = f'results/{region}_optimal_runways.tif'
    
    profile_copy = profile.copy()
    profile_copy.update(dtype=rasterio.uint8, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile_copy) as dst:
        dst.write(runway_raster, 1)
    
    print(f"Saved optimal runways GeoTIFF to: {output_path}")

def print_summary_statistics(optimal_runways):
    """
    Print summary statistics for all optimal runways.
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    lengths = [r['length'] for r in optimal_runways]
    max_grads = [r['gradient_metrics']['max_abs_gradient'] for r in optimal_runways]
    mean_grads = [r['gradient_metrics']['mean_abs_gradient'] for r in optimal_runways]
    eval_counts = [r['eval_count'] for r in optimal_runways]
    
    print(f"\nTotal optimal runways found: {len(optimal_runways)}")
    
    print(f"\nRunway Lengths:")
    print(f"  Min: {min(lengths):.1f}m")
    print(f"  Max: {max(lengths):.1f}m")
    print(f"  Mean: {np.mean(lengths):.1f}m")
    print(f"  Median: {np.median(lengths):.1f}m")
    
    print(f"\nMax Gradient Statistics:")
    print(f"  Min: {min(max_grads):.2f}°")
    print(f"  Max: {max(max_grads):.2f}°")
    print(f"  Mean: {np.mean(max_grads):.2f}°")
    print(f"  Median: {np.median(max_grads):.2f}°")
    
    print(f"\nMean Gradient Statistics:")
    print(f"  Min: {min(mean_grads):.2f}°")
    print(f"  Max: {max(mean_grads):.2f}°")
    print(f"  Mean: {np.mean(mean_grads):.2f}°")
    print(f"  Median: {np.median(mean_grads):.2f}°")
    
    print(f"\nOptimization Efficiency:")
    print(f"  Function evaluations per contour:")
    print(f"    Min: {min(eval_counts)}")
    print(f"    Max: {max(eval_counts)}")
    print(f"    Mean: {np.mean(eval_counts):.1f}")
    
    # Runway quality categories
    excellent = sum(1 for g in max_grads if g < 1.0)
    good = sum(1 for g in max_grads if 1.0 <= g < 2.0)
    acceptable = sum(1 for g in max_grads if 2.0 <= g < 3.0)
    marginal = sum(1 for g in max_grads if g >= 3.0)
    
    print(f"\nRunway Quality (by max gradient):")
    print(f"  Excellent (<1°): {excellent} ({100*excellent/len(optimal_runways):.1f}%)")
    print(f"  Good (1-2°): {good} ({100*good/len(optimal_runways):.1f}%)")
    print(f"  Acceptable (2-3°): {acceptable} ({100*acceptable/len(optimal_runways):.1f}%)")
    print(f"  Marginal (≥3°): {marginal} ({100*marginal/len(optimal_runways):.1f}%)")
    
    # Top 5 runways
    print(f"\nTop 5 Runways (by gradient quality):")
    sorted_runways = sorted(optimal_runways, key=lambda r: r['gradient_metrics']['max_abs_gradient'])
    for i, runway in enumerate(sorted_runways[:5], 1):
        print(f"  {i}. Contour {runway['contour_id']}: "
              f"length={runway['length']:.1f}m, "
              f"max_grad={runway['gradient_metrics']['max_abs_gradient']:.2f}°, "
              f"mean_grad={runway['gradient_metrics']['mean_abs_gradient']:.2f}°")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def find_optimal_runways(region='norcoast8',
                         runway_length_min=400,
                         runway_width_min=15,
                         min_area=6000,
                         erosion_distance=200,
                         optimization_method='Nelder-Mead',
                         max_iter=500):
    """
    Main pipeline: find optimal runway for each viable contour.
    
    Parameters:
    -----------
    region : str
        Region identifier
    runway_length_min : float
        Minimum runway length for filtering (meters)
    runway_width_min : float
        Minimum runway width for filtering (meters)
    min_area : float
        Minimum contour area (m²)
    erosion_distance : float
        Erosion distance for feasible center points (meters)
    optimization_method : str
        Optimization method: 'Nelder-Mead', 'Powell', 'differential_evolution'
    max_iter : int
        Maximum iterations for optimizer
    
    Returns:
    --------
    dict with results
    """
    print("=" * 60)
    print(f"RUNWAY OPTIMIZATION FOR REGION: {region}")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Minimum runway dimensions: {runway_length_min}m × {runway_width_min}m")
    print(f"  Minimum contour area: {min_area} m²")
    print(f"  Erosion distance: {erosion_distance} m")
    print(f"  Optimization method: {optimization_method}")
    print(f"  Max iterations: {max_iter}")
    
    # ========================================================================
    # Load filtered landable areas
    # ========================================================================
    filtered_path = f'results/{region}_filtered_landable_areas.tif'
    
    if not os.path.exists(filtered_path):
        print(f"\nERROR: {filtered_path} not found!")
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
    
    # ========================================================================
    # Apply min rectangle filter
    # ========================================================================
    print(f"\nApplying min rectangle filter...")
    landable_filtered = filter_by_min_rectangle(
        landable_map,
        min_width=runway_width_min / pixel_size,
        min_height=runway_length_min / pixel_size
    )
    
    num_regions_before = cv2.connectedComponents(landable_map.astype(np.uint8))[0] - 1
    num_regions_after = cv2.connectedComponents(landable_filtered.astype(np.uint8))[0] - 1
    print(f"Regions before filter: {num_regions_before}")
    print(f"Regions after filter: {num_regions_after}")
    
    landable_map = landable_filtered
    
    # ========================================================================
    # Load DEM
    # ========================================================================
    dem_path = f'dem_maps/{region}_dem_utm.tif'
    if not os.path.exists(dem_path):
        # Try alternative naming
        dem_path = f'dem_maps/{region.replace("norcoast8", "norcoast_b23")}_dem_utm.tif'
        if not os.path.exists(dem_path):
            print(f"\nERROR: DEM file not found at {dem_path}")
            return None
    
    with rasterio.open(dem_path) as src:
        elevation = src.read(1, masked=True)
        dem_transform = src.transform
    
    print(f"\nLoaded DEM: {elevation.shape}")
    print(f"Elevation range: {elevation.min():.1f} to {elevation.max():.1f} meters")
    
    # Verify transforms match
    if transform != dem_transform:
        print("\nWARNING: Landable map and DEM transforms don't match!")
        print(f"Landable: {transform}")
        print(f"DEM: {dem_transform}")
    
    # ========================================================================
    # Process contours
    # ========================================================================
    contour_data_list = process_contours(
        landable_map,
        transform,
        min_area=min_area,
        erosion_distance=erosion_distance
    )
    
    if len(contour_data_list) == 0:
        print("\nNo viable contours found after filtering!")
        return None
    
    # ========================================================================
    # Optimize runway for each contour
    # ========================================================================
    print("\n" + "=" * 60)
    print("OPTIMIZING RUNWAYS")
    print("=" * 60)
    
    optimal_runways = []
    
    for contour_data in contour_data_list:
        result = optimize_runway_for_contour(
            contour_data,
            elevation,
            transform,
            method=optimization_method,
            max_iter=max_iter
        )
        
        if result is not None:
            optimal_runways.append(result)
    
    print(f"\n" + "=" * 60)
    print(f"OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nSuccessfully optimized {len(optimal_runways)}/{len(contour_data_list)} contours")
    
    if len(optimal_runways) == 0:
        print("No valid runways found!")
        return None
    
    # ========================================================================
    # Create visualizations
    # ========================================================================
    visualize_results(region, landable_map, elevation, transform,
                     contour_data_list, optimal_runways)
    
    # ========================================================================
    # Save results
    # ========================================================================
    save_results_geotiff(region, optimal_runways, landable_map, transform, profile)
    
    # ========================================================================
    # Print statistics
    # ========================================================================
    print_summary_statistics(optimal_runways)
    
    return {
        'optimal_runways': optimal_runways,
        'contour_data': contour_data_list,
        'landable_map': landable_map,
        'elevation': elevation,
        'transform': transform,
        'region': region
    }


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == "__main__":
    """
    MAIN RUNNER - Execute runway optimization analysis
    """
    
    print("\n" + "=" * 60)
    print("RUNWAY OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    region = 'norcoast8'              # Region identifier
    runway_length_min = 400           # Minimum runway length (m)
    runway_width_min = 15             # Minimum runway width (m) - for filtering only
    min_area = 6000                   # Minimum contour area (m²)
    erosion_distance = 0            # Half runway length for center feasibility (m)
    
    # Optimization settings
    optimization_method = 'Nelder-Mead'  # Options: 'Nelder-Mead', 'Powell', 'differential_evolution'
    max_iter = 500                       # Maximum optimizer iterations
    
    # ========================================================================
    # RUN OPTIMIZATION
    # ========================================================================
    
    results = find_optimal_runways(
        region=region,
        runway_length_min=runway_length_min,
        runway_width_min=runway_width_min,
        min_area=min_area,
        erosion_distance=erosion_distance,
        optimization_method=optimization_method,
        max_iter=max_iter
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    if results is not None:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\nOutputs saved to results/ folder:")
        print(f"  1. {region}_runway_optimization.png")
        print(f"     → 4-panel visualization with optimal runways")
        print(f"  2. {region}_runway_gradient_map.png")
        print(f"     → Elevation map with runways colored by gradient")
        print(f"  3. {region}_optimal_runways.tif")
        print(f"     → Georeferenced raster of all optimal runways")
        
        # Best runway info
        best_runway = min(results['optimal_runways'], 
                         key=lambda r: r['gradient_metrics']['max_abs_gradient'])
        
        print(f"\nOverall best runway:")
        print(f"  Contour ID: {best_runway['contour_id']}")
        print(f"  Length: {best_runway['length']:.1f}m")
        print(f"  Orientation: {best_runway['theta_deg']:.1f}°")
        print(f"  Max gradient: {best_runway['gradient_metrics']['max_abs_gradient']:.2f}°")
        print(f"  Mean gradient: {best_runway['gradient_metrics']['mean_abs_gradient']:.2f}°")
        print(f"  Elevation change: {best_runway['gradient_metrics']['elevation_change']:.1f}m")
        
        print(f"\nOptimization method: {optimization_method}")
        print(f"Average function evaluations: {np.mean([r['eval_count'] for r in results['optimal_runways']]):.1f}")
        
    else:
        print("\n" + "=" * 60)
        print("ANALYSIS FAILED")
        print("=" * 60)
        print("\nPossible reasons:")
        print("  - Filtered landable areas TIF file not found")
        print("  - DEM file not found")
        print("  - No viable contours after filtering")
        print("\nPlease check:")
        print(f"  1. results/{region}_filtered_landable_areas.tif exists")
        print(f"  2. dem_maps/{region}_dem_utm.tif exists")
        print("  3. Run overlay_osm_on_binary_map.py first if needed")