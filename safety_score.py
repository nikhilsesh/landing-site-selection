import os
import rasterio
from rasterio import path
from rasterio.warp import calculate_default_transform, reproject, Resampling # NEW IMPORT

import numpy as np
# import cv2
from loguru import logger
import matplotlib.pyplot as plt

def compute_strike_dip_projected(elevation, transform):
    """
    Compute strike, dip, and dip direction from a DEM in a PROJECTED CRS.
    Assumes the CRS uses meters and the DEM is north-up.
    """
    # Ensure we're working with a masked array
    if not isinstance(elevation, np.ma.MaskedArray):
        z = np.ma.asarray(elevation)
    else:
        z = elevation

    # Get pixel size in meters (for projected CRS)
    dx = abs(transform.a)  # meters per pixel in x
    dy = abs(transform.e)  # meters per pixel in y

    # Fill masked values for gradient computation
    fill_value = np.ma.median(z)
    z_filled = z.filled(fill_value)
    
    # Gradient: returns [dz/dy, dz/dx] in units of elevation/meter
    grad_y, grad_x = np.gradient(z_filled, dy, dx)
    
    # Re-mask if needed
    if np.ma.is_masked(z):
        from scipy.ndimage import binary_dilation
        expanded_mask = binary_dilation(z.mask, structure=np.ones((3,3)))
        grad_y = np.ma.masked_where(expanded_mask, grad_y)
        grad_x = np.ma.masked_where(expanded_mask, grad_x)

    # Grdients in elevation per meter, so they represent the slope in each direction
    dz_dx_east = grad_x
    dz_dy_north = -grad_y  # Negative because y-axis points south
    
    # Dip direction (azimuth of steepest descent)
    downhill_east = -dz_dx_east
    downhill_north = -dz_dy_north
    dip_dir_deg = (np.degrees(np.ma.arctan2(downhill_east, downhill_north)) + 360.0) % 360.0

    # Strike is perpendicular to dip direction
    strike_deg = (dip_dir_deg + 90.0) % 360.0

    # Dip (slope angle from horizontal)
    slope_magnitude = np.ma.hypot(dz_dx_east, dz_dy_north)
    dip_deg = np.degrees(np.ma.arctan(slope_magnitude))

    return strike_deg, dip_deg, dip_dir_deg

def compute_variance(elevation, window_size=3, min_valid_fraction=0.5, pad_mode="reflect"):
    """
    Compute local elevation variance over a square moving window. Returns an array of shape
    equal to elevation with per-pixel variance.

    - window_size: odd integer >= 1 defining the side length of the window
    - min_valid_fraction: required fraction of finite pixels in the window to emit a value
      (otherwise the output is masked). Set to 0.0 to always compute if any valid pixels exist.
    - pad_mode: numpy pad mode for borders (e.g., "reflect", "edge", "constant")

    Returns a masked array if input is masked, otherwise returns regular array.
    
    UPDATED: Now handles masked arrays from reprojected DEMs in projected CRS (UTM).
    """
    is_masked_input = np.ma.is_masked(elevation)
    
    # Handle masked arrays properly (from rasterio with masked=True)
    if is_masked_input:
        z = np.ma.filled(elevation, np.nan)  # Replace masked values with NaN
    else:
        z = np.asarray(elevation)

    if window_size < 1 or (window_size % 2) == 0:
        raise ValueError("window_size must be an odd integer >= 1")

    # Convert to float64 and identify valid pixels
    z = z.astype(np.float64, copy=False)
    valid_mask = np.isfinite(z)
    z = np.where(valid_mask, z, 0.0)

    pad = window_size // 2

    def _box_sum(arr):
        arr_pad = np.pad(arr, ((pad, pad), (pad, pad)), mode=pad_mode)
        # Integral image with leading zero row/col for simpler region sums
        S = np.pad(arr_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0).cumsum(0).cumsum(1)
        # Vectorized window sum for each top-left position over the padded array
        return S[window_size:, window_size:] - S[:-window_size, window_size:] - S[window_size:, :-window_size] + S[:-window_size, :-window_size]

    sum_x  = _box_sum(z)
    sum_x2 = _box_sum(z*z)
    count  = _box_sum(valid_mask.astype(np.float64))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_x / count
        mean_sq = sum_x2 / count
        var = mean_sq - (mean * mean)

    # clamp numerical imprecision to zero
    var = np.where(var < 0.0, 0.0, var)
    valid_fraction = count / float(window_size * window_size)
    
    # Create output
    var_result = var.astype(np.float32)
    
    # Return masked array if input was masked
    if is_masked_input:
        output_mask = valid_fraction < float(min_valid_fraction)
        return np.ma.masked_array(var_result, mask=output_mask)
    else:
        var_result = np.where(valid_fraction >= float(min_valid_fraction), var_result, np.nan)
        return var_result
    
# Function to compute terrain ruggedness index (TRI)
def compute_TRI(elevation, window_size=3, min_valid_fraction=0.5, pad_mode="reflect"):
    """
    Compute Terrain Ruggedness Index (TRI).
    
    TRI is the mean absolute elevation difference between each cell and its neighbors.
    For window_size=3, uses the 8 immediate neighbors (classic TRI).
    For larger windows, uses all neighbors in the window.
    
    Parameters:
    -----------
    elevation : array
        Input elevation data (can be masked array)
    window_size : int
        Size of neighborhood window (default 3 for classic 8-neighbor TRI)
    min_valid_fraction : float
        Minimum fraction of valid neighbors required
    pad_mode : str
        Padding mode for edges
        
    Returns:
    --------
    tri : array
        Terrain Ruggedness Index values (mean absolute difference from center)
    """
    is_masked_input = np.ma.is_masked(elevation)
    
    # Handle masked arrays properly (from rasterio with masked=True)
    if is_masked_input:
        z = np.ma.filled(elevation, np.nan)  # Replace masked values with NaN
    else:
        z = np.asarray(elevation)

    if window_size < 1 or (window_size % 2) == 0:
        raise ValueError("window_size must be an odd integer >= 1")

    # Convert to float64 and identify valid pixels
    z = z.astype(np.float64, copy=False)
    valid_mask = np.isfinite(z)
    z_filled = np.where(valid_mask, z, 0.0)
    
    pad = window_size // 2
    
    # Pad the arrays
    z_padded = np.pad(z_filled, ((pad, pad), (pad, pad)), mode=pad_mode)
    valid_padded = np.pad(valid_mask, ((pad, pad), (pad, pad)), mode=pad_mode)
    
    # Initialize output arrays
    tri = np.zeros_like(z, dtype=np.float64)
    neighbor_count = np.zeros_like(z, dtype=np.float64)
    
    # Iterate over all offsets in the window (excluding center)
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            if di == 0 and dj == 0:
                continue  # Skip the center pixel itself
            
            # Extract the shifted neighbor array
            neighbor = z_padded[pad + di : pad + di + z.shape[0], 
                               pad + dj : pad + dj + z.shape[1]]
            neighbor_valid = valid_padded[pad + di : pad + di + z.shape[0],
                                         pad + dj : pad + dj + z.shape[1]]
            
            # Compute absolute difference where both center and neighbor are valid
            both_valid = valid_mask & neighbor_valid
            tri += np.where(both_valid, np.abs(z_filled - neighbor), 0.0)
            neighbor_count += both_valid.astype(np.float64)
    
    # Compute mean absolute difference
    with np.errstate(invalid="ignore", divide="ignore"):
        tri = tri / neighbor_count
    
    # Check if we have enough valid neighbors
    max_neighbors = window_size * window_size - 1  # Exclude center pixel
    valid_fraction = neighbor_count / max_neighbors
    
    # Create output
    tri_result = tri.astype(np.float32)
    
    # Return masked array if input was masked
    if is_masked_input:
        output_mask = valid_fraction < float(min_valid_fraction)
        return np.ma.masked_array(tri_result, mask=output_mask)
    else:
        tri_result = np.where(valid_fraction >= float(min_valid_fraction), tri_result, np.nan)
        return tri_result

def compute_safety_score(dip_deg, var, a=5, b=1):
    """
    Compute safety score from dip and variance.
    TODO: Weights to be tuned.
    """
    print('compute safety score entered')
    return  - (a*dip_deg + b*var)

def reproject_dem_to_utm(input_path, output_path=None):
    """
    Automatically reproject a DEM to the appropriate UTM zone.
    
    Parameters:
    -----------
    input_path : str
        Path to input DEM file
    output_path : str, optional
        Path for output reprojected DEM. If None, adds '_utm' to input filename.
    
    Returns:
    --------
    output_path : str
        Path to the reprojected DEM file
    """
    if output_path is None:
        output_path = input_path.replace('.tif', '_utm.tif')
    
    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"Reprojected DEM already exists at: {output_path}")
        return output_path
    
    with rasterio.open(input_path) as src:
        print(f"Original CRS: {src.crs}")
        print(f"Original bounds: {src.bounds}")
        print(f"Original pixel size: {src.transform.a} x {src.transform.e}")
        
        # Automatically estimate the best UTM zone
        # dst_crs = src.estimate_utm_crs()
        # print(f"\nEstimated UTM CRS: {dst_crs}")
        dst_crs = 'EPSG:32610'  # UTM zone 10N for California, replace with automatic if needed
        
        # Calculate the transform for the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        # Update metadata for output file
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        print(f"New pixel size (meters): {transform.a:.2f} x {abs(transform.e):.2f}")
        
        # Create the reprojected file
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
        
        print(f"\nReprojected DEM saved to: {output_path}")
    
    return output_path

def compute_safety_map(region, dem_path):
    logger.info("=" * 60)
    logger.info("Computing safety score from DEM data")
    logger.info("=" * 60)

    # Usage example
    projected_dem_path = reproject_dem_to_utm(dem_path)

    # Now read and process the projected DEM
    with rasterio.open(projected_dem_path) as src:
        elevation = src.read(1, masked=True)
        transform = src.transform
        
        print(f"\nProcessing projected DEM:")
        print(f"CRS: {src.crs}")
        print(f"Pixel size: {transform.a:.2f} x {abs(transform.e):.2f} meters")
        print(f"Elevation range: {elevation.min():.1f} to {elevation.max():.1f} meters")
    
        # Compute slope with the corrected function
        strike, dip_deg, dip_dir = compute_strike_dip_projected(elevation, transform) 
        print(f"\nSlope (dip) statistics:")
        print(f"  Min: {dip_deg.min():.2f}°")
        print(f"  Max: {dip_deg.max():.2f}°")
        print(f"  Mean: {dip_deg.mean():.2f}°")

        # Compute variance with the corrected function
        var = compute_variance(elevation, window_size=31)
        print(f"\nVariance statistics:")
        print(f"  Min: {np.ma.min(var):.4f}")
        print(f"  Max: {np.ma.max(var):.4f}")
        print(f"  Mean: {np.ma.mean(var):.4f}")
        print(f"  Median: {np.ma.median(var):.4f}")
        print(f"  Data type: {var.dtype}")

        # Compute TRI and output statistics
        tri = compute_TRI(elevation)
        print(f"\nTRI statistics:")
        print(f"  Min: {np.ma.min(tri):.4f}")
        print(f"  Max: {np.ma.max(tri):.4f}")
        print(f"  Mean: {np.ma.mean(tri):.4f}")
        
        safety_score = compute_safety_score(dip_deg, var)

    # Generate plots of dip, variance, TRI, and safety score and save to results folder
    # Dip
    plt.figure(figsize=(10, 6))
    plt.imshow(dip_deg, cmap="viridis")
    plt.colorbar(label="Dip (degrees)")
    plt.title("Dip (slope angle)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"results/{region}_dip.png", dpi=300, bbox_inches="tight")

    # Variance
    plt.figure(figsize=(10, 6))
    plt.imshow(var, cmap="viridis")
    plt.colorbar(label="Variance (m^2)")
    plt.title("Variance")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"results/{region}_var.png", dpi=300, bbox_inches="tight")

    # TRI
    plt.figure(figsize=(10, 6))
    plt.imshow(tri, cmap="viridis")
    plt.colorbar(label="TRI (mean absolute elevation difference)")
    plt.title("Terrain Ruggedness Index (TRI)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"results/{region}_tri.png", dpi=300, bbox_inches="tight")

    # Safety Score
    plt.figure(figsize=(10, 6))
    plt.imshow(safety_score, cmap="viridis")
    plt.colorbar(label="Safety Score")
    plt.title("Safety Score")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"results/{region}_safety_score.png", dpi=300, bbox_inches="tight")
    
    return safety_score

def compute_binary_safety_map(safety_score, threshold=-20, dem_path=None, region='norcoast5'):
    binary_map = (safety_score > threshold).astype(np.uint8) * 255

    # Generate plot of binary map and save to results folder
    plt.figure(figsize=(10, 6))
    plt.imshow(binary_map, cmap="gray")
    plt.title("Binary Safety Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"results/{region}_binary_safety_map.png", dpi=300, bbox_inches="tight")
    
    # NEW: Save as GeoTIFF with proper georeferencing
    if dem_path:
        utm_dem_path = dem_path.replace('.tif', '_utm.tif')
        with rasterio.open(utm_dem_path) as src:
            profile = src.profile.copy()
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=0  # Set nodata to 0 for uint8 (or use None if no nodata needed)
            )
            
            output_tif = f"osm_data/{region}_binary_safety_map.tif"
            with rasterio.open(output_tif, 'w', **profile) as dst:
                dst.write(binary_map.astype(np.uint8), 1)
            
            print(f"Saved georeferenced binary map to: {output_tif}")
    
    return binary_map

region = 'norcoast8'
dem_path = f'dem_maps/norcoast_b23_dem.tif'
output_path = f'results/{region}_safety_score_norm.png'
print('initialized variables')
safety_score = compute_safety_map(region, dem_path)
binary_safety_map = compute_binary_safety_map(safety_score, threshold=-20, dem_path=dem_path, region=region)  # Pass dem_path

# if __name__ == "__main__":
    # dem_path = 'dem_maps/davis_dem.tif'
    # dem_path = 'dem_maps/napa_dem.tif'
    # output_path = 'results/davis_safety_score.png'
    # compute_safety_map(dem_path, output_path)