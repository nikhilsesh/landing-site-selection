"""
get_satellite_imagery.py

Downloads satellite imagery matching DEM coordinate extent.
Uses contextily to fetch tiles and saves as PNG.
Also includes function to create contour maps from DEMs.
"""

import rasterio
from rasterio.warp import transform_bounds
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from PIL import Image

def download_satellite_imagery(region, dem_path, zoom='auto', source='Esri.WorldImagery', 
                                output_folder='satellite_imagery', dpi=100, 
                                png_compression=6):
    """
    Download satellite imagery matching DEM extent and save as PNG.
    
    Parameters:
    -----------
    region : str
        Region identifier
    dem_path : str
        Path to DEM file
    zoom : int or 'auto'
        Zoom level for satellite tiles. 'auto' will choose based on DEM resolution.
        Higher zoom = more detail but larger file size.
        Typical values: 13-14 (low res), 15-16 (medium), 17-18 (high)
    source : str
        Tile source. Options:
        - 'Esri.WorldImagery' (default, high quality)
        - 'Esri.WorldTopoMap' (topographic with satellite)
        - 'OpenTopoMap' (topographic only)
    output_folder : str
        Folder to save satellite imagery
    dpi : int
        Resolution for PNG output. Lower = smaller file size.
        Typical values: 72 (low), 100 (medium), 150 (high), 300 (very high)
    png_compression : int
        PNG compression level (0-9). Higher = smaller file but slower.
        6 is a good balance. Use 9 for smallest files.
    
    Returns:
    --------
    output_path : str
        Path to saved PNG file
    """
    
    print("=" * 60)
    print(f"DOWNLOADING SATELLITE IMAGERY FOR: {region}")
    print("=" * 60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Read DEM to get bounds and CRS
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        crs = src.crs
        dem_shape = src.shape
        dem_transform = src.transform
        dem_res = abs(src.transform.a)  # pixel size in meters (if projected)
        
        print(f"\nDEM Information:")
        print(f"  CRS: {crs}")
        print(f"  Bounds: {bounds}")
        print(f"  Shape: {dem_shape}")
        print(f"  Resolution: {dem_res:.2f} meters/pixel")
        
        # Check if we need to reproject to UTM for contextily
        if crs.to_epsg() not in [None] and 'UTM' not in crs.to_string().upper():
            # DEM is in geographic coordinates, need to get UTM bounds
            print(f"\nDEM is in geographic coordinates, converting to Web Mercator for contextily...")
            bounds_webmerc = transform_bounds(crs, 'EPSG:3857', *bounds)
            use_crs = 'EPSG:3857'
        elif crs.is_projected:
            # Already projected (e.g., UTM)
            bounds_webmerc = bounds
            use_crs = crs.to_string()
        else:
            # Fallback
            bounds_webmerc = transform_bounds(crs, 'EPSG:3857', *bounds)
            use_crs = 'EPSG:3857'
    
    print(f"\nFetching satellite imagery...")
    print(f"  Source: {source}")
    print(f"  Zoom level: {zoom}")
    print(f"  Target CRS: {use_crs}")
    print(f"  Output DPI: {dpi}")
    print(f"  PNG compression: {png_compression}")
    
    # Calculate appropriate zoom level if auto
    if zoom == 'auto':
        # Estimate based on DEM resolution
        # Reduced by 1-2 levels for smaller file sizes
        # Typical satellite imagery at zoom 17 ≈ 1.2m/pixel
        # zoom 18 ≈ 0.6m/pixel, zoom 16 ≈ 2.4m/pixel
        if dem_res < 1:
            zoom = 16  # Reduced from 18
        elif dem_res < 3:
            zoom = 15  # Reduced from 17
        elif dem_res < 10:
            zoom = 14  # Reduced from 16
        else:
            zoom = 13  # Reduced from 15
        print(f"  Auto-selected zoom: {zoom} (optimized for smaller file size)")
    
    # Fetch satellite imagery using contextily
    try:
        # Create a figure with the correct extent
        # Reduced figure size for smaller output
        fig, ax = plt.subplots(figsize=(12, 12))  # Reduced from (20, 20)
        ax.set_xlim(bounds_webmerc[0], bounds_webmerc[2])
        ax.set_ylim(bounds_webmerc[1], bounds_webmerc[3])
        
        # Remove axes for cleaner image
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])  # Make plot fill entire figure
        
        # Add basemap
        print("\nDownloading tiles (this may take 30-60 seconds)...")
        ctx.add_basemap(
            ax,
            crs=use_crs,
            source=source,
            zoom=zoom,
            attribution=False
        )
        
        # Get the image from the basemap
        if len(ax.images) > 0:
            satellite_image = ax.images[0].get_array()
            extent_fetched = ax.images[0].get_extent()
            
            print(f"\n✓ Satellite imagery downloaded")
            print(f"  Image shape: {satellite_image.shape}")
            print(f"  Extent: {extent_fetched}")
            
            # Save as PNG
            output_png = f'{output_folder}/{region}_satellite.png'
            
            # Save with tight layout and no padding
            plt.savefig(
                output_png,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0,
                format='png',
                pil_kwargs={'compress_level': png_compression}
            )
            
            plt.close(fig)
            
            # Get file size
            file_size_mb = os.path.getsize(output_png) / (1024 * 1024)
            
            print(f"\n✓ Saved satellite PNG to: {output_png}")
            print(f"  File size: {file_size_mb:.2f} MB")
            
            # Optional: Further compress using PIL if needed
            if file_size_mb > 10:  # If larger than 10MB, suggest optimization
                print(f"\n  Note: File is relatively large ({file_size_mb:.2f} MB)")
                print(f"  Consider:")
                print(f"    - Reducing zoom level (current: {zoom})")
                print(f"    - Reducing dpi (current: {dpi})")
                print(f"    - Increasing png_compression (current: {png_compression})")
            
            return output_png
            
        else:
            print("\n✗ Error: Could not fetch satellite imagery")
            plt.close(fig)
            return None
            
    except Exception as e:
        print(f"\n✗ Error downloading satellite imagery: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Try a different zoom level (lower = faster)")
        print("  3. Try a different source (e.g., 'OpenTopoMap')")
        return None

def create_contour_map(region, dem_path, output_folder='dem_maps', 
                       contour_interval='auto', num_contours=20,
                       colormap='terrain', dpi=150, 
                       figsize=(12, 10), show_labels=True,
                       png_compression=6, downsample_factor=1):
    """
    Create a contour map from DEM and save as PNG.
    
    Parameters:
    -----------
    region : str
        Region identifier
    dem_path : str
        Path to DEM file
    output_folder : str
        Folder to save contour map (default: 'dem_maps')
    contour_interval : float or 'auto'
        Elevation interval between contours in meters.
        'auto' will choose based on elevation range.
    num_contours : int
        Number of contour lines if contour_interval is 'auto' (default: 20)
    colormap : str
        Matplotlib colormap for elevation coloring.
        Options: 'terrain', 'viridis', 'plasma', 'gist_earth', 'rainbow'
    dpi : int
        Resolution for PNG output (default: 150)
    figsize : tuple
        Figure size in inches (width, height)
    show_labels : bool
        Whether to show elevation labels on contours
    png_compression : int
        PNG compression level (0-9)
    downsample_factor : int
        Factor to downsample DEM for faster processing (1=no downsampling, 2=half resolution, etc.)
        Higher values = faster but less detail. Recommended: 1-4
    
    Returns:
    --------
    output_path : str
        Path to saved contour map PNG
    """
    
    print("=" * 60)
    print(f"CREATING CONTOUR MAP FOR: {region}")
    print("=" * 60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Read DEM with proper masking to handle nodata values
    with rasterio.open(dem_path) as src:
        # Use masked=True to properly handle nodata values (like -999999)
        dem_data = src.read(1, masked=True)
        bounds = src.bounds
        crs = src.crs
        transform = src.transform
        
        # Downsample if requested for speed
        if downsample_factor > 1:
            print(f"\nDownsampling DEM by factor of {downsample_factor} for faster processing...")
            dem_data = dem_data[::downsample_factor, ::downsample_factor]
            print(f"  Original shape: {src.shape}")
            print(f"  Downsampled shape: {dem_data.shape}")
        
        # Get elevation statistics (using masked array methods)
        min_elev = dem_data.min()
        max_elev = dem_data.max()
        elev_range = max_elev - min_elev
        
        print(f"\nDEM Information:")
        print(f"  CRS: {crs}")
        print(f"  Shape: {dem_data.shape}")
        print(f"  Elevation range: {min_elev:.1f} to {max_elev:.1f} meters")
        print(f"  Elevation span: {elev_range:.1f} meters")
        
        # Convert bounds to lat/lon if needed
        if crs.is_projected:
            # Need to convert from projected coordinates to lat/lon
            from rasterio.warp import transform as rio_transform
            
            # Transform corners to lat/lon
            lons, lats = rio_transform(crs, 'EPSG:4326', 
                                       [bounds.left, bounds.right], 
                                       [bounds.bottom, bounds.top])
            lon_min, lon_max = min(lons), max(lons)
            lat_min, lat_max = min(lats), max(lats)
            
            print(f"  Geographic bounds:")
            print(f"    Longitude: {lon_min:.6f}° to {lon_max:.6f}°")
            print(f"    Latitude: {lat_min:.6f}° to {lat_max:.6f}°")
        else:
            # Already in lat/lon
            lon_min, lon_max = bounds.left, bounds.right
            lat_min, lat_max = bounds.bottom, bounds.top
        
        # Create coordinate arrays in lat/lon
        rows, cols = dem_data.shape
        lons = np.linspace(lon_min, lon_max, cols)
        lats = np.linspace(lat_max, lat_min, rows)
        X, Y = np.meshgrid(lons, lats)
        
        # Determine contour interval
        if contour_interval == 'auto':
            # Choose nice round numbers based on elevation range
            if elev_range < 50:
                contour_interval = 5
            elif elev_range < 100:
                contour_interval = 10
            elif elev_range < 250:
                contour_interval = 20
            elif elev_range < 500:
                contour_interval = 50
            elif elev_range < 1000:
                contour_interval = 100
            else:
                contour_interval = 200
            
            print(f"  Auto-selected contour interval: {contour_interval} meters")
        else:
            print(f"  Contour interval: {contour_interval} meters")
        
        # Generate contour levels
        contour_levels = np.arange(
            np.floor(min_elev / contour_interval) * contour_interval,
            np.ceil(max_elev / contour_interval) * contour_interval + contour_interval,
            contour_interval
        )
        
        print(f"  Number of contour lines: {len(contour_levels)}")
        print(f"\nGenerating contour map...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reduce number of filled contour levels for speed (30 is plenty)
    num_filled_levels = min(30, len(contour_levels) * 2)
    
    # Create filled contours (elevation shading) - fewer levels for speed
    # The masked array will automatically handle nodata regions
    print("  Drawing filled contours...")
    contourf = ax.contourf(X, Y, dem_data, levels=num_filled_levels, 
                           cmap=colormap, alpha=0.8)
    
    # Create contour lines - only draw every other line if there are too many
    contour_step = 1 if len(contour_levels) < 30 else 2
    contour_levels_to_draw = contour_levels[::contour_step]
    
    print("  Drawing contour lines...")
    contours = ax.contour(X, Y, dem_data, levels=contour_levels_to_draw, 
                          colors='black', linewidths=0.5, alpha=0.6)
    
    # Add contour labels - fewer labels for speed
    if show_labels:
        print("  Adding contour labels...")
        # Only label every 2nd or 3rd contour line for clarity
        label_step = 2 if len(contour_levels_to_draw) < 20 else 3
        ax.clabel(contours, inline=True, fontsize=8, fmt='%d m',
                 levels=contour_levels_to_draw[::label_step])
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Elevation (m)', shrink=0.8)
    
    # Set labels and title with degree symbols
    ax.set_xlabel('Longitude (°)', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title(f'Contour Map - {region}\nElevation: {min_elev:.0f} to {max_elev:.0f} m', 
                 fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Format tick labels to show degrees with proper precision
    ax.ticklabel_format(useOffset=False, style='plain')
    
    # Set aspect ratio to be proportional to latitude
    # (accounts for longitude compression at higher latitudes)
    avg_lat = (lat_min + lat_max) / 2
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))
    ax.set_aspect(aspect_ratio, adjustable='box')
    
    # Save as PNG
    output_png = f'{output_folder}/{region}_contour_map.png'
    
    print("  Saving PNG...")
    plt.savefig(
        output_png,
        dpi=dpi,
        bbox_inches='tight',
        format='png',
        pil_kwargs={'compress_level': png_compression}
    )
    
    plt.close(fig)
    
    # Get file size
    file_size_mb = os.path.getsize(output_png) / (1024 * 1024)
    
    print(f"\n✓ Saved contour map PNG to: {output_png}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return output_png

if __name__ == "__main__":
    # Configuration
    # region = 'norcoast_b23'
    region = 'alameda_b21_x59y418'
    dem_path = f'dem_maps/{region}_dem.tif'
    
    # Option 1: Download satellite imagery
    # Adjust these parameters to control file size:
    # - zoom: 13-14 (small), 15-16 (medium), 17-18 (large)
    # - dpi: 72 (small), 100 (medium), 150 (large)
    # - png_compression: 6 (balanced), 9 (smallest)
    
    # satellite_path = download_satellite_imagery(
    #     region=region,
    #     dem_path=dem_path,
    #     zoom='auto',  # or specify 13, 14, 15, 16, 17, 18
    #     source='Esri.WorldImagery',
    #     output_folder='satellite_imagery',
    #     dpi=100,  # Adjust: 72 (low), 100 (medium), 150 (high)
    #     png_compression=6  # Adjust: 6 (fast), 9 (smallest file)
    # )
    
    # if satellite_path:
    #     print("\n" + "=" * 60)
    #     print("SATELLITE IMAGERY DOWNLOAD COMPLETE!")
    #     print("=" * 60)
    #     print(f"\nSaved to: {satellite_path}")
    #     print("\nYou can now use this PNG in presentations or reports.")
    # else:
    #     print("\n" + "=" * 60)
    #     print("DOWNLOAD FAILED")
    #     print("=" * 60)
    
    # Option 2: Create contour map
    print("\n")
    contour_path = create_contour_map(
        region=region,
        dem_path=dem_path,
        output_folder='dem_maps',
        contour_interval='auto',  # or specify elevation interval in meters (e.g., 10, 20, 50)
        num_contours=20,
        colormap='terrain',  # Options: 'terrain', 'viridis', 'plasma', 'gist_earth', 'rainbow'
        dpi=150,
        figsize=(12, 10),
        show_labels=True,
        png_compression=6,
        downsample_factor=2  # Set to 2-4 for faster processing, 1 for full resolution
    )
    
    if contour_path:
        print("\n" + "=" * 60)
        print("CONTOUR MAP CREATION COMPLETE!")
        print("=" * 60)
        print(f"\nSaved to: {contour_path}")
        print("\nYou can now use this contour map for terrain analysis.")
    else:
        print("\n" + "=" * 60)
        print("CONTOUR MAP CREATION FAILED")
        print("=" * 60)