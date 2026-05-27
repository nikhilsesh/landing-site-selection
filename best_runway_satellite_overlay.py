"""
best_runway_satellite_overlay.py

Creates a close-up plot of the best runway overlaid on satellite imagery.
Fetches only the necessary satellite tiles for the runway area.
"""

import numpy as np
import rasterio
from rasterio.transform import xy
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Rectangle
import os


def plot_runway_on_satellite_ax(ax, runway, transform, crs, padding_meters=500, 
                                 zoom_level='auto', satellite_source='Esri.WorldImagery',
                                 show_labels=True):
    """
    Plot a runway on satellite imagery in a provided matplotlib axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    runway : dict
        Runway dictionary with 'p1', 'p2', 'length' keys (pixel coordinates)
    transform : rasterio.Affine
        Affine transform to convert pixel to geographic coordinates
    crs : rasterio.crs.CRS
        Coordinate reference system
    padding_meters : float
        Padding around runway in meters
    zoom_level : int or 'auto'
        Zoom level for satellite tiles
    satellite_source : str
        Tile source name
    show_labels : bool
        Whether to show axis labels and title
    
    Returns:
    --------
    ax : matplotlib axis
        Modified axis with satellite imagery and runway
    """
    
    # Get runway endpoints in geographic coordinates
    p1_geo = transform * (runway['p1'][0], runway['p1'][1])
    p2_geo = transform * (runway['p2'][0], runway['p2'][1])
    
    runway_length = runway['length']
    
    # Calculate extent with padding
    min_x = min(p1_geo[0], p2_geo[0]) - padding_meters
    max_x = max(p1_geo[0], p2_geo[0]) + padding_meters
    min_y = min(p1_geo[1], p2_geo[1]) - padding_meters
    max_y = max(p1_geo[1], p2_geo[1]) + padding_meters
    
    extent_utm = [min_x, max_x, min_y, max_y]
    
    # Set up the plot extent
    ax.set_xlim(extent_utm[0], extent_utm[1])
    ax.set_ylim(extent_utm[2], extent_utm[3])
    
    # Convert CRS for contextily
    crs_string = f"EPSG:{crs.to_epsg()}"
    
    # Fetch and add satellite imagery
    try:
        ctx.add_basemap(
            ax,
            crs=crs_string,
            source=satellite_source,
            zoom=zoom_level,
            attribution=False
        )
    except Exception as e:
        print(f"Warning: Could not fetch satellite imagery: {e}")
        ax.set_facecolor('lightgray')
    
    # Draw the runway line
    ax.plot([p1_geo[0], p2_geo[0]], [p1_geo[1], p2_geo[1]], 
           'r-', linewidth=4, label=f'Runway ({runway_length:.0f}m)', zorder=10)
    
    # Mark endpoints
    ax.plot(p1_geo[0], p1_geo[1], 'yo', markersize=10, 
           markeredgecolor='black', markeredgewidth=2, label='Start', zorder=11)
    ax.plot(p2_geo[0], p2_geo[1], 'ys', markersize=10, 
           markeredgecolor='black', markeredgewidth=2, label='End', zorder=11)
    
    # Add scale bar
    scale_length = 100  # meters
    scale_x = extent_utm[0] + (extent_utm[1] - extent_utm[0]) * 0.05
    scale_y = extent_utm[2] + (extent_utm[3] - extent_utm[2]) * 0.05
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
           'k-', linewidth=3, solid_capstyle='butt', zorder=12)
    ax.text(scale_x + scale_length/2, scale_y + (extent_utm[3] - extent_utm[2]) * 0.02, 
           f'{scale_length}m', ha='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=12)
    
    # Add north arrow
    arrow_x = extent_utm[1] - (extent_utm[1] - extent_utm[0]) * 0.08
    arrow_y = extent_utm[3] - (extent_utm[3] - extent_utm[2]) * 0.08
    arrow_length = (extent_utm[3] - extent_utm[2]) * 0.05
    ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - arrow_length),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=12)
    
    if show_labels:
        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        ax.set_title('Best Runway - Satellite View', fontsize=12, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.ticklabel_format(useOffset=False, style='plain')
    
    return ax


def plot_best_runway_on_satellite(region='norcoast8', 
                                   padding_meters=200,
                                   zoom_level='auto',
                                   satellite_source='Esri.WorldImagery'):
    """
    Create a close-up plot of the best runway on satellite imagery.
    
    Parameters:
    -----------
    region : str
        Region identifier
    padding_meters : float
        Padding around runway in meters
    zoom_level : int or 'auto'
        Zoom level for satellite tiles (higher = more detail, more data)
        'auto' will choose based on area size
    satellite_source : str
        Tile source. Options:
        - 'Esri.WorldImagery' (default, good quality)
        - 'Google.Satellite' (requires API key)
        - 'OpenStreetMap.Mapnik' (street map, not satellite)
        See: https://contextily.readthedocs.io/en/latest/providers_deepdive.html
    
    Returns:
    --------
    None (saves plot to file)
    """
    print("=" * 60)
    print(f"CREATING SATELLITE VIEW OF BEST RUNWAY: {region}")
    print("=" * 60)
    
    # Load the best runway GeoTIFF
    runway_path = f'results/endpoint_search/{region}_best_runway_parametric.tif'
    
    if not os.path.exists(runway_path):
        print(f"ERROR: {runway_path} not found!")
        print("Please run runway_optimization.py first.")
        return None
    
    with rasterio.open(runway_path) as src:
        runway_raster = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Find runway pixels
    runway_pixels = np.argwhere(runway_raster > 0)
    
    if len(runway_pixels) == 0:
        print("ERROR: No runway pixels found in GeoTIFF!")
        return None
    
    # Get endpoints (first and last pixel along the runway)
    rows = runway_pixels[:, 0]
    cols = runway_pixels[:, 1]
    
    # Sort by distance from first pixel to get ordered line
    first_idx = 0
    distances = np.sqrt((cols - cols[first_idx])**2 + (rows - rows[first_idx])**2)
    sorted_indices = np.argsort(distances)
    
    # Get endpoints
    p1_pixel = (cols[sorted_indices[0]], rows[sorted_indices[0]])
    p2_pixel = (cols[sorted_indices[-1]], rows[sorted_indices[-1]])
    
    # Calculate length in meters
    p1_geo = transform * p1_pixel
    p2_geo = transform * p2_pixel
    runway_length = np.sqrt((p2_geo[0] - p1_geo[0])**2 + (p2_geo[1] - p1_geo[1])**2)
    
    # Create runway dict
    runway = {
        'p1': p1_pixel,
        'p2': p2_pixel,
        'length': runway_length
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Use the new function
    plot_runway_on_satellite_ax(
        ax, runway, transform, crs, 
        padding_meters=padding_meters,
        zoom_level=zoom_level,
        satellite_source=satellite_source,
        show_labels=True
    )
    
    plt.tight_layout()
    
    # Save
    output_path = f'results/endpoint_search/{region}_best_runway_satellite.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved satellite view to: {output_path}")
    
    plt.show()
    
    return output_path


if __name__ == "__main__":
    # Configuration
    region = 'norcoast8'
    
    # Create satellite view
    plot_best_runway_on_satellite(
        region=region,
        padding_meters=500,  # How much area around runway to show
        zoom_level='auto',   # Let contextily choose appropriate zoom
        satellite_source='Esri.WorldImagery'  # High-quality free satellite imagery
    )
    
    print("\n" + "=" * 60)
    print("ALTERNATIVE SATELLITE SOURCES:")
    print("=" * 60)
    print("If you want to try different imagery, change satellite_source to:")
    print("  - 'Esri.WorldImagery' (default, updated frequently)")
    print("  - 'Esri.WorldTopoMap' (topographic map with satellite)")
    print("  - 'OpenTopoMap' (topographic, no satellite)")
    print("  - 'CartoDB.Positron' (light basemap)")
    print("\nNote: Only tiles for the specific area are downloaded (~1-5 MB)")