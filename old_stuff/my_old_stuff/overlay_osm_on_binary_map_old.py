import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def load_and_reproject_osm_data(osm_path, target_crs='EPSG:32610'):
    """Load OSM data and reproject to match DEM CRS"""
    gdf = gpd.read_file(osm_path)
    print(f"Loaded {len(gdf)} features from {osm_path}")
    print(f"Original CRS: {gdf.crs}")
    
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        print(f"Reprojected to: {target_crs}")
    
    return gdf

def rasterize_osm_features(gdf, shape, transform, buffer_distance=0):
    """
    Convert vector features to raster matching DEM dimensions.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        OSM features to rasterize
    shape : tuple
        (height, width) of output raster
    transform : Affine
        Geotransform from DEM
    buffer_distance : float
        Buffer around features in meters (e.g., 10m around buildings)
    
    Returns:
    --------
    raster : ndarray
        Binary raster (1 where features exist, 0 elsewhere)
    """
    if len(gdf) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    # Buffer features if requested
    if buffer_distance > 0:
        gdf = gdf.copy()
        gdf['geometry'] = gdf.geometry.buffer(buffer_distance)
    
    # Rasterize: assign 1 to pixels covered by features
    shapes = ((geom, 1) for geom in gdf.geometry)
    raster = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    return raster

def analyze_landing_suitability(region='norcoast5', buffer_buildings=5, buffer_roads=3):
    """
    Overlay OSM data on binary safety map to identify truly landable areas.
    
    Parameters:
    -----------
    region : str
        Region identifier
    buffer_buildings : float
        Buffer distance around buildings in meters
    buffer_roads : float
        Buffer distance around roads in meters
    """
    
    # Load the binary safety map as GeoTIFF (with georeferencing!)
    binary_map_path = f'results/{region}_binary_safety_map.tif'
    
    with rasterio.open(binary_map_path) as src:
        binary_safety = src.read(1)
        transform = src.transform
        shape = binary_safety.shape
        bounds = src.bounds
        crs = src.crs
        
        print(f"Binary Safety Map Info:")
        print(f"  CRS: {crs}")
        print(f"  Shape: {shape}")
        print(f"  Bounds: {bounds}")
    
    # Convert to binary (in case values are 0/255 instead of 0/1)
    binary_safety = (binary_safety > 0).astype(np.uint8)
    
    print(f"  Safe pixels: {np.sum(binary_safety)} ({100*np.sum(binary_safety)/binary_safety.size:.2f}%)")
    
    # Load OSM data
    print("\n" + "="*60)
    print("Loading OSM data...")
    print("="*60)
    
    buildings = load_and_reproject_osm_data('osm_data/norcoast_b23_buildings.geojson', target_crs=crs)
    features = load_and_reproject_osm_data('osm_data/norcoast_b23_features.geojson', target_crs=crs)
    
    # Load roads from GeoPackage
    roads_nodes = gpd.read_file('osm_data/norcoast_b23_roads.gpkg', layer='nodes')
    roads_edges = gpd.read_file('osm_data/norcoast_b23_roads.gpkg', layer='edges')
    
    if roads_nodes.crs != crs:
        roads_nodes = roads_nodes.to_crs(crs)
    if roads_edges.crs != crs:
        roads_edges = roads_edges.to_crs(crs)
    
    print(f"Loaded {len(roads_edges)} road segments")
    
    # Rasterize features
    print("\n" + "="*60)
    print("Rasterizing OSM features...")
    print("="*60)
    
    buildings_raster = rasterize_osm_features(buildings, shape, transform, buffer_distance=buffer_buildings)
    roads_raster = rasterize_osm_features(roads_edges, shape, transform, buffer_distance=buffer_roads)
    
    # Combine obstacle features
    # Check if features has certain types we want to exclude
    exclude_landuse = ['residential', 'commercial', 'industrial', 'retail']
    exclude_natural = ['water', 'wetland']
    
    obstacles_from_features = features.copy()
    water_features = gpd.GeoDataFrame()
    
    # Filter for obstacle landuse types
    if 'landuse' in obstacles_from_features.columns:
        obstacles_from_features = obstacles_from_features[
            obstacles_from_features['landuse'].isin(exclude_landuse)
        ]
    
    # Extract water features
    if 'natural' in features.columns:
        water_features = features[features['natural'].isin(exclude_natural)]
    
    features_raster = rasterize_osm_features(obstacles_from_features, shape, transform, buffer_distance=0)
    water_raster = rasterize_osm_features(water_features, shape, transform, buffer_distance=0) if len(water_features) > 0 else np.zeros(shape, dtype=np.uint8)
    
    # Combine all obstacles
    obstacles = np.maximum.reduce([buildings_raster, roads_raster, features_raster, water_raster])
    
    print(f"\nObstacle coverage:")
    print(f"  Buildings: {np.sum(buildings_raster)} pixels ({100*np.sum(buildings_raster)/binary_safety.size:.2f}%)")
    print(f"  Roads: {np.sum(roads_raster)} pixels ({100*np.sum(roads_raster)/binary_safety.size:.2f}%)")
    print(f"  Other features: {np.sum(features_raster)} pixels ({100*np.sum(features_raster)/binary_safety.size:.2f}%)")
    print(f"  Water: {np.sum(water_raster)} pixels ({100*np.sum(water_raster)/binary_safety.size:.2f}%)")
    print(f"  Total obstacles: {np.sum(obstacles)} pixels ({100*np.sum(obstacles)/binary_safety.size:.2f}%)")
    
    # Compute truly landable areas: safe AND no obstacles
    landable = binary_safety & (~obstacles.astype(bool))
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Terrain-safe areas: {np.sum(binary_safety)} pixels ({100*np.sum(binary_safety)/binary_safety.size:.2f}%)")
    print(f"Obstacle-free safe areas: {np.sum(landable)} pixels ({100*np.sum(landable)/binary_safety.size:.2f}%)")
    print(f"Reduction due to obstacles: {100*(1 - np.sum(landable)/max(np.sum(binary_safety), 1)):.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Binary safety map
    axes[0, 0].imshow(binary_safety, cmap='gray')
    axes[0, 0].set_title(f'Terrain Safety\n({100*np.sum(binary_safety)/binary_safety.size:.2f}% safe)')
    axes[0, 0].set_xlabel('Longitude (pixels)')
    axes[0, 0].set_ylabel('Latitude (pixels)')
    
    # 2. Buildings
    axes[0, 1].imshow(buildings_raster, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title(f'Buildings ({len(buildings)} features)')
    axes[0, 1].set_xlabel('Longitude (pixels)')
    axes[0, 1].set_ylabel('Latitude (pixels)')
    
    # 3. Roads
    axes[0, 2].imshow(roads_raster, cmap='Blues', alpha=0.7)
    axes[0, 2].set_title(f'Roads ({len(roads_edges)} segments)')
    axes[0, 2].set_xlabel('Longitude (pixels)')
    axes[0, 2].set_ylabel('Latitude (pixels)')
    
    # 4. All obstacles
    axes[1, 0].imshow(obstacles, cmap='Oranges', alpha=0.7)
    axes[1, 0].set_title(f'All Obstacles\n({100*np.sum(obstacles)/binary_safety.size:.2f}% coverage)')
    axes[1, 0].set_xlabel('Longitude (pixels)')
    axes[1, 0].set_ylabel('Latitude (pixels)')
    
    # 5. Overlay
    overlay = np.zeros((*shape, 3))
    overlay[binary_safety == 1] = [1, 1, 1]  # White for safe
    overlay[obstacles == 1] = [1, 0, 0]  # Red for obstacles
    overlay[landable == 1] = [0, 1, 0]  # Green for landable
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay\n(White=Safe, Red=Obstacles, Green=Landable)')
    axes[1, 1].set_xlabel('Longitude (pixels)')
    axes[1, 1].set_ylabel('Latitude (pixels)')
    
    # 6. Final landable areas
    axes[1, 2].imshow(landable, cmap='Greens')
    axes[1, 2].set_title(f'Landable Areas\n({100*np.sum(landable)/binary_safety.size:.2f}% suitable)')
    axes[1, 2].set_xlabel('Longitude (pixels)')
    axes[1, 2].set_ylabel('Latitude (pixels)')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Terrain-safe'),
        Patch(facecolor='red', label='Obstacles'),
        Patch(facecolor='green', label='Landable')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(f'results/{region}_landing_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: results/{region}_landing_analysis.png")
    
    # Save landable areas as GeoTIFF for GIS use
    landable_path = f'results/{region}_landable_areas.tif'
    with rasterio.open(binary_map_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, nodata=0)
        
        with rasterio.open(landable_path, 'w', **profile) as dst:
            dst.write(landable.astype(np.uint8), 1)
    
    print(f"Saved landable areas raster to: {landable_path}")
    
    return landable, obstacles, binary_safety

if __name__ == "__main__":
    # Run the analysis
    landable, obstacles, safety = analyze_landing_suitability(
        region='norcoast6',
        buffer_buildings=5,  # 5 meter buffer around buildings
        buffer_roads=3       # 3 meter buffer around roads
    )