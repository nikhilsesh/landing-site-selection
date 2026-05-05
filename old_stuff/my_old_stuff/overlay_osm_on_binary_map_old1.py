import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
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

def filter_unsafe_features(features_gdf):
    """
    Filter GeoDataFrame to only include features unsafe for landing.
    Returns filtered GeoDataFrame of obstacles.
    """
    unsafe_features = gpd.GeoDataFrame()
    
    # Categories where ALL values are unsafe - just check if column exists and has any value
    EXCLUDE_ALL_CATEGORIES = [
        'building',     # All buildings unsafe
        'natural',      # All water, wetlands, cliffs, etc are unsafe
        'waterway',     # All rivers, streams, canals unsafe
        'power',        # All power lines, towers, substations unsafe
        'man_made',     # All towers, masts, bridges, industrial structures unsafe
        'aeroway',      # Don't land on active runways/taxiways
        'railway',      # All rail lines unsafe
        'barrier',      # All fences, walls, hedges unsafe
        'leisure',      # Sports fields often have equipment/structures
    ]
    
    # Categories where SPECIFIC values are unsafe
    UNSAFE_LANDUSE = [
        'residential', 'commercial', 'industrial', 'retail', 
        'railway', 'quarry', 'landfill', 'reservoir', 
        'basin', 'vineyard', 'orchard', 'forest'
        # Note: farmland, meadow, grass might be safe
    ]
    
    UNSAFE_CROP = [
        'hops', 'grapes'
        # Low crops like wheat, barley are potentially safe
    ]
    
    # Exclude all features from "exclude all" categories
    for category in EXCLUDE_ALL_CATEGORIES:
        if category in features_gdf.columns:
            mask = features_gdf[category].notna()
            if mask.any():
                unsafe_features = pd.concat([unsafe_features, features_gdf[mask]], ignore_index=True)
                print(f"  Excluded {mask.sum()} {category} features")
    
    # Exclude specific landuse types
    if 'landuse' in features_gdf.columns:
        landuse_mask = features_gdf['landuse'].isin(UNSAFE_LANDUSE)
        if landuse_mask.any():
            unsafe_features = pd.concat([unsafe_features, features_gdf[landuse_mask]], ignore_index=True)
            print(f"  Excluded {landuse_mask.sum()} unsafe landuse features")
    
    # Exclude specific crop types
    if 'crop' in features_gdf.columns:
        crop_mask = features_gdf['crop'].isin(UNSAFE_CROP)
        if crop_mask.any():
            unsafe_features = pd.concat([unsafe_features, features_gdf[crop_mask]], ignore_index=True)
            print(f"  Excluded {crop_mask.sum()} unsafe crop features")
    
    # Exclude orchards with trees tag
    if 'trees' in features_gdf.columns:
        trees_mask = features_gdf['trees'].notna()
        if trees_mask.any():
            unsafe_features = pd.concat([unsafe_features, features_gdf[trees_mask]], ignore_index=True)
            print(f"  Excluded {trees_mask.sum()} tree/orchard features")
    
    return unsafe_features.drop_duplicates()

def filter_unsafe_roads(roads_gdf, min_lanes=6):
    """
    Filter roads that are unsafe for landing.
    Most roads are unsafe - only very wide highways without dividers are safe.
    
    Parameters:
    -----------
    roads_gdf : GeoDataFrame
        Road segments with highway tag
    min_lanes : int
        Minimum number of lanes for a road to be considered safe (default 6)
    
    Returns:
    --------
    unsafe_roads : GeoDataFrame
        Roads that should be excluded
    """
    
    if len(roads_gdf) == 0:
        return roads_gdf
    
    # Start by assuming ALL roads are unsafe
    safe_mask = pd.Series([False] * len(roads_gdf), index=roads_gdf.index)
    
    if 'highway' in roads_gdf.columns:
        # Only consider motorways and trunks
        major_roads_mask = roads_gdf['highway'].isin(['motorway', 'trunk'])
        
        for idx in roads_gdf[major_roads_mask].index:
            road = roads_gdf.loc[idx]
            
            # Check if it has enough lanes
            has_enough_lanes = False
            if 'lanes' in road and pd.notna(road['lanes']):
                try:
                    num_lanes = int(road['lanes'])
                    if num_lanes >= min_lanes:
                        has_enough_lanes = True
                except:
                    pass
            
            # Check for physical dividers (makes it unsafe)
            has_divider = False
            if 'divider' in road and road['divider'] in ['yes', 'barrier', 'kerb']:
                has_divider = True
            
            # Only safe if enough lanes AND no divider
            if has_enough_lanes and not has_divider:
                safe_mask[idx] = True
    
    # Return only the unsafe roads (inverse of safe_mask)
    unsafe_roads = roads_gdf[~safe_mask].copy()
    
    print(f"  Roads: {len(roads_gdf)} total, {safe_mask.sum()} safe, {len(unsafe_roads)} unsafe")
    
    return unsafe_roads

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
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None)
    raster = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    return raster

def analyze_landing_suitability(region='norcoast5', 
                                 buffer_distance=5,
                                 min_road_lanes=6):
    """
    Overlay OSM data on binary safety map to identify truly landable areas.
    
    Parameters:
    -----------
    region : str
        Region identifier
    buffer_distance : float
        Buffer distance around all obstacles in meters
    min_road_lanes : int
        Minimum lanes for a highway to be considered safe (default 6)
    """
    
    # Load the binary safety map as GeoTIFF (with georeferencing!)
    binary_map_path = f'osm_data/{region}_binary_safety_map.tif'
    
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
    
    # Load unified OSM data
    print("\n" + "="*60)
    print("Loading OSM data...")
    print("="*60)
    
    osm_features = load_and_reproject_osm_data(
        f'osm_data/{region}_osm_features.geojson', 
        target_crs=crs
    )
    
    # Separate roads from other features (roads need special handling)
    print("\n" + "="*60)
    print("Filtering unsafe features...")
    print("="*60)
    
    roads = osm_features[osm_features['highway'].notna()].copy() if 'highway' in osm_features.columns else gpd.GeoDataFrame()
    non_road_features = osm_features[osm_features['highway'].isna()].copy() if 'highway' in osm_features.columns else osm_features.copy()
    
    print(f"\nTotal features: {len(osm_features)}")
    print(f"  Roads: {len(roads)}")
    print(f"  Non-road features: {len(non_road_features)}")
    
    # Filter non-road features
    print("\nFiltering non-road features:")
    unsafe_non_road = filter_unsafe_features(non_road_features)
    
    # Filter roads
    print("\nFiltering roads:")
    unsafe_roads = filter_unsafe_roads(roads, min_lanes=min_road_lanes)
    
    # Combine all unsafe features
    all_obstacles = pd.concat([unsafe_non_road, unsafe_roads], ignore_index=True)
    
    print(f"\n" + "="*60)
    print(f"Total obstacles to exclude: {len(all_obstacles)}")
    print("="*60)
    
    # Rasterize all obstacles with buffer
    print("\nRasterizing obstacles...")
    obstacles_raster = rasterize_osm_features(
        all_obstacles, 
        shape, 
        transform, 
        buffer_distance=buffer_distance
    )
    
    print(f"Obstacle coverage: {np.sum(obstacles_raster)} pixels ({100*np.sum(obstacles_raster)/binary_safety.size:.2f}%)")
    
    # Compute truly landable areas: safe AND no obstacles
    landable = binary_safety & (~obstacles_raster.astype(bool))
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Terrain-safe areas: {np.sum(binary_safety)} pixels ({100*np.sum(binary_safety)/binary_safety.size:.2f}%)")
    print(f"Obstacle-free safe areas: {np.sum(landable)} pixels ({100*np.sum(landable)/binary_safety.size:.2f}%)")
    print(f"Reduction due to obstacles: {100*(1 - np.sum(landable)/max(np.sum(binary_safety), 1)):.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Binary safety map
    axes[0, 0].imshow(binary_safety, cmap='gray')
    axes[0, 0].set_title(f'Terrain Safety\n({100*np.sum(binary_safety)/binary_safety.size:.2f}% safe)')
    axes[0, 0].set_xlabel('Longitude (pixels)')
    axes[0, 0].set_ylabel('Latitude (pixels)')
    
    # 2. All obstacles
    axes[0, 1].imshow(obstacles_raster, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title(f'All Obstacles\n({len(all_obstacles)} features, {100*np.sum(obstacles_raster)/binary_safety.size:.2f}% coverage)')
    axes[0, 1].set_xlabel('Longitude (pixels)')
    axes[0, 1].set_ylabel('Latitude (pixels)')
    
    # 3. Overlay
    overlay = np.zeros((*shape, 3))
    overlay[binary_safety == 1] = [1, 1, 1]  # White for safe
    overlay[obstacles_raster == 1] = [1, 0, 0]  # Red for obstacles
    overlay[landable == 1] = [0, 1, 0]  # Green for landable
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay\n(White=Terrain Safe, Red=Obstacles, Green=Landable)')
    axes[1, 0].set_xlabel('Longitude (pixels)')
    axes[1, 0].set_ylabel('Latitude (pixels)')
    
    # 4. Final landable areas
    axes[1, 1].imshow(landable, cmap='Greens')
    axes[1, 1].set_title(f'Landable Areas\n({100*np.sum(landable)/binary_safety.size:.2f}% suitable)')
    axes[1, 1].set_xlabel('Longitude (pixels)')
    axes[1, 1].set_ylabel('Latitude (pixels)')
    
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
    
    return landable, obstacles_raster, binary_safety

if __name__ == "__main__":
    # Run the analysis
    landable, obstacles, safety = analyze_landing_suitability(
        region='norcoast8',
        buffer_buildings=5,  # 5 meter buffer around buildings
        buffer_roads=3       # 3 meter buffer around roads
    )