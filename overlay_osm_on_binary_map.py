import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2

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

def separate_obstacles_by_type(unsafe_gdf):
    """
    Separate obstacles into categories that need different buffer distances.
    
    Returns:
    --------
    dict of GeoDataFrames for each obstacle type
    """
    obstacles = {
        'buildings': gpd.GeoDataFrame(),
        'power_lines': gpd.GeoDataFrame(),
        'power_points': gpd.GeoDataFrame(),
        'waterways': gpd.GeoDataFrame(),
        'water_bodies': gpd.GeoDataFrame(),
        'other': gpd.GeoDataFrame()
    }
    
    if len(unsafe_gdf) == 0:
        return obstacles
    
    # Buildings (already polygons)
    if 'building' in unsafe_gdf.columns:
        obstacles['buildings'] = unsafe_gdf[unsafe_gdf['building'].notna()].copy()
    
    # Power infrastructure - separate lines from points
    if 'power' in unsafe_gdf.columns:
        power_features = unsafe_gdf[unsafe_gdf['power'].notna()].copy()
        if len(power_features) > 0:
            # Lines: power lines
            line_mask = power_features['power'].isin(['line', 'minor_line', 'cable'])
            obstacles['power_lines'] = power_features[line_mask].copy()
            
            # Points: towers, poles
            point_mask = power_features['power'].isin(['tower', 'pole', 'substation', 'plant', 'generator'])
            obstacles['power_points'] = power_features[point_mask].copy()
    
    # Waterways (LineStrings - streams, rivers)
    if 'waterway' in unsafe_gdf.columns:
        obstacles['waterways'] = unsafe_gdf[unsafe_gdf['waterway'].notna()].copy()
    
    # Water bodies (Polygons - lakes, wetlands)
    if 'natural' in unsafe_gdf.columns:
        natural_water = unsafe_gdf[unsafe_gdf['natural'].notna()].copy()
        if len(natural_water) > 0:
            # Only classify as water_bodies if it's actually water-related
            water_types = ['water', 'wetland', 'bay', 'strait']
            water_mask = natural_water['natural'].isin(water_types)
            obstacles['water_bodies'] = natural_water[water_mask].copy()
    
    # Everything else
    all_categorized = pd.concat([
        obstacles['buildings'],
        obstacles['power_lines'],
        obstacles['power_points'],
        obstacles['waterways'],
        obstacles['water_bodies']
    ], ignore_index=True)
    
    if len(all_categorized) > 0:
        # Find indices not yet categorized
        categorized_indices = set()
        for df in [obstacles['buildings'], obstacles['power_lines'], obstacles['power_points'], 
                   obstacles['waterways'], obstacles['water_bodies']]:
            if len(df) > 0 and 'index' in df.columns:
                categorized_indices.update(df['index'].values)
        
        # Reset index to make filtering easier
        unsafe_gdf_indexed = unsafe_gdf.reset_index()
        uncategorized_mask = ~unsafe_gdf_indexed.index.isin(all_categorized.index if len(all_categorized) > 0 else [])
        obstacles['other'] = unsafe_gdf_indexed[uncategorized_mask].copy()
    else:
        obstacles['other'] = unsafe_gdf.copy()
    
    return obstacles

def analyze_landing_suitability(region='norcoast5', 
                                 buffer_buildings=5,
                                 buffer_power_lines=20,
                                 buffer_power_points=30,
                                 buffer_roads=3,
                                 buffer_waterways=10,
                                 buffer_water_bodies=5,
                                 buffer_other=5,
                                 min_road_lanes=6):
    """
    Overlay OSM data on binary safety map to identify truly landable areas.
    
    Parameters:
    -----------
    region : str
        Region identifier
    buffer_buildings : float
        Buffer around building polygons (meters)
    buffer_power_lines : float
        Buffer around power line paths (meters) - creates safety corridor
    buffer_power_points : float
        Buffer around power towers/poles (meters) - tower footprint + clearance
    buffer_roads : float
        Buffer around road centerlines (meters) - accounts for shoulders/ditches
    buffer_waterways : float
        Buffer around stream/river centerlines (meters) - represents actual width
    buffer_water_bodies : float
        Buffer around lakes/wetlands (meters) - small safety margin
    buffer_other : float
        Buffer around all other obstacles (meters)
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
        
        # Convert UTM bounds to lat/lon (WGS84)
        bounds_latlon = transform_bounds(crs, 'EPSG:4326', *bounds)
        
        print(f"Binary Safety Map Info:")
        print(f"  CRS: {crs}")
        print(f"  Shape: {shape}")
        print(f"  Bounds (UTM): {bounds}")
        print(f"  Bounds (Lat/Lon): {bounds_latlon}")
    
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
    
    print(f"\n" + "="*60)
    print(f"Obstacles found: {len(unsafe_non_road)} non-road + {len(unsafe_roads)} roads")
    print("="*60)
    
    # Separate obstacles by type for differential buffering
    print("\nSeparating obstacles by type...")
    obstacles = separate_obstacles_by_type(unsafe_non_road)
    
    print(f"  Buildings: {len(obstacles['buildings'])}")
    print(f"  Power lines: {len(obstacles['power_lines'])}")
    print(f"  Power towers/poles: {len(obstacles['power_points'])}")
    print(f"  Waterways: {len(obstacles['waterways'])}")
    print(f"  Water bodies: {len(obstacles['water_bodies'])}")
    print(f"  Other obstacles: {len(obstacles['other'])}")
    print(f"  Roads: {len(unsafe_roads)}")
    
    # Rasterize each obstacle type with appropriate buffer
    print("\n" + "="*60)
    print("Rasterizing obstacles with buffers...")
    print("="*60)
    
    buildings_raster = rasterize_osm_features(obstacles['buildings'], shape, transform, buffer_distance=buffer_buildings)
    power_lines_raster = rasterize_osm_features(obstacles['power_lines'], shape, transform, buffer_distance=buffer_power_lines)
    power_points_raster = rasterize_osm_features(obstacles['power_points'], shape, transform, buffer_distance=buffer_power_points)
    waterways_raster = rasterize_osm_features(obstacles['waterways'], shape, transform, buffer_distance=buffer_waterways)
    water_bodies_raster = rasterize_osm_features(obstacles['water_bodies'], shape, transform, buffer_distance=buffer_water_bodies)
    other_raster = rasterize_osm_features(obstacles['other'], shape, transform, buffer_distance=buffer_other)
    roads_raster = rasterize_osm_features(unsafe_roads, shape, transform, buffer_distance=buffer_roads)
    
    # Combine all obstacles
    obstacles_raster = np.maximum.reduce([
        buildings_raster, 
        power_lines_raster, 
        power_points_raster, 
        waterways_raster, 
        water_bodies_raster, 
        other_raster, 
        roads_raster
    ])
    
    print(f"\nObstacle coverage by type:")
    print(f"  Buildings: {np.sum(buildings_raster)} pixels ({100*np.sum(buildings_raster)/binary_safety.size:.2f}%)")
    print(f"  Power lines: {np.sum(power_lines_raster)} pixels ({100*np.sum(power_lines_raster)/binary_safety.size:.2f}%)")
    print(f"  Power towers/poles: {np.sum(power_points_raster)} pixels ({100*np.sum(power_points_raster)/binary_safety.size:.2f}%)")
    print(f"  Waterways: {np.sum(waterways_raster)} pixels ({100*np.sum(waterways_raster)/binary_safety.size:.2f}%)")
    print(f"  Water bodies: {np.sum(water_bodies_raster)} pixels ({100*np.sum(water_bodies_raster)/binary_safety.size:.2f}%)")
    print(f"  Other: {np.sum(other_raster)} pixels ({100*np.sum(other_raster)/binary_safety.size:.2f}%)")
    print(f"  Roads: {np.sum(roads_raster)} pixels ({100*np.sum(roads_raster)/binary_safety.size:.2f}%)")
    print(f"  Total obstacles: {np.sum(obstacles_raster)} pixels ({100*np.sum(obstacles_raster)/binary_safety.size:.2f}%)")
    
    # Compute truly landable areas: safe AND no obstacles
    landable = binary_safety & (~obstacles_raster.astype(bool))
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Terrain-safe areas: {np.sum(binary_safety)} pixels ({100*np.sum(binary_safety)/binary_safety.size:.2f}%)")
    print(f"Obstacle-free safe areas: {np.sum(landable)} pixels ({100*np.sum(landable)/binary_safety.size:.2f}%)")
    print(f"Reduction due to obstacles: {100*(1 - np.sum(landable)/max(np.sum(binary_safety), 1)):.2f}%")

    # # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate extent in geographic coordinates (lon, lat) - USE CONVERTED BOUNDS
    extent = [bounds_latlon[0], bounds_latlon[2], bounds_latlon[1], bounds_latlon[3]]
    
    # 1. Binary safety map
    axes[0, 0].imshow(binary_safety, cmap='gray', extent=extent, aspect='auto')
    axes[0, 0].set_title(f'Terrain Safety\n({100*np.sum(binary_safety)/binary_safety.size:.2f}% safe)')
    axes[0, 0].set_xlabel('Longitude (degrees)')
    axes[0, 0].set_ylabel('Latitude (degrees)')
    axes[0, 0].ticklabel_format(useOffset=False, style='plain')  # Prevent scientific notation
    
    # 2. All obstacles
    axes[0, 1].imshow(obstacles_raster, cmap='Reds', alpha=0.7, extent=extent, aspect='auto')
    axes[0, 1].set_title(f'All Obstacles\n({100*np.sum(obstacles_raster)/binary_safety.size:.2f}% coverage)')
    axes[0, 1].set_xlabel('Longitude (degrees)')
    axes[0, 1].set_ylabel('Latitude (degrees)')
    axes[0, 1].ticklabel_format(useOffset=False, style='plain')
    
    # 3. Overlay
    overlay = np.zeros((*shape, 3))
    overlay[binary_safety == 1] = [1, 1, 1]  # White for safe
    overlay[obstacles_raster == 1] = [1, 0, 0]  # Red for obstacles
    overlay[landable == 1] = [0, 1, 0]  # Green for landable
    
    axes[1, 0].imshow(overlay, extent=extent, aspect='auto')
    axes[1, 0].set_title('Overlay\n(White=Terrain Safe, Red=Obstacles, Green=Landable)')
    axes[1, 0].set_xlabel('Longitude (degrees)')
    axes[1, 0].set_ylabel('Latitude (degrees)')
    axes[1, 0].ticklabel_format(useOffset=False, style='plain')
    
    # 4. Final landable areas
    axes[1, 1].imshow(landable, cmap='Greens', extent=extent, aspect='auto')
    axes[1, 1].set_title(f'Landable Areas\n({100*np.sum(landable)/binary_safety.size:.2f}% suitable)')
    axes[1, 1].set_xlabel('Longitude (degrees)')
    axes[1, 1].set_ylabel('Latitude (degrees)')
    axes[1, 1].ticklabel_format(useOffset=False, style='plain')
    
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


    # Filter landable area by minimum size of connected eligible voxels
    # Ensure binary image is uint8 (0 or 255)
    landable = landable.astype(np.uint8) * 255

    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(landable, connectivity=4)

    # stats columns: [x, y, width, height, area]
    min_size = 6000 # minimum area of 15 m x 400 m = 6000 m^2
    landable_filter = np.zeros_like(landable)

    count = 0
    for i in range(1, num_labels):  # Start from 1 to skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            # print(stats[i]) # debugging
            landable_filter[labels == i] = 255
            count += 1

    # Plot filtered landable areas
    print(f"\nNumber of landable areas after cleaning: {count}")
    plt.figure(figsize=(8, 6))
    plt.imshow(landable_filter, cmap='Greens', extent=extent, aspect='auto')
    plt.title(f'Filtered Landable Areas (min size {min_size} pixels)')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.ticklabel_format(useOffset=False, style='plain')
    # plt.show()
    plt.savefig(f'results/{region}_filtered_landable_areas.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved filtered landable areas visualization to: results/{region}_filtered_landable_areas.png")

    # Save filtered landable areas as GeoTIFF
    filtered_landable_path = f'results/{region}_filtered_landable_areas.tif'
    with rasterio.open(binary_map_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, nodata=0)
        
        with rasterio.open(filtered_landable_path, 'w', **profile) as dst:
            dst.write((landable_filter > 0).astype(np.uint8), 1)
    
    return landable, obstacles_raster, binary_safety

if __name__ == "__main__":
    # Run the analysis
    landable, obstacles, safety = analyze_landing_suitability(
        # region='alameda_b21_x59y418',
        region='norcoast_b23',
        buffer_buildings=5,
        buffer_power_lines=20,
        buffer_power_points=30,
        buffer_roads=3,
        buffer_waterways=10,
        buffer_water_bodies=5,
        buffer_other=5,
        min_road_lanes=6
    )