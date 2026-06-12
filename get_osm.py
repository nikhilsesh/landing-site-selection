import rasterio
from rasterio.warp import transform_bounds
import osmnx as ox
import os

region = 'norcoast_b23'

with rasterio.open(f'dem_maps/{region}_dem.tif') as src:
    # Your current bounds (UTM)
    bounds_utm = src.bounds
    print(f"UTM Bounds: {bounds_utm}")
    
    # Transform to WGS84 for OSM
    bounds_wgs84 = transform_bounds('EPSG:26910', 'EPSG:4326', *bounds_utm)
    west, south, east, north = bounds_wgs84
    
    print(f"\nBounding Box for OSM (WGS84 lat/lon):")
    print(f"West:  {west:.8f}")
    print(f"South: {south:.8f}")
    print(f"East:  {east:.8f}")
    print(f"North: {north:.8f}")
    
    # Format for Overpass API
    bbox_overpass = f"{south},{west},{north},{east}"
    print(f"\nOverpass API format: {bbox_overpass}")

# EXPANDED FEATURE EXLUSION LIST

# Create osm_data folder if it doesn't exist
os.makedirs("osm_data", exist_ok=True)

# Your coordinates
# bbox format: (left, bottom, right, top) = (west, south, east, north)
# west = -122.42889385
# south = 38.21591735
# east = -122.31368042
# north = 38.30675808

bbox = (west, south, east, north)
print(f"Bbox: (west={west}, south={south}, east={east}, north={north})")

# Define all tags we need to check for exclusion
# Each key should be set to True to get all values for that key
tags = {
    'building': True,      # All buildings
    'highway': True,       # Roads (need to filter later)
    'natural': True,       # Water, wetlands, cliffs, etc.
    'waterway': True,      # Rivers, streams, canals
    'power': True,         # Power lines, towers, substations
    'man_made': True,      # Towers, masts, bridges, industrial structures
    'aeroway': True,       # Runways, taxiways
    'railway': True,       # Rail lines
    'barrier': True,       # Fences, walls, hedges
    'leisure': True,       # Sports fields, stadiums
    'landuse': True,       # Residential, commercial, vineyards, etc.
    'crop': True,          # Crop types (hops, grapes, etc.)
    'trees': True,         # Orchards
}

print("\nDownloading all OSM features...")
print(f"Tags to extract: {list(tags.keys())}")

# Download all features with the specified tags
all_features = ox.features_from_bbox(bbox, tags=tags)

print(f"\nDownloaded {len(all_features)} total features")

# Display breakdown by feature type
print("\nFeature breakdown:")
for tag in tags.keys():
    if tag in all_features.columns:
        count = all_features[tag].notna().sum()
        print(f"  {tag}: {count} features")

# Save to single GeoJSON file
output_file = f"osm_data/{region}_osm_features.geojson"
all_features.to_file(output_file, driver='GeoJSON')
print(f"\nAll features saved to: {output_file}")

print("\n✓ OSM data extraction complete!")