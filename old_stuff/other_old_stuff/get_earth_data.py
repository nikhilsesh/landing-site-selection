import ee
from loguru import logger

########################################################
# TO AUTHENTICATE EARTH ENGINE IN TERMINAL:
# > earthengine authenticate --auth_mode=gcloud --force
########################################################

ee.Initialize(project='pivotal-stanford')

# Define your area of interest in California:

# San Francisco Bay Area - north, mostly water
# roi = ee.Geometry.Rectangle([-122.5, 37.7, -122.3, 37.9])
# Folsom/Placerville area - foothills, no large water bodies
# roi = ee.Geometry.Rectangle([-121.0, 38.6, -120.8, 38.75])
# roi = ee.Geometry.Rectangle([-120.85, 38.70, -120.845, 38.705])
roi = ee.Geometry.Rectangle([-121.765, 38.540, -121.755, 38.550])  # ~1km x 1km

# Get latest NAIP imagery (1m resolution RGB + NIR)
naip = ee.ImageCollection('USDA/NAIP/DOQQ') \
    .filterBounds(roi) \
    .filterDate('2018-01-01', '2024-12-31') \
    .sort('system:time_start', False) \
    .first()
#print(f"NAIP image date: {naip.date().format('YYYY-MM-dd').getInfo()}")

# Select RGB bands (NAIP also has NIR if you want it)
rgb = naip.select(['R', 'G', 'B'])

# Get 10m DEM (best available for California)
dem = ee.Image('USGS/3DEP/10m').select('elevation').clip(roi)

# Export RGB at 1m resolution
task_rgb = ee.batch.Export.image.toDrive(
    image=rgb,
    description='naip_1m_rgb',
    folder='earth_engine',
    fileNamePrefix='naip_1m_rgb',
    region=roi,
    scale=1,  # 1m resolution
    crs='EPSG:4326',
    maxPixels=1e9,
    fileFormat='GeoTIFF'
)

# Export DEM 
task_dem = ee.batch.Export.image.toDrive(
    image=dem,
    description='dem',
    folder='earth_engine',
    fileNamePrefix='dem',
    region=roi,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e9,
    fileFormat='GeoTIFF'
)

task_rgb.start()
task_dem.start()

logger.info("Export tasks started. Check your Google Earth Engine Tasks tab.")
logger.info("Files will be saved to your Google Drive in the 'EarthEngine' folder.")