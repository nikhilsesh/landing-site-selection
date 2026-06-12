1. Download DEM data from USGS National Map (https://apps.nationalmap.gov/downloader/)
    - Select from Elevation Products (3D Elevation Porgram Products and Services) and
    choose 1-meter DEM, download the .tif file for the desired region

2. Move the DEM .tif file to `./dem_maps` and rename it to be specific as possible 
    with the _dem suffix (e.g. alameda_b21_x59y418_dem.tif)

3. Compute terrain-based safety score by modifying the 'region' variable in safety_score.py
    to be the region name (e.g. alameda_b21_x59y418) and run 
    > uv run safety_score.py

4. Modify the 'region' variable in get_osm.py and run
    > uv run get_osm.py
    to get OSM data for the associated DEM region, which saves it to `./osm_data`

5. Generate the OSM-based binary safety map and landable regions map by modifying 
    the 'region' variable in overlay_osm_on_binary_map.py and run 
    > uv run overlay_osm_on_binary_map.py

6. Generate the final optimal runways map with satellite overlay by modifying the 
    'region' variable in runway_fit_EndpointSearch.py and running 
    > uv run runway_fit_EndpointSearch.py
    
7. All relevant results will be saved in the `./results` folder, which you should rename so
    that it is not overwritten by future runs
    - Binary maps and landable areas maps prior to runway-fitting analysis will be saved directly into this folder
    - The final map of optimal runways is saved in `./results/endpoint_search/`
    - If the objective_contours call in runway_fit_EndpointSearch.py is uncommented, the
    objective function contours will be saved to `./results/objective_contours/`
    - Data about the optimal runways and runtime will be saved to `./results/optimization_logs/`