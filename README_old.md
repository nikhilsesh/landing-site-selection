To retrieve .tif imagery and elevation data files, run:

> uv run get_earth_data.py

Manually move into corresponding `dem_maps` and `rgb_maps` folders from `EarthEngine` Google Drive folder.

To segment buildings, compute safety scores, and select optimal landing site, run:

> uv run select_landing_site.py

TODO:
- Tune safety score hyperparameters (w, window_size)
- Add terminal flags
- Add batch download/processing functionality