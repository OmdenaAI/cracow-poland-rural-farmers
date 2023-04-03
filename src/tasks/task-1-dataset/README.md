# EveryField
Delineating agricultural field boundaries from Sentinel-2 imagery. Code from Jesse Bakker's MA thesis project, University of Minnesota Department of Geography, Environment and Society.

Functions and descriptions are in the fields_functions.py and parameters are saved in the global_config.py file. The Fields Code Processing Demo notebook walks through the processing steps. You will need to point the 'prep_file_dir' parameter in the global_config dictionary to a local directory with zipped .SAFE files from Sentinel-2 or use the sample data.

# Creating the Environment in Conda
To create the conda environment to run the code, use the environment.yml file:
`conda env create --file environment.yml`

This should work on Mac or Linux OS, but for Windows you may encounter an error trying to create the environment this way.
Instead, you can manually create the environment with the following commands:
  1. Create an empty environment without any packages:
  `conda create -n geoenv`
  2. Activate the new environment:
  `conda activate geoenv`
  3. Install all the necessary packages at once so that conda handles the dependencies:
  ```conda install -c conda-forge python=3.8.3 gdal=3.0.4 geopandas=0.7.0 rasterio=1.1.5 dask=2.19.0 xarray=0.15.1 matplotlib=3.2.1 seaborn=0.10.1 scikit-learn=0.23.1 scikit-image=0.17.2 ipython=7.15.0 ipykernel=5.3.0 folium=0.11.0 bokeh=2.1.1 holoviews=1.13.3 datashader=0.11.0 psycopg2=2.8.5 sqlalchemy=1.3.17 geoalchemy2=0.6.3 descartes=1.1.0 contextily=1.0.0 memory_profiler=0.57.0 autopep8=1.5.3 netcdf4=1.5.3```

# Solange's note for the demo

Change in the global_config file the following parameters for the demo:
'prep_file_dir': where you want the data to be downloaded from the Copernicus API
'username': Copernicus Hub username
'password': Copernicus Hub password
'preproc_out_dir': where you want the preprocessed data to be saved


This parameter 'prep_tile_id' was changed manually for now for the demo, we need to find a way to make it as a dymanic variable and not a global variable.

