# Omdena-Krakow-Poland Agricultural masking (borrowed from many sources)
Delineating agricultural field boundaries from Sentinel-2 imagery. Some code from Jesse Bakker's MA thesis project, University of Minnesota Department of Geography, Environment and Society.

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
  `!pip install requirements.txt`

# Solange's note for the demo

Change in the global_config file the following parameters for the demo:
'prep_file_dir': where you want the data to be downloaded from the Copernicus API
'username': Copernicus Hub username
'password': Copernicus Hub password
'preproc_out_dir': where you want the preprocessed data to be saved


This parameter 'prep_tile_id' was changed manually for now for the demo, we need to find a way to make it as a dymanic variable and not a global variable.

