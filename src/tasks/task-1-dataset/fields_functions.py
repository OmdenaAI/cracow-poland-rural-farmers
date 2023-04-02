# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:32:15 2020

@author: jesse
"""

"""
Combined document for all final fields code for the thesis.
This will get broken out into separate documents for each step (maybe?) once organized and labeled here.

Thresholding Analysis Processing structure:
    - In different document

Cluster Analysis Processing structure:
    - Preprocessing data
    - Investigate Clusters (visualization, identify cut off points for field cluster classification)
    - 

"""


"""
Importing Packages
"""
### import libraries
import os
import numpy as np
import dask
from dask.array import map_overlap, map_blocks
import xarray as xr
import glob
import time
import datetime
import zipfile
from osgeo import gdal
from pathlib import Path

import os.path
from os import path

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

#from functools import partial # Was used during testing
import skimage
from skimage.filters import sobel, gaussian
#from skimage.segmentation import mark_boundaries
from skimage import feature, measure, morphology, segmentation, exposure
import fiona
from rasterio import features
import rasterio as rio
import folium

from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn import cluster
from sklearn.utils import shuffle
import scipy.ndimage as ndi
import pickle

# post processing 
### import packages
import geopandas as gpd
import pandas as pd
import os
import os.path
from os import path
from shapely.geometry import shape, mapping, Point

#%%

# import config module with global config parameters
from global_config import *

#%%
"""
Functions for each step
"""

"""
--------------------------------------------------
Utility and preprocessing functions
--------------------------------------------------
"""


### clip values above and below a certain percentile of the array
def clip_nan(array, percentile):
    np.seterr(divide='ignore',invalid='ignore')
    vmax = np.nanpercentile(array, 100 - percentile)
    vmin = np.nanpercentile(array, percentile)
    array_clip = np.clip((array), a_min = vmin, a_max = vmax)
    return array_clip

### apply the clip_nan() function in parallel to be lazily evaluated
def clip_nan_ufunc(array, percentile):
    return xr.apply_ufunc(clip_nan, array, percentile,
                          input_core_dims=None,
                          dask='parallelized',
                          output_dtypes=[int])
    
### Normalize array values to be passed to processing functions
def normalize(array):
    '''normalized numpy array into scale 0.0 - 1.0'''
    np.seterr(divide='ignore', invalid='ignore')
    return ((array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array)))

### Apply the normalize funtion in parallel to be lazily evaluated
def normalize_ufunc(array):
    return xr.apply_ufunc(normalize, array,
                          input_core_dims=[[]],
                          output_dtypes=[float],
                          dask='parallelized')

def zero_to_nan(array):
#        rio_da = rio_da.where(rio_da != 0, np.nan)
    array = array.astype(float)
    array[array == 0] = np.nan
    return array

def zero_to_nan_ufunc(array):
    return xr.apply_ufunc(zero_to_nan, array,
                          input_core_dims=[[]],
                          output_dtypes=[float],
                          dask='parallelized')


"""
--------------------------------------------------
Prepare dataset functions
--------------------------------------------------

These functions construct an xarray DataSet object of the 10m resolution R, G, B, NIR bands of Sentinel-2 imagery
combined along the time dimension for a single tile footprint, referenced by the three letter tile ID.
It expects to read the data from .zip files of the Sentinel-2 .SAFE format from a directory on MSI.
Each tile footprint is processed separately. 

There is spatial overlap between neighboring tile footprints.There are two ways of addressing this built into the functions. 
Setting the 'prep_remove_overlap' parameter to True in the config dictionary will remove the overlapping area with the 
neighboring tile footprint by selecting 10,000 pixels in the x and y directions from the top left corner of the tile, 
resulting in a DataSet of dimension: (time steps, 10000, 10000). This DataSet is referred to as the time stack, or ds_time_stack, 
since it is a stack of the R, G, B, NIR rasters along the time dimension.

You can also manually select a spatial subset of the data by setting the 'prep_manual_subset' parameter to True,
and specifying the index of the first x value ('prep_x_start'), y value ('prep_y_start'), and number of pixels
to include in each direction ('prep_step'). This is useful for testing and development where code can run
much faster on smaller spatial subsets of the data, instead of on the full 10000 by 10000 pixel tile. It is 
also implemented for processing large areas when there are limits on memory. You can manually divide each tile
during processing and stitch the outputs back together if memory crashes when processing a full tile, which can 
happen when there are a lot of imagery dates. Leaving both the 'prep_remove_overlap' and 'prep_manual_subset' will
return the full tile footprint.

The xr_dict_to_dataset_cloud_mask() function also includes parameters for specifying the threshold for cloud coverage
on images being read into the time stack ('prep_cloud_coverage_thresh') and for applying the cloud mask from the 
Sentinel-2 data quality information. This cloud mask raster is 20m resolution, so it must be resampled to 10m to be
applied as a mask to the imagery bands. The 'prep_load_cloud_mask' boolean parameter specifies whether the cloud mask will be
loaded from the imagery data and the 'prep_apply_cloud_mask' boolean parameter specifies whether it will be applied to the 
imagery as a mask. These are generally the same and can be combined into a single parameter. The 'prep_cloud_mask_thresh' 
parameter sets the threshold value for the cloud mask, which is a probability of the pixel being cloudy, in order to 
use it as a binary mask on the imagery. The mask tends to be conservative, missing some cloudy pixels.

This data prep process clips outliers from each band, sets the nodata values to NAN (they are 0 in the raw data), 
and clips outliers based on percentile from each imagery band. If you do not want to clip the outliers, set the 
'prep_clip_outliers' parameter to False in the cofig dictionary. The 'prep_clip_percentile' parameter specifies 
what percentile to clip for the outliers. Usually, 1% is sufficient to remove outlier pixels that can cause issues.

This process happens lazily and will not load the data until the output of an operation done on the DataSet is needed. 
The data is normalized for each variable (band) in the dataset by specifying the 'prep_normalize_bands' boolean parameter
in the config dictionary. This normalizes the data by the full time series, not by the individual time step. To normalize
each time step individually, move the normalization function into the end of the xr_dict_to_dataset_cloud_mask() function.
Nodata values are set to NaN.

Config variables:
prep_config = {'prep_file_dir': 'C:/Users/jesse/Documents/grad school/masters research/code/fields_library/data/rasters/from_MSI/', # dir with sentinel tiles
              'prep_tile_id': 'TPT',            # three letter tile ID for tile being processed
              'prep_base_chunk': 'auto',        # chunk size for x and y dimenstions
              'prep_time_chunk': 'auto',        # chunk size along the time dimension
              'prep_remove_overlap': False,     # Set to true to load 10000 by 10000 tile
              'prep_manual_subset': True,       # If prep_remove_overlap is True, set this to False. This will select spatial subset
              'prep_x_start': 0,                # This sets the start index for the x value
              'prep_y_start': 0,                # this sets the start index for the y value
              'prep_step': 500,                 # this sets the number of pixels to include in the spatial subset in the x and y dimensions
              'prep_cloud_coverage_thresh': 50, # this sets the cloud coverage threshold for reading in images
              'prep_load_cloud_mask': True,     # this specifies whether to load the cloud mask layer
              'prep_apply_cloud_mask': True,    # this specifies whether to apply the cloud mask layer to the imagery bands
              'prep_cloud_mask_thresh': 70,     # this specifies what cloud probability value to use for binarizing the cloud mask
              'prep_clip_percentile': 2}        # this specifies what percentage of values on either end of the histogram to clip for each band


### Code Reorganizing Notes:
- The code currently only works with Sentinel-2 Level 2A data (bottom of atmosphere) and hits an error parsing the metadata
  when trying to read Level 1C (top of atmosphere) data.

"""

def xr_get_zip_info_to_dict(zip_file):
    """
    Builds a dictionary of metadata from zipped sentinel tiles to pass 
    to the function that opens the data into an xarray dataset.
    Expects a list of zipped sentinel tiles and 
    an epsg code (which needs to be consistent for all tiles)
    in the format: "EPSG_32614", which can be identified from calling !gdalinfo
    """
    ### Info from file name
    file_name = zip_file[:-4]
    tile_id = zip_file[-26:-20]
    date_str = zip_file[-19:-11]
    date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
    platform = zip_file[:3]
    prod_level = zip_file[7:10]
    
    # info from gdal_info
    gdal_info = gdal.Info(zip_file, format='json')
    # set GDAL call for 10m bands in zip tile directory
    gdal_str = gdal_info['metadata']['SUBDATASETS']['SUBDATASET_1_NAME']
    epsg = gdal_str[-10:]
    cloud_coverage = float(gdal_info['metadata']['']['CLOUD_COVERAGE_ASSESSMENT'])
    nodata_pixel_percentage = float(gdal_info['metadata']['']['NODATA_PIXEL_PERCENTAGE'])
    
    ### Get string for cloud mask iamge
    # Get contents of zip file as a list
    zip_contents = zipfile.ZipFile(zip_file, 'r')
    # path to 20m cloud mask within the zipped file
    cld_path = [s for s in zip_contents.namelist() if "MSK_CLDPRB_20m.jp2" in s]
    # create string to call cloud mask .jp2 from zipped file with rasterio
    cld_mask_str = "/vsizip/" + zip_file + "/" + cld_path[0]

    # return dict of tile info with top level key: file name
    info_dict = {file_name: {'tile_id':tile_id,
                             'date_str':date_str,
                             'date_obj':date_obj,
                             'gdal_str':gdal_str,
                             'epsg':epsg,
                             'cloud_coverage':cloud_coverage,
                             'nodata_pixel_percentage':nodata_pixel_percentage,
                             'platform':platform,
                             'product_level':prod_level,
                             'cloud_mask_str':cld_mask_str}}
    
    return info_dict



### Pass tile information (gdal_str) to open file to xr.dataset and place object in ds_list
### With remove_overlap set to True, the function selects by x,y extent to remove right and bottom overlap with neighboring tiles
### The total pixel overlap in each direction with the right and bottom neighbors is 983 pixels,
### based on the difference in initial coordinates (top left) from the rasterio objects
def xr_dict_to_dataset_cloud_mask(da_name, gdal_str, date_obj, tile_id, cld_mask_str,
                                  load_cloud_mask = False, apply_cloud_mask = False, cloud_mask_thresh = 30,
                                  chunks=(-1,1000,1000), remove_overlap = True,
                                  manual_subset = False, x_start = 0, y_start = 0, step = 1000):
    """
    Pass tile information (gdal_str) to open file to xr.dataset and place object in ds_list
    With remove_overlap set to True, the function selects by x,y extent to remove right and bottom overlap with neighboring tiles
    The total pixel overlap in each direction with the right and bottom neighbors is 983 pixels,
    based on the difference in initial coordinates (top left) from the rasterio objects
    
    parameters:
        da_name - xr DataArray name, needed to load in imagery from info_dict via xr_get_zip_info_to_dict()
        gdal_str - string to read the 10m bands from zipped tile from info_dict via xr_get_zip_info_to_dict()
        date_obj - date of imagery formatted as datetime obj from info_dict via xr_get_zip_info_to_dict()
        tile_id - Sentinel tile ID from info_dict via xr_get_zip_info_to_dict()
        cld_mask_str - path to cloud mask in zipped tile from info_dict via xr_get_zip_info_to_dict()
        
        load_cloud_mask - Boolean to load the cloud mask raster
        apply_cloud_mask - Boolean to apply cloud mask to RGBN bands
        cloud_mask_thresh - int to set threshold for cloud mask probability
        
        chunks - (time dim dask chunks, x dim dask chunks, y dim dask chunks) via config
        remove_overlap - Boolean to spatially subset the tile to remove the overlapping area with neighboring tile
        manual_subset - Boolean to assign manual spatial subset, used for manual chunking or testing
        x_start - int to assign first x value, left edge of the subset
        y_start - int to assign first y value, top edge of the subset
        step - int to assign how many pixels in each direction (x, y) of the manual subset
        
    """    

    ### Define offset for cleaning up seams in cloud masking, can be an arbitrary number larger than 1, should be even and relatively small
    prep_offset = 10
    
    # Lazily read file to xarray DataArray    
    rio_da = xr.open_rasterio(gdal_str, chunks=chunks)
    # set time coordinate
    rio_da['time'] = date_obj  
    # set da name
    rio_da.name = da_name
    # set tile_id
    rio_da.attrs['tile_id'] = tile_id    
    
    if load_cloud_mask:
        # Get cloud mask
        cld_mask_da = xr.open_rasterio(cld_mask_str, chunks=('auto','auto'))
        cld_mask_da['time'] = date_obj
        cld_mask_da = cld_mask_da.where(cld_mask_da > cloud_mask_thresh, 0) 
        cld_mask_da.name = 'cloud_mask'
        cld_mask_da = cld_mask_da.astype(bool).astype(int)
    
    # this removes the area overlapping with the neighboring tile on the right and bottom
    # The overlap is 983 pixels, but using 980 results in a clean 10000 by 10000 pixel tile
    if remove_overlap:
        rio_da = rio_da[:,:-980+prep_offset,:-980+prep_offset]
        if load_cloud_mask:
            cld_mask_da = cld_mask_da[:,:-490+(prep_offset/2),:-490+(prep_offset/2)]

    if manual_subset:
        # Just for processing in manual batches
        x = x_start
        y = y_start       
        x_end = x + step + prep_offset
        y_end = y + step + prep_offset 
        rio_da = rio_da[:, x:x_end, y:y_end]
        if load_cloud_mask:
            cld_mask_da = cld_mask_da[:,int(x/2):int(x_end/2), int(y/2):int(y_end/2)]      

    if load_cloud_mask:
        # Resample cloud mask to 10m resolution to work with imagery bands
        cld_mask_da = cld_mask_da.interp_like(rio_da.isel(band=0), method='linear')        
        cld_mask_da = cld_mask_da.isel(band=0).fillna(0)
        cld_mask_da = cld_mask_da.chunk(chunks=('auto','auto')).drop('band').astype(bool)
        cld_mask_da = np.invert(cld_mask_da)
    
    # convert imagery to dataset
    ds = rio_da.to_dataset(dim='band')
    ds = ds.rename_vars({1:'red',
                         2:'green',
                         3:'blue',
                         4:'nir'})
    
    # Apply cloud mask to bands
    if apply_cloud_mask:
        ds['red'] = ds['red'].where(cld_mask_da)
        ds['green'] = ds['green'].where(cld_mask_da)
        ds['blue'] = ds['blue'].where(cld_mask_da)
        ds['nir'] = ds['nir'].where(cld_mask_da)
    
    if remove_overlap:
        ds = ds[dict(x=slice(0,10000), y=slice(0,10000))].chunk(chunks={'x':'auto','y':'auto'})
    
    if manual_subset:
        x = x_start
        y = y_start       
        x_end = x + step
        y_end = y + step
        ds = ds[dict(x=slice(0,step), y=slice(0,step))].chunk(chunks={'x':'auto','y':'auto'})
    
    return ds


def prep_data():
    # set variables from global config
    global config
    # file dir
    file_dir = config['prep_file_dir']
    tile_id = config['prep_tile_id']
    # chunking
    base_chunk = config['prep_base_chunk']
    time_chunk = config['prep_time_chunk']
    chunk_size = (time_chunk, base_chunk, base_chunk)
    rechunk_size = {'time': time_chunk, 'x': base_chunk, 'y': base_chunk}
    # data read parameters
    cloud_coverage_thresh = config['prep_cloud_coverage_thresh']
    load_cloud_mask = config['prep_load_cloud_mask']
    apply_cloud_mask = config['prep_apply_cloud_mask']
    cloud_mask_thresh = config['prep_cloud_mask_thresh']
    overlap_bool = config['prep_remove_overlap']
    manual_subset = config['prep_manual_subset']
    x_start = config['prep_x_start']
    y_start = config['prep_y_start']
    step = config['prep_step']
    clip_outliers = config['prep_clip_outliers']
    percentile = config['prep_clip_percentile']
    normalize = config['prep_normalize_bands']
    
    # set working directory to the folder with the sentinel data tiles
    # Set file_dir in the global variables at top of the code
    os.chdir(file_dir)
    glob_str = "*" + tile_id + "*.zip"
    zip_list = glob.glob(glob_str)
    
    # build dictionary of metadata from gdal_info calls for each tile in the list of zip_files
    tile_info_dict = {}
    for zip_str in zip_list:
        tile_info_dict.update(xr_get_zip_info_to_dict(zip_str))

    # date list to hold date objects of time slices in the data stack in order to avoid duplicate dates
    ### FIX ME : there should be a better long-term solution to select the best image if there are two from the same date instead of the first one that the code sees
    date_list=[]
    count_cloud_pass = 0
    count_total_files = 0
    print("Cloud Coverage Threshold:", cloud_coverage_thresh)
    for k, v in tile_info_dict.items():
        if v['cloud_coverage'] < cloud_coverage_thresh:
            # Print tile info
            print("passed cloud coverage:", v['date_str'][:4],v['date_str'][4:6],v['date_str'][6:], "with", v['cloud_coverage'], "pct | nodata pixel pct:", v['nodata_pixel_percentage'] )
            # if this is the first tile to pass the cloud coverage thresh, define it as ds_time_stack
            if count_cloud_pass == 0:
                ds_time_stack = xr_dict_to_dataset_cloud_mask(k,v['gdal_str'],v['date_obj'],v['tile_id'], v['cloud_mask_str'], 
                                                              load_cloud_mask = load_cloud_mask, apply_cloud_mask = apply_cloud_mask, cloud_mask_thresh = cloud_mask_thresh,
                                                              chunks=chunk_size, remove_overlap=overlap_bool,
                                                              manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step)
                                
                date_list.append(v['date_obj'])
                
            # if not the first tile
            if count_cloud_pass > 0:
                # if a tile with the same date is already in the ds_time_stack, then skip this tile because there can't be two tiles with the same time dim 
                ### QUESTION : is it faster to just reference ds_time_stack.coords['time'] for monitoring duplicates?
                ### EXAMPLE: if v['date_obj'] in ds_time_stack.coords['time']:
                if v['date_obj'] in date_list:
                    continue
                ds_time_stack = xr.concat([ds_time_stack,
                                           xr_dict_to_dataset_cloud_mask(k,v['gdal_str'],v['date_obj'],v['tile_id'], v['cloud_mask_str'],
                                                                         load_cloud_mask = load_cloud_mask, apply_cloud_mask = apply_cloud_mask, cloud_mask_thresh = cloud_mask_thresh,
                                                                         chunks=chunk_size, remove_overlap=overlap_bool,
                                                                         manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step)], 
                                                                         dim = 'time')
                date_list.append(v['date_obj'])
            # Add to the count so that subsequent tiles are concatenated with ds_time_stack instead of overwriting it
            count_cloud_pass += 1
            
        else:
            print("FAILED cloud coverage:", v['date_str'][:4],v['date_str'][4:6],v['date_str'][6:], "with ", v['cloud_coverage'],  "pct | nodata pixel pct:", v['nodata_pixel_percentage'] )
        count_total_files += 1
    print(count_cloud_pass, "out of", count_total_files, "time steps passed cloud coverage threshold")
    ### QUESTION : Is this the best way to handle an empty ds_time_stack?
    if count_cloud_pass == 0:
        print("no tiles below cloud threshold")
    
    ### Rechunk the full dataset and sort by time dimension
    ds_time_stack = ds_time_stack.chunk(chunks=rechunk_size).sortby('time')
    
    # Clip each band in each of the time steps to remove outliers
    # set 0 values to NaN since missing data is read in as 0
    ds_time_stack = ds_time_stack.map(zero_to_nan_ufunc, keep_attrs = True)
    # Clip outlier values in each band
    if clip_outliers:
        ds_time_stack = ds_time_stack.map(clip_nan_ufunc, percentile=percentile, keep_attrs = True)
    # Normalize each band
    if normalize:
        ds_time_stack = xr.Dataset.chunk(ds_time_stack.map(normalize_ufunc, keep_attrs=True), 
                                         chunks=rechunk_size)
    
    print(ds_time_stack)    
    return ds_time_stack



"""
Thresholding approach prep data functions. The prep_data() function should be used, but combined code for
the Thresholding approach depends on a few of these older functions for loading in the data.
These use the same overall process as the ones above but with some slight differences.
They are not as optimized, they do not evaluate lazily with the cloud mask, they do not normalize the data, 
the outliers are clipped per image date instead of over the full time series. 
"""

### Pass tile information (gdal_str) to open file to xr.dataset and place object in ds_list
### With remove_overlap set to True, the function selects by x,y extent to remove right and bottom overlap with neighboring tiles
### The total pixel overlap in each direction with the right and bottom neighbors is 983 pixels,
### based on the difference in initial coordinates (top left) from the rasterio objects
def xr_dict_to_dataset_thresh(da_name, gdal_str, date_obj, tile_id, 
                       chunks=("auto","auto","auto"), remove_overlap = True,
                       manual_subset = False, x_start = 0, y_start = 0, step = 1000):
    """
    Pass tile information (gdal_str) to open file to xr.dataset and place object in ds_list
    With remove_overlap set to True, the function selects by x,y extent to remove right and bottom overlap with neighboring tiles
    The total pixel overlap in each direction with the right and bottom neighbors is 983 pixels,
    based on the difference in initial coordinates (top left) from the rasterio objects
    """
    # Lazily read file to xarray DataArray    
    rio_da = xr.open_rasterio(gdal_str, chunks=chunks)
    # set time coordinate
    rio_da['time'] = date_obj  
    # set da name
    rio_da.name = da_name
    # set tile_id
    rio_da.attrs['tile_id'] = tile_id    
    # this removes the area overlapping with the neighboring tile on the right and bottom
    # The overlap is 983 pixels, but using 980 results in a clean 10000 by 10000 pixel tile
    if remove_overlap:
        rio_da = rio_da[:,:-980,:-980]
        
    if manual_subset:
        # Just for processing in manual batches
        x = x_start
        y = y_start       
        x_end = x + step        
        y_end = y + step       
        rio_da = rio_da[:, x:x_end, y:y_end]

    # convert to dataset
    ds = rio_da.to_dataset(dim='band')
    ds = ds.rename_vars({1:'red',
                         2:'green',
                         3:'blue',
                         4:'nir'})

    return ds

def xr_dict_to_dataset_cloud_mask_thresh(da_name, gdal_str, date_obj, tile_id, cld_mask_str, mask_bands = True,
                                  chunks=(-1,1000,1000), remove_overlap = True,
                                  manual_subset = False, x_start = 0, y_start = 0, step = 1000):
    """
    Pass tile information (gdal_str) to open file to xr.dataset and place object in ds_list
    With remove_overlap set to True, the function selects by x,y extent to remove right and bottom overlap with neighboring tiles
    The total pixel overlap in each direction with the right and bottom neighbors is 983 pixels,
    based on the difference in initial coordinates (top left) from the rasterio objects
    """
#    for k in test_dict:
#        da_name = k
#        gdal_str = v['gdal_str']
#        date_obj = v['date_obj']
#        tile_id = v['tile_id']
#        cld_mask_str = v['cloud_mask_str']
    
    # Lazily read file to xarray DataArray    
    rio_da = xr.open_rasterio(gdal_str, chunks=chunks)
    # set time coordinate
    rio_da['time'] = date_obj  
    # set da name
    rio_da.name = da_name
    # set tile_id
    rio_da.attrs['tile_id'] = tile_id    
    
    # Get cloud mask
    cld_mask_da = xr.open_rasterio(cld_mask_str)
    cld_mask_da['time'] = date_obj
    cld_mask_da.name = 'cloud_mask'
    
    ### Resample cloud mask to 10m resolution to work with imagery bands
    new_x = np.linspace((cld_mask_da.x[0].values - 5).astype(int), (cld_mask_da.x[-1].values + 5).astype(int), len(cld_mask_da.x) * 2)
    new_y = np.linspace((cld_mask_da.y[0].values + 5).astype(int), (cld_mask_da.y[-1].values - 5).astype(int), len(cld_mask_da.y) * 2)
    cld_mask_da_interp = cld_mask_da.interp(x = new_x, method = 'nearest')
    cld_mask_da_interp = cld_mask_da_interp.interp(y = new_y, method = 'nearest')
    #test_cld_da_interp.where(test_cld_da_interp > 50)[:,900:1000,0:100].plot(figsize=(12,10))
    base_chunk = chunks[1]
    ### Prep cloud mask data to join with imagery bands
    cld_mask_da_interp = cld_mask_da_interp[0,:,:]
    cld_mask_da_interp = cld_mask_da_interp.chunk(chunks=(base_chunk,base_chunk))
    del cld_mask_da_interp.attrs['transform']
    del cld_mask_da_interp.attrs['res']
    del cld_mask_da_interp.attrs['is_tiled']
    del cld_mask_da_interp.attrs['nodatavals']
    del cld_mask_da_interp.attrs['scales']
    del cld_mask_da_interp.attrs['offsets']
    cld_mask_da_interp = cld_mask_da_interp.drop('band')
    
    # this removes the area overlapping with the neighboring tile on the right and bottom
    # The overlap is 983 pixels, but using 980 results in a clean 10000 by 10000 pixel tile
    if remove_overlap:
        rio_da = rio_da[:,:-980,:-980]
        cld_mask_da_interp = cld_mask_da_interp[:-980,:-980]
        
    if manual_subset:
        # Just for processing in manual batches
        x = x_start
        y = y_start       
        x_end = x + step        
        y_end = y + step       
        rio_da = rio_da[:, x:x_end, y:y_end]
        cld_mask_da_interp = cld_mask_da_interp[x:x_end, y:y_end]

    # convert to dataset
    ds = rio_da.to_dataset(dim='band')
    ds = ds.rename_vars({1:'red',
                         2:'green',
                         3:'blue',
                         4:'nir'})
    
    if mask_bands:
        ds['cloud_mask'] = cld_mask_da_interp.astype(int)
        ds['red'] = ds['red'].where(ds.cloud_mask < 30)
        ds['green'] = ds['green'].where(ds.cloud_mask < 30)
        ds['blue'] = ds['blue'].where(ds.cloud_mask < 30)
        ds['nir'] = ds['nir'].where(ds.cloud_mask < 30)
        
    return ds

### No cloud masking in this version
def build_ds_list(tile_info_dict, cloud_coverage_thresh = 20, chunk_size = ("auto","auto","auto"), 
                  overlap_bool = True, manual_subset = False, x_start = 0, y_start = 0, step = 1000):
    """
    Loop over the dictionary of tile infromation and pass each entry to the 
    xr_dict_to_dataset function to build a list of lazy xr datasets. 
    Set cloud coverage threshold to only add tiles with cloud coverage assessment below threshold.
    Overlap_bool: xr_dict_to_dataset_thresh() parameter that removes spatial overlap between neighboring tiles.
    Manual_subset, x_start, y_start, step: parameters for xr_dict_to_dataset_thresh() that handle manual tile chunk processing.
    """
    ds_list = []
    count_cloud_pass = 0
    count_total_files = 0
    print("Cloud Coverage Threshold:", cloud_coverage_thresh)
    for k, v in tile_info_dict.items():
        if v['cloud_coverage'] < cloud_coverage_thresh:
            print("passed cloud coverage:", v['date_str'][:4],v['date_str'][4:6],v['date_str'][6:], "with", v['cloud_coverage'], "pct | nodata pixel pct:", v['nodata_pixel_percentage'] )
            count_cloud_pass += 1
            ds_list.append(xr_dict_to_dataset_thresh(k,v['gdal_str'],v['date_obj'],v['tile_id'],
                                              chunks=chunk_size, remove_overlap=overlap_bool,
                                              manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step))
        else:
            print("FAILED cloud coverage:", v['date_str'][:4],v['date_str'][4:6],v['date_str'][6:], "with ", v['cloud_coverage'],  "pct | nodata pixel pct:", v['nodata_pixel_percentage'] )
        count_total_files += 1
    print(count_cloud_pass, "out of", count_total_files, "time steps passed cloud coverage threshold")
    
#    if ds_list == None:
#        pass
    
    return ds_list


### Updating above function to integrate the cloud mask
def build_ds_list_cloud_mask(tile_info_dict, cloud_coverage_thresh = 20, chunk_size = ("auto","auto","auto"), 
                  overlap_bool = True, manual_subset = False, x_start = 0, y_start = 0, step = 1000):
    """
    Loop over the dictionary of tile infromation and pass each entry to the 
    xr_dict_to_dataset function to build a list of lazy xr datasets. 
    Set cloud coverage threshold to only add tiles with cloud coverage assessment below threshold.
    Overlap_bool: xr_dict_to_dataset() parameter that removes spatial overlap between neighboring tiles.
    Manual_subset, x_start, y_start, step: parameters for xr_dict_to_dataset() that handle manual tile chunk processing.
    """
    ds_list = []
    count_cloud_pass = 0
    count_total_files = 0
    print("Cloud Coverage Threshold:", cloud_coverage_thresh)
    for k, v in tile_info_dict.items():
        if v['cloud_coverage'] < cloud_coverage_thresh:
            print("passed cloud coverage:", v['date_str'][:4],v['date_str'][4:6],v['date_str'][6:], "with", v['cloud_coverage'], "pct | nodata pixel pct:", v['nodata_pixel_percentage'] )
            count_cloud_pass += 1
            ds_list.append(xr_dict_to_dataset_cloud_mask_thresh(k,v['gdal_str'],v['date_obj'],v['tile_id'], v['cloud_mask_str'], 
                                                         mask_bands = True,
                                                         chunks=chunk_size, remove_overlap=overlap_bool,
                                                         manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step))
        else:
            print("FAILED cloud coverage:", v['date_str'][:4],v['date_str'][4:6],v['date_str'][6:], "with ", v['cloud_coverage'],  "pct | nodata pixel pct:", v['nodata_pixel_percentage'] )
        count_total_files += 1
    print(count_cloud_pass, "out of", count_total_files, "time steps passed cloud coverage threshold")
    
#    if ds_list == None:
#        pass
    
    return ds_list

def build_time_stack_from_ds_list(ds_list, rechunk_size = {'time': "auto", 'x': "auto", 'y': "auto"}):
    """
    Takes a list of xarray datasets and concatenates them along the 'time' dimension into a single dataset.
    Then sorts the dataset by time.
    """
    date_list=[]
    ds_list_trimmed=[]
    for ds in ds_list:
#        print(ds.coords['time'])
        if ds.coords['time'] in date_list:
            continue
        else: 
            date_list.append(ds.coords['time'])
            ds_list_trimmed.append(ds)
            
    ds_time_stack = xr.concat(ds_list_trimmed, dim='time')
    
    ### Rechunk the full dataset
    ds_time_stack = ds_time_stack.chunk(chunks=rechunk_size).sortby('time')
    return ds_time_stack


def prep_ds_time_stack(file_dir, tile_id, cloud_coverage_thresh = 15, cloud_mask = False,
                        base_chunk = "auto", overlap_bool = True,
                        manual_subset = False, x_start = 0, y_start = 0, step = 2000):
    
    chunk_size = ("auto", base_chunk, base_chunk)
    rechunk_size = {'time': "auto", 'x': base_chunk, 'y': base_chunk}
    
    # set working directory to the folder with the sentinel data tiles
    # Set file_dir in the global variables at top of the code
#    os.chdir(file_dir)
    glob_str = "*" + tile_id + "*.zip"
    zip_list = glob.glob(file_dir + glob_str)
    #                print(len(zip_list), "time steps")
    
    # build dictionary of metadata from gdal_info calls for each tile in the list of zip_files
    tile_info_dict = {}
    for zip_str in zip_list:
        tile_info_dict.update(xr_get_zip_info_to_dict(zip_str))
    
    # build list of xr.dataset objects from the tile_info_dict
    # Defaults to using the cloud-masked dataset
    if cloud_mask == True:
        ds_list = build_ds_list_cloud_mask(tile_info_dict, 
                            cloud_coverage_thresh = cloud_coverage_thresh, 
                            chunk_size = chunk_size, overlap_bool = overlap_bool,
                            manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step)
    if cloud_mask == False:
        ds_list = build_ds_list(tile_info_dict, 
                                cloud_coverage_thresh = cloud_coverage_thresh, 
                                chunk_size = chunk_size, overlap_bool = overlap_bool,
                                manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step)
    
    if len(ds_list) == 0:
        print("no tiles below cloud threshold")
    
    # Clip each band in each of the time steps to remove outliers
    for ds in ds_list:
        for band in ['red','green','blue','nir']:
            ds[band] = clip_nan_ufunc(ds[band].where(ds[band] !=0), percentile = 1)
    
    # Stack each time stack into single xarray dataset with dimensions ('time','x','y')
    ds_time_stack = build_time_stack_from_ds_list(ds_list, rechunk_size = rechunk_size)
                    
    # Drop any time slices that only have null data, only really needed when running in manual chunks (I think)
    # This is to get around the error when encountering an "All NaN slice" -- a full chunk of nan data in a single time slice
    # which happens when big chunks of tile are no data due to a satellite path
    ### This takes a really long time on a full tile with 10 time slices
#    ds_time_stack = ds_time_stack.dropna(dim='time', how='all')
    
    print(ds_time_stack)    
    return ds_time_stack


"""
-------------------------------------
Mask Processing functions
-------------------------------------
"""

### QUESTION : the data in each band gets normalized multiple times (red & nir in NDVI, green & nir in NDWI, and all bands in the edge map)
###             should this step happen once up front instead?

### NDVI
def ndvi_xr(input_ds):
    '''
    computes the Normalized Difference Vegetation Index
    '''
    np.seterr(divide='ignore', invalid='ignore')
    return ((input_ds.nir - input_ds.red)/(input_ds.nir + input_ds.red))

### UPDATE : the clip and normalize functions have been moved to the data prep step, so it shouldn't be necessary here
def ndvi_xr_norm(input_ds):
    np.seterr(divide='ignore', invalid='ignore')
    nir = normalize_ufunc(input_ds.nir)
    red = normalize_ufunc(input_ds.red)
    return ((nir - red)/(nir + red))

### EVI - // Enhanced Vegetation Index  (abbrv. EVI)
#// General formula: 2.5 * (NIR - RED) / ((NIR + 6*RED - 7.5*BLUE) + 1)
#// URL https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=16
def evi_xr(input_ds):
    '''
    computes the Enhanced Vegetation Index
    '''
    np.seterr(divide='ignore', invalid='ignore')    
    input_ds = clip_nan_ufunc(input_ds, 1)
    input_ds = normalize_ufunc(input_ds)
    EVI = 2.5 * (input_ds.nir - input_ds.red) / ((input_ds.nir + 6.0 * input_ds.red - 7.5 * input_ds.blue) + 1.0)
    return EVI

### GCVI - green chlorophyll vegetation index
### https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1278&context=natrespapers
### (NIR - green) / 1
### This isn't used
def gcvi_xr(input_ds):
    np.seterr(divide='ignore', invalid='ignore')
    return ((input_ds.nir - input_ds.green) - 1)

### NDWI
def ndwi_xr(input_ds):
    """
    computes the Normalized Difference Water Index
    """
    np.seterr(divide='ignore', invalid='ignore')
    return ((input_ds.green - input_ds.nir)/(input_ds.green + input_ds.nir))

### UPDATE : the clip and normalize functions have been moved to the data prep step, so it shouldn't be necessary here
def ndwi_xr_norm(input_ds):
    """
    computes the Normalized Difference Water Index
    """
    np.seterr(divide='ignore', invalid='ignore')
    green = normalize_ufunc(input_ds.green)
    nir = normalize_ufunc(input_ds.nir)
    return ((green - nir)/(green + nir))

### Find threshold value for NDWI - 0.5 to start
    ### UPDATE : switched from ndwi_xr_norm() to ndwi_xr() because the normalization takes place in the data prep now
def ndwi_mask_func(input_ds, ndwi_thresh = 0.5):
    ### Take the mean NDWI value across the full time stack, normalized  
    return xr.where(ndwi_xr_norm(input_ds).mean(dim='time', skipna=True) > ndwi_thresh, 1, 0)
#    return xr.where(ndwi_xr(input_ds).mean(dim='time', skipna=True) > ndwi_thresh, 1, 0)

### EDGES
# Sobel edges function from skimage
def sobel_edges(input_array):
    return sobel(input_array)


### FIX ME: this can be made more generalizable for assigning what bands get included in the edge magnitude calculation
### Currently, the R G B NIR bands each get passed to the edge algorithm, but it could be useful to
### also pass a derived NDVI band to the edge algorithm as well, or to only use a subset of the bands.
### This also can take a different edge algorithm as an input, instead of just the sobel filter.
def compute_edges(ds_time_stack, edge_algo = sobel_edges, chunk_size='auto', percentile=0.1):
    for i in np.arange(0,len(ds_time_stack.coords['time']),1):
#         print('edges time step:', i)   # This is for testing
        ### Map the edge algorithm to the individual time step in the for loop
        edges = ds_time_stack.isel(time=i).fillna(0).map(edge_algo).chunk({'x':chunk_size,'y':chunk_size})
        ### Then take the sum of the edge magnitude from each band, and take the cumulative sum of all the time steps
        if i == 0:
            edges_sum = (edges['red']+edges['green']+edges['blue']+edges['nir'])
        if i > 0:
            edges_sum = edges_sum + (edges['red']+edges['green']+edges['blue']+edges['nir'])
        
    ### Then take the average edge magnitude across the full time stack
    edges_sum = edges_sum/(4*len(ds_time_stack.coords['time']))
    ### Clip outlier values based on the percentile input to remove noisy pixels with high edge values
    edges_sum = clip_nan_ufunc(edges_sum, percentile)
    ### Normalize the average edge magnitude layer to a range of 0 to 1
    edges_sum = normalize_ufunc(edges_sum)

    return edges_sum


"""
These functions are from the original thresholding approach
The cumulative edge algorithm here is slower than the one above (I think)
The edge finding approach has a lot of room for improvement.
"""
# Passes each band in each timestep of a xr dataset through the sobel edge filter in parallel, lazily evaluated
# Concatenates the sum of each timestep's edges into a xr dataarray in teh same shape as the input
### This could be broken up/made more modular for cases where there is only a single timestep, for instance
def edges_to_xr_data_array(input_ds, ndvi = False):
    # Define some lists to get populated in the loop
    edges_list = []
    edges_time_list = []
    
    bands = ['red','green','blue','nir']
    
    # List of bands to loop over
    if ndvi:
        bands = ['red','green','blue','nir','ndvi']
    
    # Loop through each time step to populate a list of summed edges for each band in each time step
    for i in np.arange(0,len(input_ds.coords['time']),1):
#        if input_ds.isnull:
#            continue
        
        # Loop through each band to compute edges and append resulting dataarray to a list
        for band in bands:
            # define coords for 
            coords = [input_ds[band].coords['y'],input_ds[band].coords['x']]
            
            # Run the sobel edges algorithm on overlapping chunks to cover seams in chunks because it is a focal operation
            # The input band (ds[band]) gets normalized so values range from 0 to 1 and then clipped to remove 
            # outliers at the highest and lowest 2% of values
            # prep edges input by normalizing the band
            edges_input_norm = normalize_ufunc(clip_nan_ufunc(input_ds[band].isel(time=i),2))
            # lazily pass array through the sobel_edges() function via the dask.map_overlap() function
            band_egdes_output = edges_input_norm.data.map_overlap(sobel_edges, depth=1)
            
            # Convert the edges array to a dataarray using the x and y coordinates of the input array
            band_egdes_output_da = xr.DataArray(band_egdes_output,
                                                dims=('y','x'),
                                                coords=coords)
            
            # Add the time step dimension to the edges array that matches the time coordinate for the time step being processed
            band_egdes_output_da['time'] = input_ds['time'][i]
            
            # append the edges to a list of DataArrays, one for each band for each time step
            edges_list.append(band_egdes_output_da)
        
        edges_list_da = xr.concat(edges_list, dim='time')
        # For each time step, sum the edges for each band
#        edges_sum = edges_list_da.sum(dim='time', skipna=True)
        edges_sum = edges_list_da.mean(dim='time', skipna=True)
        # Give this new dataarray a logical name
        edges_sum.name = 'edges_sum'
        # Append the summed edge map of each time step to a list, to be merged along the 'time' dimension
        edges_time_list.append(edges_sum)
    # concat the summed edges for each time step into an array that matches the shape (x, y, time) of the time_stack_ds
    edges_da = xr.concat(edges_time_list, dim='time')

    return edges_da


### build mask
    
def mask_ndvi_max_and_range(monthly_avg_ndvi_ds, max_thresh = 0.3, range_thresh = 0.3):
    ### Maximum monthly average NDVI
    ndvi_monthly_max = monthly_avg_ndvi_ds.max(dim='time', skipna = True)
    ### NDVI range from max monthly NDVI to mean NDVI for May
    ndvi_range_max_t0 = ndvi_monthly_max - monthly_avg_ndvi_ds.isel(time=0)
    ndvi_range_mask = xr.where(ndvi_range_max_t0 < range_thresh, 1, 0)
    ### Max monthly average ndvi
    ndvi_max_mask = xr.where(ndvi_monthly_max < max_thresh, 1, 0)
    ### Combine masks
    return xr.ufuncs.logical_or(ndvi_range_mask, ndvi_max_mask)

# This function is in the mask.py file
def create_combined_ndvi_edges(ndvi, edges, ndvi_weight=1, edges_weight=1):
    return (ndvi_weight*ndvi) + (edges_weight*edges)

def create_mask_bool_array(ndvi_edge_combo, binary_threshold):
    return xr.where(normalize_ufunc(ndvi_edge_combo) < binary_threshold, 1, 0)
#    return xr.where(normalize_ufunc(ndvi_edge_combo) < binary_threshold, False, True)

def build_mask(ndvi_range, cumulative_edges, ndvi_weight, edges_weight):
    # inputs get normalized so that each range from 0 to 1
    combined_mask_inputs = create_combined_ndvi_edges(ndvi = 1 - normalize_ufunc(ndvi_range), 
                                      edges = normalize_ufunc(cumulative_edges), 
                                      ndvi_weight = ndvi_weight, 
                                      edges_weight = edges_weight)

    # Compute mask
    return combined_mask_inputs

# This will get passed to dask arrays within the dataset through dask.map_overlap() for parallelization
def fill_holes(input_array):
    return ndi.binary_fill_holes(input_array)

# This will get passed to dask arrays within the dataset through dask.map_overlap() for parallelization
### Original preliminary run used a size = (2,2)
def min_filter(input_array, size = (2, 2)):
    return ndi.minimum_filter(input_array, size=size)


def mask_processing(input_ds, ndwi_thresh = 0.5, ndvi_max_thresh = 0.3, ndvi_range_thresh = 0.3, edges_thresh = 0.3):
    ### NDWI mask: mean NDWI across all time steps, masked for values greater than ndwi_thresh
    ndwi_mask = ndwi_mask_func(input_ds, ndwi_thresh = ndwi_thresh)
    
    ### NDVI Masking
    # Compute mean NDVI on the monthly composite images as input to the NDVI masking function
    # Compute NDVI mask: Max monthly mean NDVI > max_thresh, and Max Monthly NDVI - May monthly mean NDVI 
    combined_ndvi_mask = mask_ndvi_max_and_range(ndvi_xr_norm(input_ds.resample(time='1MS').mean(skipna = True)), 
                                                 max_thresh = ndvi_max_thresh, 
                                                 range_thresh = ndvi_range_thresh)
    
    ### Edges
    # Mean edges for full data stack
    edges_mean = compute_edges(input_ds)
    # Mask edges
    edges_mask = xr.where(edges_mean > edges_thresh, 1, 0)
    
    
    ### Combined mask is a combination of the component masks via a logical OR function
    ### If a pixel is masked out in any of the component masks it is masked out in the combined mask layer
    combined_mask = xr.ufuncs.logical_or(xr.ufuncs.logical_or(ndwi_mask, combined_ndvi_mask), edges_mask)
    # Invert mask
    combined_mask = xr.where(combined_mask == 1, 0, 1)
    
    ### Fill holes in the mask so that there aren't stray pixels
    combined_mask = combined_mask.data.map_overlap(fill_holes, depth=1)
    
    ### Set minimum filter to enforce minimum height/width of background. 
    ### This eliminates small isolated areas and expands road areas.
    ### This can be replaced with some of the morphological filters used in the cluster mask clean up process
    combined_mask = combined_mask.map_overlap(min_filter, depth = 1)
    
    ### Set mask to ds_time_stack array
    input_ds['mask'] = xr.DataArray(combined_mask, 
                                     dims=('y','x'),
                                     coords = [input_ds['red'].coords['y'],
                                               input_ds['red'].coords['x']])
    
    return input_ds



"""
Prep RGB
"""
# combine selected rgb bands into a 3 x M x N array to pass to segmentation
def prep_rgb_input_for_segmentation(input_ds, rgb_date, chunk_size, percentile = 2):

    # get RGB from the time closest to the input date
    rgb_da_list = [normalize_ufunc(input_ds.red.sel(time=rgb_date, method='nearest')),
                   normalize_ufunc(input_ds.green.sel(time=rgb_date, method='nearest')),
                   normalize_ufunc(input_ds.blue.sel(time=rgb_date, method='nearest'))]

    # Combine bands into RGB array and rescale the intensity so that image has better contrast for segmentation
    rgb_image_da = clip_nan_ufunc(xr.concat(rgb_da_list, dim='band'), percentile=percentile)
    rgb_image_da = rgb_image_da.assign_coords(band=['r','g','b'])
    rgb_image_da = rgb_image_da.chunk(chunks=chunk_size)
   
    return rgb_image_da

def rgb_image(rgb_date_str, ds_time_stack, rgb_chunk_size, gaussian_filt = True, gaussian_sigma = 2, percentile = 2):
    rgb_date_obj = datetime.datetime.strptime(rgb_date_str, '%Y%m%d')
    ### Build RGB xr DataArray to be passed to segmentation and merge functions
    ### Percentile is the amount of the rgb image to clip from either end of the histogram
    rgb_da = prep_rgb_input_for_segmentation(ds_time_stack, rgb_date_obj, chunk_size = rgb_chunk_size, percentile = percentile)
    ### Transpose to (y, x, band) in order to pass to 
    rgb_transpose = rgb_da.transpose('y', 'x', 'band')
    
    ### Apply gaussian filter to rgb image
    if gaussian_filt == True:
        dims = rgb_transpose.dims
        coords = rgb_transpose.coords
        rgb_transpose = xr.DataArray(rgb_transpose.data.map_overlap(gaussian, sigma = gaussian_sigma, multichannel = True, depth = 1),
                                     dims=dims,
                                     coords=coords)
    
    ### Apply mask to rgb image
    masked_rgb_image_da = rgb_transpose.where(ds_time_stack['mask']==1, other=0).fillna(0)
    
    return masked_rgb_image_da



"""
felzenszwalb segmentation
"""
def segment_fz(input_rgb_array, scale = 50, sigma = .8, min_size = 50):
    return segmentation.felzenszwalb(input_rgb_array, scale = scale, sigma= sigma, min_size= min_size)


"""
Write to shapefile
"""

def write_segments_to_shapefile_xr(input_array, src_transform, src_crs, output_file, mask = None):
    '''
    This function takes an array (meant for a raster that has already been segmented)
    and writes the polygonized raster to a shapefile.

    input_array: raster to by polygonized and exported
    src_transform: the "transform" spatial metadata from the rasterio.read() of the source raster
    src_crs: the coordinate reference system from the source raster, as a string. ex: 'EPSG:32614'
    output_file: file path/name ending with '.shp' for the output
    '''
    # set input array to integer
    # array_int = input_array.astype(int)
    
    # polygonize input raster to GeoJSON-like dictionary with rasterio.features.shapes
    # src_transform comes from the source raster that holds the spatial metadata
    results = ({'geometry': s, 'properties': {'raster_val': v}}
          for i, (s, v) in enumerate(features.shapes(input_array, mask=mask, transform = src_transform)))
    geoms = list(results)
    
    # establish schema to write into shapefile
    schema_template = {
    'geometry':'Polygon', 
    'properties': {
        'raster_val':'int'}}
    
    ### FIX THIS: changed driver='Shapefile' to driver='ESRI Shapefile'
    # src_crs is the coordinate reference system from the source raster that holds the spatial metadata
    with fiona.open(output_file, 'w', driver='Shapefile', schema = schema_template, crs = src_crs) as layer:
        # loop through the list of raster polygons and write to the shapefile
        for geom in geoms:
            layer.write(geom)



"""
Data preprocessing combined function
"""
### This function combines all the preprocessing steps and saves out netcdf files. 
### It also restructures the data and fits a k-means clustering model to the data, saving out the cluster centers to a separate directory.
### A larger number of analyitcal layers were processed initially, but were ultimately commented out of the function.
### Only the layers used for the clustering are computed and saved to netcdf.
### Once all the tiles have been preprocessed, they can be easily opened in xarray with xr.open_mfdataset()
def preprocess_to_netcdf(ds_time_stack):
    global config      
    out_dir = config['preproc_out_dir']
    out_file_name = config['preproc_outfile_prefix'] + config['prep_tile_id']
    sample_pct = config['preproc_sample_pct']
    n_clusters = config['preproc_n_clusters']
    cluster_tile = config['preproc_cluster_tile']
        
    ### Cloud-free (or not-cloud-masked) pixel count
    pixel_count = ds_time_stack.nir.count(dim='time')
    pixel_count.name = 'pixel_count'
    #    pixel_count_mon = ds_time_stack.nir.resample(time='1MS').count(dim='time')
    
    ### NDWI stats: 
    ### NDWI mean
    #    ndwi_mean = ndwi_xr(ds_time_stack).mean(dim='time', skipna=True)
    #    ndwi_mean.name = 'ndwi_mean'
    ### NDWI average for July and August
    ndwi_mean_jul_aug_sep = ndwi_xr(ds_time_stack.loc[dict(time=slice('2019-07-01','2019-10-01'))]).mean(dim='time', skipna=True)
    ndwi_mean_jul_aug_sep.name = 'ndwi_mean_jul_aug_sep'
    
    ### NDVI stats: 
    ### Standard Deviation appears to be the most resistant to noise/clouds (compared to min, max, range, mean)
    #    ndvi_mean = ndvi_xr(ds_time_stack).mean(dim='time', skipna = True)
    #    ndvi_max = ndvi_xr(ds_time_stack).max(dim='time', skipna = True)  # Influenced by clouds
    #    ndvi_min = ndvi_xr(ds_time_stack).min(dim='time', skipna = True)  # Influenced by clouds
    #    ndvi_range = ndvi_max - ndvi_min                                  # Influenced by clouds, might be better with monthly mean
    ndvi_std = ndvi_xr(ds_time_stack).std(dim='time', skipna = True)    # pretty resistant to cloud noise
    #    ndvi_var = ndvi_xr(ds_time_stack).var(dim='time', skipna = True)
    
    ### Monthly mean NDVI
    #    ndvi_mon_mean = ndvi_xr(ds_time_stack.resample(time='1MS').mean(dim='time', skipna = True))
    #ndvi_mon_mean.plot(x='x', y='y', col='time', col_wrap=3)
    
        ### START TIMING
    t_start = time.perf_counter()
    t_proc_start = time.process_time()
    
    ### Median - can't be computed with dask arrays, so it has to be converted to np arrays and the full stack loaded into memory and processed serially
    ds_time_stack_no_dask = xr.Dataset({'red':(['time', 'y', 'x'],ds_time_stack.red.values),
                                        'nir':(['time', 'y', 'x'],ds_time_stack.nir.values)}, coords=ds_time_stack.coords)
    ### Compute monthly median and convert output to dask by passing chunk size method
    ndvi_mon_med = ndvi_xr(ds_time_stack_no_dask.resample(time='1MS').median(dim='time', skipna = True, keep_attrs=True).sortby('time')).chunk({'x':'auto', 'y':'auto'})
    
        ### STOP TIMING
    print("Monthly Median Total CPU time:", time.process_time() - t_proc_start)
    print("Monthly Median Total Wall time:", time.perf_counter() - t_start)
    
    ### Edges - Not using this edge function because it takes a lot longer and the results are noisier
    ### Instead of passing edges over each band in each time step, passing the standard deviation of NDVI over the time dimension
    ### Reduces the temporal variation between regions with varying cloud and satellite coverage
    ### Need to test this in non-ag regions--won't work well for edges in cities, likely--but seems to capture edges around ag fields well
    #edges = compute_edges(ds_time_stack)
    #edges.plot(figsize=(8,8))
    
        ### START TIMING
#    t_start = time.perf_counter()
#    t_proc_start = time.process_time()
#        
#    ### Edges from ndvi standard deviation
#    ### Pass this to the longitudinal edge convolution workflow
#    edges_ndvi_std = xr.DataArray(normalize_ufunc(sobel(ndvi_std)),
#                                  dims=ndvi_std.dims,
#                                  coords=ndvi_std.coords).chunk({'x':'auto','y':'auto'})
#        ### STOP TIMING
#    print("Edges Total CPU time:", time.process_time() - t_proc_start)
#    print("Edges Total Wall time:", time.perf_counter() - t_start)
    
    
    ### Combine into single dataset
    preproc_ds = xr.Dataset({'pixel_count':pixel_count,
    #                             'pixel_count_mon':pixel_count_mon,
                             'ndwi_mean_jul_aug_sep':ndwi_mean_jul_aug_sep,
                             'ndvi_std':ndvi_std,
    #                             'ndvi_min':ndvi_min,
    #                             'ndvi_max':ndvi_max,
    #                             'ndvi_mon_mean':ndvi_mon_mean,
                             'ndvi_mon_med':ndvi_mon_med})
#                             'edges_ndvi_std':edges_ndvi_std})
    
    ### Set compression in the encoding dictionary to pass to the to_netcdf() call
    comp = dict(zlib=True, complevel=6)
    encoding = {var: comp for var in preproc_ds.data_vars}
    #    encoding['pixel_count'].update(dtype = 'int16')
    #    encoding['pixel_count_mon'].update(dtype = 'int16')
    
            ### START TIMING
    t_start = time.perf_counter()
    t_proc_start = time.process_time()
    
    test_out_fp = out_dir + out_file_name + '.nc'
    print("saving preprocessed tile to:", test_out_fp)
    Path(out_dir).mkdir(exist_ok=True)
    preproc_ds.to_netcdf(test_out_fp, mode='w', format='NETCDF4', encoding=encoding)
    
        ### STOP TIMING
    print("Writing NetCDF file Total CPU time:", time.process_time() - t_proc_start)
    print("Writing NetCDF file Total Wall time:", time.perf_counter() - t_start)
    
    if cluster_tile:
        out_dir_cluster_centers = out_dir + 'cluster_centers/' 
        Path(out_dir_cluster_centers).mkdir(exist_ok=True)
        
        ### KMeans clustering
        ### NOTE: before clustering, make sure each variable is normalized - NDVI is already normalized
    #        kmeans_input_array = ndvi_mon_med.stack(stack=('x','y'))
    #        kmeans_input_array = kmeans_input_array.transpose('stack','time').fillna(0).chunk(chunks={'stack':'auto','time':'auto'})
        
        ### NEW CODE - Adding NDVI std and NWDI mean Jul-Sep to cluster
        ndvi_mon_med_stack = ndvi_mon_med.stack(stack=('x','y'))
        ndvi_mon_med_stack = ndvi_mon_med_stack.transpose('stack','time').fillna(0).rename({'time':'variable'})
        ndvi_mon_med_stack = ndvi_mon_med_stack.chunk(chunks={'stack':'auto','variable':'auto'})
        #ndvi_mon_med_stack = ndvi_mon_med_stack.transpose('stack','time').fillna(0)
        print('ndvi_mon_med_stack', ndvi_mon_med_stack)
        
        to_cluster_ds = xr.Dataset(data_vars={'ndvi_std':normalize_ufunc(preproc_ds.ndvi_std), 
    #                                          'edges':normalize_ufunc(preproc_ds.edges_ndvi_std), 
                                              'ndwi_mean':normalize_ufunc(preproc_ds.ndwi_mean_jul_aug_sep)})
        to_cluster_ds = to_cluster_ds.to_array(name='nontime_vars').stack(stack=('x','y')).transpose('stack','variable').fillna(0)
        
        #### Combining ndvi time series with other cluster vars
        image_array = xr.concat([ndvi_mon_med_stack,to_cluster_ds], dim='variable')
        image_array = image_array.chunk(chunks={'stack':'auto', 'variable':'auto'})
        print('image array to cluster:',image_array)
            
        ### set width and height
        w, h = len(ds_time_stack.coords['x']), len(ds_time_stack.coords['y'])
        sample_pct = sample_pct
        n_samples = int(w * h * sample_pct)
        n_clusters = n_clusters
        print('kmeans total samples:', n_samples)
        
        t0 = time.time()
        image_array_sample = shuffle(image_array, random_state=0)[:n_samples]
        image_array_sample = image_array_sample.chunk(chunks={'stack':'auto','variable':'auto'})
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_jobs = -1).fit(image_array_sample)
        print("kmeans clustering done in %0.3fs." % (time.time() - t0))
        
        ### FIX ME: if there are not any images for a month to create a monthly median NDVI layer, the cluster vars will not be the right length
        ### build DA of kmeans clusters of monthly median NDVI time series
        cluster_centers_da = xr.DataArray(kmeans.cluster_centers_,
                                          dims=['n','cluster_vars'],
                                          coords={'cluster_vars':['ndvi_may','ndvi_jun','ndvi_jul','ndvi_aug','ndvi_sep','ndvi_oct','ndvi_std','ndwi_mean']})
    
        t_start = time.perf_counter()
        t_proc_start = time.process_time()
        
        test_out_fp = out_dir_cluster_centers + out_file_name + '_kmeans_cluster_centers.nc'
        cluster_centers_da.to_netcdf(test_out_fp, mode='w', format='NETCDF4')
    
        ### STOP TIMING
        print("Writing Cluster Centers NetCDF file Total CPU time:", time.process_time() - t_proc_start)
        print("Writing Cluster Centers NetCDF file Total Wall time:", time.perf_counter() - t_start)


### This was the original function, which is identical but included more data layers in the output.
### This can be rewritten to be more flexible about what data layers to include in the preprocessing outputs.
def preprocess_to_netcdf_original_var_list(ds_time_stack):
    global config
    out_dir = config['preproc_out_dir']
    out_file_name = config['preproc_outfile_prefix']
    sample_pct = config['preproc_sample_pct']
    n_clusters = config['preproc_n_clusters']
    cluster_tile = config['preproc_cluster_tile']
        
    ### Cloud-free (or not-cloud-masked) pixel count
    pixel_count = ds_time_stack.nir.count(dim='time')
    pixel_count.name = 'pixel_count'
    pixel_count_mon = ds_time_stack.nir.resample(time='1MS').count(dim='time')
    
    ### NDWI stats: 
    ### NDWI mean
#    ndwi_mean = ndwi_xr(ds_time_stack).mean(dim='time', skipna=True)
#    ndwi_mean.name = 'ndwi_mean'
    ### NDWI average for July and August
    ndwi_mean_jul_aug_sep = ndwi_xr(ds_time_stack.loc[dict(time=slice('2019-07-01','2019-10-01'))]).mean(dim='time', skipna=True)
    ndwi_mean_jul_aug_sep.name = 'ndwi_mean_jul_aug_sep'
    
    ### NDVI stats: 
    ### Standard Deviation appears to be the most resistant to noise/clouds (compared to min, max, range, mean)
#    ndvi_mean = ndvi_xr(ds_time_stack).mean(dim='time', skipna = True)
    ndvi_max = ndvi_xr(ds_time_stack).max(dim='time', skipna = True)  # Influenced by clouds
    ndvi_min = ndvi_xr(ds_time_stack).min(dim='time', skipna = True)  # Influenced by clouds
#    ndvi_range = ndvi_max - ndvi_min                                  # Influenced by clouds, might be better with monthly mean
    ndvi_std = ndvi_xr(ds_time_stack).std(dim='time', skipna = True)    # pretty resistant to cloud noise
#    ndvi_var = ndvi_xr(ds_time_stack).var(dim='time', skipna = True)
    
    ### Monthly mean NDVI
#    ndvi_mon_mean = ndvi_xr(ds_time_stack.resample(time='1MS').mean(dim='time', skipna = True))
    #ndvi_mon_mean.plot(x='x', y='y', col='time', col_wrap=3)
    
        ### START TIMING
    t_start = time.perf_counter()
    t_proc_start = time.process_time()
    
    ### Median - can't be computed with dask arrays, so it has to be converted to np arrays and the full stack loaded into memory and processed serially
    ds_time_stack_no_dask = xr.Dataset({'red':(['time', 'y', 'x'],ds_time_stack.red.values),
                                        'nir':(['time', 'y', 'x'],ds_time_stack.nir.values)}, coords=ds_time_stack.coords)
    ### Compute monthly median and convert output to dask by passing chunk size method
    ndvi_mon_med = ndvi_xr(ds_time_stack_no_dask.resample(time='1MS').median(dim='time', skipna = True, keep_attrs=True).sortby('time')).chunk({'x':'auto', 'y':'auto'})
    
        ### STOP TIMING
    print("Monthly Median Total CPU time:", time.process_time() - t_proc_start)
    print("Monthly Median Total Wall time:", time.perf_counter() - t_start)
    
    ### Edges - Not using this edge function because it takes a lot longer and the results are noisier
    ### Instead of passing edges over each band in each time step, passing the standard deviation of NDVI over the time dimension
    ### Reduces the temporal variation between regions with varying cloud and satellite coverage
    ### Need to test this in non-ag regions--won't work well for edges in cities, likely--but seems to capture edges around ag fields well
    #edges = compute_edges(ds_time_stack)
    #edges.plot(figsize=(8,8))
    
    ### Edges from ndvi standard deviation
    ### Pass this to the longitudinal edge convolution workflow
    edges_ndvi_std = xr.DataArray(normalize_ufunc(sobel(ndvi_std)),
                                  dims=ndvi_std.dims,
                                  coords=ndvi_std.coords).chunk({'x':'auto','y':'auto'})
    
    ### Combine into single dataset
    preproc_ds = xr.Dataset({'pixel_count':pixel_count,
                             'pixel_count_mon':pixel_count_mon,
                             'ndwi_mean_jul_aug_sep':ndwi_mean_jul_aug_sep,
                             'ndvi_std':ndvi_std,
                             'ndvi_min':ndvi_min,
                             'ndvi_max':ndvi_max,
#                             'ndvi_mon_mean':ndvi_mon_mean,
                             'ndvi_mon_med':ndvi_mon_med,
                             'edges_ndvi_std':edges_ndvi_std})
    
    ### Set compression in the encoding dictionary to pass to the to_netcdf() call
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in preproc_ds.data_vars}
#    encoding['pixel_count'].update(dtype = 'int16')
#    encoding['pixel_count_mon'].update(dtype = 'int16')

            ### START TIMING
    t_start = time.perf_counter()
    t_proc_start = time.process_time()
    
    test_out_fp = out_dir + out_file_name + '.nc'
    preproc_ds.to_netcdf(test_out_fp, mode='w', format='NETCDF4', encoding=encoding)

        ### STOP TIMING
    print("Writing NetCDF file Total CPU time:", time.process_time() - t_proc_start)
    print("Writing NetCDF file Total Wall time:", time.perf_counter() - t_start)

    if cluster_tile:
        out_dir_cluster_centers = out_dir + 'cluster_centers/' 
        Path(out_dir_cluster_centers).mkdir(exist_ok=True)
        ### KMeans clustering
        ### NOTE: before clustering, make sure each variable is normalized - NDVI is already normalized
        kmeans_input_array = ndvi_mon_med.stack(stack=('x','y'))
        kmeans_input_array = kmeans_input_array.transpose('stack','time').fillna(0).chunk(chunks={'stack':'auto','time':'auto'})
    
        ### set width and height
        w, h = len(ds_time_stack.coords['x']), len(ds_time_stack.coords['y'])
        sample_pct = sample_pct
        n_samples = int(w * h * sample_pct)
        n_clusters = n_clusters
        print('kmeans total samples:', n_samples)
        
        t0 = time.time()
        image_array_sample = shuffle(kmeans_input_array, random_state=0)[:n_samples]
        image_array_sample = image_array_sample.chunk(chunks={'stack':'auto','time':'auto'})
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_jobs = -1).fit(image_array_sample)
        print("kmeans clustering done in %0.3fs." % (time.time() - t0))
    
        ### build DA of kmeans clusters of monthly median NDVI time series
        cluster_centers_da = xr.DataArray(kmeans.cluster_centers_,
                                          dims=['n','cluster_vars'],
                                          coords={'cluster_vars':['ndvi_may','ndvi_jun','ndvi_jul','ndvi_aug','ndvi_sep','ndvi_oct']})
    
        t_start = time.perf_counter()
        t_proc_start = time.process_time()
        
        test_out_fp = out_dir_cluster_centers + out_file_name + '_kmeans_cluster_centers.nc'
        cluster_centers_da.to_netcdf(test_out_fp, mode='w', format='NETCDF4')
    
        ### STOP TIMING
        print("Writing Cluster Centers NetCDF file Total CPU time:", time.process_time() - t_proc_start)
        print("Writing Cluster Centers NetCDF file Total Wall time:", time.perf_counter() - t_start)



"""
-------------------------------------
Write processing log function
-------------------------------------

This function saves a text file log of parameters and processing time for the preprocessing step

"""

def preprocessing_log(tot_cpu_time, tot_wall_time):
    global config
    
    param_str = (f"Tile ID: {config['prep_tile_id']}\n"
             f"Output location: {config['preproc_out_dir']}\n"
             "-----------------\n"
             f"Total CPU time: {tot_cpu_time}\n"
             f"Total Wall time: {tot_wall_time}\n"
             "-----------------\n"
             "Parameters: \n"
             f"{config}")

    file_name = str(config['preproc_outfile_prefix']) + str(config['prep_tile_id']) + "_process_log.txt"
    if config['prep_manual_subset']:
        file_name = str(config['shp_out_dir']) + str(config['prep_tile_id']) + "_x" + str(config['prep_x_start']) + "_y" + str(config['prep_y_start']) + "_step" +str(config['prep_step']) + "_process_log.txt"
    
    with open(file_name, 'w') as f:
        print(param_str, file=f)
        
    





"""
-------------------------------------
Statewide K-means cluster mask process
-------------------------------------             
"""

def create_classified_image(labels, w, h):
    """Recreate the (compressed) image from the custer labels"""
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
#             image[i][j] = labels[label_idx]
            image[j][i] = labels[label_idx]
            label_idx += 1
    return image

def fit_kmeans_x_y_time_array_to_clustered_img(input_array, trained_kmeans):
    original_shape = tuple(input_array.transpose('x', 'y', 'time').shape)
    w, h, d = original_shape
    
    # This creates an array of shape 6 feature vectors length 250000
    image_array = input_array.stack(stack=('x','y'))
    # transpose to get a 2D array with x*y by time features, fill NaN values with 0 otherwise kmeans will fail
    image_array = image_array.transpose('stack','time').fillna(0)
    
    # fit input array to model to get labels for all points
    t0 = time.time()
    labels = trained_kmeans.predict(image_array)
    print("fitting data to kmeans model done in %0.3fs." % (time.time() - t0))
    
    # Reassemble clustered array to original image dimensions
    clustered_img = create_classified_image(labels, w, h)
    
    clustered_img_da = xr.DataArray(clustered_img,
                                      dims=['y','x'],
                                      coords={'x':input_array.coords['x'],
                                              'y':input_array.coords['y']})
    clustered_img_da = clustered_img_da.chunk({'x': 'auto', 'y': 'auto'}).astype(int)
    
    return clustered_img_da

def fit_kmeans_preproc_array_to_clustered_img_8_vars(input_array, trained_kmeans):
    original_shape = tuple(input_array.ndvi_mon_med.transpose('x', 'y', 'time').shape)
    w, h, d = original_shape
       
    ndvi_mon_med_stack = input_array.ndvi_mon_med.stack(stack=('x','y'))
    ndvi_mon_med_stack = ndvi_mon_med_stack.transpose('stack','time').fillna(0).rename({'time':'variable'})
    ndvi_mon_med_stack = ndvi_mon_med_stack.chunk(chunks={'stack':'auto','variable':'auto'})
    #ndvi_mon_med_stack = ndvi_mon_med_stack.transpose('stack','time').fillna(0)
#     print('ndvi_mon_med_stack', ndvi_mon_med_stack)

    to_cluster_ds = xr.Dataset(data_vars={'ndvi_std':input_array.ndvi_std, 
#                                          'edges':normalize_ufunc(preproc_ds.edges_ndvi_std), 
                                          'ndwi_mean':input_array.ndwi_mean_jul_aug_sep})
    to_cluster_ds = to_cluster_ds.to_array(name='nontime_vars').stack(stack=('x','y')).transpose('stack','variable').fillna(0)    
    
    #### Combining ndvi time series with other cluster vars
    image_array = xr.concat([ndvi_mon_med_stack,to_cluster_ds], dim='variable')
    image_array = image_array.chunk(chunks={'stack':'auto', 'variable':'auto'})
    
#     # This creates an array of shape 6 feature vectors length 250000
#     image_array = input_array.stack(stack=('x','y'))
#     # transpose to get a 2D array with x*y by time features, fill NaN values with 0 otherwise kmeans will fail
#     image_array = image_array.transpose('stack','time').fillna(0)
    
    # fit input array to model to get labels for all points
    t0 = time.time()
    labels = trained_kmeans.predict(image_array)
    print("done in %0.3fs." % (time.time() - t0))
    
    # Reassemble clustered array to original image dimensions
    clustered_img = create_classified_image(labels, w, h)
    
    clustered_img_da = xr.DataArray(clustered_img,
                                      dims=['y','x'],
                                      coords={'x':input_array.coords['x'],
                                              'y':input_array.coords['y']})
    clustered_img_da = clustered_img_da.chunk({'x': 'auto', 'y': 'auto'}).astype(int)
    
    return clustered_img_da

def plot_line_graph_cluster_centers(kmeans_cluster_centers, title_str):
    print('N Clusters =', len(kmeans_cluster_centers))
    tab20 = cm.get_cmap('tab20', len(kmeans_cluster_centers))
    
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    
    for idx, val in enumerate(kmeans_cluster_centers):
        ax.plot(val, label=idx, linewidth=2, c=tab20(idx))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(title="CLuster ID", fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.ylabel("Monthly Median NDVI")
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
    plt.grid(alpha=1, linewidth=0.5)
    plt.title(title_str)
    plt.show()


### Map clustered image
def plot_kmeans_clusters(clustered_img_input, title_str=None):
    plt.figure(figsize=(10,8))
    img = plt.imshow(clustered_img_input, cmap='tab20')
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.title(title_str)
    plt.show()

#### Functions for cleaning up the binary mask
def binary_closing(input_array):
    return ndi.binary_closing(input_array)

def binary_closing_2(input_array):
    return ndi.binary_closing(input_array, iterations=2)

def binary_closing_3(input_array):
    return ndi.binary_closing(input_array, iterations=3)

def binary_dilation(input_array):
    return ndi.binary_dilation(input_array)

def binary_dilation_2(input_array):
    return ndi.binary_dilation(input_array, iterations = 2)

def binary_fill_holes(input_array):
    return ndi.binary_fill_holes(input_array)

def binary_erosion_2(input_array):
    return ndi.binary_erosion(input_array, iterations = 2)

"""
Prepare KMeans model from statewide cluster samples

config parameters:
    - cluster_centers_dir
    - n_clusters
    - kmeans_model_out_dir
    - kmeans_file_name
"""

def write_statewide_kmeans_model():
    
    global config
    
    ### Define variables from config
    cluster_file_dir = config['preproc_out_dir'] + 'cluster_centers/'
    n_clusters = config['kmeans_n_clusters']
    kmeans_model_out_dir = config['kmeans_model_out_dir']
    kmeans_file_name = 'statewide_kmeans_' + str(n_clusters) + 'clusters.sav'
#     kmeans_file_name = kmeans_config['kmeans_file_name']
    
    # Load statewide clusters to DataArray
    cluster_file_dir_call = cluster_file_dir + '*.nc*'
    
    ### Open cluster center files into single xarray dataset
    cluster_merge = xr.open_mfdataset(cluster_file_dir_call, concat_dim = 'n', combine='nested', chunks={'n':'auto','cluster_vars':'auto'})
    cluster_merge = cluster_merge.rename_vars({'__xarray_dataarray_variable__':'clusters'})
    
    ### convert coordinate to datetime objects
    #keys = cluster_merge.coords['cluster_vars'].values
#    date_obj_list = []
#    for i in range(5, 11, 1):
#        date_obj_list.append(datetime.datetime(2019, i, 1))
#    cluster_merge = cluster_merge.assign_coords({'cluster_vars':date_obj_list})
    print(cluster_merge)
    
    # Fit model to cluster centers - SELECT K for statewide
    # fit a kmeans model on a sample of the data
    t0 = time.time()
    #image_array_sample = shuffle(image_array, random_state=0)[:n_samples]
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(cluster_merge.clusters)
    print("KMeans training on statewide samples done in %0.3fs." % (time.time() - t0))
    
    agg_clusters_da = xr.DataArray(kmeans.cluster_centers_,
                                  dims=['cluster_ID', 'cluster_vars'])
    agg_clusters_da = agg_clusters_da.assign_coords({'cluster_vars':cluster_merge.cluster_vars})
    print('Cluster Centers:', agg_clusters_da)
    print(agg_clusters_da.cluster_vars.values)
    
    ### Save KMeans model
    Path(kmeans_model_out_dir).mkdir(exist_ok=True)
    #kmeans_file_name = 'statewide_kmeans_' + str(n_clusters) + 'clusters.sav'
    kmeans_fp = kmeans_model_out_dir + kmeans_file_name
    pickle.dump(kmeans, open(kmeans_fp, 'wb'))
    
    ### Save statewide cluster centers for reference
    agg_clusters_da_out_fp = kmeans_model_out_dir + kmeans_file_name[:-4] + '_cluster_centers.nc'
    agg_clusters_da.to_netcdf(agg_clusters_da_out_fp, mode='w', format='NETCDF4')
    
'''
Create mask from Kmeans model

- load preprocessed data into merged tile
- Apply kmeans:
    - load kmeans model from preprocessing outputs
    - identify non-crop cluster labels based on temporal NDVI statistics
    - 
- 

config parameters:
    - preprocessed_data_dir
    - tile_id
    - std_thresh
    - min_thresh
    - max_thresh
    - range_thresh
    - kmeans_8var_clusters ### This is a boolean parameter to specify if the kmeans model includes only the 6 months of NDVI median, or 2 additional vars
    - kmeans_model_out_dir  ### already in config from first step
    - kmeans_file_name  ### already in config from first step
    - ndwi_thresh
    - mask_out_dir

'''

def create_mask_from_kmeans():
    global config
    
    # Load preprocessed imagery data into single dataset 
    preproc_tile_dir = config['preproc_out_dir']
    tile_id = config['prep_tile_id']
    preproc_fp_str = preproc_tile_dir + '*' + tile_id + '*.nc'
    preproc_merge_ds = xr.open_mfdataset(preproc_fp_str, chunks={'x':'auto', 'y':'auto'}, combine='by_coords')
    
    ### Apply kmeans
    # Load kmeans model
    kmeans_model_out_dir = config['kmeans_model_out_dir']
    n_clusters = config['kmeans_n_clusters']
    kmeans_file_name = 'statewide_kmeans_' + str(n_clusters) + 'clusters.sav'
    kmeans_fp = kmeans_model_out_dir + kmeans_file_name
    kmeans = pickle.load(open(kmeans_fp, 'rb'))
    
    # define crop and non-crop clusters
    std_thresh = config['kmeans_std_thresh']
    min_thresh = config['kmeans_min_thresh']
    max_thresh = config['kmeans_max_thresh']
    range_thresh = config['kmeans_range_thresh']
    kmeans_8var_clusters = config['kmeans_8var_clusters']
    
    print('Cluster Centers just NDVI Mon Med:', kmeans.cluster_centers_.T[:-2].T)
    
    non_crop_cluster_labels = []
    crop_cluster_labels = []
    print('label | ', 'standard deviation | ', 'minimum | ', 'maximum | ', 'range')
    
    if kmeans_8var_clusters:
        for idx, cluster_center in enumerate(kmeans.cluster_centers_.T[:-2].T):
            print(idx,
                  np.std(cluster_center), 
                  np.min(cluster_center), 
                  np.max(cluster_center), 
                  (np.max(cluster_center)-np.min(cluster_center)))
            if np.std(cluster_center) < std_thresh or np.min(cluster_center) > min_thresh or np.max(cluster_center) < max_thresh or (np.max(cluster_center)-np.min(cluster_center)) < range_thresh:
                non_crop_cluster_labels.append(idx)
            else:
                crop_cluster_labels.append(idx)
    
    if not kmeans_8var_clusters:
        for idx, cluster_center in enumerate(kmeans.cluster_centers_):
            print(idx, 
                  np.std(cluster_center), 
                  np.min(cluster_center), 
                  np.max(cluster_center), 
                  (np.max(cluster_center)-np.min(cluster_center)))
            if np.std(cluster_center) < std_thresh or np.min(cluster_center) > min_thresh or np.max(cluster_center) < max_thresh or (np.max(cluster_center)-np.min(cluster_center)) < range_thresh:
                non_crop_cluster_labels.append(idx)
            else:
                crop_cluster_labels.append(idx)
    
    print('non-crop cluster labels:', non_crop_cluster_labels)
    print('crop cluster labels:', crop_cluster_labels)
    
    # apply kmeans to imagery tile data
    clustered_img_output = fit_kmeans_preproc_array_to_clustered_img_8_vars(preproc_merge_ds, kmeans)
    
    ### Bin clusters into crop/non-crop binary mask
    cluster_mask = clustered_img_output.isin(non_crop_cluster_labels)
    
    ### NDWI mask
    ndwi_thresh = config['kmeans_ndwi_thresh']
    ndwi_mask = preproc_merge_ds.ndwi_mean_jul_aug_sep.where(preproc_merge_ds.ndwi_mean_jul_aug_sep > ndwi_thresh, 0)
    ndwi_mask = ndwi_mask.where(ndwi_mask <= ndwi_thresh, 1)
    
    ### clean up the NDWI mask for salt and pepper noise
    # invert binary ndwi mask to close small holes
    ndwi_mask = xr.where(ndwi_mask == 1, 0, 1)
    # Apply binary_closing algorithm with 3 iterations to remove small slivers
    ndwi_mask = xr.DataArray(ndwi_mask.data.map_overlap(binary_closing, depth = 1),
                                   dims=ndwi_mask.dims,
                                   coords=ndwi_mask.coords)
    # invert the mask back in order to dilate masked regions by 1 pixel
    ndwi_mask = xr.where(ndwi_mask == 1, 0, 1)
    # dilate water mask 
    ndwi_mask = xr.DataArray(ndwi_mask.data.map_overlap(binary_dilation, depth = 1),
                                   dims=ndwi_mask.dims,
                                   coords=ndwi_mask.coords)
    # Apply binary_closing algorithm again to remove small holes within water bodies
    ndwi_mask = xr.DataArray(ndwi_mask.data.map_overlap(binary_closing_3, depth = 1),
                                   dims=ndwi_mask.dims,
                                   coords=ndwi_mask.coords)
    #ndwi_mask[4500:5000,1500:2000].plot(figsize=(10,8))  ### plotting intermediate mask
    
    ### Combine cluster and NDWI masks
    mask_combined = xr.ufuncs.logical_or(cluster_mask, ndwi_mask)
    
    ### Mask clean up
    # Binary Closing 2
    mask_combined = xr.DataArray(mask_combined.data.map_overlap(binary_closing, depth = 1),
                                   dims=ndwi_mask.dims,
                                   coords=ndwi_mask.coords)
    # invert
    mask_combined = xr.where(mask_combined == 1, 0, 1)
    # fill holes
    mask_combined = xr.DataArray(mask_combined.data.map_overlap(binary_fill_holes, depth = 1),
                               dims=ndwi_mask.dims,
                               coords=ndwi_mask.coords)
    # erode mask
    mask_combined = xr.DataArray(mask_combined.data.map_overlap(binary_erosion_2, depth = 1),
                               dims=ndwi_mask.dims,
                               coords=ndwi_mask.coords)
    # dilate mask
    mask_combined = xr.DataArray(mask_combined.data.map_overlap(binary_dilation_2, depth = 1),
                               dims=ndwi_mask.dims,
                               coords=ndwi_mask.coords)
    
    #mask_combined[4500:5000,1500:2000].plot(figsize=(10,8)) ### plotting mask
    
    
    ### Save out mask
    mask_out_dir = config['kmeans_mask_out_dir']
    tile_id = config['prep_tile_id']
    mask_fp = mask_out_dir + str(tile_id) + '_mask.nc'
    mask_combined.name = 'mask'
    encoding = {'mask': {'dtype': bool, 'zlib': True, 'complevel': 9}}
    Path(mask_out_dir).mkdir(exist_ok=True)
    mask_combined.to_netcdf(mask_fp, mode='w', format='NETCDF4', encoding=encoding)
    

"""
Segmenting fields from K-Means clusters
"""

def load_mask():
    global config
    tile_id = config['prep_tile_id']
    mask_file_dir = config['kmeans_mask_out_dir']
    
    mask_fp = mask_file_dir + '*' + tile_id + '*.nc' 
    mask_file_list = glob.glob(mask_fp)
    print("there are {!s} patches for tile: {!s}".format(len(mask_file_list), tile_id))
    
    mask = xr.open_dataset(mask_file_list[0], chunks = {'x':'auto','y':'auto'})

    ### spatially subset the mask to match the ds_time_stack 
    if config['kmeans_from_full_tile_mask']:
        if config['prep_manual_subset']:
            x_start = config['prep_x_start']
            y_start = config['prep_y_start']
            step = config['prep_step']
    
            mask = mask.mask[x_start:x_start+step, y_start:y_start+step]
    
    ### Mask files for map v6 are already ivnerted correctly
    #mask = np.invert(mask)
    
    print(mask)
    return mask

def prep_nir_dates_for_rgb_image():
    # set variables from global config
    global config

    file_dir = config['prep_file_dir']
    tile_id = config['prep_tile_id']
    # set working directory to the folder with the sentinel data tiles
    # Set file_dir in the global variables at top of the code
#    os.chdir(file_dir)
    glob_str = "*" + tile_id + "*.zip"
    zip_list = glob.glob(file_dir + glob_str)
    
    # build dictionary of metadata from gdal_info calls for each tile in the list of zip_files
    tile_info_dict = {}
    for zip_str in zip_list:
        tile_info_dict.update(xr_get_zip_info_to_dict(zip_str))
    
    # Set date values for selecting imagery
    may_31 = datetime.datetime.strptime('20190531', '%Y%m%d')
    oct_1 = datetime.datetime.strptime('20191001', '%Y%m%d')
    date_1 = datetime.datetime.strptime('20190715', '%Y%m%d')
    date_2 = datetime.datetime.strptime('20190815', '%Y%m%d')
    date_3 = datetime.datetime.strptime('20190915', '%Y%m%d')
    
    date_list = []
    cloud_coverage_list = []
    nodata_pct_list = []
    
    rgb_cloud_thresh = 5
    rgb_nodata_thresh = 0.5
    
    while len(date_list) < 3:
        for k, v in tile_info_dict.items():
            if v['cloud_coverage'] < rgb_cloud_thresh:
                if v['nodata_pixel_percentage'] < rgb_nodata_thresh:
                    if v['date_obj'] > may_31 and v['date_obj'] < oct_1:
                        if v['date_obj'] not in date_list:
                            date_list.append(v['date_obj'])
                            cloud_coverage_list.append(v['cloud_coverage'])
                            nodata_pct_list.append(v['nodata_pixel_percentage'])
        rgb_cloud_thresh += 1
        rgb_nodata_thresh += 0.5
       
    print('date list:', date_list)
    print('cloud coverage:', cloud_coverage_list)
    print('no data pct:', nodata_pct_list)
   
    cloud_da = xr.DataArray(cloud_coverage_list,
                            coords=[date_list],
                            dims=['time'],
                            name='cloud_coverage')
            
    nodata_da = xr.DataArray(nodata_pct_list,
                            coords=[date_list],
                            dims=['time'],
                            name='nodata_pct')
    
    data_ds = xr.merge([cloud_da, nodata_da]).sortby('time')
    
    data_ds_nonan= data_ds.dropna(dim='time') 
    
    test_date_1 = data_ds_nonan.sel(time=date_1, method='nearest').time.values
    data_ds_nonan = data_ds_nonan.where(data_ds_nonan.time != test_date_1).dropna(dim='time') 
    test_date_2 = data_ds_nonan.sel(time=date_2, method='nearest').time.values
    data_ds_nonan = data_ds_nonan.where(data_ds_nonan.time != test_date_2).dropna(dim='time') 
    test_date_3 = data_ds_nonan.sel(time=date_3, method='nearest').time.values
    nir_date_list = [test_date_1, test_date_2, test_date_3]
    print(nir_date_list)
    
    return nir_date_list


def segment_data(ds_time_stack, mask):
    print("-------Segmentation-------")
    ### Prep RGB
    rgb_date_str = config['seg_rgb_date_str']
    gaussian_filt = config['seg_rgb_gaussian_filt'] # True
    gaussian_sigma = config['seg_rgb_gaussian_sigma'] # 2
    percentile = config['seg_rgb_percentile'] # 2
    # chunking
    base_chunk = config['prep_base_chunk']
    chunk_size = {'band': "auto", 'x': base_chunk, 'y': base_chunk}
    # segmentation parameters
    seg_scale = config['seg_fz_scale']
    seg_sigma = config['seg_fz_sigma']
    seg_min_size = config['seg_fz_min_size']
    
    use_nir = config['seg_use_nir']
    
    if not use_nir:
        rgb_date = datetime.datetime.strptime(rgb_date_str, '%Y%m%d')
        ### Build RGB xr DataArray to be passed to segmentation and merge functions
        ### Percentile is the amount of the rgb image to clip from either end of the histogram
        # Combine bands into RGB array and rescale the intensity so that image has better contrast for segmentation
        rgb_da = clip_nan_ufunc(xr.concat([ds_time_stack.red.sel(time=rgb_date, method='nearest'),
                                           ds_time_stack.green.sel(time=rgb_date, method='nearest'),
                                           ds_time_stack.blue.sel(time=rgb_date, method='nearest')], dim='band'), percentile=percentile)
    
    if use_nir:
        nir_date_list = prep_nir_dates_for_rgb_image()
        nir_date_1 = nir_date_list[0]
        nir_date_2 = nir_date_list[1]
        nir_date_3 = nir_date_list[2]
        ### Build RGB xr DataArray to be passed to segmentation and merge functions
        ### Percentile is the amount of the rgb image to clip from either end of the histogram
        # Combine bands into RGB array and rescale the intensity so that image has better contrast for segmentation
        rgb_da = clip_nan_ufunc(xr.concat([ds_time_stack.nir.sel(time=nir_date_1, method='nearest'),
                                           ds_time_stack.nir.sel(time=nir_date_2, method='nearest'),
                                           ds_time_stack.nir.sel(time=nir_date_3, method='nearest')], dim='band'), percentile=percentile)
        
    rgb_da = rgb_da.assign_coords(band=['r','g','b'])
    rgb_da = rgb_da.chunk(chunks=chunk_size)
    
    ### Transpose to (y, x, band) in order to pass to 
    rgb_da = rgb_da.transpose('y', 'x', 'band')
    
    ### Apply gaussian filter to rgb image
    if gaussian_filt == True:
        dims = rgb_da.dims
        coords = rgb_da.coords
        rgb_da = xr.DataArray(rgb_da.data.map_overlap(gaussian, sigma = gaussian_sigma, multichannel = True, depth = 1),
                                     dims=dims,
                                     coords=coords)
    
    ### Apply mask to rgb image
    rgb_da = rgb_da.where(mask==1, other=0).fillna(0)
    rgb_da = rgb_da.mask
    rgb_da.name = 'masked_rgb'

    ### FIX ME : figure out how to run segmentation in parallel??? I think the skimage module only takes np arrays
    return segmentation.felzenszwalb(rgb_da, scale = seg_scale, sigma= seg_sigma, min_size= seg_min_size), rgb_da


"""
-------------------------------------
Write to shapefile
-------------------------------------
"""

def write_shapefile(segmented_array, ds_time_stack, mask_input):
    '''
    This function takes an array (meant for a raster that has already been segmented)
    and writes the polygonized raster to a shapefile.

    input_array: raster to by polygonized and exported
    src_transform: the "transform" spatial metadata from the rasterio.read() of the source raster
    src_crs: the coordinate reference system from the source raster, as a string. ex: 'EPSG:32614'
    output_file: file path/name ending with '.shp' for the output
    '''
    print("-------Writing to Shapefile-------")
    tile_id = config['prep_tile_id']
    out_dir = config['shp_out_dir']
    file_out_str = config['shp_file_out_str']
    manual_subset = config['prep_manual_subset']
    x_start = config['prep_x_start']
    y_start = config['prep_y_start']
    
        ### Save to shapefile
    # set file paths for output
    out_file_name = tile_id + file_out_str + ".shp"
    output_str = out_dir + out_file_name

    # set parameters for spatial projection and mask    
    transform = ds_time_stack.attrs['transform']
    ### The transform will be incorrect if the raster has been subset from the top or left at all
    if manual_subset:
        transform_update = (ds_time_stack.attrs['transform'][0],
                            ds_time_stack.attrs['transform'][1],
                            ds_time_stack.coords['x'].values[0]-5,
                            ds_time_stack.attrs['transform'][3],
                            ds_time_stack.attrs['transform'][4],
                            ds_time_stack.coords['y'].values[0]+5)
        transform  = transform_update
        # Update outfile name to reflect spatial subset
        out_file_name = tile_id + file_out_str + "_x" + str(x_start) + "_y" + str(y_start) + "_step" + str(config['prep_step']) + ".shp"
        output_str = out_dir + out_file_name
    
    # CRS from main data stack attributes
    crs = ds_time_stack.attrs['crs']
    # Define mask and set type to bool as required for shapefile writing to work
    ### QUESTION : Data Type, should this data type be changed in the mask processing or does it not really make a difference? 
    if type(mask_input) == xr.core.dataset.Dataset:
        mask_input = mask_input.mask
    
    shp_mask = mask_input.fillna(value=0).astype(bool)
    
    # polygonize input raster to GeoJSON-like dictionary with rasterio.features.shapes
    # update segmented_array data type based on error message: "image dtype must be one of: int16, int32, uint8, uint16, float32"   
    results = ({'geometry': s, 'properties': {'raster_val': v}}
          for i, (s, v) in enumerate(features.shapes(segmented_array.astype('int16'), mask=shp_mask.values, transform = transform)))
    geoms = list(results)
    
    # establish schema to write into shapefile
    schema_template = {
    'geometry':'Polygon', 
    'properties': {
        'raster_val':'int'}}
    
    Path(out_dir).mkdir(exist_ok=True)
    
    # src_crs is the coordinate reference system from the source raster that holds the spatial metadata
    with fiona.open(output_str, 'w', driver='Shapefile', schema = schema_template, crs = crs) as layer:
        # loop through the list of raster polygons and write to the shapefile
        for geom in geoms:
            layer.write(geom)
    
    print("Output saved to:", output_str)


"""
Thresholding approach combined functions
"""

def process_all(file_dir, tile_id, out_dir, file_out_str,
                overlap_bool = False, manual_subset = True,
                x_start = 0, y_start = 0, step = 1000,
                base_chunk="auto", cloud_coverage_thresh = 15,
                ndwi_thresh = 0.5, ndvi_max_thresh = 0.3, ndvi_range_thresh = 0.3, edges_thresh = 0.3,
                rgb_date_str = '20190803', rgb_gaussian_sigma = 2, 
                fz_scale = 200, fz_sigma = .5, fz_min_size = 400):
    
    ### START TIMING
    t_start = time.perf_counter()
    t_proc_start = time.process_time()
    
    ### put code for timing here
    
    ### Load tile data
    ds_time_stack = prep_ds_time_stack(file_dir, 
                                       tile_id, 
                                       cloud_coverage_thresh = cloud_coverage_thresh,
                                       cloud_mask = False,
                                       base_chunk = base_chunk, 
                                       overlap_bool = overlap_bool, 
                                       manual_subset = manual_subset, x_start = x_start, y_start = y_start, step = step)
    
    ### Preprocessing to compute crop mask
    ds_time_stack = mask_processing(ds_time_stack, 
                                    ndwi_thresh = ndwi_thresh, 
                                    ndvi_max_thresh = ndvi_max_thresh, 
                                    ndvi_range_thresh = ndvi_range_thresh, 
                                    edges_thresh = edges_thresh)
    
    ### Prep masked rgb image for segmentation
    ds_time_stack['rgb'] = rgb_image(rgb_date_str = rgb_date_str, 
                                              ds_time_stack = ds_time_stack, 
                                              rgb_chunk_size = {'band': "auto", 'x': base_chunk, 'y': base_chunk}, 
                                              gaussian_filt = True, 
                                              gaussian_sigma = rgb_gaussian_sigma, 
                                              percentile = 1)
    
    ### Segment the RGB image
    fz_segments = ds_time_stack.rgb.data.map_blocks(segment_fz, scale = fz_scale, sigma = fz_sigma, min_size = fz_min_size, drop_axis = 2)
    ds_time_stack['fz_segments'] = xr.DataArray(fz_segments.astype('int32'), 
                                                dims=('y','x'),
                                                coords = [ds_time_stack['red'].coords['y'],ds_time_stack['red'].coords['x']])
    
    print("fz_segments:", ds_time_stack['fz_segments'].dtype)
    
    ### Save to shapefile
    # set file paths for output
    out_file_name = tile_id + file_out_str + ".shp"
    output_str = out_dir + out_file_name
    
    ### The transform will be incorrect if the raster has been subset from the top or left at all
    transform_update = (ds_time_stack.attrs['transform'][0],
                        ds_time_stack.attrs['transform'][1],
                        ds_time_stack.coords['x'].values[0]-5,
                        ds_time_stack.attrs['transform'][3],
                        ds_time_stack.attrs['transform'][4],
                        ds_time_stack.coords['y'].values[0]+5)
    
    transform  = transform_update
    
    # set parameters for spatial projection and mask
#     transform  = ds_time_stack.attrs['transform']
    crs = ds_time_stack.attrs['crs']

    ds_time_stack['mask'] = ds_time_stack['mask'].fillna(value=0).astype(bool)
    shp_mask = ds_time_stack['mask'].values
#    print("shp_mask:", shp_mask.dtype)

    # write out labels to shapefile
    write_segments_to_shapefile_xr(ds_time_stack.fz_segments.values, transform, crs, output_str, mask = shp_mask)

    t_stop = time.perf_counter()
    t_proc_stop = time.process_time()
    tot_cpu_time = t_proc_stop - t_proc_start
    tot_wall_time = t_stop - t_start
    ### END TIMING
    print("Total CPU time:", t_proc_stop - t_proc_start)
    print("Total Wall time:", t_stop - t_start)

    param_str = (f"Tile ID: {tile_id}\n"
                 f"Output location: {out_file_name}\n"
                 "-----------------\n"
                 f"Total CPU time: {tot_cpu_time}\n"
                 f"Total Wall time: {tot_wall_time}\n"
                 "-----------------\n"
                 "Parameters: \n"
                 f"Chunk size (x,y): {base_chunk}\n"
                 f"Cloud Coverage Threshold: {cloud_coverage_thresh}\n"
                 f"Mask Weights - NDWI: {ndwi_thresh}, NDVI Max: {ndvi_max_thresh}, NDVI range: {ndvi_range_thresh}, Edges: {edges_thresh}\n"
                 f"RGB Segmentation Imagery Date: {rgb_date_str}\n"
                 f"RGB Preprocessing gaussian sigma: {rgb_gaussian_sigma}\n"
                 f"Felzenszwalb Parameters - Scale: {fz_scale}, Sigma: {fz_sigma}, Min Size: {fz_min_size}\n")
    
    test_file_name = out_dir + tile_id + "_process_log.txt"
    
    with open(test_file_name, 'w') as f:
        print(param_str, file=f)


def process_all_no_data_read(input_ds, tile_id, out_dir, file_out_str, cloud_coverage_thresh, 
                             subset = False, x_start = 0, y_start = 0, step = 2000, base_chunk="auto", 
                             ndwi_thresh = 0.5, ndvi_max_thresh = 0.3, ndvi_range_thresh = 0.3, edges_thresh = 0.3,
                             rgb_date_str = '20190803', rgb_gaussian_sigma = 2, 
                             fz_scale = 200, fz_sigma = .5, fz_min_size = 400):
    
    ### START TIMING
    t_start = time.perf_counter()
    t_proc_start = time.process_time()
    
    ds_time_stack = input_ds.sel(time=slice('2019-05-01', '2019-10-01'))
    
    ### put code for timing here
    if subset == True:
        x_0 = x_start
        x_1 = x_start + step
        y_0 = y_start
        y_1 = y_start + step
        
        ds_time_stack = ds_time_stack.isel(x=slice(x_0,x_1), y=slice(y_0,y_1))
    
    ds_time_stack = ds_time_stack.chunk(chunks=({'time':'auto', 'x':'auto', 'y':'auto'}))
    
    ### Preprocessing to compute crop mask
    ds_time_stack = mask_processing(ds_time_stack, 
                                    ndwi_thresh = ndwi_thresh, 
                                    ndvi_max_thresh = ndvi_max_thresh, 
                                    ndvi_range_thresh = ndvi_range_thresh, 
                                    edges_thresh = edges_thresh)
    
    ### Prep masked rgb image for segmentation
    ds_time_stack['rgb'] = rgb_image(rgb_date_str = rgb_date_str, 
                                              ds_time_stack = ds_time_stack, 
                                              rgb_chunk_size = {'band': "auto", 'x': base_chunk, 'y': base_chunk}, 
                                              gaussian_filt = True, 
                                              gaussian_sigma = rgb_gaussian_sigma, 
                                              percentile = 1)
    
    ### Segment the RGB image
    fz_segments = ds_time_stack.rgb.data.map_blocks(segment_fz, scale = fz_scale, sigma = fz_sigma, min_size = fz_min_size, drop_axis = 2)
    ds_time_stack['fz_segments'] = xr.DataArray(fz_segments.astype('int32'), 
                                                dims=('y','x'),
                                                coords = [ds_time_stack['red'].coords['y'],ds_time_stack['red'].coords['x']])
    
    
    ### Save to shapefile
    # set file paths for output
    out_file_name = tile_id + file_out_str + "_x" + str(x_start) + "_y" + str(y_start) + ".shp"
    output_str = out_dir + out_file_name
    print("output file:", output_str)
    
    ### The transform will be incorrect if the raster has been subset from the top or left at all
    if subset == True:
        transform_update = (ds_time_stack.attrs['transform'][0],
                            ds_time_stack.attrs['transform'][1],
                            ds_time_stack.coords['x'].values[0]-5,
                            ds_time_stack.attrs['transform'][3],
                            ds_time_stack.attrs['transform'][4],
                            ds_time_stack.coords['y'].values[0]+5)

        transform  = transform_update

    # set parameters for spatial projection and mask
    if subset == False:
        transform  = ds_time_stack.attrs['transform']
    
    crs = ds_time_stack.attrs['crs']

    ds_time_stack['mask'] = ds_time_stack['mask'].fillna(value=0).astype(bool)
    shp_mask = ds_time_stack['mask'].values
    print("shp_mask:", shp_mask.dtype)

    # write out labels to shapefile
    write_segments_to_shapefile_xr(ds_time_stack.fz_segments.values, transform, crs, output_str, mask = shp_mask)

    t_stop = time.perf_counter()
    t_proc_stop = time.process_time()
    tot_cpu_time = t_proc_stop - t_proc_start
    tot_wall_time = t_stop - t_start
    ### END TIMING
    print("Total CPU time:", t_proc_stop - t_proc_start)
    print("Total Wall time:", t_stop - t_start)

    param_str = (f"Tile ID: {tile_id}\n"
                 f"Output location: {out_file_name}\n"
                 "-----------------\n"
                 f"Total CPU time: {tot_cpu_time}\n"
                 f"Total Wall time: {tot_wall_time}\n"
                 "-----------------\n"
                 "Parameters: \n"
                 f"Chunk size (x,y): {base_chunk}\n"
                 f"Cloud Coverage Threshold: {cloud_coverage_thresh}\n"
                 f"Mask Weights - NDWI: {ndwi_thresh}, NDVI Max: {ndvi_max_thresh}, NDVI range: {ndvi_range_thresh}, Edges: {edges_thresh}\n"
                 f"RGB Segmentation Imagery Date: {rgb_date_str}\n"
                 f"RGB Preprocessing gaussian sigma: {rgb_gaussian_sigma}\n"
                 f"Felzenszwalb Parameters - Scale: {fz_scale}, Sigma: {fz_sigma}, Min Size: {fz_min_size}\n")
    
    test_file_name = out_dir + tile_id + "_x" + str(x_start) + "_y" + str(y_start) + "_process_log.txt"
    
    with open(test_file_name, 'w') as f:
        print(param_str, file=f)




"""
Post-processing functions
"""

### Shapefile cleanup

# read a shapefile filepath string to a geodataframe
def shapefile_to_gpd_df(shp_fp, bbox=None):
    """
    read a shapefile filepath string to a geodataframe
    """
    gpd_df = gpd.read_file(shp_fp, bbox = bbox)
    return gpd_df

# convert both shp to same crs, in this case 'epsg:32614' WGS 84 / UTM zone 14N
def convert_crs(input_gpd_df, crs_dest = {'init': 'epsg:32614'}):
    """
    Change the coordinate reference system of a GeoDataFrame.
    By default it will change it to 'epsg:32614' WGS 84 / UTM zone 14N
    in order to calculate area accurately for regions in Minnesota.
    """
    if input_gpd_df.crs == crs_dest:
        output_gpd_df = input_gpd_df.copy()
    if input_gpd_df.crs != crs_dest:
        output_gpd_df = input_gpd_df.to_crs(crs_dest)        
    return output_gpd_df

def field_area_sqmeters(input_gpd_df):
    """
    Returns the input geopandas dataframe with area of each feature in sq m
    """
    # check crs
    print(input_gpd_df.crs)
    
#    input_copy = input_gpd_df.copy()
##    input_copy['area_sqkm'] = input_copy['geometry'].area/10**6 # in sq km
#    input_copy['area_sq_m'] = input_copy['geometry'].area # in sq m
#    return input_copy    
    
    input_gpd_df['area_sq_m'] = input_gpd_df['geometry'].area # in sq m
    return input_gpd_df



def min_polygon_size_filter(input_gpd_df, min_area_sq_m=1000):
    """
    Returns a copy of the input with only features larger than the minimum area.
    This is to filter out very small features.
    """
    input_feature_count = input_gpd_df.count()[0]
    df_trim = input_gpd_df[input_gpd_df['area_sq_m'] > min_area_sq_m]
    print("Reduced total field features from:", input_feature_count, "to", df_trim.count()[0])
    return df_trim

def simplify_polygons(input_gpd_df, tolerance = 10):
    input_gpd_df['geometry'] = input_gpd_df['geometry'].simplify(tolerance)
    return input_gpd_df

def apply_convex_hull(input_gpd_df):
    input_gpd_df['geometry'] = input_gpd_df['geometry'].convex_hull
    return input_gpd_df

def prep_shp_for_merge(shp_filepath, crs_dest = {'init': 'epsg:32614'}, min_area_sq_m = 1000, tolerance = 10):
    """
    Combines shapefile prep steps into single function
    """
    prepped_df = simplify_polygons(min_polygon_size_filter(field_area_sqmeters(convert_crs(shapefile_to_gpd_df(shp_filepath),crs_dest)), min_area_sq_m), tolerance)
    return prepped_df

def prep_shp_for_merge_no_crs_check(shp_filepath, min_area_sq_m = 1000, tolerance = 10):
    """
    Combines shapefile prep steps into single function
    """
    print('Prep shapefile WITH simplify')
    prepped_df = simplify_polygons(min_polygon_size_filter(field_area_sqmeters(shapefile_to_gpd_df(shp_filepath)), min_area_sq_m), tolerance)
    return prepped_df

def prep_shp_for_merge_no_simplify(shp_filepath, min_area_sq_m = 1000):
    """
    Combines shapefile prep steps into single function
    """        
    print('Prep shapefile WITHOUT simplify')
    prepped_df = min_polygon_size_filter(field_area_sqmeters(shapefile_to_gpd_df(shp_filepath)), min_area_sq_m)
    return prepped_df

### Fixing topology error, ref: https://stackoverflow.com/questions/49099049/geopandas-shapely-spatial-difference-topologyexception
def around(coords, precision=5):
    result = []
    try:
        return round(coords, precision)
    except TypeError:
        for coord in coords:
            result.append(around(coord, precision))
    return result


def layer_precision(geometry, precision=5):
    geojson = mapping(geometry)
    geojson['coordinates'] = around(geojson['coordinates'],precision)
    return shape(geojson)

def merge_batch_processing(input_tile_list, input_dir, out_dir, min_area_sq_m, tolerance):
    for row_list in input_tile_list:
        for tile_id in row_list:
            tiles = glob.glob(input_dir + "/" + tile_id + "*.shp")
            print("There are", len(tiles), "chunks for tile:", tile_id)
            if len(tiles) == 0:
                continue
            
            outname = out_dir + tile_id + "_merged.shp"
            
            ### get crs for input, this assumes all shapefiles in the list have the same crs
            crs = shapefile_to_gpd_df(tiles[0]).crs
            
            ### START TIMING
            t_start = time.perf_counter()
            t_proc_start = time.process_time()
            
            ### put code for timing here          
            gdf = pd.concat([prep_shp_for_merge_no_crs_check(shp, min_area_sq_m = min_area_sq_m, tolerance = tolerance) for shp in tiles]).pipe(gpd.GeoDataFrame)
            
            ### Set crs for output
            gdf.crs = crs
            
            t_stop = time.perf_counter()
            t_proc_stop = time.process_time()
            ### END TIMING
            print("CPU time:", t_proc_stop - t_proc_start)
            print("Wall time:", t_stop - t_start)

            
            print(outname)
            gdf.to_file(outname)

### This is run after the chunks of each tile have been cleaned up and merged into a single shp for each tile
### This function takes the T14 or T15 list as an input and merges all the cleaned tiles into a single shp for each UTM zone            
def combine_merged_tiles(input_tile_list, input_dir, out_dir, out_name):    
    # shapefile path list
    shp_path_list = []
    # blank df list
    df_list = []
    
    # add filepath for each merged shp to list to the shp_path_list
    for row_list in input_tile_list:
        for tile_id in row_list:
            shp_paths = glob.glob(input_dir + "/" + tile_id + "*.shp")
            print("There are", len(shp_paths), ".shp for tile:", tile_id)
            
            for shp_path in shp_paths:
                shp_path_list.append(shp_path)
    
    # open each shapefile in the shp_path_list to df and append to a df_list
    for shp_fp in shp_path_list:    
        df = shapefile_to_gpd_df(shp_fp, bbox=None)
        df_list.append(df)
    
    # Concatenate dfs in the df_list into single geodataframe
    print("merging:", len(df_list), "tiles")
    rdf = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=df_list[0].crs)      
    
    # define outfile for merged shapefile
    out_file = out_dir + out_name
    print("saving to:", out_file)
    rdf.to_file(out_file)            

### Updated merge to statewide shapefile function to process tiles from the two different UTM zones: T14 and T15
def merge_tiles_to_statewide_clipped_shp():
    # set global vars
    global config
    home_dir = config['merge_home_dir']
    input_dir = config['merge_shp_input_dir']
    mn_bdry_utm_zones_fp = config['merge_mn_bdry_utm_zones_fp']
    min_area_sq_m = config['merge_min_area_sq_m']
    tolerance = config['merge_simplify_tolerance']
    out_file = config['merge_statewide_outfile']
    merge_simplify_bool = config['merge_simplify_bool']
    
    # Minnesota tiles in UTM Zone T14
    T14 = ['UPV','UQV','UPU','UQU','TPT','TQT','TPS','TQS','TPR','TQR','TPQ','TQQ','TPP','TQP']
    # Minnesota tiles in UTM Zone T15
    T15 = ['UUQ','UUP','UVP','UWP','UXP','UYP','TUN','TVN','TWN','TXN','TUM','TVM','TWM','TUL','TVL','TWL','TUK','TVK','TWK','TUJ','TVJ','TWJ','TXJ']
    
    # set file paths
    fields_lib_dir = home_dir
    os.chdir(fields_lib_dir)
    print(os.getcwd())
    
    crs_utm14 = {'init': 'epsg:32614'}
    crs_utm15 = {'init': 'epsg:32615'}
    
    # Load reference shapefiles for clipping
    mn_bdry_utm_zones_df = shapefile_to_gpd_df(mn_bdry_utm_zones_fp).to_crs(crs=crs_utm14)
    mn_bdry_utm_zones_df = mn_bdry_utm_zones_df.drop(columns=['FID_cb_201', 'STATEFP', 'STATENS', 'AFFGEOID', 'GEOID', 'STUSPS',
           'NAME', 'LSAD', 'ALAND', 'AWATER', 'FID_MN_til', 'FID_MN_t_1',
           'Shape_Leng', 'Shape_Area'])
    print('MN UTM Zone crs:',mn_bdry_utm_zones_df.crs)
    
    # get all T14 tiles into a single df
    df_list_t14 = []
    for tile_id in T14:
        tiles = glob.glob(input_dir + "/" + tile_id + "*.shp")
        print("There are", len(tiles), "chunks for tile:", tile_id)
        if len(tiles) == 0:
            continue
        ### get crs for input, this assumes all shapefiles in the list have the same crs
        crs = crs_utm14
        ### Merge with or without simplifying polygons          
        if merge_simplify_bool:
            gdf = pd.concat([prep_shp_for_merge_no_crs_check(shp, min_area_sq_m = min_area_sq_m, tolerance = tolerance) for shp in tiles]).pipe(gpd.GeoDataFrame)
        if not merge_simplify_bool:
            gdf = pd.concat([prep_shp_for_merge_no_simplify(shp, min_area_sq_m = min_area_sq_m) for shp in tiles]).pipe(gpd.GeoDataFrame)        
        ### Set crs for output
        gdf.crs = crs
        ### Append to df list to be concatenated into single df
        df_list_t14.append(gdf)
    print('count of tiles in T14 list:', len(df_list_t14))
    
    # clip to T14 non-overlapping footprint
    t14_df = gpd.GeoDataFrame(pd.concat(df_list_t14, ignore_index=True), crs=crs_utm14)
    t14_df = gpd.overlay(t14_df, mn_bdry_utm_zones_df.loc[mn_bdry_utm_zones_df['area_label']=='T14'], how='intersection')
    
    # prep T15 shapefile for clipping
    t15_footprint_df = mn_bdry_utm_zones_df[mn_bdry_utm_zones_df['area_label'] != 'T14'].copy()
    t15_footprint_df['area_label'] = 'T15'
    t15_footprint_df = t15_footprint_df.dissolve(by='area_label')
    t15_footprint_df = t15_footprint_df.to_crs(crs=crs_utm15)
    t15_footprint_df['area_label'] = 'T15'
    
    df_list_t15 = []
    for tile_id in T15:
        tiles = glob.glob(input_dir + "/" + tile_id + "*.shp")
        print("There are", len(tiles), "chunks for tile:", tile_id)
        if len(tiles) == 0:
            continue
        ### get crs for input, this assumes all shapefiles in the list have the same crs
        crs = crs_utm15
        ### Merge with or without simplifying polygons          
        if merge_simplify_bool:
            gdf = pd.concat([prep_shp_for_merge_no_crs_check(shp, min_area_sq_m = min_area_sq_m, tolerance = tolerance) for shp in tiles]).pipe(gpd.GeoDataFrame)
        if not merge_simplify_bool:
            gdf = pd.concat([prep_shp_for_merge_no_simplify(shp, min_area_sq_m = min_area_sq_m) for shp in tiles]).pipe(gpd.GeoDataFrame)            ### Set crs for output
        gdf.crs = crs
        ### Append to df list to be concatenated into single df
        df_list_t15.append(gdf)
    print('count of tiles in T15 list:', len(df_list_t15))
        
    # join into single gdf
    print('merging T15 tiles')
    t15_df = gpd.GeoDataFrame(pd.concat(df_list_t15, ignore_index=True), crs=crs_utm15)
    # clip to T15 non-overlapping footprint
    t15_df = gpd.overlay(t15_df, t15_footprint_df, how='intersection')
    
    print('joining t14 and t15')
    ### Combine the two UTM zone field output dataframes into a single df
    fields_df = gpd.GeoDataFrame(pd.concat([t14_df.to_crs(crs=t15_df.crs), t15_df], ignore_index=True), crs=crs_utm15)
    
    # Save clipped & combined file out
    print("saving to:", out_file)
    fields_df.to_file(out_file)

