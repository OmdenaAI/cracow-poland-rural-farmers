# -*- coding: utf-8 -*-
"""
@author: jesse bakker
"""

### global config parameters

config = {
        ### Data prep parameters
          'prep_file_dir': '/home/prep_file_dir/', 
          'prep_tile_id': 'UEB',
          'prep_base_chunk': 'auto',
          'prep_time_chunk': 'auto',
          'prep_remove_overlap': False,
          'prep_manual_subset': True,
          'prep_x_start': 7500,
          'prep_y_start': 7500,
          'prep_step': 500,
          'prep_cloud_coverage_thresh': 50,
          'prep_load_cloud_mask': True,
          'prep_apply_cloud_mask': True,
          'prep_cloud_mask_thresh': 70,
          'prep_clip_outliers': True,
          'prep_clip_percentile': 1,
          'prep_normalize_bands': True,
          'username': '...',
          'password':'...',
        ### Data preprocessing parameters for mask creation
          'preproc_out_dir': '/home/preproc_out_dir/',
          'preproc_outfile_prefix':'fields_preproc_demo_',
          'preproc_sample_pct':0.05,
          'preproc_n_clusters':15,
          'preproc_cluster_tile':True,
        ### Kmeans Clustering mask processing parameters
          'kmeans_n_clusters': 15,
          'kmeans_model_out_dir': 'kmeans_model_dir/',
          'kmeans_8var_clusters':True,
          'kmeans_std_thresh':0.2,
          'kmeans_min_thresh':0,
          'kmeans_max_thresh':0.3,
          'kmeans_range_thresh':0.7,
          'kmeans_ndwi_thresh':0.2,
          'kmeans_mask_out_dir':'mask_out_dir/',
          'kmeans_from_full_tile_mask':False,   ### this is if you are only running a subset area during mask processing
        ### Segmentation parameters
          'seg_rgb_date_str': '20221016',
          'seg_rgb_gaussian_filt': False,
          'seg_rgb_gaussian_sigma': 1,
          'seg_rgb_percentile': 1,
          'seg_use_nir': True,
          'seg_fz_scale': 200,
          'seg_fz_sigma': 0.5,
          'seg_fz_min_size': 400,
        ### Shapefile save out parameters
          'shp_out_dir':'shp_dir/',
          'shp_file_out_str':'_code_demo_clustering'}


### Dictionary with date strings for the most cloud-free, least no-data pixel tile for segmentation
rgb_date_dict = {'UPV':'20190819',
                 'UQV':'20190707',
                 'UPU':'20190819',
                 'UQU':'20190816',
                 'TPT':'20190806',
                 'TQT':'20190806',
                 'TPS':'20190816',
                 'TQS':'20190816',
                 'TPR':'20190816',
                 'TQR':'20190816',
                 'TPQ':'20190905',
                 'TQQ':'20190905',
                 'TPP':'20190905',
                 'TQP':'20190808',
                 'UUQ':'20190707',
                 'UUP':'20190702',
                 'UVP':'20190823',
                 'UWP':'20190724',
                 'UXP':'20190731',
                 'UYP':'20190611',
                 'TUN':'20190816',
                 'TVN':'20190823',
                 'TWN':'20190823',
                 'TXN':'20190706',
                 'TUM':'20190828',
                 'TVM':'20190803',
                 'TWM':'20190731',
                 'TUL':'20190828',
                 'TVL':'20190828',
                 'TWL':'20190830',
                 'TUK':'20190808',
                 'TVK':'20190808',
                 'TWK':'20190830',
                 'TUJ':'20190808',
                 'TVJ':'20190808',
                 'TWJ':'20190711',
                 'TXJ':'20190711'}