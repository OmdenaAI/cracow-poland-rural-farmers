from shapely.geometry import shape
from shapely.geometry import Point, Polygon
import simplekml
import geopandas as gpd
from xml.dom import minidom
import os
import shutil
from operator import itemgetter
# open shapefile
shp_path = "/Users/aaronl/Documents/GitHub/cracow-poland-rural-farmers/src/tasks/task-1-dataset/pl_10km.shp"
sf = gpd.read_file(shp_path)

# create KML file
kml = simplekml.Kml()

# Iterate over the rows of the GeoDataFrame
for _, row in sf.iterrows():
    # Convert the geometry to a shapely object
    geom = shape(row["geometry"])

    if not geom.is_valid:
        print(f"Invalid geometry at index {_}: {geom}")
    # Extract the coordinates from the exterior of the geometry
    name = row["CELLCODE"]
    description = row["NOFORIGIN"]
    coords = [(x, y) for x, y in geom.exterior.coords]

    pm = kml.newpoint(name=name, description=description, coords=coords)
    print(pm)
# save KML file
kml_path = "/Users/aaronl/Documents/GitHub/cracow-poland-rural-farmers/src/tasks/task-1-dataset/kml/shptokml.kml"
#kml.save(kml_path)
kml_dir = "/Users/aaronl/Documents/GitHub/cracow-poland-rural-farmers/src/tasks/task-1-dataset/kml/"

def convertkml2shp(path):

    '''
    Extract and name and boundary coordinates of each kml file
    For each chili field, create a dictionary of field name (key) and list of boundary coordinates (value)
    Update farmDictionary with each field dictionary
    Convert all KML files to shapefiles
    '''

    gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"

    directory = os.listdir(path) # path to directory containing kml files
    #os.mkdir(path + "shapefiles/") # create directory in path2 (inside kml files directory) to store shapefiles
    shapefile_directory = path # call the directory for the shapefiles
    FieldBounds_dic = {} # Empty dictionary to contain all field names (key): boundary coordinate (value) pairs

    gdf = gpd.read_file("/Users/aaronl/Documents/GitHub/cracow-poland-rural-farmers/src/tasks/task-1-dataset/kml/shapefiles/shptokml.kml", driver='KML')

    for index, row in gdf.iterrows():
        field_name = row['Name']

        field_boundary = [row.geometry.bounds]

        # do something with the field_boundary values
        minx, miny, maxx, maxy = min(field_boundary, key=itemgetter(0))[0], min(field_boundary, key=itemgetter(1))[1], max(field_boundary, key=itemgetter(2))[2], max(field_boundary, key=itemgetter(3))[3]
        field_borders = [minx, miny, maxx, maxy]

        field_dic = {field_name:field_borders}

        FieldBounds_dic.update(field_dic) # update FieldName:BorderCoordinates dictionary

    # create a geopandas dataframe from the dictionary
    print(next(iter(FieldBounds_dic.items())))
    gdf = gpd.GeoDataFrame.from_dict(FieldBounds_dic, orient='index', columns=['name','geometry'])

    # set the coordinate reference system (CRS) of the dataframe
    gdf.crs = 'EPSG:4326'

    # write the dataframe to a shapefile
    gdf.to_file('fields.shp')
    return FieldBounds_dic



convertkml2shp(kml_dir)

