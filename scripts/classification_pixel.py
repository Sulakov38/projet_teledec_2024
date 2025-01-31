import os
import geopandas as gpd
from my_function import (
    make_id,
    classif_pixel
)

classif_tif = '/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif'
shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
output_id_tif = '/home/onyxia/work/results/data/img_pretraitees/forest_id.tif'

if os.path.exists(output_id_tif):
    pass
else: 
    output_id_shp = '/home/onyxia/work/results/data/sample/forest_id.shp'
    gdf = gpd.read_file(shapefile)
    make_id(gdf, output_id_shp)

image_filename = '/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_allbands.tif'

out_folder = '/home/onyxia/work/results/data/classif/'
classif_pixel(image_filename, classif_tif, output_id_tif, out_folder, 2, 2)

