import os
import geopandas as gpd
from my_function import (
    rasterize,
    make_id,
    classif_pixel_20
)

classif_tif = '/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif'
shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
output_id_tif = '/home/onyxia/work/results/data/img_pretraitees/forest_id.tif'

if os.path.exists(output_id_tif):
    pass
else: 
    output_id_shp = '/home/onyxia/work/results/data/sample/forest_id.shp'
    gdf = gpd.read_file(shapefile)
    make_id(gdf, output_id_shp, output_id_tif)

image_filename = '/home/onyxia/work/data/images/Serie_temp_S2_allbands.tif'

img_ds = '/home/onyxia/work/data/images/SENTINEL2B_20220326-105856-076_L2A_T31TCJ_C_V3-0_FRE_B2_10_2154.tif'

classif_pixel_20(image_filename, classif_tif, output_id_tif, 5, 3)


