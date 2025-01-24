import os
import geopandas as gpd
from fonc import (
    rasterize,
    make_id,
    filter_polygons_by_code,
    classif_pixel_19,
    classif_pixel_20
)
import sys
sys.path.append('/home/onyxia/work/libsigma/')
import classification as cla
import read_and_write as rw
import plots



classif_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/Sample_BD_foret_T31TCJ_classif.tif'
output_id_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/forest_id.tif'


if os.path.exists(classif_tif):
    pass
else: 
    emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
    classif_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ_classif.shp'

    shapefile = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ.shp'
    gdf = gpd.read_file(shapefile)
    code_column = "code"
    gdf['code'] = gdf['code'].astype(int)
    allowed_codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]
    filtered_gdf = filter_polygons_by_code(gdf, code_column, allowed_codes)
    filtered_gdf.to_file(classif_shp)
    field_name = 'code'
    type_data = 'Byte'
    rasterize(classif_shp, classif_tif, field_name, 10, emprise, type_data)

if os.path.exists(output_id_tif):
    pass
else: 
    emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
    output_id_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/forest_id.shp'

    make_id(filtered_gdf, emprise, output_id_shp, output_id_tif)

image_filename = '/home/onyxia/work/data/images/Serie_temp_S2_allbands.tif'
#img_classif = rw.load_img_as_array(image_filename)
#image_3_bandes = img_classif[:, :, :3]
out_file = '/home/onyxia/work/data/images/Serie_temp_S2_allbands2.tif'
img_ds = '/home/onyxia/work/data/images/SENTINEL2B_20220326-105856-076_L2A_T31TCJ_C_V3-0_FRE_B2_10_2154.tif'
#data_set = rw.open_image(img_ds)
#rw.write_image(out_file,image_3_bandes,data_set)

#classif_pixel3(image_filename, classif_tif, output_id_tif, 30, 5)
#classif_pixel_19(image_filename, classif_tif, output_id_tif, 5)

classif_pixel_20(image_filename, classif_tif, output_id_tif, 5, 3)

