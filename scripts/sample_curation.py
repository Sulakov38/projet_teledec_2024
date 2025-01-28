import os
from my_function import (
    create_vegetation_code,
    rasterize,
    rasterize_emprise)


# Paramètres pour code_vege
shapefile_path_vege = '/home/onyxia/work/data/project/FORMATION_VEGETALE.shp'
shapefile_path_emprise = '/home/onyxia/work/data/project/emprise_etude.shp'
output_path_shp = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
output_path_tif = '/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif'
create_vegetation_code(shapefile_path_vege, shapefile_path_emprise, output_path_shp)

field_name = 'code'
type_data = 'Byte'
spatial_resolution = 10

rasterize(output_path_shp, output_path_tif, field_name, spatial_resolution, type_data)
print(f"Le fichier SHP a été enregistrée dans {output_path_shp}")
print(f"Le fichier TIF a été enregistrée dans {output_path_tif}")