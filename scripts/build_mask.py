from osgeo import gdal
import os
import geopandas as gpd
from my_function import (
    rasterize_emprise,
    rasterize,
    masque
)
shapefile_path_vege = '/home/onyxia/work/data/project/FORMATION_VEGETALE.shp'
output_vegetation = '/home/onyxia/work/data/project/FORMATION_VEGETALE_binaire.shp'

if os.path.exists(output_vegetation):
    pass
else:
    vegetation = gpd.read_file(shapefile_path_vege)
    vegetation['binaire'] = 1
    vegetation.to_file(output_vegetation)

emprise_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'

if os.path.exists(emprise_tif):
    pass
else:
    emprise = '/home/onyxia/work/data/project/emprise_etude.shp'
    field_name_emprise = 'minx'
    sptial_resolution = 10.0
    rasterize_emprise(emprise, emprise_tif, field_name_emprise, sptial_resolution)

out_image = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/binaire.tif'
field_name = 'binaire'
sptial_resolution = 10.0
typedata = 'Byte'
rasterize(output_vegetation, out_image, field_name, sptial_resolution, emprise, typedata)

input_image = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/binaire.tif'
output_masque = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/masque_foret.tif'
masque(input_image, output_masque)
os.remove(input_image)
print(f"Le masque a été enregistrée dans {output_masque}")