import os
import geopandas as gpd
from my_function import (
    rasterize,
    masque
)
shapefile_path_vege = '/home/onyxia/work/data/project/FORMATION_VEGETALE.shp'
output_vegetation = '/home/onyxia/work/data/project/FORMATION_VEGETALE_binaire.shp'

vegetation = gpd.read_file(shapefile_path_vege)
vegetation['binaire'] = 1
vegetation.to_file(output_vegetation)

out_image = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/binaire.tif'
field_name = 'binaire'
sptial_resolution = 10.0
typedata = 'Byte'
rasterize(output_vegetation, out_image, field_name, sptial_resolution, typedata)

output_masque = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/masque_foret.tif'
masque(out_image, output_masque)
os.remove(out_image)
print(f"Le masque a été enregistrée dans {output_masque}")