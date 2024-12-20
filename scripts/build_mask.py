from osgeo import gdal
from my_function import (
    rasterize_emprise,
    rasterize,
    masque
)

# Paramètres pour rasterize_emprise
emprise = '/home/onyxia/work/data/project/emprise_etude.shp'
field_name = 'minx'
output_emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
sptial_resolution = 10.0
rasterize_emprise(emprise, output_emprise, field_name, sptial_resolution)

# Paramètres pour rasterize
in_vector = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ.shp'
out_image = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/raster.tif'
field_name = 'code'
emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
sptial_resolution = 10.0
rasterize(in_vector,out_image,field_name,sptial_resolution,emprise)

# Paramètres pour masque
data_type_match = {'uint8': gdal.GDT_Byte,
                   'uint16': gdal.GDT_UInt16,
                   'uint32': gdal.GDT_UInt32,
                   'int16': gdal.GDT_Int16,
                   'int32': gdal.GDT_Int32,
                   'float32': gdal.GDT_Float32,
                   'float64': gdal.GDT_Float64}
input_image = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/raster.tif'
output_masque = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/masque_foret.tif'
masque(input_image, output_masque)
print(f"Le masque a été enregistrée dans {output_masque}")