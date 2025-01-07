from my_function import (
    rasterize,
    make_id,
    filter_polygons_by_code,
    classif_pixel
)
classif_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ_classif.shp'
classif_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/Sample_BD_foret_T31TCJ_classif.tif'
emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'

if os.path.exists(classif_tif):
    print("Le fichier tif pour la classification existe déjà, on ne lance pas make_id().")
else: 
    shapefile = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ.shp'
    gdf = gpd.read_file(shapefile)
    code_column = "code"
    allowed_codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]
    filtered_gdf = filter_polygons_by_code(gdf, code_column, allowed_codes)
    filtered_gdf.to_file(classif_shp)
    field_name = 'code'
    type_data = 'Byte'
    rasterize(classif_shp, classif_tif, field_name, 10, emprise, type_data)


# Create samples with ID if they didn't exist 
emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
output_id_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/forest_id.shp'
output_id_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/forest_id.tif'

if os.path.exists(output_id_shp) and os.path.exists(output_id_tif):
    print("Au moins un des fichiers existe déjà, on ne lance pas make_id().")
else:
    make_id(filtered_gdf, emprise, output_id_shp, output_id_tif)

image_filename = '/home/onyxia/work/data/images/Serie_temp_S2_allbands.tif'

classif_pixel(image_filename, classif_tif, output_id_tif, 2, 5)