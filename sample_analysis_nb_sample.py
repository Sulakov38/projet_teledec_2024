from my_function import (
    load_shapefile,
    filter_polygons_by_code,
    calculate_polygons_per_class,
    create_bar_chart_matplotlib,
    pixels_per_class,
    pixels_per_polygons_per_class
)

# Paths and parameters
shapefile_path = "/home/onyxia/work/projet_teledec_2024/Sample_BD_foret_T31TCJ.shp"
raster_path = "/home/onyxia/work/projet_teledec_2024/raster.tif"
code_column = "code"
class_column = "nom"
allowed_codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]

# Load and filter shapefile
gdf = load_shapefile(shapefile_path)
gdf[code_column] = gdf[code_column].astype(int)
filtered_gdf = filter_polygons_by_code(gdf, code_column, allowed_codes)

# Count polygons per class and create bar chart
class_counts = calculate_polygons_per_class(filtered_gdf, class_column)
output_poly_path = "/home/onyxia/work/projet_teledec_2024/diag_baton_nb_poly_by_class.png"
create_bar_chart_matplotlib(class_counts, output_poly_path)
print("Diagramme en bâton créé et enregistré avec succès !")

# Analyze pixels per class and create bar chart
output_pix_path = "/home/onyxia/work/projet_teledec_2024/diag_baton_nb_pix_by_class.png"
pixels_per_class(raster_path, shapefile_path, output_pix_path)
print("Diagramme en bâton créé et enregistré avec succès !")

# Compute pixel distribution and create violin plot
output_violin_path = "/home/onyxia/work/projet_teledec_2024/violin_plot_nb_pix_par_poly_par_classe.png"
pixels_per_polygons_per_class(raster_path, shapefile_path, output_violin_path)
print("Diagramme en violon créé et enregistré avec succès !")
