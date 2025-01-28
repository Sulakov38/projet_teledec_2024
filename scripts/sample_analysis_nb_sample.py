from my_function import (
    load_shapefile,
    calculate_polygons_per_class,
    create_bar_chart_matplotlib,
    pixels_per_class,
    pixels_per_polygons_per_class
)

# Paths and parameters
shapefile_path = "/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp"
raster_path = "/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif"
class_column = "nom"

# Load and filter shapefile
gdf = load_shapefile(shapefile_path)

# Count polygons per class and create bar chart
class_counts = calculate_polygons_per_class(gdf, class_column)
output_poly_path = "/home/onyxia/work/figure/diag_baton_nb_poly_by_class.png"
create_bar_chart_matplotlib(class_counts, output_poly_path)
print(f"Diagramme en bâton créé et enregistré dans {output_poly_path}")

# Analyze pixels per class and create bar chart
output_pix_path = "/home/onyxia/work/figure/diag_baton_nb_pix_by_class.png"
pixels_per_class(raster_path, gdf, output_pix_path)
print(f"Diagramme en bâton créé et enregistré dans {output_pix_path}")

# Compute pixel distribution and create violin plot
output_violin_path = "/home/onyxia/work/figure/violin_plot_nb_pix_par_poly_par_class.png"
pixels_per_polygons_per_class(gdf, output_violin_path)
print(f"Diagramme en violon créé et enregistré dans {output_violin_path}")
