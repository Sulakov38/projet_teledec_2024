from my_function import (
    compute_class_statistics,
    map_class_names,
    create_class_to_color_mapping,
    plot_ndvi_results,
    load_class_names
)
import matplotlib.pyplot as plt

# Paths to input and output files
ndvi_path = "/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
classes_path = "/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif"
shapefile_path = "/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp"
output_path = "/home/onyxia/work/results/figure/temp_mean_ndvi.png"

# Parametres
selected_classes = [12, 13, 14, 23, 24, 25]
band_dates = ["26/03/2022", "05/04/2022", "03/08/2022", "11/11/2022", "16/11/2022", "09/02/2023"]

# Statistiques du NDVI
results_df = compute_class_statistics(ndvi_path, classes_path, selected_classes, band_dates)

# Map class names using shapefile
results_df = map_class_names(results_df, shapefile_path)

# Create color mapping for classes
colors = plt.cm.tab10.colors
class_to_color = create_class_to_color_mapping(selected_classes, colors)

code_to_name = load_class_names(shapefile_path)
plot_ndvi_results(results_df, selected_classes, class_to_color, code_to_name, output_path)
