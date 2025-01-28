from my_function import (
    compute_class_statistics,
    map_class_names,
    create_class_to_color_mapping,
    plot_ndvi_results
)
import matplotlib.pyplot as plt
import geopandas as gpd
import sys
sys.path.append('/home/onyxia/work/libsigma/')
import read_and_write as rw

# Paths to input and output files
ndvi_path = "/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
classes_path = "/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif"
shapefile_path = "/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp"
output_path = "/home/onyxia/work/results/figure/temp_mean_ndvi.png"

# Parameters
selected_classes = [12, 13, 14, 23, 24, 25]
band_dates = ["26/03/2022", "05/04/2022", "03/08/2022", "11/11/2022", "16/11/2022", "09/02/2023"]

# Load raster data
ndvi = rw.load_img_as_array(ndvi_path)
classes = rw.load_img_as_array(classes_path)

# Compute NDVI statistics
results_df = compute_class_statistics(ndvi, classes, selected_classes, band_dates)

# Map class names using shapefile
results_df = map_class_names(results_df, shapefile_path)

# Create color mapping for classes
colors = plt.cm.tab10.colors
class_to_color = create_class_to_color_mapping(selected_classes, colors)

# Plot and save results
shapefile = gpd.read_file(shapefile_path)
shapefile["code"] = shapefile["code"].astype(int)
code_to_name = dict(zip(shapefile["code"], shapefile["nom"]))
plot_ndvi_results(results_df, selected_classes, class_to_color, code_to_name, output_path)
