import os
import geopandas as gpd
from my_function import (
    load_class_names,
    calculate_average_distances,
    calculate_average_distances_class_poly,
    plot_average_distances,
)
import sys
sys.path.append('/home/onyxia/work/libsigma/')
import read_and_write as rw

# Paths to input and output files
ndvi = "/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
classes = "/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ_all.tif"
shapefile_path = "/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ_all.shp"
output_path_q1 = "/home/onyxia/work/results/figure/diag_baton_dist_centroide_classe.png"
output_path_q2 = "/home/onyxia/work/results/figure/violin_plot_dist_centroide_classe.png"

# Load class names
code_to_name = load_class_names(shapefile_path)

# Define classes of interest
classes_of_interest_1 = [11, 12, 13, 14, 23, 24, 25]
classes_of_interest_2 = [15, 26, 28, 29]
# Calculate average distances
average_distances_q1 = calculate_average_distances(ndvi, classes, classes_of_interest_1, classes_of_interest_2)
plot_average_distances(average_distances_q1, code_to_name, classes_of_interest_1, classes_of_interest_2, output_path_q1)

# Plot the results
calculate_average_distances_class_poly(ndvi, classes, classes_of_interest_1, classes_of_interest_2, output_path_q2)

