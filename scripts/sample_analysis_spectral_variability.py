from my_function import (
    load_class_names,
    calculate_average_distances,
    plot_average_distances,
)
import sys
sys.path.append('/home/onyxia/work/libsigma/')
import read_and_write as rw

# Paths to input and output files
ndvi_file = "/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
classes_file = "/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif"
shapefile_path = "/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ.shp"
output_path = "/home/onyxia/work/projet_teledec_2024/results/figure/diag_baton_dist_centroide_classe.png"

# Load raster data
ndvi_data = rw.load_img_as_array(ndvi_file)  # NDVI data (shape: [bands, height, width])
classes_data = rw.load_img_as_array(classes_file)  # Classes data (shape: [height, width])

# Load class names
code_to_name = load_class_names(shapefile_path)

# Define classes of interest
classes_of_interest = [11, 12, 13, 14, 23, 24, 25]

# Calculate average distances
average_distances = calculate_average_distances(ndvi_data, classes_data, classes_of_interest)

# Plot the results
plot_average_distances(average_distances, code_to_name, output_path)