from my_function import (
    plot_ndvi_results,
    load_class_names
)

# Paths to input and output files
ndvi_path = "/home/onyxia/work/results/data/img_pretraitees/Serie_temp_S2_ndvi.tif"
classes_path = "/home/onyxia/work/results/data/img_pretraitees/Sample_BD_foret_T31TCJ.tif"
shapefile_path = "/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp"
output_path = "/home/onyxia/work/results/figure/temp_mean_ndvi.png"

# Parametres
selected_classes = [12, 13, 14, 23, 24, 25]
band_dates = ["26/03/2022", "05/04/2022", "03/08/2022", "11/11/2022", "16/11/2022", "09/02/2023"]

plot_ndvi_results(ndvi_path, classes_path, selected_classes, band_dates, shapefile_path, output_path)

