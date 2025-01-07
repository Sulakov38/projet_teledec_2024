import plots
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from my_function import (
    make_id,
    filter_polygons_by_code,
    classif_pixel
)
import sys
sys.path.append('/home/onyxia/work/libsigma/')
import classification as cla
import read_and_write as rw

shapefile = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ.shp'
output_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ_classif.shp'

# Create filtered samples
gdf = gpd.read_file(shapefile)
code_column = "code"
allowed_codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]
filtered_gdf = filter_polygons_by_code(gdf, code_column, allowed_codes)
filtered_gdf.to_file(output_shp)

# Create samples with ID if they didn't exist 
emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
output_id_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/forest_id.shp'
output_id_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/forest_id.tif'

if os.path.exists(output_id_shp) and os.path.exists(output_id_tif):
    print("Au moins un des fichiers existe déjà, on ne lance pas make_id().")
else:
    make_id(filtered_gdf, emprise, output_id_shp, output_id_tif)

image_filename = '/home/onyxia/work/data/images/Serie_temp_S2_allbands.tif'

classif_pixel(image_filename, output_id_tif)