import os
import numpy as np
import read_and_write as rw
from my_function import (
    load_img_as_array,
    write_image,
    calculate_band_index,
    compute_ndvi,
    data_type_match,
)

# Paramètres pour preparation
emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
bands = [2, 3, 4, 5, 6, 7, 8, '8A', 11, 12]
releves = [
    'B_20220326-105856-076_L2A_T31TCJ_C_V3-0_FRE_B',
    'B_20220405-105855-542_L2A_T31TCJ_C_V3-0_FRE_B',
    'B_20220803-105903-336_L2A_T31TCJ_C_V3-0_FRE_B',
    'B_20221111-105858-090_L2A_T31TCJ_C_V3-1_FRE_B',
    'A_20221116-105900-865_L2A_T31TCJ_C_V3-1_FRE_B',
    'B_20230209-105857-157_L2A_T31TCJ_C_V3-1_FRE_B'
]
img_all_band = []
forest = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/masque_foret.tif'
preparation(releves, bands, emprise, img_all_band)

# Concaténation des bandes
output_img = '/home/onyxia/work/images/Serie_temp_S2_allbands_concat.tif'
image_filename = '/home/onyxia/work/images/SENTINEL2B_20220326-105856-076_L2A_T31TCJ_C_V3-0_FRE_B2_10_2154.tif'
img = np.concatenate(img_all_band, axis=-1)
data_set = rw.open_image(image_filename)
rw.write_image(output_img, img, data_set=data_set)

# Paramètres pour nodata
input_raster = '/home/onyxia/work/images/Serie_temp_S2_allbands_concat.tif'
output_raster = '/home/onyxia/work/images/Serie_temp_S2_allbands.tif'
nodata(input_raster, output_raster)

print(f"L'image Concaténé a été enregistrée dans {output_raster}")

# Définition des paramètres
dirname = '/home/onyxia/work/images'  # Dossier contenant l'image d'entrée
out_dirname = '/home/onyxia/work/images'  # Dossier de sortie
filename = os.path.join(dirname, 'Serie_temp_S2_allbands.tif')  # Nom du fichier d'entrée
out_ndvi_filename = os.path.join(out_dirname, 'Serie_temp_S2_ndvi.tif')  # Nom du fichier de sortie

# Chargement des données
data, data_set = load_img_as_array(filename)

# Définir les paramètres de traitement
num_dates = 6  # Nombre de dates
bands_per_date = 10  # Nombre de bandes par date

# Calcul des indices des bandes B4 (rouge) et B8 (NIR) pour chaque date
red_band_indices = [
    calculate_band_index(i, 2, bands_per_date, data.shape[0]) for i in range(num_dates)
]
nir_band_indices = [
    calculate_band_index(i, 6, bands_per_date, data.shape[0]) for i in range(num_dates)
]

# Calcul du NDVI pour chaque date
ndvi_stack = []
for i in range(num_dates):
    # Extraction des bandes rouge et infrarouge
    r = data[red_band_indices[i], :, :].astype('float32')
    ir = data[nir_band_indices[i], :, :].astype('float32')
    
    # Calcul du NDVI en utilisant la fonction dédiée
    ndvi = compute_ndvi(r, ir)
    
    # Ajout du NDVI calculé à la pile
    ndvi_stack.append(ndvi)

# Conversion de la pile NDVI en tableau numpy
ndvi_stack = np.array(ndvi_stack)

# Écriture de l'image NDVI dans un fichier de sortie
write_image(out_ndvi_filename, ndvi_stack, reference_ds=data_set, gdal_dtype=data_type_match['float32'])

print(f"L'image NDVI a été enregistrée dans {out_ndvi_filename}")
