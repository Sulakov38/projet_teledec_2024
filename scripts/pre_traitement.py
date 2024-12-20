import os
import numpy as np
from my_function import (
    load_img_as_array,
    write_image,
    calculate_band_index,
    compute_ndvi,
    data_type_match,
)

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
