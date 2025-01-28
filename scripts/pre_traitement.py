import os
from my_function import (
    preparation,
    create_ndvi
)

out_dirname = '/home/onyxia/work/results/data/img_pretraitees/'  # Dossier de sortie
filename = os.path.join(out_dirname, 'Serie_temp_S2_allbands.tif')  # Nom du fichier d'entrée
if os.path.exists(filename):
    pass
else:
    bands = [2, 3, 4, 5, 6, 7, 8, '8A', 11, 12]
    releves = [
        'B_20220326-105856-076_L2A_T31TCJ_C_V3-0_FRE_B',
        'B_20220405-105855-542_L2A_T31TCJ_C_V3-0_FRE_B',
        'B_20220803-105903-336_L2A_T31TCJ_C_V3-0_FRE_B',
        'B_20221111-105858-090_L2A_T31TCJ_C_V3-1_FRE_B',
        'A_20221116-105900-865_L2A_T31TCJ_C_V3-1_FRE_B',
        'B_20230209-105857-157_L2A_T31TCJ_C_V3-1_FRE_B'
    ]

    forest = '/home/onyxia/work/results/data/img_pretraitees/masque_foret.tif'
    preparation(releves, bands, out_dirname, forest)
    print(f"L'image Concaténé a été enregistrée dans {out_dirname}")

num_dates = 6 
bands_per_date = 10  
total_bands = 60
create_ndvi(out_dirname, num_dates, bands_per_date, total_bands)
