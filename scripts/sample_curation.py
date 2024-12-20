from my_function import code_vege

# Paramètres pour code_vege
shapefile_path_vege = '/home/onyxia/work/data/project/FORMATION_VEGETALE.shp'
shapefile_path_emprise = '/home/onyxia/work/data/project/emprise_etude.shp'
output_path = '/home/onyxia/work/projet_teledec_2024/results/data/sample/Sample_BD_foret_T31TCJ.shp'
code_vege(shapefile_path_vege, shapefile_path_emprise, output_path)
print(f"Le fichier SHP a été enregistrée dans {output_path}")