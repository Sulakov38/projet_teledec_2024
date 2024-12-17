import matplotlib.pyplot as plt
import geopandas as gpd
import sys
sys.path.append('libsigma/')
import read_and_write as rw
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
import seaborn as sns

# Chargement des données du shapefile
def load_shapefile(file_path):
    return gpd.read_file(file_path)

# Calcul du nombre de polygones par classe
def calculate_polygons_per_class(dataframe, class_column):
    return dataframe[class_column].value_counts()

# Création du diagramme en bâton
def create_bar_chart_matplotlib(data, output_poly_path):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data.values, color='skyblue')
    plt.title('Nombre de polygones par classe')
    plt.xlabel("Essences d'arbres")
    plt.ylabel('Nombre de polygones')
    plt.xticks(rotation=45, ha='right')

    # Ajout des étiquettes sur les bâtons
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height(), 
            str(bar.get_height()), 
            ha='center', 
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(output_poly_path)
    plt.close()

def pixels_per_class(input_image, shapefile_path, output_pix_path) :

    # Chargement du fichier raster
    data_set = rw.open_image(input_image)
    img = rw.load_img_as_array(input_image)

    # Afficher les valeurs uniques du raster
    unique_values, counts = np.unique(img, return_counts=True)

    # Chargement du fichier shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Correspondance code -> nom
    shapefile["code"] = shapefile["code"].astype(int)
    code_to_name = dict(zip(shapefile["code"], shapefile["nom"]))

    # Création d'un DataFrame
    data = pd.DataFrame({"code": unique_values, "count": counts})
    data["name"] = data["code"].map(code_to_name)

    # Suppréssion des codes non trouvés
    data = data.dropna(subset=["name"])

    # Trie par nombre de pixels
    data = data.sort_values(by="count", ascending=False)

    # Créeation du diagramme
    plt.figure(figsize=(12, 8))
    bars = plt.bar(data["name"], data["count"], width=0.8, color='skyblue')

    # Ajout des valeurs au-dessus des barres
    for bar, count in zip(bars, data["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")

    # Ajout des labels et du titre
    plt.xlabel("Essences d'arbres")
    plt.ylabel("Nombre de pixels")
    plt.title("Nombre de pixels par essence d'arbres")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_pix_path)
    plt.close()

def pixels_per_polygons_per_class(raster_path, shapefile_path, output_violin_path) :

    # Chargement du raster
    data_set = rw.open_image(raster_path)
    img = rw.load_img_as_array(raster_path)

    # Chargement du shapefile
    shapefile = gpd.read_file(shapefile_path)
    shapefile["code"] = shapefile["code"].astype(int)

    # Calcul des statistiques pour compter les pixels par polygone
    stats = zonal_stats(
        shapefile,  # shapefile contenant les polygones
        raster_path,  # chemin du raster
        stats=['count'],  # count = nombre de pixels dans chaque polygone
        nodata=0  # Exclut les pixels nodata
    )

    # Ajout du nombre de pixels au shapefile
    stats_df = pd.DataFrame(stats)  # Création d'un DataFrame des statistiques
    shapefile["nombre_pixels"] = stats_df["count"]

    # Calcul de la moyenne du nombre de pixels par polygone pour chaque classe
    resume = shapefile.groupby("nom").agg(
        total_polygones=("geometry", "count"),  # Nombre total de polygones par classe
        total_pixels=("nombre_pixels", "sum"),  # Nombre total de pixels par classe
        moyenne_pixels_par_polygone=("nombre_pixels", "mean")  # Moyenne des pixels par polygone
    ).reset_index()

    # Création du diagramme en violon pour la distribution des pixels par polygone par classe
    plt.figure(figsize=(14, 8))
    sns.violinplot(x="nom", y="nombre_pixels", data=shapefile, cut=0, scale="width", inner="quartile", linewidth=1.2, color='skyblue')
    plt.yscale("log")  # Utilisation d'une échelle logarithmique pour mieux visualiser les valeurs extrêmes
    plt.title("Distribution du nombre de pixels par polygone pour chaque essences d'arbres")
    plt.xlabel("Essences d'arbres")
    plt.ylabel("Nombre de pixels par polygone (Echelle logarithmique)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_violin_path)


shapefile_path = "/home/onyxia/work/projet_teledec_2024/Sample_BD_foret_T31TCJ.shp"
raster_path = "/home/onyxia/work/projet_teledec_2024/raster.tif"
class_column = "nom"
gdf = load_shapefile(shapefile_path)
class_counts = calculate_polygons_per_class(gdf, class_column)
output_poly_path = "/home/onyxia/work/projet_teledec_2024/diag_baton_nb_poly_by_class.png"
create_bar_chart_matplotlib(class_counts, output_poly_path)
print("Diagramme en bâton créé et enregistré avec succès !")

output_pix_path = "/home/onyxia/work/projet_teledec_2024/diag_baton_nb_pix_by_class.png"
pixels_per_class(raster_path, shapefile_path, output_pix_path)
print("Diagramme en bâton créé et enregistré avec succès !")

output_violin_path = "/home/onyxia/work/projet_teledec_2024/violin_plot_nb_pix_par_poly_par_classe.png"
pixels_per_polygons_per_class(raster_path, shapefile_path, output_violin_path)
print("Diagramme en violon créé et enregistré avec succès !")