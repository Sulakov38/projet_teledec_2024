import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.lines import Line2D

import sys
sys.path.append('/home/onyxia/work/libsigma/')
import classification as cla
import read_and_write as rw
import plots

# Dictionnaire de correspondance entre les types de données et GDAL
data_type_match = {
    'uint8': gdal.GDT_Byte,
    'uint16': gdal.GDT_UInt16,
    'uint32': gdal.GDT_UInt32,
    'int16': gdal.GDT_Int16,
    'int32': gdal.GDT_Int32,
    'float32': gdal.GDT_Float32,
    'float64': gdal.GDT_Float64,
}


def reproject_raster(minx, miny, maxx, maxy, input_raster, output_raster, src_epsg, dst_epsg):
    """
    Reprojette un raster dans un nouveau système de coordonnées, en définissant 
    une étendue spécifique et une résolution spatiale de 10 x 10.

    Paramètres
    ----------
    minx, miny, maxx, maxy : float
        Coordonnées de la zone d’intérêt dans le système de coordonnées de destination.
        Chemin vers le raster en entrée.
    output_raster : str
        Chemin vers le raster reprojeté en sortie (fichier GeoTIFF).
    src_epsg : int
        Code EPSG du système de coordonnées source.
    dst_epsg : int
        Code EPSG du système de coordonnées de destination.
    """
    # Define the command pattern with parameters
    cmd_pattern = ("gdalwarp -s_srs EPSG:{src_epsg} "
                   "-t_srs EPSG:{dst_epsg} "
                   "-te {minx} {miny} {maxx} {maxy} -te_srs EPSG:{dst_epsg} "
                   "-tr 10 10 "
                   "-r near "
                   "-of GTiff "
                   "{input_raster} {output_raster}")
    
    # Fill the command pattern with parameters
    cmd = cmd_pattern.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                             input_raster=input_raster, output_raster=output_raster,
                             src_epsg=src_epsg, dst_epsg=dst_epsg)
    
    # Print and execute the command
    print(cmd)
    os.system(cmd)


def merge(img_all_band, output_raster):
    """
    Fusionne plusieurs rasters en un seul fichier GeoTIFF, en créant des bandes séparées
    pour chaque image.

    Paramètres
    ----------
    img_all_band : list of str
        Liste des chemins vers les différents fichiers raster à fusionner.
    output_raster : str
        Chemin du fichier de sortie (GeoTIFF) contenant l’ensemble des rasters fusionnés.
    """
    cmd_pattern = ("gdal_merge -o {output_raster} "
               "-of GTiff "
               "-n 0 "
               "-separate "
               "{input_rasters}")

    # Joindre tous les fichiers raster de img_all_band en une seule chaîne
    input_rasters = ' '.join(img_all_band)
    # Remplir la commande avec les paramètres
    cmd = cmd_pattern.format(output_raster=output_raster, input_rasters=input_rasters)
    # Afficher la commande générée
    print(cmd)
    # Exécuter la commande
    os.system(cmd)

def preparation(releves, bands, dirname, forest_mask):
    """
    Prépare un ensemble d’images raster Sentinel-2 en reprojetant, recadrant et appliquant 
    un masque forestier, puis fusionne le tout en une seule image finale.

    Paramètres
    ----------
    releves : list of str
        Liste des chaînes identifiant les différentes dates de relevés Sentinel-2 (ex.: ['_20200101', '_20200115']).
    bands : list of str
        Liste des bandes Sentinel-2 à traiter (ex.: ['B2', 'B3', 'B4', 'B8']).
    dirname : str
        Chemin vers le répertoire dans lequel sera sauvegardée l’image finale fusionnée.
    forest_mask : str
        Chemin vers le raster contenant un masque forestier (même étendue et résolution 
        que les images Sentinel-2) afin de filtrer les zones non forestières.
    """
    suffixe = '.tif'
    emprise_tif = '/home/onyxia/work/results/data/img_pretraitees/emprise.tif'
    emprise = '/home/onyxia/work/data/project/emprise_etude.shp'
    field_name_emprise = 'minx'
    sptial_resolution = 10.0
    rasterize_emprise(emprise, emprise_tif, field_name_emprise, sptial_resolution)
    minx, miny, maxx, maxy = get_img_extend(emprise_tif)
    src_epsg = 32631  # EPSG 32631 (UTM zone 31N)
    dst_epsg = 2154   # EPSG 2154 (RGF93 / Lambert-93)
    img_all_band = []
    for r in releves:
        for B in bands:
            # Construct file paths
            band_name = f'/home/onyxia/work/data/images/SENTINEL2{r}{B}'
            band_file = f'{band_name}{suffixe}'
            output_filename_inter = f"{band_name}_10_2154_inter{suffixe}"
            output_filename = f"{band_name}_10_2154{suffixe}"
            # Vérifier si le fichier de sortie existe
            if os.path.exists(output_filename):
                # Ajouter le chemin du fichier existant à la liste
                img_all_band.append(output_filename)
            else:
                # Reprojeter le raster si le fichier n'existe pas
                reproject_raster(minx, miny, maxx, maxy, band_file, output_filename_inter, src_epsg, dst_epsg)          
                img_all_band.append(output_filename)
                img = rw.load_img_as_array(output_filename_inter)
                masque_foret = rw.load_img_as_array(forest_mask).astype('bool')
                band_masked = img.copy()
                band_masked[~masque_foret] = 0
                dataset = rw.open_image(output_filename_inter)
                rw.write_image(output_filename, band_masked, data_set=dataset)
                os.remove(output_filename_inter)
    img_merge = os.path.join(dirname, 'Serie_temp_S2_allbands_merge.tif')  
    output_img = os.path.join(dirname, 'Serie_temp_S2_allbands.tif')
    merge(img_all_band, img_merge)
    nodata(img_merge, output_img, 0)
    os.remove(img_merge)
    os.remove(emprise_tif)

def create_ndvi(dirname, num_dates, bands_per_date, total_bands):
    """
    Génère une série temporelle d’images NDVI à partir d’un fichier de bandes Sentinel-2 et 
    enregistre le résultat dans un fichier GeoTIFF.

    Paramètres
    ----------
    dirname : str
        Chemin du répertoire contenant le fichier d’entrée et où seront sauvegardés les résultats.
    num_dates : int
        Nombre de dates ou de scènes temporelles à traiter.
    bands_per_date : int
        Nombre total de bandes par date (ex. 10, 13 selon la configuration Sentinel-2).
    total_bands : int
        Nombre total de bandes dans le fichier d’entrée (num_dates × bands_per_date).
    """ 
    out_ndvi_concat = os.path.join(dirname, 'Serie_temp_S2_ndvi_concat.tif')  # Nom du fichier de sortie
    filename = os.path.join(dirname, 'Serie_temp_S2_allbands.tif')  # Nom du fichier d'entrée
    # Chargement des données
    data = rw.load_img_as_array(filename)
    # Définir les paramètres de traitement

    # Calcul des indices des bandes B4 (rouge) et B8 (NIR) pour chaque date
    red_band_indices = [
        calculate_band_index(i, 2, bands_per_date, total_bands) for i in range(num_dates)
    ]
    nir_band_indices = [
        calculate_band_index(i, 6, bands_per_date, total_bands) for i in range(num_dates)
    ]

    # Calcul du NDVI pour chaque date
    ndvi_stack = []
    for i in range(num_dates):
        # Extraction des bandes rouge et infrarouge
        r = data[:, :, red_band_indices[i]].astype('float32')
        ir = data[:, :, nir_band_indices[i]].astype('float32')
        
        # Calcul du NDVI en utilisant la fonction dédiée
        ndvi = compute_ndvi(r, ir)
        
        # Ajout du NDVI calculé à la pile
        ndvi_stack.append(ndvi)

    # Conversion de la pile NDVI en tableau numpy
    ndvi_stack = np.dstack(ndvi_stack)

    data_set = rw.open_image(filename)
    # Écriture de l'image NDVI dans un fichier de sortie
    rw.write_image(out_ndvi_concat, ndvi_stack, data_set=data_set, gdal_dtype=data_type_match['float32'])
    out_ndvi_filename = os.path.join(dirname, 'Serie_temp_S2_ndvi.tif')  # Nom du fichier de sortie
    nodata(out_ndvi_concat, out_ndvi_filename, -9999)
    os.remove(out_ndvi_concat)
    print(f"L'image NDVI a été enregistrée dans {out_ndvi_filename}")

def nodata(input_raster, output_raster, value):
    """
    Assigne une valeur NoData à un raster et génère un nouveau fichier GeoTIFF en sortie.

    Paramètres
    ----------
    input_raster : str
        Chemin vers le fichier raster d'entrée (fichier .tif).
    output_raster : str
        Chemin du fichier de sortie (fichier .tif) avec la nouvelle valeur NoData.
    value : int ou float
        Valeur à attribuer comme NoData dans le raster.
    """
    # Define the command pattern for setting no-data value
    cmd_pattern = ("gdal_translate -a_nodata {value} "
                   "-of GTiff "
                   "{input_raster} {output_raster}")
    
    # Fill the command with the parameters
    cmd = cmd_pattern.format(input_raster=input_raster, output_raster=output_raster, value=value)
    
    # Print and execute the command
    print(cmd)
    os.system(cmd)

def rasterize_emprise(in_vector, out_image, field_name_emprise, sptial_resolution):
    """
    Convertit une couche vectorielle en raster (GeoTIFF) en appliquant une 
    résolution spatiale spécifiée.

    Paramètres
    ----------
    in_vector : str
        Chemin vers la couche vectorielle (fichier .shp, GeoPackage, etc.) à rasteriser.
    out_image : str
        Chemin du fichier de sortie au format GeoTIFF (fichier .tif).
    field_name_emprise : str
        Nom du champ d’attribut à utiliser pour renseigner la valeur des pixels dans le raster.
    sptial_resolution : int
        Résolution spatiale (en unités de la projection de la couche) à utiliser pour le raster 
        (taille d’un pixel sur l’axe X et Y).
    """
    # Define the command pattern for rasterization
    cmd_pattern = ("gdal_rasterize -a {field_name_emprise} "
                   "-tr {sptial_resolution} {sptial_resolution} "
                   "-ot Byte -of GTiff "
                   "-a_nodata 0 "
                   "{in_vector} {out_image}")
    
    # Format the command with provided parameters
    cmd = cmd_pattern.format(in_vector=in_vector, out_image=out_image,
                             field_name_emprise=field_name_emprise, sptial_resolution=sptial_resolution)
    
    # Print and execute the command
    print(cmd)
    os.system(cmd)


def get_img_extend(image_filename):
    """
    Récupère l’emprise spatiale d’une image raster.

    Paramètres
    ----------
    image_filename : str
        Chemin vers l’image raster dont on veut extraire l’emprise.
    """
    # Open the image file using GDAL
    dataset = gdal.Open(image_filename)
    
    # Get the geotransform for spatial information
    geotransform = dataset.GetGeoTransform()
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * dataset.RasterXSize
    miny = maxy + geotransform[5] * dataset.RasterYSize
    
    return minx, miny, maxx, maxy

def create_vegetation_code(shapefile_path_vege, shapefile_path_emprise, output_path, output_path_all):
    """
    Crée et applique un système de codification sur les entités d’un shapefile de végétation
    en fonction de leur champ 'CODE_TFV', puis génère deux shapefiles de sortie (complet 
    et filtré) en les ‘clippant’ sur l’emprise géographique spécifiée.

    Paramètres
    ----------
    shapefile_path_vege : str
        Chemin vers le shapefile de végétation à traiter.
    shapefile_path_emprise : str
        Chemin vers le shapefile représentant l’emprise géographique de référence.
    output_path : str
        Chemin de sortie du shapefile filtré contenant uniquement certains codes d’intérêt.
    output_path_all : str
        Chemin de sortie du shapefile complet incluant l’ensemble des codes traités.
    """
    # Load the vegetation and extent shapefiles
    gdf_vegetation = gpd.read_file(shapefile_path_vege)
    gdf_emprise = gpd.read_file(shapefile_path_emprise)
    
    # Add empty columns for names and codes
    gdf_vegetation['nom'] = None
    gdf_vegetation['code'] = None

    # Define a mapping dictionary for assigning values based on the 'CODE_TFV' field
    mapping = {
        "FF1-49-49": ("Autres feuillus", "11"),
        "FF1-10-10": ("Autres feuillus", "11"),
        "FF1-09-09": ("Autres feuillus", "11"),
        "FF1G01-01": ("Chêne", "12"),
        "FF1-14-14": ("Robinier", "13"),
        "FP": ("Peupleraie", "14"),
        "FF1-00-00": ("Mélange de feuillus", "15"), 
        "FF1-00": ("Feuillus en îlots", "16"),
        "FF2-91-91": ("Autres conifères autre que pin", "21"),
        "FF2-63-63": ("Autres conifères autre que pin", "21"),
        "FF2G61-61": ("Autres conifères autre que pin", "21"),
        "FF2-90-90": ("Autres conifères autre que pin", "21"),
        "FF2-81-81": ("Autres Pin", "22"),
        "FF2-52-52": ("Autres Pin", "22"),
        "FF2-80-80": ("Autres Pin", "22"),
        "FF2-64-64": ("Douglas", "23"),
        "FF2G53-53": ("Pin laricio ou pin noir", "24"),
        "FF2-51-51": ("Pin maritime", "25"),
        "FF2-00-00": ("Mélange conifères", "26"),
        "FF2-00": ("Conifères en îlots", "27"),
        "FF32": ("Mélange de conifères prépondérants et feuillus", "28"),
        "FF31": ("Mélange de feuillus prépondérants et conifères", "29"),
    }

    # Map values to the 'nom' and 'code' columns
    gdf_vegetation['nom'] = gdf_vegetation['CODE_TFV'].map(lambda x: mapping[x][0] if x in mapping else None)
    gdf_vegetation['code'] = gdf_vegetation['CODE_TFV'].map(lambda x: mapping[x][1] if x in mapping else None)
    
    # Remove rows with missing codes
    gdf_vegetation = gdf_vegetation.dropna(subset=['code'])
    
    # Clip the vegetation shapefile to the extent
    vege_emprise = gpd.clip(gdf_vegetation, gdf_emprise)
    
    # Save the processed shapefile
    vege_emprise.to_file(output_path_all)
    codes_of_interest = ["11", "12", "13", "14", "21", "22", "23", "24", "25"]
    filtered_gdf = gdf_vegetation[gdf_vegetation['code'].isin(codes_of_interest)]

    # Clipper le shapefile de végétation avec l'emprise
    vege_emprise_interest = gpd.clip(filtered_gdf, gdf_emprise)

    # Sauvegarder le shapefile traité
    vege_emprise_interest.to_file(output_path)

def rasterize(in_vector, out_image, field_name, sptial_resolution, type_data):
    """
    Convertit un shapefile en raster en utilisant un champ d’attribut pour la valeur des pixels,
    et applique une résolution et une emprise préalablement définie.

    Paramètres
    ----------
    in_vector : str
        Chemin vers la couche vectorielle (fichier .shp, GeoPackage, etc.) à rasteriser.
    out_image : str
        Chemin du fichier GeoTIFF en sortie.
    field_name : str
        Nom du champ d’attribut utilisé pour définir la valeur des pixels dans le raster.
    sptial_resolution : float
        Résolution spatiale (taille d’un pixel, en unités du système de coordonnées).
    type_data : str
        Type de données de sortie (ex. "Byte", "UInt16", "Float32", etc.).
    """
    emprise_tif = '/home/onyxia/work/results/data/img_pretraitees/emprise.tif'

    if os.path.exists(emprise_tif):
        pass
    else:
        emprise = '/home/onyxia/work/data/project/emprise_etude.shp'
        field_name_emprise = 'minx'
        sptial_resolution = 10.0
        rasterize_emprise(emprise, emprise_tif, field_name_emprise, sptial_resolution)
    minx, miny, maxx, maxy = get_img_extend(emprise_tif)
    
    # Define the command pattern for rasterization
    cmd_pattern = ("gdal_rasterize -a {field_name} "
                    "-te {minx} {miny} {maxx} {maxy} "
                    "-tr {sptial_resolution} {sptial_resolution} "
                    "-ot {type_data} -of GTiff "
                    "-a_nodata 0 "
                    "{in_vector} {out_image}")
    
    # fill the string with the parameter thanks to format function
    cmd = cmd_pattern.format(minx=minx,miny=miny,maxx=maxx,maxy=maxy,in_vector=in_vector, out_image=out_image, 
                            field_name=field_name,sptial_resolution=sptial_resolution,type_data=type_data)
    
    # Print and execute the command
    print(cmd)
    os.system(cmd)
    os.remove(emprise_tif)

def masque(input_dir, output_dir):
    """
    Crée un shapefile binaire à partir d’un shapefile de végétation existant, puis le 
    convertit en masque raster pour distinguer la forêt des autres formations.

    Paramètres
    ----------
    input_dir : str
        Chemin vers le répertoire contenant le shapefile de végétation (FORMATION_VEGETALE.shp).
    output_dir : str
        Chemin vers le répertoire où seront sauvegardés le shapefile binaire et le masque raster.
    """
    # Open the input image and load it as a numpy array
    shapefile_path_vege = os.path.join(input_dir,'FORMATION_VEGETALE.shp')
    output_vegetation = os.path.join(input_dir,'FORMATION_VEGETALE_binaire.shp')

    vegetation = gpd.read_file(shapefile_path_vege)

    vegetation['binaire'] = 1

    codes = ["LA4", "LA6", "F01", "F02", "F03", "FF0", "FO0"]
    vegetation.loc[vegetation['CODE_TFV'].isin(codes), 'binaire'] = 0

    vegetation.to_file(output_vegetation)

    out_image = os.path.join(output_dir,'img_pretraitees/binaire.tif')
    field_name = 'binaire'
    sptial_resolution = 10.0
    typedata = 'Byte'
    rasterize(output_vegetation, out_image, field_name, sptial_resolution, typedata)

    output_masque = os.path.join(output_dir,'img_pretraitees/masque_foret.tif')
    data_set = rw.open_image(out_image)
    img = rw.load_img_as_array(out_image)
    
    # Create a binary mask where values are non-zero
    masque_foret = (img != 0).astype(int)
    
    # Write the mask to an output file
    rw.write_image(output_masque, masque_foret, data_set=data_set,
                   gdal_dtype=data_type_match['uint8'])
    os.remove(out_image)
    print(f"Le masque a été enregistrée dans {output_masque}")


def load_shapefile(file_path):
    """
    Charge un shapefile dans un GeoDataFrame via Geopandas.

    Paramètres
    ----------
    file_path : str
        Chemin vers le shapefile (.shp) à charger.
    """
    return gpd.read_file(file_path)

def calculate_polygons_per_class(shapefile_path, class_column):
    """
    Counts the number of polygons per class in a GeoDataFrame.
    Args:
        dataframe (GeoDataFrame): The GeoDataFrame containing polygons.
        class_column (str): The column containing class labels.
    Returns:
        Series: Count of polygons per class.
    """
    dataframe = load_shapefile(shapefile_path)
    return dataframe[class_column].value_counts()

def create_bar_chart_matplotlib(data, output_poly_path):
    """
    Creates and saves a bar chart showing the number of polygons per class.
    Args:
        data (Series): Data containing class labels and counts.
        output_poly_path (str): Path to save the output bar chart.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data.values, color='skyblue')
    plt.title('Nombre de polygones par classe')
    plt.xlabel("Essences d'arbres")
    plt.ylabel('Nombre de polygones')
    plt.xticks(rotation=45, ha='right')

    # Adding labels on top of the bars
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

def pixels_per_class(input_image, shapefile_path, output_pix_path):
    """
    Analyzes the number of pixels for each class in a raster and generates a bar chart.
    Args:
        input_image (str): Path to the raster image.
        shapefile_path (str): Path to the shapefile containing class information.
        output_pix_path (str): Path to save the output bar chart.
    """
    # Load raster and compute unique pixel values
    img = rw.load_img_as_array(input_image)
    unique_values, counts = np.unique(img, return_counts=True)
    dataframe = load_shapefile(shapefile_path)

    dataframe["code"] = dataframe["code"].astype(int)

    # Map codes to class names
    code_to_name = dict(zip(dataframe["code"], dataframe["nom"]))
    data = pd.DataFrame({"code": unique_values, "count": counts})
    data["name"] = data["code"].map(code_to_name)
    data = data.dropna(subset=["name"])  # Remove unmatched codes
    data = data.sort_values(by="count", ascending=False)

    # Generate bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(data["name"], data["count"], width=0.8, color='skyblue')
    for bar, count in zip(bars, data["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")

    plt.xlabel("Essences d'arbres")
    plt.ylabel("Nombre de pixels")
    plt.title("Nombre de pixels par essence d'arbres")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_pix_path)
    plt.close()

def make_id(dataframe, output_id_shp):
    """
    Attribue un identifiant unique à chaque entité d’un GeoDataFrame, enregistre 
    le résultat sous forme de shapefile, puis génère un raster basé sur cet identifiant.

    Paramètres
    ----------
    dataframe : geopandas.GeoDataFrame
        Objet contenant des entités géographiques avec une colonne 'code'.
    output_id_shp : str
        Chemin cible pour enregistrer le shapefile contenant la colonne 'unique_id'.
    """
    dataframe["code"] = dataframe["code"].astype(int)
    dataframe["unique_id"] = range(1, len(dataframe) + 1)
    output_id_shp = '/home/onyxia/work/results/data/sample/forest_id.shp'
    dataframe.to_file(output_id_shp)

    field = "unique_id"
    output_id_tif = '/home/onyxia/work/results/data/img_pretraitees/forest_id.tif'
    resolution = 10
    type_data = 'Uint16'
    rasterize(output_id_shp, output_id_tif, field, resolution, type_data)

def pixels_per_polygons_per_class(shapefile_path, output_violin_path):
    """
    Computes and visualizes the pixel distribution per polygon for each class.
    Args:
        shapefile_path (str): Path to the shapefile containing class information.
        output_violin_path (str): Path to save the output violin plot.
    """
    output_id_shp = '/home/onyxia/work/results/data/sample/forest_id.shp'
    output_id_tif = '/home/onyxia/work/results/data/img_pretraitees/forest_id.tif'

    if os.path.exists(output_id_shp):
        dataframe = load_shapefile(output_id_shp)
    else: 
        shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ.shp'
        gdf = gpd.read_file(shapefile)
        make_id(gdf, output_id_shp)
        dataframe = load_shapefile(output_id_shp)

    array = rw.load_img_as_array(output_id_tif)
    unique_values, counts = np.unique(array, return_counts=True)

    val_to_count = dict(zip(unique_values, counts))

    dataframe["nombre_pixels"] = dataframe["unique_id"].map(val_to_count)

    # Préparation des données pour le diagramme en violon
    unique_noms = dataframe["nom"].unique()
    data_for_violin = [dataframe.loc[dataframe["nom"] == n, "nombre_pixels"].values for n in unique_noms]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Création du diagramme en violon
    parts = ax.violinplot(data_for_violin, showmeans=True, showmedians=True, showextrema=True)

    # Paramétrages de l'axe et des titres
    ax.set(
        yscale="log",
        title="Distribution du nombre de pixels par polygone pour chaque essence d'arbres",
        xlabel="Essences d'arbres",
        ylabel="Nombre de pixels par polygone (Échelle logarithmique)"
    )

    # Ajustement des labels de l'axe X
    ax.set_xticks(np.arange(1, len(unique_noms) + 1))
    ax.set_xticklabels(unique_noms, rotation=45, ha='right')

    # Personnalisation des violons
    for violin_body in parts['bodies']:
        violin_body .set(facecolor='skyblue', edgecolor='black', alpha=0.7)

    # Personnalisation des lignes (médianes, moyennes, barres)
    for line_type in ['cbars', 'cmins', 'cmaxes']:
        parts[line_type].set(lw=0.5, ls='--', color='gray')

    # Style spécifique pour la moyenne
    parts['cmeans'].set(lw=1, ls='-', color='black')  # Ligne plus épaisse et rouge pour la moyenne

    # Style spécifique pour la médiane
    parts['cmedians'].set(lw=1, ls='-', color='darkgray')  # Ligne plus épaisse et bleue pour la médiane

    plt.tight_layout()
    plt.savefig(output_violin_path)

def compute_class_statistics(ndvi_path, classes_path, selected_classes, band_dates):
    """
    Compute mean and standard deviation of NDVI for each class and temporal band.

    Args:
        ndvi_path (str): path of the NDVI geotiff data.
        classes_path (str) : path of the Class geotiff data.
        selected_classes (list): List of selected classes.
        band_dates (list): List of dates corresponding to each band.

    Returns:
        pd.DataFrame: DataFrame containing the computed statistics.
    """
    ndvi = rw.load_img_as_array(ndvi_path)
    classes = rw.load_img_as_array(classes_path)
    classes = np.squeeze(classes)
    results = []
    for band_idx in range(ndvi.shape[2]):
        band_data = ndvi[:, :, band_idx]
        band_data = np.squeeze(band_data)
        for cls in selected_classes:
            class_mask = (classes == cls)
            masked_ndvi = band_data[class_mask]

            mean_ndvi = np.mean(masked_ndvi)
            std_ndvi = np.std(masked_ndvi)

            results.append({
                "class": cls,
                "band": band_dates[band_idx],
                "mean": mean_ndvi,
                "std": std_ndvi
            })
    return pd.DataFrame(results)

def map_class_names(results_df, shapefile_path):
    """
    Map class codes to names using a shapefile.

    Args:
        results_df (pd.DataFrame): DataFrame containing class codes.
        shapefile_path (str): Path to the shapefile with class names.

    Returns:
        pd.DataFrame: Updated DataFrame with class names.
    """
    shapefile = gpd.read_file(shapefile_path)
    shapefile["code"] = shapefile["code"].astype(int)
    code_to_name = dict(zip(shapefile["code"], shapefile["nom"]))
    results_df["class_name"] = results_df["class"].map(code_to_name)
    return results_df

def create_class_to_color_mapping(classes, colormap):
    """
    Create a dictionary mapping classes to unique colors.

    Args:
        classes (list): List of classes.
        colormap (list): List of colors.

    Returns:
        dict: Dictionary mapping each class to a specific color.
    """
    return {cls: colormap[i % len(colormap)] for i, cls in enumerate(classes)}

def plot_ndvi_results(ndvi_path, classes_path, selected_classes, band_dates, shapefile_path, output_path):
    """
    Plot the NDVI results with mean and standard deviation for each class.

    Args:
        results_df (pd.DataFrame): DataFrame containing NDVI results.
        selected_classes (list): List of selected classes.
        class_to_color (dict): Mapping of classes to colors.
        code_to_name (dict): Mapping of class codes to names.
        output_path (str): Path to save the output plot.
    """
    if os.path.exists(ndvi_path):
        code_to_name = load_class_names(shapefile_path)
        # Statistiques du NDVI
        results_df = compute_class_statistics(ndvi_path, classes_path, selected_classes, band_dates)

        # Map class names using shapefile
        results_df = map_class_names(results_df, shapefile_path)

        # Create color mapping for classes
        colors = plt.cm.tab10.colors
        class_to_color = create_class_to_color_mapping(selected_classes, colors)
        plt.figure(figsize=(12, 8))

        # Plot curves for each class
        for cls in selected_classes:
            class_data = results_df[results_df["class"] == cls]
            color = class_to_color[cls]

            # Plot mean
            plt.plot(
                class_data["band"],
                class_data["mean"],
                marker="o",
                color=color,
            )
            # Plot standard deviation
            plt.plot(
                class_data["band"],
                class_data["std"],
                marker="x",
                linestyle="--",
                color=color,
            )

        # Build custom legend
        custom_legend = [
            Line2D([0], [0], color=class_to_color[cls], marker="o", linestyle="-", label=f"{code_to_name[cls]}")
            for cls in selected_classes
        ]

        # Add legend styles for mean and std
        legend_title = [
            Line2D([0], [0], color="black", marker="o", label="Moyenne"),
            Line2D([0], [0], color="black", linestyle="--", marker="x", label="Ecart type"),
        ]

        # Configure legend
        plt.legend(
            handles=legend_title + custom_legend,
            title="Essences d'arbres",
            loc="center",
            frameon=True,
            framealpha=0.8,
            fontsize=10
        )

        # Configure plot
        plt.title("Signature temporelle de la moyenne et de l'écart type du NDVI par classe")
        plt.xlabel("Dates")
        plt.ylabel("NDVI")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Diagramme créé et enregistré dans {output_path}")
        plt.close()
    else:
        print("Le fichier NDVI n'existe pas, veuillez lancer d'abord le script 'pre_traitement.py'.")

# Fonction pour calculer l'indice d'une bande
def calculate_band_index(date_index, band_position, bands_per_date, total_bands):
    """
    Calcule l’indice d’une bande spécifique dans un tableau multidimensionnel
    qui contient plusieurs dates.

    Paramètres
    ----------
    date_index : int
        Indice de la date.
    band_position : int
        Position de la bande dans la liste des bandes pour une date donnée.
    bands_per_date : int
        Nombre total de bandes pour une seule date.
    total_bands : int
        Nombre total de bandes (toutes dates confondues).
    """
    band_index = date_index * bands_per_date + band_position
    if band_index >= total_bands:
        raise ValueError(
            f"L'indice de bande {band_index} dépasse le nombre total de bandes ({total_bands})."
        )
    return band_index

def compute_ndvi(red_band, nir_band):
    """
    Calcule l’Indice de Végétation par Différence Normalisée (NDVI).

    Paramètres
    ----------
    red_band : numpy.ndarray
        Tableau correspondant à la bande rouge.
    nir_band : numpy.ndarray
        Tableau correspondant à la bande infrarouge proche (NIR).
    """
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return np.nan_to_num(ndvi, nan=-9999)

def report_from_dict_to_df(dict_report):
    """
    Convertit le rapport de classification (sous forme de dictionnaire) en un DataFrame Pandas, 
    tout en supprimant les lignes et colonnes jugées non nécessaires.

    Paramètres
    ----------
    dict_report : dict
        Dictionnaire contenant les métriques de classification (précision, recall, f1-score, etc.)
    """
    # Conversion du rapport en DataFrame
    report_df = pd.DataFrame.from_dict(dict_report)

    # Suppression de lignes/colonnes non nécessaires
    try:
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    except KeyError:
        print(dict_report)
        report_df = report_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis=1)

    report_df = report_df.drop(['support'], axis=0)

    return report_df

def classif_pixel(image_filename, sample_filename, id_filename, out_folder, nb_folds, nb_iter):
    """
    Effectue une classification pixel à partir d’une image et de données d’échantillonnage, 
    en utilisant un classifieur par forêts aléatoires (RandomForest) avec validation croisée 
    stratifiée et regroupée.

    Paramètres
    ----------
    image_filename : str
        Chemin vers l’image à classer (format compatible GDAL).
    sample_filename : str
        Chemin vers le shapefile ou raster contenant les échantillons de classes.
    id_filename : str
        Chemin vers le shapefile ou raster contenant les identifiants (IDs) de regroupement 
        (pour la StratifiedGroupKFold).
    out_folder : str
        Chemin du dossier de sortie pour les résultats (cartes, matrices de confusion, rapports).
    nb_folds : int
        Nombre de plis (folds) utilisés lors de la validation croisée.
    nb_iter : int
        Nombre d’itérations de la validation croisée répétée.
    """      

    if os.path.exists(image_filename):
        suffix = '_CV{}folds_stratified_group_x{}times'.format(nb_folds, nb_iter)
        out_classif = os.path.join(out_folder, 'carte_essences_echelle_pixel.tif')
        out_matrix = os.path.join(out_folder, 'matrice{}.png'.format(suffix))
        out_qualite = os.path.join(out_folder, 'qualites{}.png'.format(suffix))

        X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
        _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)


        groups = np.squeeze(groups)
        # Supposons que Y contient toutes les classes

        list_cm = []
        list_accuracy = []
        list_report = []
        Y = Y.squeeze()
        unique_classes = np.unique(Y)

        # Cross-validation stratifiée avec groupes
        for iter_num in range(nb_iter):
            kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)
            
            for fold_num, (train, test) in enumerate(kf.split(X, Y, groups=groups)):
                X_train, X_test = X[train], X[test]
                Y_train, Y_test = Y[train], Y[test]

                # Entraînement
                clf = RF(max_depth=50, oob_score=True, max_samples=0.75, class_weight='balanced', n_jobs=-1)
                clf.fit(X_train, Y_train)

                # Test
                Y_predict = clf.predict(X_test)
                # Imprimer la composition des Y_test et Y_predict
                print(f"Iteration {iter_num}, Fold {fold_num}")
                print("Composition de Y_test :")
                print(dict(zip(*np.unique(Y_test, return_counts=True))))
                print("Composition de Y_predict :")
                print(dict(zip(*np.unique(Y_predict, return_counts=True))))
                # Compute confusion matrix
                cm = confusion_matrix(Y_test, Y_predict)
                
                # Ajouter des classes absentes avec des zéros dans la matrice de confusion
                cm_full = np.zeros((len(unique_classes), len(unique_classes)))
                for i in range(len(unique_classes)):
                    if i < cm.shape[0]:
                        cm_full[i, :cm.shape[1]] = cm[i]
                
                list_cm.append(cm_full)

                # Compute accuracy
                list_accuracy.append(accuracy_score(Y_test, Y_predict))

                # Classification report
                report = classification_report(Y_test, Y_predict,
                                            labels=unique_classes,
                                            output_dict=True)


                # Passer le rapport rempli à la fonction
                list_report.append(report_from_dict_to_df(report))

        # compute mean of cm
        array_cm = np.array(list_cm)
        mean_cm = array_cm.mean(axis=0)

        # compute mean and std of overall accuracy
        array_accuracy = np.array(list_accuracy)
        mean_accuracy = array_accuracy.mean()
        std_accuracy = array_accuracy.std()

        # compute mean and std of classification report
        array_report = np.array(list_report)
        mean_report = array_report.mean(axis=0)
        std_report = array_report.std(axis=0)
        a_report = list_report[0]
        mean_df_report = pd.DataFrame(mean_report, index=a_report.index,
                                columns=a_report.columns)
        std_df_report = pd.DataFrame(std_report, index=a_report.index,
                                columns=a_report.columns)

        # Display confusion matrix
        plots.plot_cm(mean_cm, np.unique(Y_predict))
        plt.savefig(out_matrix, bbox_inches='tight')

        # Display class metrics
        fig, ax = plt.subplots(figsize=(10, 7))
        ax = mean_df_report.T.plot.bar(ax=ax, yerr=std_df_report.T, zorder=2)
        ax.set_ylim(0, 1)
        _ = ax.text(1.5, 0.95, 'OA : {:.2f} +- {:.2f}'.format(mean_accuracy,
                                                            std_accuracy),
                    fontsize=14)
        ax.set_title('Class quality estimation')

        # custom : cuteness
        # background color
        ax.set_facecolor('ivory')
        # labels
        x_label = ax.get_xlabel()
        ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
        y_label = ax.get_ylabel()
        ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
        # borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
        ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
        # grid
        ax.minorticks_on()
        ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                    linewidth=0.5, zorder=1)
        ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                    linewidth=0.3, zorder=1)
        plt.savefig(out_qualite, bbox_inches='tight')

        X_img, _, t_img = cla.get_samples_from_roi(image_filename, image_filename)
        Y_predict = clf.predict(X_img)
        ds = rw.open_image(image_filename)
        nb_row, nb_col, _ = rw.get_image_dimension(ds)

        #initialization of the array
        img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
        #np.Y_predict

        img[t_img[0], t_img[1], 0] = Y_predict
        rw.write_image(out_classif, img, data_set=ds, gdal_dtype=None,
                    transform=None, projection=None, driver_name=None,
                    nb_col=None, nb_ligne=None, nb_band=1)
        print(f"Classification réalisée et enregistré dans {out_classif}")
        print(f"Performances du modèle enregistrées dans {out_folder}")
    else:
        print("L'image pré traitées n'existe pas, veuillez d'abord lancer le script 'pré_traitement.py'.")



def calculate_band_means(ndvi_data, mask):
    """
    Calculate the mean NDVI values for each band based on a mask.

    Args:
        ndvi_data (np.ndarray): NDVI data array (shape: [bands, height, width]).
        mask (np.ndarray): Boolean mask indicating valid pixels.

    Returns:
        np.ndarray: Array of mean values for each band.
    """
    mask = np.squeeze(mask)
    means = []
    for band in range(ndvi_data.shape[2]):
        band_data = ndvi_data[:, :, band]
        band_data = np.squeeze(band_data)
        means.append(np.mean(band_data[mask]))
    return np.array(means)

def calculate_distances(ndvi_data, band_means, mask):
    """
    Calculate the Euclidean distance of each pixel to the centroid for a given class.

    Args:
        ndvi_data (np.ndarray): NDVI data array (shape: [bands, height, width]).
        band_means (np.ndarray): Array of mean values for each band.
        mask (np.ndarray): Boolean mask indicating valid pixels.

    Returns:
        np.ndarray: Array of distances for the masked pixels.
    """
    distances = np.zeros(mask.shape, dtype=np.float32)
    distances = np.squeeze(distances)
    mask = np.squeeze(mask)
    for band in range(ndvi_data.shape[2]):
        distances += np.squeeze(band_means[band] - (ndvi_data[:, :, band])) ** 2
    distances = np.sqrt(distances)
    return distances[mask]


def plot_average_distances(ndvi, classes, code_to_name, classes_of_interest_1, classes_of_interest_2, output_path):
    """
    Create a bar chart showing average distances to centroid for each class, with two color groups.

    Args:
        average_distances (dict): Dictionary of class IDs and their average distances.
        code_to_name (dict): Mapping of class codes to names.
        classes_of_interest_1 (list): First list of class IDs of interest.
        classes_of_interest_2 (list): Second list of class IDs of interest.
        output_path (str): Path to save the plot.
    """
    if os.path.exists(ndvi): 
        average_distances = calculate_average_distances(ndvi, classes, classes_of_interest_1, classes_of_interest_2)
        # Replace class codes with names
        class_names = [code_to_name[class_id] for class_id in average_distances.keys()]
        values = list(average_distances.values())

        # Create a color list based on the two groups
        colors = []
        for class_id in average_distances.keys():
            if class_id in classes_of_interest_1:
                colors.append('peachpuff')  # Color for first list
            elif class_id in classes_of_interest_2:
                colors.append('skyblue')    # Color for second list
            else:
                colors.append('gray')    # Default color if class doesn't belong to any group

        # Plot bar chart with specific colors
        plt.figure(figsize=(14, 8))
        plt.bar(class_names, values, align='center', color=colors)
        plt.xlabel("Essences d'arbres")
        plt.ylabel("Distance moyenne au centroïde")
        plt.title("Distance moyenne au centroïde par essences d'arbres")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    else:
        print("L'image NDVI n'existe pas, veuillez d'abord lancer le script 'pre_traitement.py'.")

def load_class_names(shapefile_path):
    """
    Load class names and their corresponding codes from a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile.

    Returns:
        dict: Mapping of class codes to class names.
    """
    shapefile = gpd.read_file(shapefile_path)
    shapefile["code"] = shapefile["code"].astype(int)
    return dict(zip(shapefile["code"], shapefile["nom"]))

def calculate_average_distances(ndvi, classes, classes_of_interest_1, classes_of_interest_2):
    """
    Calculate the average distances to the centroid for the given classes, divided into two groups.

    Args:
        ndvi_data (np.ndarray): NDVI data array (shape: [bands, height, width]).
        classes_data (np.ndarray): Class data array (shape: [height, width]).
        classes_of_interest_1 (list): First list of class IDs to process.
        classes_of_interest_2 (list): Second list of class IDs to process.

    Returns:
        dict: Dictionary with class IDs as keys and average distances as values.
    """
    ndvi_data = rw.load_img_as_array(ndvi)  
    classes_data = rw.load_img_as_array(classes)  

    average_distances = {}

    # Loop through both lists of classes
    for class_id in (classes_of_interest_1 + classes_of_interest_2):  # Combine both lists
        class_mask = classes_data == class_id
        if not np.any(class_mask):
            print(f"No pixels found for class {class_id}")
            continue

        # Calculate band means for the class
        band_means = calculate_band_means(ndvi_data, class_mask)
        # Calculate distances to centroid
        distances = calculate_distances(ndvi_data, band_means, class_mask)

        # Calculate average distance
        average_distance = np.mean(distances)
        average_distances[class_id] = average_distance

    return average_distances


def calculate_average_distances_class_poly(ndvi, classes, classes_of_interest_1, classes_of_interest_2, output_violin_path):
    """
    Calcule les distances moyennes au centroïde (en termes de valeurs NDVI) pour différentes 
    classes d’intérêt, puis génère un diagramme en violon illustrant la distribution 
    de ces distances par classe et par polygone.

    Paramètres
    ----------
    ndvi : str
        Chemin vers le fichier raster NDVI (fichier .tif).
    classes : str
        Chemin vers le fichier raster des classes (fichier .tif).
    classes_of_interest_1 : list
        Liste des identifiants de classes d’intérêt (ex. [1,2,3]) correspondant à un premier 
        groupe de classes.
    classes_of_interest_2 : list
        Liste des identifiants de classes d’intérêt correspondant à un second groupe de classes.
    output_violin_path : str
        Chemin où sera enregistré le diagramme en violon (fichier .png ou .jpg).
    """
    if os.path.exists(ndvi):
        output_id_tif = '/home/onyxia/work/results/data/img_pretraitees/forest_id_all.tif' 
        shapefile = '/home/onyxia/work/results/data/sample/Sample_BD_foret_T31TCJ_all.shp'
        output_id_shp = '/home/onyxia/work/results/data/sample/forest_id_all.shp'
        gdf = gpd.read_file(shapefile)
        make_id(gdf, output_id_shp)

        ndvi_data = rw.load_img_as_array(ndvi)  
        classes_data = rw.load_img_as_array(classes)
        # Dictionnaire pour stocker les distances au centroïde par classe
        distances_by_class = {}
        
        # Remplir le dictionnaire distances_by_class avec les distances par polygone et par classe
        for class_id in classes_of_interest_1 + classes_of_interest_2:  # Union des deux listes
            class_mask = classes_data == class_id
            distances_per_polygon = []  # Liste pour stocker les distances par polygone de cette classe

            for zone_id in np.unique(output_id_tif):
                zone_mask = output_id_tif == zone_id
                combined_mask = class_mask & zone_mask
                
                # Calculate band means for the class
                band_means = calculate_band_means(ndvi_data, combined_mask)

                # Calculate distances to centroid
                distances = calculate_distances(ndvi_data, band_means, combined_mask)
                distances_per_polygon.extend(distances)  # Ajouter les distances du polygone dans la liste
            
            # Ajouter les distances de tous les polygones d'une classe
            distances_by_class[class_id] = distances_per_polygon
        
        # Transformation des données pour le violon plot
        data_for_violin = []
        unique_classes = list(distances_by_class.keys())

        for class_id in unique_classes:
            distances = distances_by_class[class_id]
            data_for_violin.append(distances)
        
        # Création du diagramme en violon
        fig, ax = plt.subplots(figsize=(14, 8))
        parts = ax.violinplot(data_for_violin, showmeans=True, showmedians=True, showextrema=True)

        # Paramétrages de l'axe et des titres
        ax.set(
            title="Distribution des distances au centroïde par polygone et par classe",
            xlabel="Classes d'intérêt",
            ylabel="Distance au centroïde"
        )

        # Ajustement des labels de l'axe X
        ax.set_xticks(np.arange(1, len(unique_classes) + 1))
        ax.set_xticklabels([f"Classe {cls}" for cls in unique_classes], rotation=45, ha='right')

        # Personnalisation des violons : appliquer des couleurs distinctes pour les deux listes
        for i, violin_body in enumerate(parts['bodies']):
            if unique_classes[i] in classes_of_interest_1:
                violin_body.set(facecolor='peachpuff', edgecolor='black', alpha=0.7)  # Orange pour la 1ère liste
            elif unique_classes[i] in classes_of_interest_2:
                violin_body.set(facecolor='skyblue', edgecolor='black', alpha=0.7)  # Bleu pour la 2ème liste

        for line_type in ['cbars', 'cmins', 'cmaxes']:
            parts[line_type].set(lw=0.5, ls='--', color='gray')

        # Style spécifique pour la moyenne
        parts['cmeans'].set(lw=1, ls='-', color='black')  # Ligne plus épaisse et rouge pour la moyenne

        # Style spécifique pour la médiane
        parts['cmedians'].set(lw=1, ls='-', color='darkgray')  # Ligne plus épaisse et bleue pour la médiane
        plt.tight_layout()
        plt.savefig(output_violin_path)
        print(f"Diagramme en violon créé et enregistré dans {output_violin_path}")
    else:
        print("L'image NDVI n'existe pas, veuillez d'abord lancer le script 'pre_traitement.py'.")


