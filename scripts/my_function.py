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
    Reprojects and clips a raster to a new coordinate reference system and extent.
    Args:
        minx (float): Minimum X-coordinate of the extent.
        miny (float): Minimum Y-coordinate of the extent.
        maxx (float): Maximum X-coordinate of the extent.
        maxy (float): Maximum Y-coordinate of the extent.
        input_raster (str): Path to the input raster file.
        output_raster (str): Path to the output raster file.
        src_epsg (int): EPSG code of the source coordinate system.
        dst_epsg (int): EPSG code of the target coordinate system.
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

def preparation(releves, bands, emprise, dirname, out_dirname, forest_mask):
    """
    Prepares Sentinel-2 images by reprojecting, masking, and storing them in a list.
    Args:
        releves (list): List of Sentinel-2 acquisition identifiers.
        bands (list): List of band indices to process.
        emprise (str): Path to the raster defining the study extent.
    """
    suffixe = '.tif'
    minx, miny, maxx, maxy = get_img_extend(emprise)
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
    img_merge = os.path.join(out_dirname, 'Serie_temp_S2_allbands_merge.tif')  
    output_img = os.path.join(out_dirname, 'Serie_temp_S2_allbands.tif')
    merge(img_all_band, img_merge)
    nodata(img_merge, output_img, 0)
    os.remove(img_merge)


def nodata(input_raster, output_raster, value):
    """
    Sets the no-data value of a raster.
    Args:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path to the output raster file.
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

def rasterize_emprise(in_vector, out_image, field_name, sptial_resolution):
    """
    Rasterizes a vector layer based on a specified field and spatial resolution.
    Args:
        in_vector (str): Path to the input vector file.
        out_image (str): Path to the output raster file.
        field_name (str): Name of the field to rasterize.
        sptial_resolution (float): Spatial resolution for the output raster.
    """
    # Define the command pattern for rasterization
    cmd_pattern = ("gdal_rasterize -a {field_name} "
                   "-tr {sptial_resolution} {sptial_resolution} "
                   "-ot Byte -of GTiff "
                   "-a_nodata 0 "
                   "{in_vector} {out_image}")
    
    # Format the command with provided parameters
    cmd = cmd_pattern.format(in_vector=in_vector, out_image=out_image,
                             field_name=field_name, sptial_resolution=sptial_resolution)
    
    # Print and execute the command
    print(cmd)
    os.system(cmd)

def get_img_extend(image_filename):
    """
    Retrieves the geographic extent of an image file.
    Args:
        image_filename (str): Path to the image file.
    Returns:
        tuple: (minx, miny, maxx, maxy) geographic extent of the image.
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

def code_vege(shapefile_path_vege, shapefile_path_emprise, output_path):
    """
    Processes a vegetation shapefile and assigns codes and names based on a predefined mapping.
    Clips the processed shapefile to a given extent and saves the output.
    Args:
        shapefile_path_vege (str): Path to the vegetation shapefile.
        shapefile_path_emprise (str): Path to the extent shapefile.
        output_path (str): Path to save the processed shapefile.
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
    vege_emprise.to_file(output_path)

def rasterize(in_vector, out_image, field_name, sptial_resolution, emprise, type_data):
    """
    Rasterizes a vector layer with a specified extent, field, and spatial resolution.
    Args:
        in_vector (str): Path to the input vector file.
        out_image (str): Path to the output raster file.
        field_name (str): Name of the field to rasterize.
        sptial_resolution (float): Spatial resolution for the output raster.
        emprise (str): Path to the vector file defining the extent.
    """
    # Get the extent of the emprise (bounding box)
    minx, miny, maxx, maxy = get_img_extend(emprise)
    
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

def masque(input_image, output_masque):
    """
    Generates a mask from an input image where non-zero values are retained.
    Args:
        input_image (str): Path to the input raster file.
        output_masque (str): Path to save the mask raster file.
    """
    # Open the input image and load it as a numpy array
    data_set = rw.open_image(input_image)
    img = rw.load_img_as_array(input_image)
    
    # Create a binary mask where values are non-zero
    masque_foret = img != 0
    
    # Write the mask to an output file
    rw.write_image(output_masque, masque_foret, data_set=data_set,
                   gdal_dtype=data_type_match['uint8'])

def load_shapefile(file_path):
    """
    Loads a shapefile as a GeoDataFrame.
    Args:
        file_path (str): Path to the shapefile.
    Returns:
        GeoDataFrame: Loaded shapefile as a GeoDataFrame.
    """
    return gpd.read_file(file_path)

def filter_polygons_by_code(dataframe, code_column, allowed_codes):
    """
    Filters polygons in a GeoDataFrame based on allowed codes.
    Args:
        dataframe (GeoDataFrame): The GeoDataFrame to filter.
        code_column (str): The column containing the codes to filter by.
        allowed_codes (list): List of allowed codes.
    Returns:
        GeoDataFrame: Filtered GeoDataFrame containing only polygons with allowed codes.
    """
    return dataframe[dataframe[code_column].isin(allowed_codes)]

def calculate_polygons_per_class(dataframe, class_column):
    """
    Counts the number of polygons per class in a GeoDataFrame.
    Args:
        dataframe (GeoDataFrame): The GeoDataFrame containing polygons.
        class_column (str): The column containing class labels.
    Returns:
        Series: Count of polygons per class.
    """
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

def pixels_per_class(input_image, dataframe, output_pix_path):
    """
    Analyzes the number of pixels for each class in a raster and generates a bar chart.
    Args:
        input_image (str): Path to the raster image.
        dataframe (str): Path to the GeoDataFrame containing class information.
        output_pix_path (str): Path to save the output bar chart.
    """
    # Load raster and compute unique pixel values
    img = rw.load_img_as_array(input_image)
    unique_values, counts = np.unique(img, return_counts=True)

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

def make_id(dataframe, emprise, output_id_shp, output_id_tif):
    dataframe["code"] = dataframe["code"].astype(int)
    dataframe["unique_id"] = range(1, len(dataframe) + 1)
    output_id_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/forest_id.shp'
    dataframe.to_file(output_id_shp)

    field = "unique_id"
    output_id_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/forest_id.tif'
    resolution = 10
    type_data = 'Uint16'
    rasterize(output_id_shp, output_id_tif, field, resolution, emprise, type_data)

def pixels_per_polygons_per_class(dataframe, output_violin_path):
    """
    Computes and visualizes the pixel distribution per polygon for each class.
    Args:
        dataframe (str): Path to the GeoDataFrame containing class information.
        output_violin_path (str): Path to save the output violin plot.
    """
    output_id_shp = '/home/onyxia/work/projet_teledec_2024/results/data/sample/forest_id.shp'
    output_id_tif = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/forest_id.tif'
    emprise = '/home/onyxia/work/projet_teledec_2024/results/data/img_pretraitees/emprise.tif'
    make_id(dataframe, emprise, output_id_shp, output_id_tif)
    array = rw.load_img_as_array(output_id_tif)
    counts = np.bincount(array.flatten())

    # Ajout de l'information "nombre_pixels" au GeoDataFrame
    dataframe["nombre_pixels"] = dataframe["unique_id"].apply(lambda x: counts[x] if x < len(counts) else 0)

    # Préparation des données pour le diagramme en violon
    unique_noms = dataframe["nom"].unique()
    data_for_violin = [dataframe.loc[dataframe["nom"] == n, "nombre_pixels"].values for n in unique_noms]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Création du diagramme en violon
    parts = ax.violinplot(data_for_violin, showmeans=True, showmedians=True, showextrema=True)

    # Paramétrages de l'axe et des titres
    ax.set(
        yscale='log',
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
    for line_type in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[line_type].set(lw=0.5, ls='--', color='black')

    plt.tight_layout()
    plt.savefig(output_violin_path)

def load_raster_as_array(raster_path):
    """
    Load a raster as a NumPy array using GDAL.

    Args:
        raster_path (str): Path to the raster file.

    Returns:
        np.ndarray: Array containing raster data. If multiple bands exist, a 3D array is returned.
    """
    dataset = gdal.Open(raster_path)
    array = []
    for i in range(1, dataset.RasterCount + 1):  # Read all bands
        band = dataset.GetRasterBand(i)
        array.append(band.ReadAsArray())
    return np.array(array) if dataset.RasterCount > 1 else np.array(array[0])

def compute_class_statistics(ndvi, classes, selected_classes, band_dates):
    """
    Compute mean and standard deviation of NDVI for each class and temporal band.

    Args:
        ndvi (np.ndarray): NDVI data as a 3D array (bands, rows, columns).
        classes (np.ndarray): Class data as a 2D array.
        selected_classes (list): List of selected classes.
        band_dates (list): List of dates corresponding to each band.

    Returns:
        pd.DataFrame: DataFrame containing the computed statistics.
    """
    results = []
    for band_idx in range(ndvi.shape[0]):
        band_data = ndvi[band_idx]
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

def plot_ndvi_results(results_df, selected_classes, class_to_color, code_to_name, output_path):
    """
    Plot the NDVI results with mean and standard deviation for each class.

    Args:
        results_df (pd.DataFrame): DataFrame containing NDVI results.
        selected_classes (list): List of selected classes.
        class_to_color (dict): Mapping of classes to colors.
        code_to_name (dict): Mapping of class codes to names.
        output_path (str): Path to save the output plot.
    """
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
    plt.close()

# Fonction pour charger une image en tant que tableau numpy
def load_img_as_array(filename):
    """
    Charge une image et retourne un tableau numpy et son dataset GDAL.
    """
    dataset = gdal.Open(filename)
    if not dataset:
        raise FileNotFoundError(f"Impossible d'ouvrir le fichier {filename}")
    return dataset.ReadAsArray(), dataset

# Fonction pour écrire une image à partir d'un tableau numpy
def write_image(output_filename, array, reference_ds, gdal_dtype):
    """
    Enregistre un tableau numpy dans un fichier TIFF en utilisant un dataset de référence.
    """
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_filename,
        reference_ds.RasterXSize,
        reference_ds.RasterYSize,
        array.shape[0],  # Nombre de bandes
        gdal_dtype
    )
    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())

    for i in range(array.shape[0]):
        out_ds.GetRasterBand(i + 1).WriteArray(array[i])

    out_ds.FlushCache()
    out_ds = None

# Fonction pour calculer l'indice d'une bande
def calculate_band_index(date_index, band_position, bands_per_date, total_bands):
    """
    Calcule l'indice de bande pour une date et une position données.
    Vérifie que l'indice ne dépasse pas le nombre total de bandes.
    """
    band_index = date_index * bands_per_date + band_position
    if band_index >= total_bands:
        raise ValueError(f"L'indice de bande {band_index} dépasse le nombre total de bandes ({total_bands}).")
    return band_index

# Fonction pour calculer le NDVI
def compute_ndvi(red_band, nir_band):
    """
    Calcule le NDVI à partir des bandes rouge et infrarouge.
    Gère les divisions par zéro en remplaçant les NaN par -9999.
    """
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return np.nan_to_num(ndvi, nan=-9999)  # Remplace les NaN par -9999

def report_from_dict_to_df(dict_report):

    # convert report into dataframe
    report_df = pd.DataFrame.from_dict(dict_report)

    # drop unnecessary rows and columns
    try :
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    except KeyError:
        print(dict_report)
        report_df = report_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis=1)

    report_df = report_df.drop(['support'], axis=0)

    return report_df

def classif_pixel(image_filename, sample_filename, id_filename, nb_iter, nb_folds):

    # outputs
    suffix = '_CV{}folds_stratified_group_x{}times'.format(nb_folds, nb_iter)
    out_folder = '/home/onyxia/work/data/'
    out_classif = os.path.join(out_folder, 'ma_classif{}.tif'.format(suffix))
    out_matrix = os.path.join(out_folder, 'ma_matrice{}.png'.format(suffix))
    out_qualite = os.path.join(out_folder, 'mes_qualites{}.png'.format(suffix))

    X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)

    list_cm = []
    list_accuracy = []
    list_report = []
    groups = np.squeeze(groups)

    # Iter on stratified K fold
    for _ in range(nb_iter):
        kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)
        for train, test in kf.split(X, Y, groups=groups):
            X_train, X_test = X[train], X[test]
            Y_train, Y_test = Y[train], Y[test]

            # 3 --- Train
            #clf = SVC(cache_size=6000)
            clf = RF(max_depth=50,oob_score=True,max_samples=0.75,class_weight='balanced')
            clf.fit(X_train, Y_train)

            # 4 --- Test
            Y_predict = clf.predict(X_test)

            # compute quality
            list_cm.append(confusion_matrix(Y_test, Y_predict))
            list_accuracy.append(accuracy_score(Y_test, Y_predict))
            report = classification_report(Y_test, Y_predict,
                                            labels=np.unique(Y_predict),
                                            output_dict=True)

            # store them
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
    ax.set_ylim(0.5, 1)
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

def classif_pixel_19(image_filename,sample_filename,id_filename,nb_folds):
    suffix = '_CV{}folds_stratified_group'.format(nb_folds)
    out_folder = '/home/onyxia/work/data/'
    out_classif = os.path.join(out_folder, 'ma_classif2_{}.tif'.format(suffix))
    out_matrix = os.path.join(out_folder, 'ma_matrice2_{}.png'.format(suffix))
    out_qualite = os.path.join(out_folder, 'mes_qualites2_{}.png'.format(suffix))

    X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)

    groups = np.squeeze(groups)
    # Iter on stratified K fold
    kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)
    for train, test in kf.split(X, Y, groups=groups):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]

        # 3 --- Train
        #clf = SVC(cache_size=6000)
        clf = RF(max_depth=50, oob_score=True, max_samples=0.75, class_weight='balanced', n_jobs=50)
        clf.fit(X_train, Y_train)

        # 4 --- Test
        Y_predict = clf.predict(X_test)

        # compute quality
    cm = confusion_matrix(Y_test, Y_predict)
    report = classification_report(Y_test, Y_predict, labels=np.unique(Y_predict), output_dict=True)
    accuracy = accuracy_score(Y_test, Y_predict)

    # display and save quality
    plots.plot_cm(cm, np.unique(Y_predict), out_filename=out_matrix)
    plots.plot_class_quality(report, accuracy, out_filename=out_qualite)

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


def classif_pixel_20(image_filename, sample_filename, id_filename, nb_folds, nb_iter):
    suffix = '_CV{}folds_stratified_group_x{}times'.format(nb_folds, nb_iter)
    out_folder = '/home/onyxia/work/data/'
    out_classif = os.path.join(out_folder, 'ma_classif{}.tif'.format(suffix))
    out_matrix = os.path.join(out_folder, 'ma_matrice{}.png'.format(suffix))
    out_qualite = os.path.join(out_folder, 'mes_qualites{}.png'.format(suffix))

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
            clf = RF(max_depth=50, oob_score=True, max_samples=0.75, class_weight='balanced', n_jobs=50)
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
    ax.set_ylim(0.5, 1)
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

def load_raster(file_path):
    """
    Load a raster file and return its data as a NumPy array.

    Args:
        file_path (str): Path to the raster file.

    Returns:
        np.ndarray: Array containing the raster data.
    """
    dataset = gdal.Open(file_path)
    if not dataset:
        raise FileNotFoundError(f"Unable to open file: {file_path}")
    data = dataset.ReadAsArray()
    return data

def calculate_band_means(ndvi_data, mask):
    """
    Calculate the mean NDVI values for each band based on a mask.

    Args:
        ndvi_data (np.ndarray): NDVI data array (shape: [bands, height, width]).
        mask (np.ndarray): Boolean mask indicating valid pixels.

    Returns:
        np.ndarray: Array of mean values for each band.
    """
    means = []
    for band in range(ndvi_data.shape[0]):
        band_data = ndvi_data[band, :, :]
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
    for band in range(ndvi_data.shape[0]):
        distances += (ndvi_data[band, :, :] - band_means[band]) ** 2
    distances = np.sqrt(distances)
    return distances[mask]

def plot_average_distances(average_distances, code_to_name, output_path):
    """
    Create a bar chart showing average distances to centroid for each class.

    Args:
        average_distances (dict): Dictionary of class IDs and their average distances.
        code_to_name (dict): Mapping of class codes to names.
        output_path (str): Path to save the plot.
    """
    # Replace class codes with names
    class_names = [code_to_name[class_id] for class_id in average_distances.keys()]
    values = list(average_distances.values())

    # Plot bar chart
    plt.bar(class_names, values, align='center', color='skyblue')
    plt.xlabel("Essences d'arbres")
    plt.ylabel("Distance moyenne au centroide")
    plt.title("Distance moyenne au centroide par essences d'arbres")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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

def calculate_average_distances(ndvi_data, classes_data, classes_of_interest):
    """
    Calculate the average distances to the centroid for the given classes.

    Args:
        ndvi_data (np.ndarray): NDVI data array (shape: [bands, height, width]).
        classes_data (np.ndarray): Class data array (shape: [height, width]).
        classes_of_interest (list): List of class IDs to process.

    Returns:
        dict: Dictionary with class IDs as keys and average distances as values.
    """
    average_distances = {}

    for class_id in classes_of_interest:
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
