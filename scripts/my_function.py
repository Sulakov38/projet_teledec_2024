import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal
from rasterstats import zonal_stats
import seaborn as sns
import sys
sys.path.append('libsigma/')
import read_and_write as rw

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

def pixels_per_class(input_image, shapefile_path, output_pix_path):
    """
    Analyzes the number of pixels for each class in a raster and generates a bar chart.
    Args:
        input_image (str): Path to the raster image.
        shapefile_path (str): Path to the shapefile containing class information.
        output_pix_path (str): Path to save the output bar chart.
    """
    # Load raster and compute unique pixel values
    data_set = rw.open_image(input_image)
    img = rw.load_img_as_array(input_image)
    unique_values, counts = np.unique(img, return_counts=True)

    # Load and filter shapefile
    shapefile = gpd.read_file(shapefile_path)
    allowed_codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]
    shapefile["code"] = shapefile["code"].astype(int)
    shapefile = shapefile[shapefile["code"].isin(allowed_codes)]

    # Map codes to class names
    code_to_name = dict(zip(shapefile["code"], shapefile["nom"]))
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

def pixels_per_polygons_per_class(raster_path, shapefile_path, output_violin_path):
    """
    Computes and visualizes the pixel distribution per polygon for each class.
    Args:
        raster_path (str): Path to the raster image.
        shapefile_path (str): Path to the shapefile containing class information.
        output_violin_path (str): Path to save the output violin plot.
    """
    # Load raster and shapefile
    data_set = rw.open_image(raster_path)
    img = rw.load_img_as_array(raster_path)
    shapefile = gpd.read_file(shapefile_path)

    # Filter shapefile by allowed codes
    allowed_codes = [11, 12, 13, 14, 21, 22, 23, 24, 25]
    shapefile["code"] = shapefile["code"].astype(int)
    shapefile = shapefile[shapefile["code"].isin(allowed_codes)]

    # Compute pixel counts per polygon
    stats = zonal_stats(
        shapefile,
        raster_path,
        stats=['count'],
        nodata=0
    )
    stats_df = pd.DataFrame(stats)
    shapefile["nombre_pixels"] = stats_df["count"]

    # Create a violin plot for pixel distribution
    plt.figure(figsize=(14, 8))
    sns.violinplot(x="nom", y="nombre_pixels", data=shapefile, cut=0, scale="width", inner="quartile", linewidth=1.2, color='skyblue')
    plt.yscale("log")
    plt.title("Distribution du nombre de pixels par polygone pour chaque essences d'arbres")
    plt.xlabel("Essences d'arbres")
    plt.ylabel("Nombre de pixels par polygone (Echelle logarithmique)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_violin_path)

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
