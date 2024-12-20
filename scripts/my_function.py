import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal
from rasterstats import zonal_stats
import seaborn as sns
import os
import sys
sys.path.append('libsigma/')
import read_and_write as rw

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

def preparation(releves, bands, emprise, forest):
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
            band_name = f'/home/onyxia/work/data/images//SENTINEL2{r}{B}'
            band_file = f'{band_name}{suffixe}'
            output_filename = f"{band_name}_10_2154{suffixe}"
            
            # Reproject the raster
            reproject_raster(minx, miny, maxx, maxy, band_file, output_filename, src_epsg, dst_epsg)
            
            # Load reprojected image and apply the forest mask
            band_reproj = rw.load_img_as_array(output_filename)
            masque_foret = rw.load_img_as_array(forest).astype('bool')
            band_masked = band_reproj.copy()
            band_masked[~masque_foret] = 0
            
            # Append the masked band to the list
            img_all_band.append(band_masked)
    return img_all_band

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

def rasterize(in_vector, out_image, field_name, sptial_resolution, emprise):
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
                   "-ot Byte -of GTiff "
                   "-a_nodata 0 "
                   "{in_vector} {out_image}")
    
    # Format the command with provided parameters
    cmd = cmd_pattern.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy,
                             in_vector=in_vector, out_image=out_image,
                             field_name=field_name, sptial_resolution=sptial_resolution)
    
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
