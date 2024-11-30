import sys
sys.path.append('libsigma/')
import geopandas as gpd
import os
import read_and_write as rw

my_folder = 'data/project/'
in_vector = os.path.join(my_folder, 'vege.shp')
out_image = os.path.splitext(in_vector)[0] + '_v2.tif'

field_name = 'code'  # field containing the numeric label of the classes

sptial_resolution = 10.0

# define command pattern to fill with paremeters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-ot Byte -of GTiff "
               "-a_nodata 0 "
               "{in_vector} {out_image}")

# fill the string with the parameter thanks to format function
cmd = cmd_pattern.format(in_vector=in_vector, out_image=out_image, field_name=field_name,
                         sptial_resolution=sptial_resolution)

# execute the command in the terminal
print(cmd)
os.system(cmd)
