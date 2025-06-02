'''
The osmium and GDAL libraries are recommended for preparatory works with OpenStreetMap (OSM) data.
The further data processing and visualisation works will be conducted in GIS tool like ArcGIS Pro.

Links:
Download OSM datasets: https://download.geofabrik.de
Official website Osmium Tool: https://osmcode.org/osmium-tool/
Github Osmium Tool: https://github.com/osmcode/osmium-tool
Official website GDAL: https://gdal.org/en/stable/
Github GDAL: https://github.com/OSGeo/gdal

Note:
    When downloading OSM datasets, click on the underlined name of a (sub-)region to
    access the defined sub-regions within that scope.
    The downloaded file is in the .osm.pbf format, which is not directly supported by ArcGIS Pro for loading.
'''
from numpy.f2py.crackfortran import updatevars

#----- MacOS -----#
# install libraries (they can also be installed by following the Github instructions, see link above)
brew install osmium-tool
brew install gdal

# use osmium to filter the data (example: primary road for Italy)
osmium tags-filter /path/to/italy-latest.osm.pbf w/highway=primary -o italy-highway-primary.osm.pbf

# use osmium to convert the data from .osm.pbf to .osm, which can be directly loaded into QGIS and ArcMap with expansion tools
osmconvert italy-highway-primary.osm.pbf -o=italy-highway-primary.osm

# use gdal to convert the road data from .osm to shapefile, which can be directly loaded into ArcGIS Pro
ogr2ogr -f "ESRI Shapefile" /Users/ykk/Desktop/Thesis/data/italy-highway-primary.shp /Users/ykk/Desktop/Thesis/data/italy-highway-primary.osm -nlt LINESTRING

#----- Ubuntu -----#
# install dependencies
sudo apt update
sudo apt install osmium-tool gdal-bin

# use osmium to filter the data (example: promary road for Italy)
osmium tags-filter /path/to/italy-latest.osm.pbf w/highway=primary -o italy-highway-primary.osm.pbf

# convert from .osm.pbf to .osm
osmconvert italy-highway-primary.osm.pbf -o=italy-highway-primary.osm

# convert the road data from .osm to shapefile
ogr2ogr -f "ESRI Shapefile" /home/username/Desktop/Thesis/data/italy-highway-primary.shp /home/username/Desktop/Thesis/data/italy-highway-primary.osm -nlt LINESTRING

#----- Windows -----#
# install dependencies from the above links

# use osmium to filter the data (example: primary road for Italy)
osmium tags-filter C:\path\to\italy-latest.osm.pbf w/highway=primary -o italy-highway-primary.osm.pbf

# convert from .osm.pbf to .osm
osmconvert italy-highway-primary.osm.pbf -o=italy-highway-primary.osm

# convert the road data from .osm to shapefile
ogr2ogr -f "ESRI Shapefile" C:\Users\username\Desktop\Thesis\data\italy-highway-primary.shp C:\Users\username\Desktop\Thesis\data\italy-highway-primary.osm -nlt LINESTRING
