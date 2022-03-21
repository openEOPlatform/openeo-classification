import pandas as pd
from openeo_classification.features import *
import ipywidgets as widgets
import datetime
from openeo_classification.connection import connection
import utm
import pyproj
import shapely
import numpy as np
from shapely.geometry import Point

# df = pd.read_csv("lucas/EU_2018_20200213.CSV")
# print(df.columns)
# print(df.head())

# print(len(df))

# pd.set_option('display.max_rows', None)
# print(df.groupby(["LC1"]).count()["POINT_ID"])

lookup_lucas = {
	"A00": "Artificial land",
	"A10": "Roofed built-up areas",
	"A20": "Artificial non-built up areas",
	"A30": "Other artificial areas",
	"B00": "Cropland",
	"B10": "Cereals",
	"B20": "Root crops",
	"B30": "Non-permanent industrial crops",
	"B40": "Dry pulses, vegetables and flowers",
	"B50": "Fodder crops",
	"B70": "Permanent crops: fruit trees",
	"B80": "Other permanent crops",
	"C00": "Woodland",
	"C10": "Broadleaved woodland",
	"C20": "Coniferous woodland",
	"C30": "Mixed woodland",
	"D00": "Shrubland",
	"D10": "Shrubland with sparse tree cover",
	"B20": "Shrubland without tree cover",
	"E00": "Grassland",
	"E10": "Grassland with sparse tree/shrub cover",
	"E20": "Grassland without tree/shrub cover",
	"E30": "Spontaneously re-vegetated surfaces",
	"F00": "Bare land and lichens/moss",
	"F10": "Rocks and stones",
	"F20": "Sand",
	"F30": "Lichens and moss",
	"F40": "Other bare soil",
	"G00": "Water areas",
	"G10": "Inland water bodies",
	"G20": "Inland running water",
	"G30": "Transitional water bodies",
	"G40": "Sea and ocean",
	"G50": "Glaciers, permanent snow",
	"H00": "Wetlands",
	"H10": "Inland wetlands",
	"H20": "Coastal wetlands"
}




# generate meaningful and consistent predictive features from S1 and S2 data; I.e. derive pixel based time series metrics(e.g. median, 
# percentiles, standard deviation) on a stack of S1 (Sigma0 or Gamma0 VV & VH backscatter, potentially complemented with 6-day coherence) 
# and S2 ARD data (spatially co-registered, atmospherically corrected, and cloud/shadow masked time series of 10 & 20m surface reflectance 
# bands and a set of derived vegetation indices) for the reference period;  CHECK

# generate a set of relevant training data taking into account the user specified legend and area of interest. This mainly contains a 
# set of filtering & cleaning operations based on the 2018 EUROSTAT LUCAS land survey data, using the available tools in the platform; 

### SPECIFIC LEGEND
### SPECIFIC AREA OF INTEREST

### ZOEK UIT WELKE COORDINATEN JE NODIG HEBT TH_LONG TH_LAT
### ZOEK UIT WELKE RIJEN JE WILT HOUDEN (INTERESTING FEATURE: GPS_ALTITUDE)
### ALLEEN EEN LC1 die geen LC2 heeft? (waar LC1_perc 100% is)
### CONVERT NAAR POINT
### CLIPPEN MET GEBIED GESELECTEERD DOOR GEBRUIKER ?
### OOK STRATIFICATIELAGEN DOOR GEBRUIKER LATEN TEKENEN

# generate a set of validation data to assess the map accuracy. For this task, a user defined percentage of the input training data will 
# be used to split into a final training and validation dataset taking into account the spatial location of the training points; 

# rasterize ancillary datasets such as the World Settlement Footprint (Esch et al. 2017) or the Global Surface Water (GSW) product 
# (Pekel et al., 2016) to the desired target resolution. These ancillary layers can either be used to pre-classify certain pixels or 
# used within expert rules (see below); 

# parameterize machine learning models. Herein, the user will have the option to 
	# Select its preferred machine learning algorithm out a list of state of the art algorithms (e.g. random forest, support vector machines, NN, ...); 
	# either select custom settings for hyper-parameter tuning or use an automated approached in which a combined gridand random search 
		# within a k-folded cross-validation is applied to identify the optimal model parameter; 
	# parameterize the machine learning algorithms either using the full training data set, or within customized regions (e.g. ecozones); 
	# Select either feature fusion or decision fusion so users can select either to train their machine learning algorithms using the stack
		# of S1 & S2 predictive features together or separately and combine the output of both models. 

# predict the target class as well its probability based on the parametrized machine learning models and spatially explicit S1 & S2 predictive 
# features, potentially complemented with expert rules taking into account the ancillary datasets; 

# Assess the map accuracy using the validation data set. Well-known accuracy values such as kappa, accuracy or F-score will be generated 
# per class and overall.  

# As a test dataset, a 10 m land cover over Europe (EEA39) will be generated using random forest, feature fusion and ecozone optimized 
# hyper-parameter tuning based on successful experiences with this methodology. As a minimum we consider the following 13 classes within 
# this test dataset, but more classes can be added after discussion at kickoff: 1) Broadleaved woodland; 2) Coniferous woodland; 
# 3) Mixed woodland; 4) Shrubland; 5) Grassland; 6) Cropland; 7) Bare land; 8) Lichens and Moss; 9) Built-up; 10) Inland Water; 
# 11) Sea and ocean; 12) Glaciers and Permanent Snow; 13) Wetlands. 


def load_lc_features(feature_raster, aoi, start_date, end_date, stepsize_s2=10, stepsize_s1=12):
    provider = "terrascope"
    
    idx_dekad = sentinel2_features(start_date, end_date, connection, provider, processing_opts={}, sampling=True, stepsize=stepsize_s2)
    idx_features = compute_statistics(idx_dekad, start_date, end_date, stepsize=stepsize_s2)

    s1_dekad = sentinel1_features(start_date, end_date, connection, provider, processing_opts={}, orbitDirection="ASCENDING", sampling=True, stepsize=stepsize_s1)
    s1_dekad = s1_dekad.resample_cube_spatial(idx_dekad)
    s1_features = compute_statistics(s1_dekad, start_date, end_date, stepsize=stepsize_s1)

    features = idx_features.merge_cubes(s1_features)

    if feature_raster == "s1":
        return s1_features, features.metadata.band_names
    elif feature_raster == "s2":
        return idx_features, idx_features.metadata.band_names
    else:
        return features, features.metadata.band_names
    


# https://github.com/jupyter-widgets/ipywidgets/blob/master/python/ipywidgets/ipywidgets/widgets/widget_upload.py
def getStartingWidgets():
    train_test_split = widgets.FloatSlider(value=0.3, min=0, max=1.0, step=0.05)
    algorithm = widgets.Dropdown(options=['Random Forest'], value='Random Forest', description='Model:', disabled=True)
    nrtrees = widgets.IntText(value=250, description='Nr trees:')
    mtry = widgets.IntText(value=3, description="Mtry:")
    feature_raster = widgets.RadioButtons(options=['Feature fusion', 'Decision fusion'])
    aoi = widgets.FileUpload(accept='.geojson,.shp',multiple=False, #style=widgets.ButtonStyle(button_color='#F0F0F0'), 
                             layout=widgets.Layout(width='20em'), description="Upload AOI")
    strat_layer = widgets.FileUpload(accept='.geojson,.shp',multiple=False, 
                                     layout=widgets.Layout(width='20em'), description="Upload stratification")
    include_mixed_pixels = widgets.RadioButtons(options=['Yes', 'No'])
    start_date = widgets.DatePicker(description='Start date', value=datetime.date(2018,1,1))
    end_date = widgets.DatePicker(description='End date', value=datetime.date(2018,12,31))
    nr_targets = widgets.IntSlider(value=10, min=2, max=37, step=1)
    nr_spp = widgets.IntSlider(value=2, min=1, max=6, step=1)

    display(widgets.Box( [ widgets.Label(value='Train / test split:'), train_test_split ]))
    display(algorithm)
    display(widgets.Box( [ widgets.Label(value="Hyperparameters RF model:"), nrtrees, mtry ]))
    display(widgets.Box( [ widgets.Label(value='S1 / S2 fusion:'), feature_raster ]))
    display(aoi)
    display(strat_layer)
    display(widgets.Box( [ widgets.Label(value='Include mixed pixels:'), include_mixed_pixels ]))
    display(start_date)
    display(end_date)
    display(widgets.Box( [ widgets.Label(value='Select the amount of target classes:'), nr_targets ]))
    display(widgets.Box( [ widgets.Label(value='Select the amount of times you want to point sample each reference polygon:'), nr_spp ]))
    return train_test_split, algorithm, nrtrees, mtry, feature_raster, aoi, strat_layer, include_mixed_pixels, start_date, end_date, nr_targets, nr_spp


def getSelectMultiple():
    return widgets.SelectMultiple(
    options=['A00: Artificial land', 
             'A10: Roofed built-up areas',
             'A20: Artificial non-built up areas',
             'A30: Other artificial areas',
             'B00: Cropland',
             'B10: Cereals',
             'B20: Root crops',
             'B30: Non-permanent industrial crops',
             'B40: Dry pulses, vegetables and flowers',
             'B50: Fodder crops',
             'B70: Permanent crops: fruit trees',
             'B80: Other permanent crops',
             'C00: Woodland',
             'C10: Broadleaved woodland',
             'C20: Coniferous woodland',
             'C30: Mixed woodland',
             'D00: Shrubland',
             'D10: Shrubland with sparse tree cover',
             'D20: Shrubland without tree cover',
             'E00: Grassland',
             'E10: Grassland with sparse tree/shrub cover',
             'E20: Grassland without tree/shrub cover',
             'E30: Spontaneously re-vegetated surfaces',
             'F00: Bare land and lichens/moss',
             'F10: Rocks and stones',
             'F20: Sand',
             'F30: Lichens and moss',
             'F40: Other bare soil',
             'G00: Water areas',
             'G10: Inland water bodies',
             'G20: Inland running water',
             'G30: Transitional water bodies',
             'G40: Sea and ocean',
             'G50: Glaciers, permanent snow',
             'H00: Wetlands',
             'H10: Inland wetlands',
             'H20: Coastal wetlands'
            ],
    rows=10,
    description='Target class')

def getTargetClasses(nr_targets):
	target_classes = {}
	for i in range(nr_targets.value):
	    target_classes["target"+str(i)] = getSelectMultiple()

	for target_selector in target_classes.values():
	    display(target_selector)
	return target_classes



def _get_epsg(lat, zone_nr):
    if lat >= 0:
        epsg_code = '326' + str(zone_nr)
    else:
        epsg_code = '327' + str(zone_nr)
    return int(epsg_code)


def extract_point_from_polygon(shp):
    """
    Extracts a shapely Point from a single shapely Polygon, with a 50% chance of that being the centroid of the polygon and 50%
    of it being a random pixel within the polygon
    """
    within = False
    while not within:
        utm_zone_nr = utm.from_latlon(*shp.bounds[0:2])[2]
        epsg_utm = _get_epsg(shp.bounds[0], utm_zone_nr)
        project_latlon_to_utm = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'),
                    pyproj.CRS(epsg_utm),
            always_xy=True).transform
        shp_utm = shapely.ops.transform(project_latlon_to_utm, shp)
        shp_utm_b = shp_utm.buffer(-10, resolution=4, cap_style=3, join_style=3)

        project_utm_to_latlon = pyproj.Transformer.from_crs(pyproj.CRS(epsg_utm),
                    pyproj.CRS('EPSG:4326'),
            always_xy=True).transform
        shp_latlon = shapely.ops.transform(project_utm_to_latlon, shp_utm_b)

        if shp_latlon.is_empty:
            shp_latlon = shp

        x = np.random.uniform(shp_latlon.bounds[0], shp_latlon.bounds[2])
        y = np.random.uniform(shp_latlon.bounds[1], shp_latlon.bounds[3])
        p = Point(x, y)
        within = shp_latlon.contains(p)
    return p.buffer(10**-10)



## C10 Broadleaved woodland    59082
## C20 Coniferous woodlandC21    12782 + 20286 + 4626 = 37694
## C30 Mixed woodland 			8081 + 8013 + 7208 = 23302
## D00 D10 D20 Shrubland;    		7346 + 11700 = 19046
## E00 E10 E20 E30 Grassland;        9134 + 54347 + 9624 = 73105
## B00 B10 B20 B30 B40 B50 B70 B80 Cropland;    16148 + 2331 + 8848 + 2162 + 2370 + 11211 + 287 + 1197 + 397 + 1455 + 1572 + 441 + 2567 + 5339 + 613 + 365 + 407 + 51 + 166 + 1590 + 157 + 838 + 111 + 80 + 620 + 2000 + 1206 + 860 + 4152 + 822 + 124 + 239 + 798 + 724 + 115 + 72 + 1758 + 1189 + 120 + 106 + 10331 + 756 = 86695
## F00 (ZONDER F30) F10 F20 F40 Bare land;    5933 + 307 + 1216 = 7456
## F30 Lichens and Moss; 511
## A00 A10 A20 A30 Built-up; 6199 + 511 + 273 + 4896 + 8892 + 798 = 21569
## G10 G20 G30 Inland Water;   1530 + 35 + 1017 + 3 + 30 = 2615
## G40 Sea and ocean;
## G50 Glaciers and Permanent Snow;   23
## H00 H10 H20 Wetlands.    2307 + 4135 + 202 + 48 + 27 = 6719


# lucas_data = gpd.read_file('lucas_with_projection.geojson')
# lucas_data_clipped = lucas_data.clip(aoi)
# lucas_data_clipped.to_file("lucas/clipped_lucas_data.geojson",driver='GeoJSON')
# gpd.read_file("https://artifactory.vgt.vito.be/auxdata-public/grids/LAEA-20km.gpkg",mask=countries)
# lucas_data.to_file("lucas.gpkg", driver="GPKG")


