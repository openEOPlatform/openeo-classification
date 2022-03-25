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
import geopandas as gpd
import json

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


def load_lc_features(feature_raster, aoi, start_date, end_date, stepsize_s2=10, stepsize_s1=12):
    provider = "terrascope"
    c = lambda: connection("openeo-dev.vito.be")

    idx_dekad = sentinel2_features(start_date, end_date, c, provider, processing_opts={}, sampling=True, stepsize=stepsize_s2)
    idx_features = compute_statistics(idx_dekad, start_date, end_date, stepsize=stepsize_s2)

    s1_dekad = sentinel1_features(start_date, end_date, c, provider, processing_opts={}, orbitDirection="ASCENDING", sampling=True, stepsize=stepsize_s1)
    s1_dekad = s1_dekad.resample_cube_spatial(idx_dekad)
    s1_features = compute_statistics(s1_dekad, start_date, end_date, stepsize=stepsize_s1)

    features = idx_features.merge_cubes(s1_features)

    if feature_raster == "s1":
        return s1_features, features.metadata.band_names
    elif feature_raster == "s2":
        return idx_features, idx_features.metadata.band_names
    else:
        return features, features.metadata.band_names
    
def getStartingWidgets():
    train_test_split = widgets.FloatSlider(value=0.75, min=0, max=1.0, step=0.05)
    algorithm = widgets.Dropdown(options=['Random Forest'], value='Random Forest', description='Model:', disabled=True)
    nrtrees = widgets.IntText(value=1000, description='Nr trees:')
    mtry = widgets.IntText(value=10, description="Mtry:")
    fusion_technique = widgets.RadioButtons(options=['Feature fusion', 'Decision fusion'])
    aoi = widgets.FileUpload(accept='.geojson,.shp',multiple=False, #style=widgets.ButtonStyle(button_color='#F0F0F0'), 
                             layout=widgets.Layout(width='20em'), description="Upload AOI")
    strat_layer = widgets.FileUpload(accept='.geojson,.shp',multiple=False, 
                                     layout=widgets.Layout(width='20em'), description="Upload stratification")
    # include_mixed_pixels = widgets.RadioButtons(options=['Yes', 'No'])
    start_date = widgets.DatePicker(description='Start date', value=datetime.date(2018,1,1))
    end_date = widgets.DatePicker(description='End date', value=datetime.date(2018,12,31))
    nr_targets = widgets.IntSlider(value=10, min=2, max=37, step=1)
    nr_spp = widgets.IntSlider(value=2, min=1, max=6, step=1)

    display(widgets.Box( [ widgets.Label(value='Train / test split:'), train_test_split ]))
    display(algorithm)
    display(widgets.Box( [ widgets.Label(value="Hyperparameters RF model:"), nrtrees, mtry ]))
    display(widgets.Box( [ widgets.Label(value='S1 / S2 fusion:'), fusion_technique ]))
    display(aoi)
    display(strat_layer)
    # display(widgets.Box( [ widgets.Label(value='Include mixed pixels:'), include_mixed_pixels ]))
    display(start_date)
    display(end_date)
    display(widgets.Box( [ widgets.Label(value='Select the amount of target classes:'), nr_targets ]))
    display(widgets.Box( [ widgets.Label(value='Select the amount of times you want to point sample each reference polygon:'), nr_spp ]))
    return train_test_split, algorithm, nrtrees, mtry, fusion_technique, aoi, strat_layer, start_date, end_date, nr_targets, nr_spp


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
    return p


def mapTargetDataToNumerical(target_classes):
    mapper = {}
    counter = 0
    for key in target_classes:
        for value in target_classes[key].value:
            abb = value[:3]
            if abb in mapper.keys():
                raise Exception("You selected one target class multiple times")
            mapper[abb] = counter
        counter += 1
    all_lucas_classes = ['A00', 'A10', 'A20', 'A30', 'B00', 'B10', 'B20', 'B30', 'B40', 'B50', 'B70', 'B80', 'C00', 'C10', 'C20', 'C30', 'D00', 'D10', 'D20', 'E00', 'E10', 'E20', 'E30', 'F00', 'F10', 'F20', 'F30', 'F40', 'G00', 'G10', 'G20', 'G30', 'G40', 'G50', 'H00', 'H10', 'H20']
    dif = sorted(list(set(all_lucas_classes) - set(mapper.keys())))
    if len(dif) != 0:
        raise Exception("You haven't contributed distributed all LUCAS classes over your output target variables. You are missing the classes: {}".format(dif))
    return mapper

def getReferenceSet(aoi, nr_samples_per_polygon, target_classes):
    if len(aoi.value) == 0:
        raise ValueError("Please upload an area of interest first in the widget menu above!")

    mask = gpd.GeoDataFrame.from_features(json.loads(list(aoi.value.values())[0]["content"])).set_crs('epsg:4326')

    print("Loading in the LUCAS Copernicus dataset...")
    data = gpd.read_file("https://artifactory.vgt.vito.be/auxdata-public/openeo/LUCAS_2018_Copernicus.gpkg",mask=mask)

    if data.empty:
        raise ValueError("Your masked area is located outside of Europe or so small that no training data can be found within it")

    print("Finished loading data.")
    print("Extracting points and converting target labels...")
    lucas_points = pd.concat([data]+[data.copy()]*(nr_samples_per_polygon.value-1), ignore_index=True)
    lucas_points["geometry"] = lucas_points["geometry"].apply(extract_point_from_polygon)
    y = lucas_points[["LC1", "geometry"]].copy()
    mapper = mapTargetDataToNumerical(target_classes)
    y["LC1"] = y["LC1"].apply(lambda x: mapper[x[:2]+"0"])
    y = y.rename(columns={"LC1":"target"})
    print("Finished extracting points and converting target labels")
    return y
