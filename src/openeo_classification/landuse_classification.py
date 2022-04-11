import datetime
import json
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from openeo.processes import array_concat, ProcessBuilder, if_, is_nodata
import xarray as xr

import geopandas as gpd
import ipywidgets as widgets
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
import utm
from shapely.geometry import Point
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from openeo_classification.connection import connection
from openeo_classification.features2 import sentinel2_features, compute_statistics, sentinel1_features

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



def compute_statistics_fill_nan(base_features, start_date, end_date, stepsize):
    """
    Computes statistics over a datacube.
    For correct statistics, the datacube needs to be preprocessed to contain observation at equitemporal intervals, without nodata values.

    @param base_features:
    @return:
    """
    def computeStats(input_timeseries: ProcessBuilder, sample_stepsize, offset):
        tsteps = list([input_timeseries.array_element(offset + sample_stepsize * index) for index in range(0, 6)])
        tsteps[1] = if_(is_nodata(tsteps[1]), tsteps[2], tsteps[1])
        tsteps[4] = if_(is_nodata(tsteps[4]), tsteps[3], tsteps[4])
        tsteps[0] = if_(is_nodata(tsteps[0]), tsteps[1], tsteps[0])
        tsteps[5] = if_(is_nodata(tsteps[5]), tsteps[4], tsteps[5])
        return array_concat(
            array_concat(input_timeseries.quantiles(probabilities=[0.25, 0.5, 0.75]), input_timeseries.sd()), tsteps)

    tot_samples = (end_date - start_date).days // stepsize
    nr_tsteps = 6
    sample_stepsize = tot_samples // nr_tsteps
    offset = int(sample_stepsize/2 + (tot_samples%nr_tsteps)/2)

    features = base_features.apply_dimension(dimension='t', target_dimension='bands', process=lambda x: computeStats(x, sample_stepsize, offset))#.apply(lambda x: x.linear_scale_range(-500, 500, -50000, 50000))
    tstep_labels = ["t" + str(offset + sample_stepsize * index) for index in range(0, 6)]
    all_bands = [band + "_" + stat for band in base_features.metadata.band_names for stat in
                 ["p25", "p50", "p75", "sd"] + tstep_labels]
    features = features.rename_labels('bands', all_bands)
    return features

def load_lc_features(provider, feature_raster, start_date, end_date, stepsize_s2=10, stepsize_s1=12, processing_opts={}, index_dict=None):
    """
    Loads the features used in the dynamic land use cover service
    @return: features, a datacube containing all calculated features, and the band names of the feature cube returned
    """
    c = lambda: connection("openeo-dev.vito.be")


    ## NIEUWE WOW
    if not index_dict:
        idx_list = ["NDVI", "NDMI", "NDGI", "NDRE1", "NDRE2", "NDRE5"]
        s2_list = ["B06", "B12"]
        index_dict = {idx: [-1,1] for idx in idx_list}
        index_dict["ANIR"] = [0,1]

    final_index_dict = {
        "collection": {
            "input_range": [0, 8000],
            "output_range": [0, 30000]
        },
        "indices": {
            index: {"input_range": index_dict[index], "output_range": [0, 30000]} for index in index_dict
        }
    }

    idx_dekad = sentinel2_features(start_date, end_date, c, provider, final_index_dict, s2_list, processing_opts=processing_opts, sampling=True, stepsize=stepsize_s2, luc=True)
    idx_features = compute_statistics_fill_nan(idx_dekad, start_date, end_date, stepsize=stepsize_s2)

    s1_dekad = sentinel1_features(start_date, end_date, c, provider, processing_opts=processing_opts, orbitDirection="ASCENDING", sampling=True, stepsize=stepsize_s1)
    s1_dekad = s1_dekad.resample_cube_spatial(idx_dekad)
    s1_features = compute_statistics_fill_nan(s1_dekad, start_date, end_date, stepsize=stepsize_s1)

    features = idx_features.merge_cubes(s1_features)

    if feature_raster == "s1":
        return s1_features, features.metadata.band_names
    elif feature_raster == "s2":
        return idx_features, idx_features.metadata.band_names
    else:
        return features, features.metadata.band_names
    
def get_starting_widgets():
    """
    A helper function that initializes and displays a number of widgets used for determining the specifics of the model building process
    """
    train_test_split = widgets.FloatSlider(value=0.75, min=0, max=1.0, step=0.05)
    algorithm = widgets.Dropdown(options=['Random Forest'], value='Random Forest', description='Model:', disabled=True)
    nrtrees = widgets.IntText(value=200, description='Nr trees:')
    # mtry = widgets.IntText(value=10, description="Mtry:")
    fusion_technique = widgets.RadioButtons(options=['Feature fusion', 'Decision fusion'], disabled=True)
    aoi_sampling = widgets.FileUpload(accept='.geojson,.shp',multiple=False, #style=widgets.ButtonStyle(button_color='#F0F0F0'),
                             layout=widgets.Layout(width='20em'), description="Upload AOI sampling")
    aoi_inference = widgets.FileUpload(accept='.geojson,.shp',multiple=False, #style=widgets.ButtonStyle(button_color='#F0F0F0'),
                             layout=widgets.Layout(width='20em'), description="Upload AOI inference")
    # include_mixed_pixels = widgets.RadioButtons(options=['Yes', 'No'])
    start_date = widgets.DatePicker(description='Start date', value=datetime.date(2018,3,1))
    end_date = widgets.DatePicker(description='End date', value=datetime.date(2018,10,31))
    nr_targets = widgets.IntSlider(value=8, min=2, max=37, step=1)
    nr_spp = widgets.IntSlider(value=2, min=1, max=6, step=1)

    display(widgets.Box( [ widgets.Label(value='Train / test split:'), train_test_split ]))
    display(algorithm)
    display(widgets.Box( [ widgets.Label(value="Hyperparameters RF model:"), nrtrees ]))
    display(widgets.Box( [ widgets.Label(value='S1 / S2 fusion:'), fusion_technique ]))
    display(aoi_sampling)
    display(aoi_inference)
    # display(widgets.Box( [ widgets.Label(value='Include mixed pixels:'), include_mixed_pixels ]))
    display(start_date)
    display(end_date)
    display(widgets.Box( [ widgets.Label(value='Select the amount of target classes:'), nr_targets ]))
    display(widgets.Box( [ widgets.Label(value='Select the amount of times you want to point sample each reference polygon:'), nr_spp ]))
    return train_test_split, algorithm, nrtrees, fusion_technique, aoi_sampling, aoi_inference, start_date, end_date, nr_targets, nr_spp


def get_select_multiple():
    """
    The widget of the custom target classes
    """
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

def get_target_classes(nr_targets):
    """
    Used for the determination of custom target classes
    """
    target_classes = {}
    for i in range(nr_targets.value):
        target_classes["target"+str(i)] = get_select_multiple()

    for target_selector in target_classes.values():
        display(target_selector)
    return target_classes


def _get_epsg(lat, zone_nr):
    """
    Calculates the epsg code corresponding to a certain latitude given the zone nr
    """
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
        utm_zone_nr = utm.from_latlon(*shp.bounds[0:2][::-1])[2]
        epsg_utm = _get_epsg(shp.bounds[1], utm_zone_nr)
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


def map_target_data_to_numerical(target_classes):
    """
    Maps the selected target classes to integers that can be used for model training and testing
    """
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

def get_reference_set(aoi, nr_samples_per_polygon, target_classes):
    """
    Loads in the reference data corresponding to an AOI, samples points from it and transforms the labels to numerical values
    """
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
    mapper = map_target_data_to_numerical(target_classes)
    y["LC1"] = y["LC1"].apply(lambda x: mapper[x[:2]+"0"])
    y = y.rename(columns={"LC1":"target"})
    print("Finished extracting points and converting target labels")
    return y

def get_strata(aoi_sampling, aoi_inference, strat_col_label="stratum"):
    """
    Loads the strata from the geojsons provided, and checks whether they are valid
    """
    strata_sampling = gpd.GeoDataFrame.from_features(json.loads(aoi_sampling.data[0]))
    strata_inference = gpd.GeoDataFrame.from_features(json.loads(aoi_inference.data[0]))
    
    ## als beiden lengte 1: prima
    if len(strata_sampling) == 1 and len(strata_inference)==1:
        strata_sampling["stratum"] = ["stratum0"]
        strata_inference["stratum"] = ["stratum0"]
        return strata_sampling, strata_inference

    ## als een van twee langer: raise exception
    if len(strata_sampling) < len(strata_inference):
        raise ValueError("Your inference AOI has more strata then your sampling AOI.")

    ## als meerdere strata maar hebben geen stratum kolom
    if strat_col_label not in strata_sampling.columns:
        raise ValueError("Your sampling AOI contains stratification polygons, however does not contain a field called {}".format(strat_col_label))
    if strat_col_label not in strata_inference.columns:
        raise ValueError("Your inference AOI contains stratification polygons, however does not contain a field called {}".format(strat_col_label))


    if len(set(strata_sampling[strat_col_label]) - set(strata_inference[strat_col_label])) > 0:
        raise ValueError("Your sampling set contains strata that do not occur in your inference strata")

    return strata_sampling, strata_inference



def buf(x):
    """
    Creates a 10m buffer around the points so that it can be used within filter_spatial
    """
    shp = x.iloc[0]
    utm_zone_nr = utm.from_latlon(*shp.bounds[0:2][::-1])[2]
    epsg_utm = _get_epsg(shp.bounds[1], utm_zone_nr)
    return x.to_crs(epsg_utm).buffer(10,cap_style=3).to_crs(4326)



def calculate_validation_metrics(path_to_test_geojson='validation_prediction/y_test.geojson', 
                                 path_to_test_raster='validation_prediction/y_test/openEO.nc', output_type="netCDF"):
    """
    Calculates a number of validation metrics from the test set and test predictions
    """
    gdf = gpd.read_file(path_to_test_geojson)
    utm_zone_nr = utm.from_latlon(*gdf.geometry[0].bounds[:2][::-1])[2]
    epsg_utm = _get_epsg(gdf.geometry[0].bounds[1], utm_zone_nr)
    gdf = gdf.to_crs(epsg_utm)
    coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
    if output_type == "GTiff":
        src = rasterio.open(path_to_test_raster)
        gdf['predicted'] = [x[0] for x in src.sample(coord_list)]
    else:
        ds = xr.open_dataset(path_to_test_raster)
        new_coords = list(map(lambda x: (5+x[0]//10 * 10, 5+x[1]//10 * 10), coord_list))
        gdf["predicted"] = [ds.sel(x=i[0],y=i[1])["var"].values.tolist() for i in new_coords]
        gdf["predicted"] = gdf["predicted"].where(gdf["predicted"] < 100, np.nan)
    print("The total amount of test samples you supplied is {}. Of these, {} could not be matched to the coordinates of your y samples. If this is more than a few samples, please check if your CRS is aligned.".format(
        str(len(gdf)), str(gdf["predicted"].isnull().sum())))
    gdf = gdf.dropna().drop("geometry",axis=1)
    gdf["predicted"] = gdf["predicted"].astype(int)
    gdf["target"] = gdf["target"].astype(int)

    acc = accuracy_score(gdf["target"],gdf["predicted"])
    print("Accuracy on test set: "+str(acc)[0:5])
    prec, rec, fscore, sup = precision_recall_fscore_support(gdf["target"],gdf["predicted"])

    final_res = {
            "accuracy": acc,
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "fscore": fscore.tolist(),
            "support": sup.tolist()
    }
    return gdf, final_res
