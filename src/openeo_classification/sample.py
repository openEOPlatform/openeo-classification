from pathlib import Path

import pandas as pd
from shapely.geometry import Point
import json
from explore import all_crop_codes
import glob
import re
import geopandas as gpd
import numpy as np
import os

def read_f(directory:str = "resources/reference_data/") -> pd.DataFrame:
    """
    Takes in an input path and from that input path, loads all reference data
    (note required format)
    Returns one dataframe containing all data combined, as well as the file names
    of each of the files included
    """
    print("Loading input data...")
    files = []
    fns = []
    for fn in glob.glob(directory+"*"):
        fns.append(re.search(r".*(20.*_POLY_[0-9]{3})_samples.json", fn).group(1))
        files.append(gpd.read_file(fn))
    print("Finished loading data.")
    return pd.concat(files), fns


def sample_polygons(crop_id, tot_samp=2000, repeat_per_sample=1, input_df=None) -> pd.DataFrame:
    """
    Builds a pandas dataframe containing up to TOT_SAMP number of samples, combined from all TOT_FILES (the different 
    data files, e.g. AT 2017, FR 2018, etc.) that are loaded into input_df (input_df does not need to be specified, because it can be
    done in this script as well).
    If some files contain less data then the average that should be loaded to get up to 2000 samples, the script will try
    to take more than the average from the remaining files.
    
    param crop_ids: the ID of the crop for which you want to sample polygons
    param tot_samp: the total amount of polygons you want to sample. If less polygons are present in total, all polygons
        that are present will be selected
    param repeat_per_sample: the amount of times you are willing to repeat a polygon (take it up multiple times) if there are less
        rows then the tot_samp you selected (minimum is one!)
    param tot_input_files: the number of input files which were used to build up the input_df
    param input_df: all input data combined into one pandas dataframe
    
    returns: a pandas dataframe containing tot_samp (or less, if not enough data is present) polygons of one specific crop type
    """
    if repeat_per_sample < 1:
        raise ValueError("Repeat per sample has a minimum value of 1!")

    if input_df is None:
        input_df = _read_f()
    crop_df = input_df[input_df["CT"] == crop_id]
    
    tot_input_files=len(crop_df["ref_id"].unique())+1
    
    if len(crop_df) == 0:
        raise ValueError("The crop ID you selected - {} - does not exist in the dataset you supplied".format(crop_id))

    tot_df = pd.concat([crop_df]+[crop_df.copy()]*(repeat_per_sample-1), ignore_index=True)
    if tot_samp > len(tot_df):
        print("The amount of {} samples you want is more than {} times the total amount of samples, {}. Hence the amount of features sampled for crop {} is: {}".format(
            crop_id, repeat_per_sample, len(crop_df), crop_id, len(tot_df)))
        tot_samp = len(tot_df)
    samples = tot_df.sample(tot_samp)
    return samples


def _extract_point_from_polygon(shp):
    """
    Extracts a shapely Point from a single shapely Polygon, with a 50% chance of that being the centroid of the polygon and 50%
    of it being a random pixel within the polygon
    """
    if np.random.rand() > .5:
        return shp.centroid.buffer(10**-10)
    else:
        within = False
        while not within:
            try:
                x = np.random.uniform(shp.bounds[0], shp.bounds[2])
                y = np.random.uniform(shp.bounds[1], shp.bounds[3])
                within = shp.contains(Point(x, y))
            except:
                print("I found an invalid shape in your dataframe!")
                within = True
        return Point(x,y).buffer(10**-10)


def write_to_json(df, ds_nr, crop_id, year, zonenumber, folder="resources/training_data/"):
    geom = np.asarray(df["sample_polygon"]).tolist()
    ref_id = df['ref_id']
    metadata = {
                    "ids": str(crop_id),
                    "year": str(year),
                    "title": str(crop_id),
                    "name": str(crop_id)
               }
    el = json.loads(gpd.GeoDataFrame({"ref_id":ref_id,"zone_num":zonenumber,"geometry":geom}).to_json())
    el.update(metadata)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder / ("sampleable_polygons_year"+str(year)+"_zone"+str(zonenumber)+"_id"+str(crop_id)+"_p"+str(ds_nr)+".json"), 'w') as fn:
        json.dump(el, fn)

def get_crop_codes(crop_list: list, f: pd.DataFrame):
    """
    This function takes a list of crops and the dataframe containing all polygons and 
    returns two lists of ID's: one of the crop classes selected by the user, and one of 
    the other classes. These can be used to sample polygons using sample_polygons.
    """
    print("Retrieving crop ID's of the classes you supplied, as well as the crop ID's of the other crops, in a separate list of lists.")
    rel_ids = [i for i,e in all_crop_codes.items() if e in crop_list]
    non_rel_ids = [i for i in f["CT"].unique() if i not in np.hstack(rel_ids)]
    non_rel_ids.sort()

    return rel_ids, non_rel_ids

    # rel_counts = np.unique([i // 100 for i in rel_ids], return_counts=True)
    # print(rel_counts)
    # non_rel_counts = np.unique([i // 100 for i in non_rel_ids], return_counts=True)
    # rel_final = [[j for j in rel_ids if j//100==i] for i in list(rel_counts[0])]
    # print(rel_final)
    # non_rel_final = [[j for j in non_rel_ids if j//100==i] for i in list(non_rel_counts[0])]
    # return rel_final, non_rel_final



def store_id(crop_id, zone, years, crop_df, output_folder):
    for year in years:
        crop_year = crop_df[crop_df['validityTi'] == str(year)+"-06-01"]
        if len(crop_year) == 0:
            continue
        split_amount = len(crop_year) // 150
        if split_amount > 0:
            dfs = np.array_split(crop_year, split_amount+1)
            for i,df in enumerate(dfs):
                write_to_json(df=df, ds_nr=i, crop_id=crop_id, year=year, zonenumber=zone, folder=output_folder)
        else:
            write_to_json(df=crop_year, ds_nr=0, crop_id=crop_id, year=year, zonenumber=zone, folder=output_folder)


def sample_and_store_id(crop_id, zones, years, input_df, output_folder, tot_samp=2000, repeat_per_sample=3):
    print("Starting to sample polygons for id {}".format(crop_id))
    crop = sample_polygons(crop_id=crop_id, tot_samp=tot_samp, repeat_per_sample=repeat_per_sample, input_df=input_df)
    crop["sample_polygon"] = crop["geometry"].apply(_extract_point_from_polygon)
    crop_belgium = crop[crop["ref_id"].str.slice(4,8) == "_BE_"]
    crop_rest = crop[crop["ref_id"].str.slice(4,8) != "_BE_"]
    store_id(crop_id, "31", years, crop_belgium, output_folder / "terrascope")
    for zone in zones:
        crop_zone = crop_rest.query('zonenumber=='+str(zone))
        store_id(crop_id, zone, years, crop_zone, output_folder / "sentinelhub")


def sample_and_store_polygons(crop_list, zones, years, input_df, output_folder=Path("resources")/ "training_data", tot_samp_crops=500, tot_samp_other=200, repeat_per_sample=3):
    """
    Sample polygons and store them on disk
    (note that this step could be integrated in the training loop if wanted)
    """
    crop_ids, other_crop_ids = get_crop_codes(crop_list, input_df)

    for crop_id in crop_ids:
        sample_and_store_id(crop_id, zones, years, input_df, output_folder=output_folder  / "crops_of_interest", tot_samp=tot_samp_crops, repeat_per_sample=repeat_per_sample)
    for crop_id in other_crop_ids:
        sample_and_store_id(crop_id, zones, years, input_df, output_folder=output_folder / "other_crops", tot_samp=tot_samp_other, repeat_per_sample=repeat_per_sample)
    return crop_ids, other_crop_ids