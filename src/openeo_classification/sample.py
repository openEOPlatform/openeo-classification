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


def sample_polygons(crop_ids, tot_samp=2000, tot_repeat=1, input_df=None) -> pd.DataFrame:
    """
    Builds a pandas dataframe containing up to TOT_SAMP number of samples, combined from all TOT_FILES (the different 
    data files, e.g. AT 2017, FR 2018, etc.) that are loaded into input_df (input_df does not need to be specified, because it can be
    done in this script as well).
    If some files contain less data then the average that should be loaded to get up to 2000 samples, the script will try
    to take more than the average from the remaining files.
    
    param crop_ids: the ID of the crop for which you want to sample polygons
    param tot_samp: the total amount of polygons you want to sample. If less polygons are present in total, all polygons
        that are present will be selected
    param tot_repeat: the amount of times you are willing to repeat a polygon (take it up multiple times) if there are less
        rows then the tot_samp you selected
    param tot_input_files: the number of input files which were used to build up the input_df
    param input_df: all input data combined into one pandas dataframe
    
    returns: a pandas dataframe containing tot_samp (or less, if not enough data is present) polygons of one specific crop type
    """
    if input_df is None:
        input_df = _read_f()
    crop_df = input_df[input_df["CT"].isin(crop_ids)]
    
    tot_input_files=len(crop_df["ref_id"].unique())+1
    
    if len(crop_df) == 0:
        raise ValueError("The crop ID you selected - {} - does not exist in the dataset you supplied".format(crop_ids))

    samples = []
    def create_samples(crop_df=crop_df,tot_input_files=tot_input_files,tot_samp=tot_samp, samples=samples):
        for ref_id, size in crop_df.groupby("ref_id").size().sort_values().iteritems():
            tot_input_files -= 1
            crop_per_ds = crop_df[crop_df["ref_id"] == ref_id]
            ## This is to equally divide the number of samples you want to select over all input datasets
            amount_sampl = tot_samp // tot_input_files
            if amount_sampl > size:
                ## Size is the amount of samples in one specific dataset (e.g., 2017 AT). So if the amount you want to
                ## select from that dataset is bigger than the amount that is actually there, then just sample the amount
                ## that is actually there and no more than that.
                samples.append(crop_per_ds.sample(size))
                tot_samp -= size
            else:
                ## If there are enough samples than just sample the amount you really want to select
                samples.append(crop_per_ds.sample(amount_sampl))
                tot_samp -= amount_sampl
        return samples, tot_samp

    while tot_samp > 0 and tot_repeat > 0:
        samples, tot_samp = create_samples(tot_samp=tot_samp,samples=samples)
        tot_repeat -= 1
    print("Polygon sampling completed.")        
    return pd.concat(samples)

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


def write_to_json(df, ds_nr, ids, year, zonenumber, zoneletter, folder="resources/training_data/"):
    geom = np.asarray(df["sample_polygon"]).tolist()
    ref_id = df['ref_id']
    metadata = {
                    "ids": str(ids), 
                    "year": str(year),
                    "title": str(ids[0]),
                    "name": str(ids[0])
               }
    el = json.loads(gpd.GeoDataFrame({"ref_id":ref_id,"zone_num":zonenumber,"zone_let":zoneletter,"geometry":geom}).to_json())
    el.update(metadata)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder / ("sampleable_polygons_year"+str(year)+"_zone"+str(zonenumber)+str(zoneletter)+"_id"+str(ids[0]//100)+"00"+"_p"+str(ds_nr)+".json"), 'w') as fn:
        json.dump(el, fn)

def get_crop_codes(crop_list: list, f: pd.DataFrame):
    """
    This function takes a list of crops and the dataframe containing all polygons and 
    returns two lists of ID's: one of the crop classes selected by the user, and one of 
    the other classes. These can be used to sample polygons using sample_polygons.
    """
    print("Retrieving crop ID's of the classes you supplied, as well as the crop ID's of the other crops, in a separate list of lists.")
    gen_ids = [i for i,e in all_crop_codes.items() if e in crop_list]
    rel_ids = [i for i in all_crop_codes.keys() if i//100 in [j // 100 for j in gen_ids]]
    non_rel_ids = [i for i in f["CT"].unique() if i not in np.hstack(rel_ids)]
    non_rel_ids.sort()

    rel_counts = np.unique([i // 100 for i in rel_ids], return_counts=True)
    non_rel_counts = np.unique([i // 100 for i in non_rel_ids], return_counts=True)
    rel_final = [[j for j in rel_ids if j//100==i] for i in list(rel_counts[0])]
    non_rel_final = [[j for j in non_rel_ids if j//100==i] for i in list(non_rel_counts[0])]
    print("Crop ID's retrieved.")
    return rel_final, non_rel_final



def sample_and_store_ids(ids, zones, years, input_df, output_folder="resources/training_data/", tot_samp=500, tot_repeat=3):
    print("Starting to sample polygons for ids {}".format(ids))
    crop = sample_polygons(crop_ids=ids, tot_samp=tot_samp, tot_repeat=tot_repeat, input_df=input_df)
    crop["sample_polygon"] = crop["geometry"].apply(_extract_point_from_polygon)
    for zone in zones:
        crop_zone = crop.query('zonenumber=='+zone[:2]+' & zoneletter=="'+zone[-1]+'" ')
        for year in years:
            crop_year = crop_zone[crop_zone['validityTi'] == str(year)+"-06-01"]
            if len(crop_year) == 0:
                continue
            split_amount = len(crop_year) // 150
            if split_amount > 0:
                dfs = np.array_split(crop_year, split_amount+1)
                for i,df in enumerate(dfs):
                    write_to_json(df=df, ds_nr=i, ids=ids, year=year, zonenumber=zone[:2], zoneletter=zone[-1], folder=output_folder)
            else:
                write_to_json(df=crop_year, ds_nr=0, ids=ids, year=year, zonenumber=zone[:2], zoneletter=zone[-1], folder=output_folder)
    print("ids {} have been written to JSON".format(ids))


def sample_and_store_polygons(crop_list, zones, years, input_df, output_folder=Path("resources")/ "training_data", tot_samp_crops=500, tot_samp_other=200, tot_repeat=3):
    """
    Sample polygons and store them on disk
    (note that this step could be integrated in the training loop if wanted)
    """
    crop_ids, other_crop_ids = get_crop_codes(crop_list, input_df)

    for ids in crop_ids:
        sample_and_store_ids(ids, zones, years, input_df, output_folder=output_folder  / "crops_of_interest", tot_samp=tot_samp_crops, tot_repeat=tot_repeat)
    for ids in other_crop_ids:
        sample_and_store_ids(ids, zones, years, input_df, output_folder=output_folder / "other_crops", tot_samp=tot_samp_other, tot_repeat=tot_repeat)
    return crop_ids, other_crop_ids