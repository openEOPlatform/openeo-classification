from pathlib import Path

import pandas as pd
from shapely.geometry import Point
import shapely.ops
import json
from explore import all_crop_codes
import glob
import re
import geopandas as gpd
import numpy as np
import os
import utm
import pyproj


def _get_coords(shp):
    p = [coord for coords_list in shp.coords[:] for coord in coords_list]
    coord_info = utm.from_latlon(*p)
    return pd.Series([coord_info[2], coord_info[3]])

def _get_epsg(lat, zone_nr):
    if lat >= 0:
        epsg_code = '326' + str(zone_nr)
    else:
        epsg_code = '327' + str(zone_nr)
    return int(epsg_code)

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
    for fn in glob.glob(directory+"*.json"):
        fns.append(re.search(r".*(20.*_POLY_[0-9]{3})_samples.json", fn).group(1))
        files.append(gpd.read_file(fn))
    print("Finished loading data.")

    ### LPIS "CT" , "ref_id" , , "year" , "geometry", "zonenumber", "zoneletter" ,  "sample_polygon" (= een punt dat gesamplet is uit de geometry)
    df_lucas = gpd.read_file(directory+"2018_EU_LUCAS_POINT_110.gpkg")[["CT","geometry"]]
    df_lucas["year"] = 2018
    df_lucas["ref_id"] = "2018_EU_LUCAS_POINT_110"
    df_lucas[["zonenumber","zoneletter"]] = df_lucas["geometry"].apply(_get_coords)

    return pd.concat(files)[["CT","geometry","year","ref_id","zonenumber","zoneletter"]], df_lucas, fns

class SamplePolygons():
    def __init__(self, crop_list, zones, years, input_dfs, aez_stratification, output_folder = Path("resources")/ "training_data", 
        tot_samp_crops_lpis=500, tot_samp_other_lpis=200, tot_samp_crops_lucas=500, tot_samp_other_lucas=200, repeat_per_sample_lpis=3):
        self.crop_list = crop_list
        self.zones = zones
        self.years = years
        self.input_df_lpis, self.input_df_lucas = input_dfs
        self.aez_stratification = aez_stratification
        self.output_folder = output_folder
        self.tot_samp_crops_lpis = tot_samp_crops_lpis
        self.tot_samp_other_lpis = tot_samp_other_lpis
        self.tot_samp_crops_lucas = tot_samp_crops_lucas
        self.tot_samp_other_lucas = tot_samp_other_lucas
        self.repeat_per_sample_lpis = repeat_per_sample_lpis

        if input_dfs is None:
            self.input_df_lpis, self.input_df_lucas, self.fns = read_f()


    def sample_polygons(self, crop_id, tot_samp_lpis, tot_samp_lucas) -> pd.DataFrame:
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
        param input_df: all input data combined into one pandas dataframe
        
        returns: a pandas dataframe containing tot_samp (or less, if not enough data is present) polygons of one specific crop type
        """
        if self.repeat_per_sample_lpis < 1:
            raise ValueError("Repeat per sample has a minimum value of 1!")

        crop_df_lpis = self.input_df_lpis[self.input_df_lpis["CT"] == crop_id]
        tot_df_lucas = self.input_df_lucas[self.input_df_lucas["CT"] == crop_id]

        if len(crop_df_lpis) == 0 and len(tot_df_lucas) == 0:
            raise ValueError("The crop ID you selected - {} - does not exist in the datasets you supplied".format(crop_id))

        tot_df_lpis = pd.concat([crop_df_lpis]+[crop_df_lpis.copy()]*(self.repeat_per_sample_lpis-1), ignore_index=True)

        if tot_samp_lpis > len(tot_df_lpis):
            print("The amount of {} samples you want is more than {} times the total amount of LPIS samples, {}, \
                Hence the amount of LPIS features sampled for crop {} is: {}".format(
                crop_id, self.repeat_per_sample_lpis, len(crop_df_lpis), crop_id, len(tot_df_lpis)))
            tot_samp_lpis = len(tot_df_lpis)
        if tot_samp_lucas > len(tot_df_lucas):
            print("The amount of {} samples you want is more than the total amount of LUCAS samples, {}, \
                Hence the amount of LUCAS features sampled for crop {} is: {}".format(
                crop_id, len(tot_df_lucas), crop_id, len(tot_df_lucas)))
            tot_samp_lucas = len(tot_df_lucas)

        samples_lpis = tot_df_lpis.sample(tot_samp_lpis)
        samples_lpis.to_csv("the_lpis_samples.csv")
        # samples_lpis["sample_polygon"] = samples_lpis["geometry"].apply(self._extract_point_from_polygon)
        samples_lpis["geometry"] = samples_lpis["geometry"].apply(self._extract_point_from_polygon)

        samples_lucas = tot_df_lucas.sample(tot_samp_lucas)
        samples_lucas.to_csv("the_lucas_samples.csv")
        return samples_lpis, samples_lucas


    def _extract_point_from_polygon(self, shp):
        """
        Extracts a shapely Point from a single shapely Polygon, with a 50% chance of that being the centroid of the polygon and 50%
        of it being a random pixel within the polygon
        """
        if np.random.rand() > .5:
            return shp.centroid.buffer(10**-10)
        else:
            within = False
            while not within:
                # epsg_code = utm.from_latlon(*list(*p.coords))
                ## Negative buffer, so reprojecting to UTM then back to lat/lon
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


    def write_to_json(self, df, ds_nr, crop_id, year, zonenumber, folder="resources/training_data/"):
        geom = np.asarray(df["geometry"]).tolist()
        # geom = np.asarray(df["sample_polygon"]).tolist()
        metadata = {
                        "ids": str(crop_id),
                        "year": str(year),
                        "title": str(crop_id),
                        "name": str(crop_id),
                   }
        el = json.loads(gpd.GeoDataFrame({
            "ref_id": df['ref_id'],
            "zone_num":zonenumber,
            "zoneID":df["zoneID"],
            "groupID":df["groupID"],
            "geometry":geom
        }).to_json())
        el.update(metadata)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder / ("sampleable_polygons_year"+str(year)+"_zone"+str(zonenumber)+"_id"+str(crop_id)+"_p"+str(ds_nr)+".json"), 'w') as fn:
            json.dump(el, fn)


    def get_crop_codes(self):
        """
        This function takes a list of crops and the dataframe containing all polygons and 
        returns two lists of ID's: one of the crop classes selected by the user, and one of 
        the other classes. These can be used to sample polygons using sample_polygons.
        """
        print("Retrieving crop ID's of the classes you supplied, as well as the crop ID's of the other crops, in a separate list of lists.")
        rel_ids = [i for i,e in all_crop_codes.items() if e in self.crop_list]
        # non_rel_ids = [i for i in self.input_df["CT"].unique() if i not in np.hstack(rel_ids)]
        non_rel_ids = [i for i in list(set.union(set(self.input_df_lpis["CT"]), set(self.input_df_lucas["CT"]))) if i not in np.hstack(rel_ids)]
        non_rel_ids.sort()

        return rel_ids, non_rel_ids


    def store_id(self, crop_id, zone, crop_df, output_folder):
        for year in self.years:
            crop_year = crop_df[crop_df['year'] == year]
            if len(crop_year) == 0:
                continue
            split_amount = len(crop_year) // 150
            if split_amount > 0:
                dfs = np.array_split(crop_year, split_amount+1)
                for i,df in enumerate(dfs):
                    self.write_to_json(df=df, ds_nr=i, crop_id=crop_id, year=year, zonenumber=zone, folder=output_folder)
            else:
                self.write_to_json(df=crop_year, ds_nr=0, crop_id=crop_id, year=year, zonenumber=zone, folder=output_folder)


    def sample_and_store_id(self,crop_id, output_folder, tot_samp_lpis, tot_samp_lucas):
        print("Starting to sample polygons for id {}".format(crop_id))

        ## SPLIT SAMPLING IN : SAMPLE & REPEAT SAMPLES
        ## SAMPLE WITHOUT REPEATING SAMPLES

        crop_lpis, crop_lucas = self.sample_polygons(crop_id=crop_id, tot_samp_lpis=tot_samp_lpis, tot_samp_lucas=tot_samp_lucas)

        if crop_lpis.empty:
            crop = crop_lucas
        elif crop_lucas.empty:
            crop = crop_lpis
        else:
            crop = pd.concat([crop_lpis,crop_lucas])

        if not crop.empty:
            crop = gpd.sjoin(crop, self.aez_stratification,how="left").drop(columns="index_right")
        crop.to_csv("final_samples.csv")
        crop_belgium = crop[crop["ref_id"].str.slice(4,8) == "_BE_"]
        crop_rest = crop[crop["ref_id"].str.slice(4,8) != "_BE_"]
        self.store_id(crop_id, "31", crop_belgium, output_folder / "terrascope")
        for zone in self.zones:
            crop_zone = crop_rest.query('zonenumber=='+str(zone))
            self.store_id(crop_id, zone, crop_zone, output_folder / "sentinelhub")


    def sample_and_store_polygons(self):
        """
        Sample polygons and store them on disk
        (note that this step could be integrated in the training loop if wanted)
        """
        crop_ids, other_crop_ids = self.get_crop_codes()

        for crop_id in crop_ids:
            self.sample_and_store_id(crop_id, output_folder=self.output_folder  / "crops_of_interest", tot_samp_lpis=self.tot_samp_crops_lpis, tot_samp_lucas=self.tot_samp_crops_lucas)
        for crop_id in other_crop_ids:
            self.sample_and_store_id(crop_id, output_folder=self.output_folder / "other_crops", tot_samp_lpis=self.tot_samp_other_lpis, tot_samp_lucas=self.tot_samp_other_lucas)
        return crop_ids, other_crop_ids