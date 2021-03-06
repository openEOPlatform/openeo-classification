from pathlib import Path

from features import *
from sample import *
from connection import creo
from job_management import *
import openeo_classification
import os
from explore import count_croptypes
import pandas as pd
from openeo_classification.connection import connection
import geopandas as gpd

# Load in the data
input_data_lpis, input_data_lucas, fns = read_f(directory = "resources/reference_data/")
aez_df = gpd.read_file("resources/AEZ2.geojson")[["groupID","zoneID","geometry"]]

# Setting parameters for running jobs
years = [2017, 2018, 2019]

zones = sorted(list(set.union(set(input_data_lpis["zonenumber"]), set(input_data_lucas["zonenumber"])))) #[31]
base_path = Path(openeo_classification.__file__).parent / "resources" / "training_data"
os.makedirs(base_path,exist_ok=True)
## Indicate which crops you want to classify and get their respective ID's, as well as the ID's of the crops you don't want to classify (the "other" class)
crop_list = [
                "Winter wheat", "Winter barley", "Winter cereal", "Winter rye" # Winter cereals   : 1110, 1510, 1910,
                "Spring wheat", "Spring barley", "Spring cereal", "Spring rye" # Spring / summer cereals : 1120, 1520, 1920, 
                "Winter rapeseed", "Maize", "Potatoes", "Sugar beet", # 4351, 1200, 5100, 8100
                # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses : 9100, 9110, 9120
    ]

sample = SamplePolygons(crop_list=crop_list, zones=zones, years=years, input_dfs=[input_data_lpis, input_data_lucas], 
    aez_stratification=aez_df, output_folder=base_path, tot_samp_crops_lpis=3000, tot_samp_other_lpis=1500, 
    tot_samp_crops_lucas=2000, tot_samp_other_lucas=1000, repeat_per_sample_lpis=3)

first_time = False
if first_time:
    ## Do some preliminary data exploration
    # pd.set_option('display.max_rows', None)
    # print(count_croptypes(input_data_lpis, fns).sort_values(by=["TOTAL"],ascending=False))

    ## Create the JSON files containing point samples that will be used for the feature calculation of the training / test data
    ## Note! This only needs to be run once.
    crop_ids, other_crop_ids = sample.sample_and_store_polygons()
else:
    crop_ids, other_crop_ids = sample.get_crop_codes()

crops_of_interest = True
if crops_of_interest:
    ids = crop_ids
    fp = base_path / "crops_of_interest"
else:
    ids = other_crop_ids
    fp = base_path / "other_crops"


def run_jobs(df, fnp, features, year, output_job_stat_path):
    while True:
        if len(df[(df["status"]=="running") | (df["status"]=="queued")]) < 2:
            pols = gpd.read_file(fnp,crs=4326)
            pols["geometry"] = pols["geometry"].centroid
            sampled_features = features.aggregate_spatial(json.loads(pols.to_json()), reducer="mean")
            job = sampled_features.send_job(
                title="Punten extraheren van sentinelhub",
                out_format="CSV",
                job_options=job_options,
            )
            job.start_job()
            print("Job ID: {}; Year: {}; Crop ID: {}".format(job.job_id, year, fnp))
            df = df.append({
                "fp": fnp,
                "status": job.status(),
                "id": job.job_id,
                "cpu": "tbd",
                "memory": "tbd",
                "duration": "tbd",
                "source": "tbd"
            },ignore_index=True)
            df.to_csv(output_job_stat_path,index=False)
            return df
        else:
            time.sleep(60)
            df = update_statuses(df)
            df.to_csv(output_job_stat_path,index=False)


corrupt_jsons = [
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\crops_of_interest\sentinelhub\sampleable_polygons_year2018_zone39_id5100_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone31_id4351_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone31_id4351_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone32_id2900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id1120_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id1700_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id1900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id2260_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id2900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id4100_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id4351_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id4380_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id7700_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id7900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id9100_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id9100_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id9100_p2.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id9110_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id9110_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2017_zone33_id9120_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone31_id9213_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id0_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id0_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1100_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1500_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1500_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1700_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1700_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id1900_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id2230_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id2900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id3300_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id3300_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id3490_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id3690_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id3690_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4380_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4380_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4420_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4420_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4420_p2.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4420_p3.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id4490_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id5000_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id7900_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id9100_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id9100_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id9110_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id9211_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone37_id9998_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone38_id0_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone38_id0_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone38_id0_p2.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone38_id1100_p0.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone38_id1100_p1.json",
    r"c:\users\driesseb\dev\openeo-classification\src\openeo_classification\resources\training_data\other_crops\sentinelhub\sampleable_polygons_year2018_zone38_id1100_p2.json"
]


for year in years:
    for prov in ["terrascope", "sentinelhub"]:
        for zone in zones:
            features = load_features(year, connection_provider = connection, provider = prov, sampling=True,processing_opts=dict(tile_size=128))
            for fnp in glob.glob(str(fp / prov / ("*"+str(year)+"_zone"+str(zone)+"*"))):
                if fnp in corrupt_jsons:
                    continue
                else:
                    with open(fnp) as fn:
                        pol = fn.readlines()[0]
                        df = create_or_load_job_statistics(path = base_path / "job_statistics.csv")
                        if not df.empty and fnp in df["fp"].tolist():
                            continue
                        df = run_jobs(df, fnp, features, year, base_path / "job_statistics.csv")

