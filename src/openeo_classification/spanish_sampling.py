from pathlib import Path

from features import *
from sample_spain import *
from connection import creo
from job_management import *
import os
import pandas as pd
from connection import connection, terrascope_dev, creo, creo_new
import geopandas as gpd
import openeo

# Load in the data
input_data_lpis, input_data_lucas, fns = read_f(directory = "resources/reference_data/")
print(input_data_lpis.head())
aez_df = gpd.read_file("resources/AEZ2.geojson")[["groupID","zoneID","geometry"]]

# Setting parameters for running jobs
years = [2019]

zones = sorted(list(set.union(set(input_data_lpis["zonenumber"])))) #[31]

base_path = Path("resources") / "training_data"
# os.makedirs(base_path,exist_ok=True)
## Indicate which crops you want to classify and get their respective ID's, as well as the ID's of the crops you don't want to classify (the "other" class)
crop_list = [
                "Winter wheat", "Winter barley", "Winter cereal", "Winter rye" # Winter cereals   : 1110, 1510, 1910,
                # "Spring wheat", "Spring barley", "Spring cereal", "Spring rye" # Spring / summer cereals : 1120, 1520, 1920, 
                "Winter rapeseed", "Maize", "Potatoes",# "Sugar beet", # 4351, 1200, 5100, 8100
                # "Grasses and other fodder crops", "Temporary grass crops", "Permanent grass crops" # Grasses : 9100, 9110, 9120
    ]

sample = SamplePolygons(crop_list=crop_list, zones=zones, years=years, input_dfs=[input_data_lpis, input_data_lucas],
    aez_stratification=aez_df, output_folder=base_path, tot_samp_crops_lpis=3000, tot_samp_other_lpis=1500, 
    tot_samp_crops_lucas=2000, tot_samp_other_lucas=1000, repeat_per_sample_lpis=3)

# crop_ids, other_crop_ids = sample.sample_and_store_polygons()

crop_ids, other_crop_ids = sample.get_crop_codes()

# crops_of_interest = True
# if crops_of_interest:
#     ids = crop_ids
#     fp = base_path / "crops_of_interest"
# else:
#     ids = other_crop_ids
#     fp = base_path / "other_crops"


def run_jobs(df, fnp, features, year, output_job_stat_path, c):
    while True:
        print("run jobs")
        if len(df[(df["status"]=="running") | (df["status"]=="queued") | (df["status"] == "created")]) < 2:
            print("meer dan drie")
            print("Starting a job "+fnp)
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
            df = update_statuses(df, c)
            df.to_csv(output_job_stat_path,index=False)


typ = "crops_of_interest" #"other_crops"
if typ == "crops_of_interest":
    ids = crop_ids
    fp = base_path / "crops_of_interest_spain"
    con = terrascope_dev
    pth = "job_statistics_spain.csv"
else:
    ids = other_crop_ids
    fp = base_path / "other_crops_spain"
    con = creo_new
    pth = "job_statistics_spain_creo.csv"
    job_options = creo_job_options_production
for year in years:
    for zone in zones:
        c = lambda: openeo.connect("openeo-dev.vito.be").authenticate_oidc()
        features = load_features(year, connection_provider = c, provider = "sentinelhub", sampling=True,processing_opts=dict(tile_size=128))
        for fnp in glob.glob(str(fp / "sentinelhub" / ("*"+str(year)+"_zone"+str(zone)+"*"))):
            with open(fnp) as fn:
                pol = fn.readlines()[0]
                df = create_or_load_job_statistics(path = base_path / pth)
                if not df.empty and fnp in df["fp"].tolist():
                    continue
                df = run_jobs(df, fnp, features, year, base_path / pth, c)

                
# c = lambda: openeo.connect("openeo.vito.be").authenticate_oidc()
# year = 2019
# zone = 31
# features = load_features(year, connection_provider = c, provider = "sentinelhub", sampling=True,processing_opts=dict(tile_size=128))
# fnp = str(fp / "sentinelhub" / ("sampleable_polygons_year"+str(year)+"_zone"+str(zone)+"_id1510_p1.json"))
# with open(fnp) as fn:
#     pol = fn.readlines()[0]
# #     df = create_or_load_job_statistics(path = str(base_path / pth))
#     df = pd.DataFrame({
#         "fp": [],
#         "status": [],
#         "start_time": [],
#         "id": [],
#         "cpu": [],
#         "memory": [],
#         "duration": []
#     })
#     df = run_jobs(df, fnp, features, year, str(base_path / pth), c)