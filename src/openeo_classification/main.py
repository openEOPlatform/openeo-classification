from pathlib import Path

from features import *
from sample import *
from connection import creo
from job_management import *
import openeo_classification
import os
from explore import count_croptypes


## Load in the data
input_data, fns = read_f(directory = "resources/reference_data/")

## Setting parameters for running jobs
years = [2017,2018,2019]
zones = ["31U"]
base_path = Path(openeo_classification.__file__).parent / "resources"/"training_data"
os.makedirs(base_path,exist_ok=True)
## Indicate which crops you want to classify and get their respective ID's, as well as the ID's of the crops you don't want to classify (the "other" class)
crop_list = ["Maize", "Wheat", "Barley","Soya beans", "Potatoes", "Sugar beet", "Grasses and other fodder crops"]
## Load or create the file that will monitor job statistics
df = create_or_load_job_statistics(path = base_path / "job_statistics.csv")

first_time = False

if first_time:
    ## Do some preliminary data exploration
    print(count_croptypes(input_data, fns))

    ## Create the JSON files containing point samples that will be used for the feature calculation of the training / test data
    ## Note! This only needs to be run once.
    crop_ids, other_crop_ids = sample_and_store_polygons(crop_list=crop_list, zones=zones, years=years, input_df=input_data, output_folder=base_path)
else:
    crop_ids, other_crop_ids = get_crop_codes(crop_list, input_data)

crops_of_interest = True
if crops_of_interest:
    ids = crop_ids
    fp = base_path / "crops_of_interest"
else:
    ids = other_crop_ids
    fp = base_path / "other_crops"


for year in years:
    features = load_features(year)
    for zone in zones:
        for i in ids:
            for fnp in glob.glob(str(fp/ ("sampleable_polygons_year"+str(year)+"_zone"+zone+"_id"+str(i)+"*.json"))):
                if not df.empty and fnp in df["fp"].tolist():
                    continue
                while True:
                    if len(df[(df["status"]=="running") | (df["status"]=="queued")]) < 2:
                        with open(fnp) as fn:
                            pol = fn.readlines()[0]
                        print(len(eval(pol)["features"]))
                        sampled_features = features.filter_spatial(json.loads(pol))
                        job = sampled_features.send_job(
                            title="Punten extraheren van sentinelhub",
                            out_format="netCDF",
                            sample_by_feature=True,
                            job_options=job_options)
                        job.start_job()
                        print("Job ID: {}; Year: {}; Crop ID: {}; Part: ?".format(job.job_id, year, i))
                        df = df.append({
                            "fp": fnp,
                            "status": job.status(),
                            "id": job.job_id,
                            "cpu": "tbd",
                            "memory": "tbd",
                            "duration": "tbd"
                        },ignore_index=True)
                        df.to_csv(base_path/"job_statistics.csv",index=False)
                        break
                    else:
                        time.sleep(60)
                        df = update_statuses(df)
                        df.to_csv(base_path/"job_statistics.csv",index=False)
