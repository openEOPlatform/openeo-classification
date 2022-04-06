from pathlib import Path

from openeo_classification.features import *
from openeo_classification.sample import *
from openeo_classification.connection import creo
from openeo_classification.job_management import *
import openeo_classification
import os
import pandas as pd
from openeo_classification.connection import connection
import geopandas as gpd

# Load in the data
resource_path = Path(openeo_classification.__file__).parent / "resources"
input_data_lpis, input_data_lucas, fns = read_f(directory =resource_path / "reference_data")
aez_df = gpd.read_file(resource_path / "AEZ2.geojson")[["groupID","zoneID","geometry"]]

# Setting parameters for running jobs
years = [2017, 2018, 2019]

zones = sorted(list(set.union(set(input_data_lpis["zonenumber"]), set(input_data_lucas["zonenumber"])))) #[31]
base_path = resource_path / "training_data"
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


dataframe = pd.DataFrame({'year':years}).merge(pd.DataFrame({'provider':["sentinelhub"]}), how='cross').merge(pd.DataFrame({'zone':zones}), how='cross').astype({'year': 'int32','zone': 'int32'},copy=False)

files = dataframe.apply(lambda row:glob.glob(str(fp / row['provider'] / ("*"+str(row['year'])+"_zone"+str(row['zone'])+"*"))),axis=1)
dataframe['sample_locations'] = files
dataframe = dataframe.explode(column='sample_locations',ignore_index=True).dropna(subset=['sample_locations'])
extents = dataframe.apply(lambda row: gpd.read_file(row['sample_locations'], crs=4326),axis=1)
dataframe = gpd.GeoDataFrame(dataframe,geometry=extents.apply((lambda s: s.unary_union.convex_hull)))
dataframe['sample_count'] = extents.apply(lambda x:len(x))
#dataframe = dataframe[~dataframe.zone.isin([30,31]) ]


print(f'Found {len(dataframe)} files to sample.' )

def run(row):
    year = row['year']
    zone = row['zone']
    provider = row['provider']
    fnp = row['sample_locations']
    features = load_features(year, connection_provider=creo, provider="creodias", sampling=True,
                             processing_opts=dict(tilesize=64))

    pols = gpd.read_file(fnp, crs=4326)
    pols["geometry"] = pols["geometry"].centroid
    sampled_features = features.aggregate_spatial(json.loads(pols.to_json()), reducer="mean")
    print(f"Year: {year}; Crop ID: {fnp}")
    errors = sampled_features.validate()
    tiles = list(set([msg['message'][6:66] for msg in errors]))
    print(len(tiles))
    print(tiles)
    if len(tiles) == 0:
        job = sampled_features.send_job(
            title="Punten extraheren {year} - {zone}",
            description="Punten extraheren voor {year} - {zone} - {fnp} ",
            out_format="CSV",
            job_options=creo_job_options,
        )

        job.start_job()
        return job
    else:
        return None


run_jobs(dataframe, run, Path("sampling_creo.csv"), parallel_jobs=1, connection_provider=creo)