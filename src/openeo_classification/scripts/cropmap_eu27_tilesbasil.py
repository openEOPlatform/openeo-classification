from pathlib import Path
import logging
import fire
import geopandas as gpd
from matplotlib.colors import ListedColormap

from openeo.processes import ProcessBuilder, array_modify
from openeo.rest.mlmodel import MlModel

from openeo_classification import features
from openeo_classification.connection import terrascope_dev#,creo_new
from openeo_classification.job_management import run_jobs, MultiBackendJobManager
from openeo_classification.resources import read_json_resource

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("openeo_classification.cropmap")

configs = {
    "reduced_featureset": {
        "input":[{
            "collection": "SENTINEL2_L2A",
            "bands": ["B03","B04","B05","B06","B08","B11","B12","SCL"]
        },
        {
            "collection": "SENTINEL1_GRD",
            "bands": ["VV","VH"]
        }
        ]
    }
}

### ALL TILES FOR BASIL
TILES_TO_PROCESS = ["29SMD_3_3", "29SNC_0_0", "33TWM_0_2", "33TVL_3_0", "33VXG_1_4", "33UVB_2_2", "35VME_4_3", "35VMF_1_2", "31UDS_4_2",
            "31UES_3_3", "31UFS_1_1", "32UMD_1_2", "32UNC_4_0", "32UMC_3_3", "32ULC_3_1", "34UFF_1_1", "34UEG_1_2", "31UFU_3_0", "31UFT_1_1", 
            "34UEV_0_3", "34UCV_2_3"]

### THE NL / SI TILES
# TILES_TO_PROCESS = ["31UFU_3_0", "33TVL_3_0"] # NL, SI

### 
# MEER_NL_TILES = ["31UDF_4_4", "31UFU_4_0", "31UFU_4_1", "31UFU_4_2",
# "31UFU_4_3", "31UFU_4_4", "31UFT_4_0", "31UFT_4_1", "31UFT_4_2", 
# "31UFT_4_3", "31UFT_4_4", "31UFS_4_0", "31UFT_3_4", "31UFT_3_3",
# "31UFT_3_2", "31UFT_3_1", "31UFT_3_0", "31UFT_2_3", "31UFT_2_2",
# "31UFT_2_1", "31UFT_2_0", "31UFT_1_3", "31UFT_1_2", "31UFT_1_1",
# "31UFT_1_0", "31UFU_1_4", "31UFU_1_3", "31UFU_1_2"]


def produce_on_terrascope():
    produce_eu27_croptype_map(provider="terrascope", year=2021, parallel_jobs=3, status_file="eu27_2021_terrascope_klein.csv")

def produce_on_sentinelhub():
    produce_eu27_croptype_map(provider="sentinelhub", year=2021, parallel_jobs=4, status_file="eu27_2021_shub.csv")

def produce_on_creodias():
    produce_eu27_croptype_map(provider="creodias", year=2021, parallel_jobs=1, status_file="eu27_creodias.csv")

def produce_eu27_croptype_map(provider="terrascope",year=2021, parallel_jobs = 20, status_file = "eu27_2021_terrascope_klein.csv"):
    """
    Script to start and monitor jobs for the EU27 croptype map project in openEO platform CCN.
    The script can use multiple backends, to maximize throughput. Jobs are tracked in a CSV file, upon failure, the script can resume
    processing by pointing to the same csv file. Delete that file to start processing from scratch.

    @param provider: The data provider: terrascope - sentinelhub - creodias
    @param year:  The year for which to generate a cropmap
    @param parallel_jobs:
    @param status_file: The local file where status should be tracked.
    @return:
    """

    terrascope_tiles = gpd.GeoDataFrame.from_features(read_json_resource("openeo_classification.resources.grids", "cropland_20km_utm.geojson"))
    terrascope_tiles = terrascope_tiles[terrascope_tiles["name"].isin(TILES_TO_PROCESS)]

    if "terrascope" == provider:
        terrascope_tiles = terrascope_tiles[(terrascope_tiles.sentinel2count>70) & (terrascope_tiles.sentinel1count>80)]
    else:
        terrascope_tiles = terrascope_tiles[(terrascope_tiles.sentinel2count <= 70) | (terrascope_tiles.sentinel1count <= 80)]
    terrascope_tiles = terrascope_tiles.sort_values(by=["cropland_perc"], ascending=False)
    print(terrascope_tiles)

    logger.info(f"Found {len(terrascope_tiles)} tiles to process using {provider}. Year: {year}")

    connection = creo_new if provider == "creodias" else terrascope_dev


    def predict_catboost(cube,model):
        if not isinstance(model, MlModel):
            model = MlModel.load_ml_model(connection=connection, id=model)

        reducer = lambda data, context: ProcessBuilder.process("predict_catboost",data=data, model=context)
        return cube.reduce_dimension(dimension="bands", reducer=reducer, context=model)


    col_palette = [
        "#FF6699",#pink - other
        "#FF9900",#orange - maize
        "#66FF33",# green -winter cereals
        "#CCFF33",# light green - spring cereals
        "#FFFF00",# yellow - rapeseed
        "#CC9900",# brown - potato
        "#990033",# darkred - sugarbeet
        "#FFFF4c"# darker yellow - grassland (but we don't need to predict it necessarily so we can merge it with "other")
    ]
    cmap = ListedColormap(col_palette)
    classification_colors = {x: cmap(x) for x in range(0, len(col_palette))}

    def run(row):
        box = row.geometry.bounds

        #cropland = row.cropland_perc
        title = f"EU27 croptypes {row['name']} {provider}"
        if(Path(title + ".tif").exists() or Path("creotesting/" + title ).exists()):
            return None

        job_options = {
            "driver-memory": "6G",
            "driver-memoryOverhead": "4G",
            "driver-cores": "1",
            "executor-memory": "6g",
            "executor-memoryOverhead": "4g",
            "executor-cores": "2",
            "max-executors": "20"
        } if provider != "creodias" else features.creo_job_options_production

        print(f"submitting job to {provider}")
        cube_raw = features.load_features(year, connection, provider=provider)

        ### TODO: dit moet uit AEZ2.json geextract worden in bijv. de centroid pixel van de box (row.geometry.bounds), en dat is de waarde die "values" hoort te hebben (43000 is Spanje)
        cube_raw = cube_raw.apply_dimension(dimension="bands", process=lambda x: array_modify(data=x, values=43000.,index=120))
        cube_raw = cube_raw.apply_dimension(dimension="bands", process=lambda x: array_modify(data=x, values=43153.,index=121))

        cube = predict_catboost(cube_raw,model="https://artifactory.vgt.vito.be/auxdata-public/openeo/catboost_test/ml_model_groot.json")

        ## linear_scale_range for converting data type to smallest possible
        job = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).linear_scale_range(0,20,0,20).create_job(out_format="GTiff",
                                                                                                                              title=title,
                                                                                                                              description=f"Croptype map for 5 crops in EU27.",
                                                                                                                              job_options=job_options, overviews="ALL", colormap=classification_colors, overview_method="mode")
        job.start_job()
        job.logs()
        return job

    # manager = MultiBackendJobManager()
    # manager.add_backend(provider, connection=terrascope_dev, parallel_jobs=4)
    # if(provider!="terrascope"):
    #     manager.add_backend("creodias", connection=creo_new, parallel_jobs=10)

    # manager.run_jobs(
    #     df=terrascope_tiles,
    #     start_job=run,
    #     output_file=Path(status_file)
    # )
    run_jobs(
        df=terrascope_tiles,
        start_job=run,
        outputFile=Path(status_file),
        connection_provider=connection,
        parallel_jobs=parallel_jobs,
    )



if __name__ == '__main__':
  fire.Fire(produce_on_sentinelhub())