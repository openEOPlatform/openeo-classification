from pathlib import Path
import logging
import fire
import geopandas as gpd
from matplotlib.colors import ListedColormap

from openeo.processes import ProcessBuilder
from openeo.rest.mlmodel import MlModel

from openeo_classification import features
from openeo_classification.connection import terrascope_dev,creo_new
from openeo_classification.job_management import run_jobs, MultiBackendJobManager
from openeo_classification.resources import read_json_resource

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

def produce_on_terrascope():
    produce_eu27_croptype_map(provider="terrascope", year=2021, parallel_jobs=3, status_file="eu27_2021_terrascope_klein.csv")

def produce_on_sentinelhub():
    produce_eu27_croptype_map(provider="sentinelhub", year=2021, parallel_jobs=8, status_file="eu27_2021_shub2.csv")

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

    if "terrascope" == provider:
        terrascope_tiles = terrascope_tiles[(terrascope_tiles.sentinel2count>70) & (terrascope_tiles.sentinel1count>80)]
    else:
        terrascope_tiles = terrascope_tiles[(terrascope_tiles.sentinel2count <= 70) | (terrascope_tiles.sentinel1count <= 80)]
    terrascope_tiles = terrascope_tiles.sort_values(by=["cropland_perc"], ascending=False)


    logger.info(f"Found {len(terrascope_tiles)} tiles to process using {provider}. Year: {year}")

    connection = creo_new if provider == "creodias" else terrascope_dev


    def predict_catboost(cube,model):
        if not isinstance(model, MlModel):
            model = MlModel.load_ml_model(connection=connection, id=model)

        reducer = lambda data, context: ProcessBuilder.process("predict_catboost",data=data, model=context)
        return cube.reduce_dimension(dimension="bands", reducer=reducer, context=model)


    col_palette = [
        "#FFFFFF",
        "#CCFF33",
        "#66FF33",
        "#FF9900",
        "#CC9900",
        "#990033",
        "#FFFF00",
        "#FF6699"
    ]
    cmap = ListedColormap(col_palette)
    classification_colors = {x: cmap(x) for x in range(0, len(col_palette))}

    def run(row,connection_provider,connection, provider):
        box = row.geometry.bounds

        #cropland = row.cropland_perc
        title = f"EU27 croptypes {row['name']} {provider}"
        if(Path(title + ".tif").exists() or Path("creotesting/" + title ).exists()):
            return None

        job_options = {
            "driver-memory": "2G",
            "driver-memoryOverhead": "2G",
            "driver-cores": "1",
            "executor-memory": "3584m",
            "executor-memoryOverhead": "2560m",
            "executor-cores": "2",
            "max-executors": "20"
        } if provider != "creodias" else features.creo_job_options_production

        cube = predict_catboost(features.load_features(year, connection_provider, provider=provider),
                                model="https://raw.githubusercontent.com/openEOPlatform/openeo-classification/main/models/ml_model_groot.json")
        job = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).linear_scale_range(0,20,0,20).create_job(out_format="GTiff",
                                                                                                                              title=title,
                                                                                                                              description=f"Croptype map for 5 crops in EU27.",
                                                                                                                              job_options=job_options, overviews="ALL", colormap=classification_colors)
        job.start_job()
        job.logs()
        return job

    manager = MultiBackendJobManager()
    manager.add_backend(provider, connection=terrascope_dev, parallel_jobs=8)
    if(provider!="terrascope"):
        manager.add_backend("creodias", connection=creo_new, parallel_jobs=4)

    manager.run_jobs(
        df=terrascope_tiles,
        start_job=run,
        output_file=Path(status_file)
    )



if __name__ == '__main__':
  fire.Fire(produce_on_sentinelhub())