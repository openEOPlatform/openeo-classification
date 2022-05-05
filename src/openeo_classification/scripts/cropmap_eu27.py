from pathlib import Path
import logging
import fire
import geopandas as gpd

from openeo.processes import ProcessBuilder
from openeo.rest.mlmodel import MlModel

from openeo_classification import features
from openeo_classification.connection import terrascope_dev,creo_new
from openeo_classification.job_management import run_jobs
from openeo_classification.resources import read_json_resource

logger = logging.getLogger("openeo_classification.cropmap")

def produce_eu27_croptype_map(provider="terrascope",year=2021, parallel_jobs = 2, status_file = "eu27_terrascope.csv"):
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

    terrascope_tiles = gpd.GeoDataFrame.from_features(read_json_resource("openeo_classification.scripts", "terrascope_data_2021.geojson"))
    terrascope_tiles = terrascope_tiles[(terrascope_tiles.sentinel2count>70) & (terrascope_tiles.sentinel1count>80)]

    logger.info(f"Found {len(terrascope_tiles)} tiles to process using {provider}. Year: {year}")

    connection = creo_new if provider == "creodias" else terrascope_dev


    def predict_catboost(cube,model):
        if not isinstance(model, MlModel):
            model = MlModel.load_ml_model(connection=connection, id=model)

        reducer = lambda data, context: ProcessBuilder.process("predict_catboost",data=data, model=context)
        return cube.reduce_dimension(dimension="bands", reducer=reducer, context=model)


    cube = predict_catboost(features.load_features(year, connection, provider=provider),model="")

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": "2G",
        "executor-memoryOverhead": "1G",
        "executor-cores": "2",
        "max-executors": "20"
    }

    def run(row):
        box = row.geometry.bounds
        cropland = row.cropland_perc
        job = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).create_job(out_format="GTiff",
                                                                                              title=f"EU27 croptypes {row.name} - {cropland:.1f}",
                                                                                              description=f"Croptype map for 5 crops in EU27.",
                                                                                              job_options=job_options, overviews="ALL")
        job.start_job()
        job.logs()
        return job

    run_jobs(
        df=terrascope_tiles,
        start_job=run,
        outputFile=Path(status_file),
        connection_provider=connection,
        parallel_jobs=parallel_jobs,
    )



if __name__ == '__main__':
  fire.Fire(produce_eu27_croptype_map)