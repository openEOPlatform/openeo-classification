from functools import partial
from pathlib import Path
import datetime
from openeo import DataCube

from openeo_classification import features
from openeo_classification.connection import connection, terrascope_dev, creo
from openeo_classification.job_management import run_jobs, MultiBackendJobManager
from .conftest import block25_31UFS
from datetime import date

def test_classification_features(some_20km_tiles_with_cropland):
    cube:DataCube = features.load_features(2019, partial(connection,"openeo.creo.vito.be"),provider="creodias")
    for polygon in some_20km_tiles_with_cropland:
        box = polygon.bounds

        errors = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).validate()
        print(f"Found {len(errors)} missing products")
        print(errors)


creo_job_options = {
        "driver-memory": "4G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": "1G",
        "executor-memoryOverhead": "4500m",
        "executor-cores": "3",
        "executor-request-cores": "400m",
        "max-executors": "40"
    }

def test_benchmark_creo(some_20km_tiles_in_belgium):
    #cube = features.load_features(2021, creo,provider="creodias")
    s2_cube, idx_list, s2_list = features.sentinel2_features(date(2020,1,1),date(2020,12,31), creo, provider="creodias")
    cube = features.compute_statistics(s2_cube).linear_scale_range(0, 30000, 0, 30000)

    for index,tile in some_20km_tiles_in_belgium.iterrows():
        box = tile['geometry'].bounds
        cube.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3]).download("block25.tiff")


def test_benchmark_creo_sentinel1(some_20km_tiles_in_belgium):
    """
    NOTE: for 2021, it seems that a large number of GRD products are missing on CreoDIAS!!
    @return:
    """
    cube = features.sentinel1_features(date(2020,1,1),date(2020,12,31), creo,provider="creodias",relativeOrbit=88)
    stats = features.compute_statistics(cube, date(2020,1,1),date(2020,12,31), 10)

    def run(row):
        box = row.geometry.bounds
        cropland = row.cropland_perc
        job = stats.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).send_job(out_format="GTiff",
                                                                                              title=f"Croptype Sentinel-1 features {cropland:.1f}",
                                                                                               description=f"Sentinel-1 features for croptype detection.",
                                                                                              job_options=creo_job_options)
        job.start_job()
        return job


    run_jobs(
        df=some_20km_tiles_in_belgium,
        start_job=run,
        outputFile=Path("benchmarks_sentinel1_creo.csv"),
        connection_provider=creo,
        parallel_jobs=1,
    )

def test_benchmark_creo_20km_tile(some_20km_tiles_in_belgium):
    cube = features.load_features(2020, creo, provider="creodias")


    def run(row):
        box = row.geometry.bounds
        cropland = row.cropland_perc
        job = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).send_job(out_format="GTiff",
                                                                                              title=f"Croptype features {cropland:.1f}",
                                                                                               description=f"Features for croptype detection.",
                                                                                              job_options=creo_job_options)
        job.start_job()
        return job

    run_jobs(
        df=some_20km_tiles_in_belgium,
        start_job=run,
        outputFile=Path("benchmarks_creo.csv"),
        connection_provider=creo,
        parallel_jobs=1,
    )

def test_benchmark_creo_20km_tile_sentinel2(some_20km_tiles_in_belgium):
    s2_cube, idx_list, s2_list = features.sentinel2_features(2020, creo, provider="creodias")
    stats = features.compute_statistics(s2_cube, date(2020,1,1), date(2020,12,31), 10).linear_scale_range(0,30000,0,30000)



    def run(row):
        box = row.geometry.bounds
        cropland = row.cropland_perc
        job = stats.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).send_job(out_format="GTiff",
                                                                                              title=f"Croptype Sentinel-2 features {cropland:.1f}",
                                                                                               description=f"Sentinel-2 features for croptype detection.",
                                                                                              job_options=creo_job_options)
        job.start_job()
        return job


    run_jobs(
        df=some_20km_tiles_in_belgium,
        start_job=run,
        outputFile=Path("benchmarks_sentinel2_creo.csv"),
        connection_provider=creo,
        parallel_jobs=1,
    )

def test_benchmark_terrascope_20km_tile_sentinel2(some_20km_tiles_in_belgium):
    s2_cube, idx_list, s2_list = features.sentinel2_features(date(2020, 1,1), date(2020,12,31), terrascope_dev, provider="terrascope")
    stats = features.compute_statistics(s2_cube,date(2020, 1,1), date(2020,12,31), 10).linear_scale_range(0,30000,0,30000)

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": "1G",
        "executor-memoryOverhead": "1G",
        "executor-cores": "2"
    }

    def run(row):
        box = row.geometry.bounds
        cropland = row.cropland_perc
        job = stats.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).send_job(out_format="GTiff",
                                                                                              title=f"Croptype Sentinel-2 features {cropland:.1f}",
                                                                                               description=f"Sentinel-2 features for croptype detection.",
                                                                                              job_options=job_options)
        job.start_job()
        return job


    run_jobs(
        df=some_20km_tiles_in_belgium,
        start_job=run,
        outputFile=Path("benchmarks_sentinel2_terrascope_masked.csv"),
        connection_provider=terrascope_dev,
    )

    #for polygon in some_20km_tiles_in_belgium:
        #box = polygon.bounds
        #stats.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3] ).execute_batch("tile_20km.tiff",title="Sentinel-2 features",job_options=job_options)

def test_benchmark_terrascope_20km_tile_features(some_20km_tiles_in_belgium):
    cube = features.load_features(2020, terrascope_dev, provider="sentinelhub")

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
        job = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).send_job(out_format="GTiff",
                                                                                              title=f"Croptype features {cropland:.1f}",
                                                                                               description=f"Features for croptype detection.",
                                                                                              job_options=job_options)
        job.start_job()
        job.logs()
        return job


    run_jobs(
        df=some_20km_tiles_in_belgium,
        start_job=run,
        outputFile=Path("benchmarks_sentinelhub_masked.csv"),
        connection_provider=terrascope_dev,
        parallel_jobs=3,
    )

def test_benchmark_creo_samples(some_polygons):
    """
    NOTE: for 2021, it seems that a large number of GRD products are missing on CreoDIAS!!
    @return:
    """
    cube = features.load_features(2020, creo, provider="creodias")



    cube.filter_spatial(some_polygons).execute_batch(title="Punten extraheren",
                            out_format="netCDF",
                            sample_by_feature=True,job_options=creo_job_options)
    #job.download_results("/tmp/")

def test_benchmark_terrascope_sentinel1():
    cube = features.sentinel1_features(date(2020,1,1),date(2020,12,31), terrascope_dev,provider="terrascope",relativeOrbit=88)#
    stats = features.compute_statistics(cube)

    box = block25_31UFS
    stats.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25_terrascope_2020_88_short.nc")

def test_sentinel1_inputs():
    cube = features.sentinel1_inputs(date(2021,1,1), date(2021,12,31), terrascope_dev, provider="terrascope", relativeOrbit=88)
    box = block25_31UFS
    cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3], crs='EPSG:32631')\
        .download("block25_terrascope_inputs.nc")

def test_sentinel1_inputs_creo():
    cube = features.sentinel1_inputs(date(2021,1,1), date(2021,12,31), creo, provider="creodias", relativeOrbit=88)
    box = block25_31UFS
    cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3], crs='EPSG:32631')\
        .download("block25_creo_inputs.nc")


def test_benchmark_multibackend_20km_tile_sentinel2(some_20km_tiles_in_belgium):
    def start_job(row, connection_provider, provider, **kwargs):
        start_date = datetime.date(int(2020), 3, 15)
        end_date = datetime.date(int(2020), 10, 31)

        s2_cube = features.sentinel2_features(start_date=start_date, end_date=end_date, connection_provider=connection_provider, provider=provider)
        stats = features.compute_statistics(base_features=s2_cube, start_date=start_date, end_date=end_date, stepsize=10).linear_scale_range(0, 30000, 0, 30000)

        box = row.geometry.bounds
        cropland = row.cropland_perc
        stats = stats.filter_bbox(bbox=box)
        job = stats.send_job(
            out_format="GTiff",
            title=f"Croptype Sentinel-2 features {cropland:.1f}",
            description=f"Sentinel-2 features for crop type detection.",
            job_options=creo_job_options
        )
        job.start_job()
        return job

    manager = MultiBackendJobManager()
    manager.add_backend("creodias", connection=creo, parallel_jobs=1)
    manager.add_backend("terrascope", connection=terrascope_dev, parallel_jobs=1)

    manager.run_jobs(
        df=some_20km_tiles_in_belgium,
        start_job=start_job,
        output_file=Path("benchmark_multibackend_20km_tile_sentinel2.csv")
    )
