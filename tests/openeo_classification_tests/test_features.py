from pathlib import Path

from openeo_classification import features
from openeo_classification.connection import connection, terrascope_dev, creo
from functools import partial
from openeo_classification_tests import block25_31UFS
from openeo import DataCube, RESTJob as BatchJob
from openeo_classification.job_management import run_jobs

def test_classification_features(some_20km_tiles_with_cropland):
    cube:DataCube = features.load_features(2019, partial(connection,"openeo.creo.vito.be"),provider="creodias")
    for polygon in some_20km_tiles_with_cropland:
        box = polygon.bounds

        errors = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).validate()
        print(f"Found {len(errors)} missing products")
        print(errors)




def test_benchmark_creo():
    cube = features.load_features(2021, creo,provider="creodias")

    box = block25_31UFS
    cube.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25.tiff")


def test_benchmark_creo_sentinel1():
    """
    NOTE: for 2021, it seems that a large number of GRD products are missing on CreoDIAS!!
    @return:
    """
    cube = features.sentinel1_features(2020, creo,provider="creodias",orbitDirection="ascending",relativeOrbit=88)
    stats = features.compute_statistics(cube)

    box = block25_31UFS
    stats.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25_creo_int16_2020.nc")
    #job.download_results("/tmp/")

def test_benchmark_creo_20km_tile(some_20km_tiles):
    cube = features.load_features(2020, creo, provider="creodias")

    for polygon in some_20km_tiles:
        box = polygon.bounds
        cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3] ).download("tile_20km.tiff")

def test_benchmark_creo_20km_tile_sentinel2(some_20km_tiles):
    s2_cube, idx_list, s2_list = features.sentinel2_features(2020, creo, provider="creodias")
    stats = features.compute_statistics(s2_cube)

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": "1G",
        "executor-memoryOverhead": "3G",
        "executor-cores": "2",
        "max-executors": "10"
    }

    for polygon in some_20km_tiles:
        box = polygon.bounds
        stats.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3] ).execute_batch("tile_20km.tiff",title="Sentinel-2 features",job_options=job_options)

def test_benchmark_terrascope_20km_tile_sentinel2(some_20km_tiles_in_belgium):
    s2_cube, idx_list, s2_list = features.sentinel2_features(2020, terrascope_dev, provider="terrascope")
    stats = features.compute_statistics(s2_cube).linear_scale_range(0,30000,0,30000)

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
                                                                                              title=f"Croptype Sentinel-2 features {cropland}",
                                                                                               description=f"Sentinel-2 features for croptype detection.",
                                                                                              job_options=job_options)
        job.start_job()
        return job


    run_jobs(some_20km_tiles_in_belgium,run,Path("benchmarks_sentinel2_terrascope_masked.csv"))

    #for polygon in some_20km_tiles_in_belgium:
        #box = polygon.bounds
        #stats.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3] ).execute_batch("tile_20km.tiff",title="Sentinel-2 features",job_options=job_options)

def test_benchmark_creo_sentinel1_samples(some_polygons):
    """
    NOTE: for 2021, it seems that a large number of GRD products are missing on CreoDIAS!!
    @return:
    """
    cube = features.sentinel1_features(2020, creo,provider="creodias",orbitDirection="ascending",relativeOrbit=88)
    stats = features.compute_statistics(cube)

    job_options = {
        "driver-memory": "1G",
        "driver-memoryOverhead": "1G",
        "driver-cores": "1",
        "executor-memory": "1G",
        "executor-memoryOverhead": "3G",
        "executor-cores": "1",
        "max-executors": "8"
    }

    stats.filter_spatial(some_polygons).execute_batch(title="Punten extraheren Sentinel-1",
                            out_format="netCDF",
                            sample_by_feature=True,job_options=job_options)
    #job.download_results("/tmp/")

def test_benchmark_terrascope_sentinel1():
    cube = features.sentinel1_features(2020, terrascope_dev,provider="terrascope",relativeOrbit=88)#
    stats = features.compute_statistics(cube)

    box = block25_31UFS
    stats.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25_terrascope_2020_88_short.nc")

def test_sentinel1_inputs():
    cube = features.sentinel1_inputs(2021, terrascope_dev, provider="terrascope", relativeOrbit=88)
    box = block25_31UFS
    cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3], crs='EPSG:32631')\
        .download("block25_terrascope_inputs.nc")

def test_sentinel1_inputs_creo():
    cube = features.sentinel1_inputs(2021, creo, provider="creodias", relativeOrbit=88)
    box = block25_31UFS
    cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3], crs='EPSG:32631')\
        .download("block25_creo_inputs.nc")