from openeo_classification import features
from openeo_classification.connection import connection, terrascope_dev, creo
from functools import partial
from openeo_classification_tests import block25_31UFS
from openeo import DataCube, RESTJob as BatchJob


def test_classification_features():
    cube:DataCube = features.load_features(2019, partial(connection,"openeo.creo.vito.be"),provider="creodias")
    box = [3.0,54.0,4.0,55.0]
    errors = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).validate()

    print(errors)


def test_benchmark_creo():
    cube = features.load_features(2021, creo,provider="creo")

    box = block25_31UFS
    cube.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25.tiff")


def test_benchmark_creo_sentinel1():
    """
    NOTE: for 2021, it seems that a large number of GRD products are missing on CreoDIAS!!
    @return:
    """
    cube = features.sentinel1_features(2020, creo,provider="creo",orbitDirection="ascending",relativeOrbit=88)
    stats = features.compute_statistics(cube)

    box = block25_31UFS
    stats.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25_creo_int16_2020.nc")
    #job.download_results("/tmp/")

def test_benchmark_terrascope_sentinel1():
    cube = features.sentinel1_features(2020, terrascope_dev,provider="terrascope",relativeOrbit=88)
    stats = features.compute_statistics(cube)

    box = block25_31UFS
    stats.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25_terrascope_2020.nc")

def test_sentinel1_inputs():
    cube = features.sentinel1_inputs(2021, terrascope_dev, provider="terrascope", relativeOrbit=88)
    box = block25_31UFS
    cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3], crs='EPSG:32631')\
        .download("block25_terrascope_inputs.nc")

def test_sentinel1_inputs_creo():
    cube = features.sentinel1_inputs(2021, creo, provider="creo", relativeOrbit=88)
    box = block25_31UFS
    cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3], crs='EPSG:32631')\
        .download("block25_creo_inputs.nc")