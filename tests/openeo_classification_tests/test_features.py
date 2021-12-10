from openeo_classification import features
from openeo_classification.connection import connection
from functools import partial
from openeo_classification_tests import block25_31UFS
from openeo import DataCube

def test_classification_features():
    cube:DataCube = features.load_features(2019, partial(connection,"openeo.creo.vito.be"),provider="creo")
    errors = cube.validate()
    print(errors)


def test_benchmark_creo():
    cube = features.load_features(2021, partial(connection,"openeo.creo.vito.be"),provider="creo")

    box = block25_31UFS
    cube.filter_bbox(west=box[0],south=box[1],east=box[2],north=box[3],crs='EPSG:32631').download("block25.tiff")