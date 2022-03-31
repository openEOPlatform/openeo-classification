from openeo_classification.landuse_classification import _get_epsg


def test_get_epsg():
    assert _get_epsg(50, 32) == 32632
    assert _get_epsg(-50, 32) == 32732
