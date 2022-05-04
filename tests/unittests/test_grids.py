from openeo_classification import grids


def test_LAEA_20km(tmp_path):
    # TODO: this is technically an integration test because it depends on external resources on artifactory
    df = grids.LAEA_20km()
    df.to_file(tmp_path / "LAEA_20km.geojson", driver='GeoJSON')
    assert len(df) == 11286


def test_utm_100km_eu27(tmp_path):
    # TODO: this is technically an integration test because it depends on external resources on artifactory
    df = grids.UTM_100km_EU27()
    df.to_file(tmp_path / "UTM_100km_EU27.geojson", driver='GeoJSON')
    assert len(df) == 662


def test_utm_100km_world(tmp_path):
    # TODO: this is technically an integration test because it depends on external resources on artifactory
    df = grids.UTM_100km_World()
    df.to_file(tmp_path / "UTM_100km_World.geojson", driver='GeoJSON')
    assert len(df) == 16786


def test_cropland_EU27(tmp_path):
    # TODO: this is technically an integration test because it depends on external resources on artifactory
    df = grids.cropland_EU27()
    df.to_file(tmp_path / "cropland_EU27.geojson", driver='GeoJSON')
    assert len(df) == 11286
