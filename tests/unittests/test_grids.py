from openeo_classification import grids


def test_LAEA_20km():
    # TODO: this is technically an integration test because it depends on external resources on artifactory
    df = grids.LAEA_20km()
    assert len(df) == 11286
    #print(df)
    #df.to_file("laea_grid.geojson", driver='GeoJSON')


def test_utm_100km():
    # TODO: this is technically an integration test because it depends on external resources on artifactory
    df = grids.UTM_100km_EU27()
    assert len(df) == 662


def test_utm_100km_geojson(tmp_path):
    df = grids.UTM_100km_World()
    df.to_file(tmp_path / "world.geojson", driver='GeoJSON')
    assert len(df) == 662


def test_cropland_EU27():
    df = grids.cropland_EU27()
    assert len(df) == 11286
