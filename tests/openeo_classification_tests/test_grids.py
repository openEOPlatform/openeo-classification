from openeo_classification import grids

def test_LAEA_20km():
    df = grids.LAEA_20km()
    assert len(df) == 11286
    #print(df)
    #df.to_file("laea_grid.geojson", driver='GeoJSON')

def test_utm_100km():
    df = grids.UTM_100km_EU27()
    assert len(df) == 681
    #print(df)

