from openeo_classification import grids

def test_LAEA_20km():
    df = grids.LAEA_20km()
    assert len(df) == 11286
    #print(df)
    #df.to_file("laea_grid.geojson", driver='GeoJSON')
