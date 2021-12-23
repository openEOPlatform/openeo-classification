import pytest
import openeo_classification
from importlib.resources import read_text,open_text, read_binary
from openeo_classification.resources.training_data import crops_of_interest
import json
from shapely.geometry import mapping,shape,GeometryCollection
import geopandas as gpd

@pytest.fixture
def some_polygons():

    with open_text(crops_of_interest,"sampleable_polygons_year2019_zone31U_id8100_p0.json") as f:
        return GeometryCollection([ shape(feature["geometry"]) for feature in json.load(f)["features"]][0:10])

@pytest.fixture
def some_20km_tiles():
    from openeo_classification.grids import LAEA_20km
    return GeometryCollection(LAEA_20km().geometry.sample(10).values)



@pytest.fixture
def some_20km_tiles_with_cropland():
    from openeo_classification.grids import cropland_EU27
    df = cropland_EU27()
    sample = df[df['cropland_perc']>50].geometry.sample(10)
    return GeometryCollection(sample.values)

@pytest.fixture
def some_20km_tiles_in_belgium():
    """
    A few tiles in Belgium, stratified by cropland percentage
    @return:
    """

    from openeo_classification.grids import cropland_EU27
    df = cropland_EU27()

    belgium = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    belgium = belgium[belgium.name == "Belgium"]

    df = df.sjoin(belgium, how="inner")
    import pandas as pd
    cut = pd.qcut(df.cropland_perc,5 , retbins=False)
    df['bins'] = cut
    sample = df.drop_duplicates('bins')
    return sample