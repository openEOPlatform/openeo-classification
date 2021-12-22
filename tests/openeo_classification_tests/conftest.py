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
