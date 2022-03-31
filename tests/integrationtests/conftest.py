import json
from importlib.resources import open_text

import geopandas as gpd
import pytest
from shapely.geometry import shape, GeometryCollection

from openeo_classification.resources.training_data.crops_of_interest import sentinelhub

block25_31UFS = [630720, 5669280, 640960, 5679520]


@pytest.fixture
def some_polygons():

    with open_text(sentinelhub,"sampleable_polygons_year2019_zone31_id8100_p0.json") as f:
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
    cut = pd.qcut(df.cropland_perc,10 , retbins=False)
    df['bins'] = cut
    bins = df.groupby(['bins'])
    sample = df[df.cropland_perc.isin(bins.cropland_perc.quantile(interpolation='nearest'))]
    return sample