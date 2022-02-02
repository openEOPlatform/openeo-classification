import geopandas as gpd
from importlib.resources import open_text
import openeo_classification.resources.grids as grids
import json

EU27 = [
"France",
"Sweden",
"Poland",
"Austria",
"Hungary",
"Romania",
"Lithuania",
"Latvia",
"Estonia",
"Germany",
"Bulgaria",
"Greece",
"Croatia",
"Luxembourg",
"Belgium",
"Netherlands",
"Portugal",
"Spain",
"Ireland",
"Italy",
"Denmark",
"Slovenia",
"Finland",
"Slovakia",
"Czechia"
]

def LAEA_20km()->gpd.GeoDataFrame:
    europe = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    europe = europe[europe.continent=="Europe"]
    countries = europe[europe.name.isin(EU27)]
    df = gpd.read_file("https://artifactory.vgt.vito.be/auxdata-public/grids/LAEA-20km.gpkg",mask=countries)

    return df


def UTM_100km_EU27()->gpd.GeoDataFrame:
    europe = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    europe = europe[europe.continent=="Europe"]
    countries = europe[europe.name.isin(EU27)]
    df = gpd.read_file("https://artifactory.vgt.vito.be/auxdata-public/grids/utm-tiling-grid-100km.gpkg",mask=countries)

    return df

def cropland_EU27()->gpd.GeoDataFrame:
    f = open_text(grids,"cropland_20km.geojson")
    features = json.load(f)
    df = gpd.GeoDataFrame.from_features(features)
    return df