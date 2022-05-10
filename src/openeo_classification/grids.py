import geopandas as gpd

from openeo_classification.resources import read_json_resource

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
    df = df.cx[-14:35, 33:72] #rough EU bbox to get rid of overseas areas
    return df


def UTM_100km_World()->gpd.GeoDataFrame:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[world.continent!='Antarctica'][world.continent!='Seven seas (open ocean)']
    countries = world[~world.name.isin(["Greenland","Iceland"])]
    df = gpd.read_file("https://artifactory.vgt.vito.be/auxdata-public/grids/utm-tiling-grid-100km.gpkg",mask=countries)
    return df


def cropland_EU27() -> gpd.GeoDataFrame:
    features = read_json_resource("openeo_classification.resources.grids", "cropland_20km.geojson")
    df = gpd.GeoDataFrame.from_features(features)
    return df

def UTM_20km_EU27()->gpd.GeoDataFrame:
    europe = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    europe = europe[europe.continent=="Europe"]
    countries = europe[europe.name.isin(EU27)]
    df = gpd.read_file("/home/driesj/data/tiling-grid-0/utm-tiling-grid-20km.gpkg",mask=countries)
    df = df.cx[-14:35, 33:72] #rough EU bbox to get rid of overseas areas
    return df
