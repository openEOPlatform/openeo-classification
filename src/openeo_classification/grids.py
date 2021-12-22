import geopandas as gpd

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
