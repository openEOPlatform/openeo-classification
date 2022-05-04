"""
Queries number of available Sentinel-2 and Sentinel-1 backscatter products on Terrascope, to speed up processing.

"""
import time

from openeo_classification.grids import cropland_EU27
from openeo_classification import features
from openeo_classification.connection import terrascope_dev
grid = cropland_EU27()
#grid = grid[grid['cropland_perc']>5]
print(len(grid))


grid["provider"]="notset"
grid["sentinel1count"]=0
grid["sentinel2count"]=-1

from terracatalogueclient import Catalogue
catalogue = Catalogue()

year = 2021

for i in grid.index:
    row = grid.loc[i]
    box = row.geometry.bounds
    cropland = row.cropland_perc
    print(row)
    print(box)
    count = catalogue.get_product_count(
        "urn:eop:VITO:CGS_S1_GRD_SIGMA0_L1",
        start=str(year) + "-01-01",
        end=str(year) + "-12-31",
        bbox = list(box)
    )
    print(count)
    grid.loc[i, "sentinel1count"] = count


    #cube = features.load_features(2019, terrascope_dev, provider="terrascope")
    #validation = cube.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).validate()
    s2_count = catalogue.get_product_count(
        "urn:eop:VITO:TERRASCOPE_S2_TOC_V2",
        start=str(year) + "-01-01",
        end=str(year) + "-12-31",
        bbox=list(box)
    )
    grid.loc[i, "sentinel2count"] = s2_count
    if s2_count > 100 and count > 80:
        grid.loc[i, "provider"] = "terrascope"

    grid.to_csv(f"jobsplit_{year}_all.csv", index=False)
    grid.to_file(f"jobsplit_{year}_all.geojson", driver='GeoJSON')