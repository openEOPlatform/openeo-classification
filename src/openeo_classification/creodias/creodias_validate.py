import json

from shapely.geometry import mapping

from openeo.rest import OpenEoApiError
from openeo_classification.connection import creo, terrascope_dev
from openeo_classification.features import sentinel2_features
from openeo_classification.grids import UTM_100km_EU27, UTM_100km_World
tiles = UTM_100km_World()

result = []

#with open("result_2019.json", "r+") as f:
#    data = json.load(f)
#    result = data["features"]
previously_loaded = {f["id"]:f for f in result}

idx_dekad, idx_list, s2_list  = sentinel2_features(2021,creo,provider="creodias")
for idx,row in tiles.iterrows():
    box = row.geometry.bounds
    tileID = row['name'][:5]
    print(tileID)
    if tileID not in previously_loaded:
        try:
            errors = idx_dekad.filter_bbox(west=box[0], south=box[1], east=box[2], north=box[3]).validate()
            tiles = list(set([ msg['message'][6:66] for msg in errors if tileID in msg['message'] ]))
            print(len(tiles))
            print(tiles)
            result.append({"id":tileID,"properties": {"count":len(tiles), "tiles":tiles},"geometry":mapping(row.geometry),"type":"Feature"})
            fc = {
                "type": "FeatureCollection",
                "features": result
            }
            #check planetary computer?
            #https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_20220125T015621_R117_T52NEK_20220125T105712
            with open("result_2021.json","w") as f:
                json.dump(fc,f)
        except OpenEoApiError as e:
            print(e)
            pass
