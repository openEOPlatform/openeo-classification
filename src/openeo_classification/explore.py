import json
from pathlib import Path

import openeo
from openeo.processes import ProcessBuilder, array_modify, quantiles, sd, array_concat
import pandas as pd
import glob
import geopandas as gpd
import re
from openeo_classification.connection import terrascope_dev

## Derived from metadata excel file
all_crop_codes = {
	0: "Unknown",
	991: "NaN",
	1000: "Cereals",
	1100: "Wheat",1110: "Winter wheat",1120: "Spring wheat",
	1200: "Maize",
	1300: "Rice",
	1400: "Sorghum",
	1500: "Barley",1510: "Winter barley",1520: "Spring barley",
	1600: "Rye",1610: "Winter rye",1620: "Spring rye",
	1700: "Oats",
	1800: "Millets",
	1900: "Other cereals",1910: "Winter cereal",1920: "Spring cereal",
	2000: "Vegetables and melons",
	2100: "Leafy or stem vegetables",2110: "Artichokes",2120: "Asparagus",2130: "Cabages",2140: "Cauliflowers & brocoli",2150: "Lettuce",2160: "Spinach",2170: "Chicory",2190: "Other leafy/stem vegetables",
	2200: "Fruit-bearing vegetables",2210: "Cucumbers",2220: "Eggplants",2230: "Tomatoes",2240: "Watermelons",2250: "Cantaloupes and other melons",2260: "Pumpkin, squash and gourds",2290: "Other fruit-bearing vegetables",
	2300: "Root, bulb or tuberous vegetables",2310: "Carrots",2320: "Turnips",2330: "Garlic",2340: "Onions and shallots",2350: "Leeks & other alliaceous vegetables",2390: "Other root, bulb or tuberous vegetables",
	2400: "Mushrooms and truffles",
	2900: "Other vegetables",
	3000: "Fruit and nuts",
	3100: "Tropical and subtropical fruits",3110: "Avocados",3120: "Bananas & plantains",3130: "Dates",3140: "Figs",3150: "Mangoes",3160: "Papayas",3170: "Pineapples",3190: "Other tropical and subtropical fruits",
	3200: "Citrus fruits",3210: "Grapefruit & pomelo",3220: "Lemons & Limes",3230: "Oranges",3240: "Tangerines, mandarins, clementines",3290: "Other citrus fruit",
	3300: "Grapes",
	3400: "Berries",3410: "Currants",3420: "Gooseberries",3430: "Kiwi fruit",3440: "Raspberries",3450: "Strawberries",3460: "Blueberries",3490: "Other berries",
	3500: "Pome fruits and stone fruits",3510: "Apples",3520: "Apricots",3530: "Cherries & sour cherries",3540: "Peaches & nectarines",3550: "Pears & quinces",3560: "Plums and sloes",3590: "Other pome fruits and stone fruits",
	3600: "Nuts",3610: "Almonds",3620: "Cashew nuts",3630: "Chestnuts",3640: "Hazelnuts",3650: "Pistachios",3660: "Walnuts",3690: "Other nuts",
	3900: "Other fruit",
	4000: "Oilseed crops",
	4100: "Soya beans",
	4200: "Groundnuts",
	4300: "Temporary oilseed crops",4310: "Castor bean",4320: "Linseed",4330: "Mustard",4340: "Niger seed",4350: "Rapeseed",4351: "Winter rapeseed",4352: "Spring rapeseed",4360: "Safflower",4370: "Sesame",4380: "Sunflower",4390: "Other temporary oilseed crops",
	4400: "Permanent oilseed crops",4410: "Coconuts",4420: "Olives",4430: "Oil palms",4490: "Other oleaginous fruits",
	5000: "Root/tuber crops",
	5100: "Potatoes",
	5200: "Sweet potatoes",
	5300: "Cassava",
	5400: "Yams",
	5900: "Other roots and tubers",
	6000: "Beverage and spice crops",
	6100: "Beverage crops",6110: "Coffee",6120: "Tea",6130: "Maté",6140: "Cocoa",6190: "Other beverage crops",
	6200: "Spice crops",6211: "Chilies & peppers",6212: "Anise, badian, fennel",6219: "Other temporary spice crops",6221: "Pepper",6222: "Nutmeg, mace, cardamoms",6223: "Cinnamon",6224: "Cloves",6225: "Ginger",6226: "Vanilla",6229: "Other permanent spice crops",
	7000: "Leguminous crops",
	7100: "Beans",
	7200: "Broad beans",
	7300: "Chick peas",
	7400: "Cow peas",
	7500: "Lentils",
	7600: "Lupins",
	7700: "Peas",
	7800: "Pigeon peas",
	7900: "Other Leguminous crops",7910: "Other Leguminous crops - Temporary",7920: "Other Leguminous crops - Permanent",
	8000: "Sugar crops",
	8100: "Sugar beet",
	8200: "Sugar cane",
	8300: "Sweet sorghum",
	8900: "Other sugar crops",
	9000: "Other crops",
	9100: "Grasses and other fodder crops",9110: "Temporary grass crops",9120: "Permanent grass crops",
	9200: "Fibre crops",9210: "Temporary fibre crops",9211: "Cotton",9212: "Jute, kenaf and similar",9213: "Flax, hemp and similar",9219: "Other temporary fibre crops",9220: "Permanent fibre crops",
	9300: "Medicinal, aromatic, pesticidal crops",9310: "Temporary medicinal etc crops",9320: "Permanent medicinal etc crops",
	9400: "Rubber",
	9500: "Flower crops",9510: "Temporary flower crops",9520: "Permanent flower crops",
	9600: "Tobacco",
	9900: "Other other crops",9910: "Other crops - temporary",9920: "Other crops - permanent",9998: "mixed cropping"
}


def count_croptypes(f, fns):
	print("Creating an overview of crop types with their respective count in reference set...")
	df = pd.DataFrame(columns = fns+["TOTAL"], index=[[all_crop_codes[i] for i in f.CT.unique()]])
	for source in f["source"].unique():
	    tmp = f[f["source"]==source]
	    for i in tmp.CT.unique():
	        df[source].loc[all_crop_codes[i]] = len(tmp[tmp["CT"]==i])

	for i in f.CT.unique():
	    df["TOTAL"].loc[all_crop_codes[i]] = len(f[f["CT"]==i])

	df.sort_values("TOTAL",ascending=False)
	print("Crop type count overview completed")
	return df


def grid_statistics_mean():
	from .grids import LAEA_20km
	# from shapely.geometry import GeometryCollection
	# grid_df = LAEA_20km()
	wc = terrascope_dev().load_collection("ESA_WORLDCOVER_10M_2020_V1", bands="MAP",
										  temporal_extent=["2020-12-30", "2021-01-01"])

	statsfile = "cropland_mean.json"
	if (not Path(statsfile).exists()):

		(wc.band("MAP") == 40).polygonal_mean_timeseries(
			"https://artifactory.vgt.vito.be/auxdata-public/grids/LAEA-20km-EU27.geojson").execute_batch(
			statsfile)

	with open(statsfile,'r') as f:
		mean_list = json.load(f)["2020-12-31T00:00:00Z"]
		mean_list = [bands[0] if len(bands) > 0 else 0.0 for bands in mean_list]

	import geopandas as gpd
	grid_df = gpd.read_file("LAEA-20km-EU27.geojson")  # https://artifactory.vgt.vito.be/auxdata-public/grids/
	grid_df["cropland_perc"] = mean_list
	grid_df["cropland_perc"]=100.0*grid_df["cropland_perc"]
	grid_df.to_file("cropland_20km.geojson", driver='GeoJSON')

def grid_statistics():
	from .grids import LAEA_20km
	#from shapely.geometry import GeometryCollection
	#grid_df = LAEA_20km()
	wc = terrascope_dev().load_collection("ESA_WORLDCOVER_10M_2020_V1",bands="MAP",temporal_extent=["2020-12-30","2021-01-01"])

	statsfile = "cropland_mean.json"
	if(not Path(statsfile).exists()):
		#wc.aggregate_spatial("https://artifactory.vgt.vito.be/auxdata-public/grids/LAEA-20km-EU27.geojson",reducer='histogram').execute_batch(
		#	statsfile)
		(wc.band("MAP") == 40).polygonal_mean_timeseries("https://artifactory.vgt.vito.be/auxdata-public/grids/LAEA-20km-EU27.geojson").execute_batch(
			statsfile)
	with open(statsfile,'r') as f:
		histogram_list = json.load(f)["2020-12-31T00:00:00Z"]
		counts = [ len(bands) for bands in histogram_list]
		histogram_list = [ bands[0] if len(bands)>0 else {} for bands in histogram_list]

	totalvalidcells = sum(counts)
	print(totalvalidcells)

	import geopandas as gpd
	grid_df = gpd.read_file("LAEA-20km-EU27.geojson")#https://artifactory.vgt.vito.be/auxdata-public/grids/

	grassland_count = [h.get("30.0",0) for h in histogram_list]
	cropland_count = [h.get("40.0",0) for h in histogram_list]
	water_count = [h.get("80.0",0) for h in histogram_list]

	total_pixels = 2000*2000
	grid_df["cropland"] = cropland_count
	grid_df["grassland"] = grassland_count
	grid_df["water"] = water_count


	grid_df["cropland_perc"]=100.0*grid_df["cropland"]/total_pixels
	grid_df.to_file("statistics_20km.geojson", driver='GeoJSON')
	#grid_df.plot(column="cropland_perc")

