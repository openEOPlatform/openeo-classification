from pathlib import Path

import openeo_classification
import pandas as pd
import re
from openeo_classification.connection import connection
import glob
import netCDF4 as nc
import numpy as np


base_path = Path(openeo_classification.__file__).parent / "resources"/"training_data"

job_file = "job_statistics - Copy.csv"
download_data = False

def download_data():
	job_stats = pd.read_csv(str(base_path / job_file))
	finished = job_stats[job_stats["status"]=="finished"]
	if len(finished) != len(job_stats):
		print("Warning! Be aware of the fact that not all of your jobs were completed succesfully!")

	con = connection()
	for index, row in finished.iterrows():
		job_path = str(base_path / "netcdfs" / row["id"])
		if not os.path.exists(job_path):
			con.job(row["id"]).get_results().download_files(job_path)

def match_ids():
	job_stats = pd.read_csv(str(base_path / "job_statistics - Copy.csv"))
	finished = job_stats[job_stats["status"]=="finished"]
	p = re.compile(r'.*\\([a-z_]+)\\([a-z]+)\\sampleable_polygons_year([0-9]+)_zone([0-9]+)_id([0-9]+)_p([0-9]).json')
	### load in the geojsons here and get features in the geojsons : zoneID and groupID
	df = pd.DataFrame(columns=["jobid","crops_of_interest","platform","year","zone","ids","part"])
	for index, row in finished.iterrows():
		m = p.match(row["fp"])
		row = {
			"jobid": row["id"],
			"crops_of_interest": m.group(1) == "crops_of_interest",
			"platform": m.group(2),
			"year": m.group(3),
			"zone": m.group(4),
			"ids": m.group(5),
			"part": m.group(6)
		}
		df = df.append(row, ignore_index=True)
	return df

def load_netcdfs():
	id_df = match_ids()
	count = 0
	band_names = ["B06", "B12"] + ["NDVI", "NDMI", "NDGI", "ANIR", "NDRE1", "NDRE2", "NDRE5"] + ["ratio", "VV", "VH"]
	tstep_labels = ["t" + str(4 * index) for index in range(0, 6)]
	features = [band + "_" + stat for band in band_names for stat in ["p10", "p50", "p90", "sd"] + tstep_labels]
	df = pd.DataFrame(columns=list(id_df.columns) + list(features))
	for fnp in glob.glob(str(base_path / "netcdfs" / "*" / "*")):
		ds_orig = nc.Dataset(fnp)
		p = re.compile(r'.*\\netcdfs\\([0-9a-z-]+)\\openEO_[0-9]+.nc')
		job_id = p.match(fnp).group(1)

		row = {}
		for feat in features:
			row[feat] = [np.mean(ds_orig[feat])]
		final = pd.concat([
			id_df[id_df["jobid"] == job_id].reset_index(drop=True), 
			pd.DataFrame(row)
		],axis=1)
		df = df.append(final)
	print(df)
	df.to_csv("resources/training_data/final_features.csv")

if download_data:
	download_data()