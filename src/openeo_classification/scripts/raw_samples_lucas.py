
from pathlib import Path
from shutil import copy2

import fire

from openeo_classification.features import creo_job_options,job_options
from openeo_classification.lucas import split_lucas
from openeo_classification.connection import creo,openeo,terrascope_dev,openeo_platform
from openeo_classification.job_management import MultiBackendJobManager

import json
import geopandas as gpd


TIMERANGE_LUT = {
    '2016': {'start': '2015-09-01',
             'end': '2016-08-30'},
    '2017': {'start': '2016-09-01',
             'end': '2017-08-30'},
    '2018': {'start': '2017-09-01','end': '2019-03-31'},
    '2019': {'start': '2018-09-01',
             'end': '2019-08-30'},
    '2020': {'start': '2019-09-01',
             'end': '2020-08-30'},
    '2021': {'start': '2020-09-01',
             'end': '2021-08-30'}
}


def get_input_TS(eoconn, time_range, geo,provider):
    '''Function that will build the
    input timeseries for the model'''

    # don't consider images with more than 85% cloud coverage
    s2_properties = {"eo:cloud_cover": lambda v: v <= 85}
    S2_L2A = eoconn.load_collection('SENTINEL2_L2A',
                                    bands=["B01", "B02", "B03",
                                           "B04", "B05", "B06",
                                           "B07", "B08", "B09",
                                           "B11", "B12", "B8A",
                                           "SCL"],
                                    properties=s2_properties)
    S2_L2A._pg.arguments['featureflags'] = {"tilesize":16}
    S2_L2A_masked = S2_L2A.process("mask_scl_dilation", data=S2_L2A,
                                   scl_band_name="SCL")
    s1properties = {"polarization": lambda p: p == "DV"}
    S1_GRD = eoconn.load_collection('SENTINEL1_GRD',
                                    bands=['VH', 'VV'],
                                    properties=s1properties)
    S1_GRD._pg.arguments['featureflags'] = {"tilesize": 16}
    isCreo = (provider.upper() == "CREODIAS")
    S1_GRD = S1_GRD.sar_backscatter(
        coefficient="sigma0-ellipsoid",
        local_incidence_angle=not isCreo,options={"implementation_version":"1","tile_size":16, "otb_memory":128,"debug":False})
    S1_GRD = S1_GRD.apply(lambda x: 10 * x.log(base=10))

    if isCreo:
        #Creo has corrupt Sentinel-2 manifests, so we only extract sentinel-1 for now
        merged_cube = S1_GRD
    else:
        merged_cube = S1_GRD.merge_cubes(S2_L2A_masked.resample_cube_spatial(S1_GRD))

    return merged_cube.filter_temporal(
        time_range).aggregate_spatial(geo, reducer='mean')


def run(row,connection_provider,connection, provider):

    fnp = row['FILENAME']
    year = "2018"
    time_start = TIMERANGE_LUT[year]['start']
    time_end = TIMERANGE_LUT[year]['end']
    time_range = [time_start, time_end]
    pols = gpd.read_file(fnp)
    filename = str(Path(fnp).name)
    pols["geometry"] = pols["geometry"].centroid
    pols = pols
    features = get_input_TS(connection,geo=json.loads(pols.to_json()),time_range=time_range,provider=provider)

    print(f"Year: {year}; Crop ID: {filename}")
    job_options = {
        "driver-memory": "4G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "2",
        "executor-memory": "2G",
        "executor-memoryOverhead": "1G",
        "executor-cores": "2",
        "max-executors": "40",
        "soft-errors": "true"
    }
    if provider.upper() == "CREODIAS":
        job_options = creo_job_options

    job = features.create_job(
        title=f"Lucas {filename} {provider}",
        description=f"Sampling Lucas {filename}",
        out_format="NetCDF",
        job_options=job_options,
    )
    print(job)
    return job




class CustomJobManager(MultiBackendJobManager):

    def on_job_done(self, job, row):
        fnp = row['FILENAME']
        base_dir = Path(fnp).parent
        job_metadata = job.describe_job()
        target_dir = base_dir / job_metadata['title']
        job.get_results().download_files(target=target_dir)

        with open(target_dir / f'job_{job.job_id}.json', 'w') as f:
            json.dump(job_metadata, f, ensure_ascii=False)

        #copy geometry to result directory
        try:
            copy2(fnp,target_dir)
        except:
            print(f'COPY ERROR {fnp} {target_dir}')

def extract_samples(provider="sentinelhub", status_file="sampling_lucas.csv",parallel_jobs=1, working_dir=Path(".")):
    output_file = working_dir / status_file
    dataframe=None
    if not output_file.exists() or not output_file.is_file():
        dataframe = split_lucas(working_dir,["10","11","12"])

    manager = CustomJobManager()
    c = openeo_platform
    if provider.upper() == "CREODIAS":
        c = creo
    manager.add_backend(provider, connection=c, parallel_jobs=parallel_jobs)

    manager.run_jobs(
        df=dataframe,
        start_job=run,
        output_file=output_file
    )

def produce_on_sentinelhub():
    extract_samples(provider="terrascope",  parallel_jobs=1, status_file="sampling_lucas.csv")

def produce_on_creodias():
    extract_samples(provider="creodias", parallel_jobs=1, status_file="sampling_lucas_creodias.csv", working_dir=Path('/home/driesj/python/openeo-classification/lucas_creo'))

if __name__ == '__main__':
  fire.Fire(produce_on_creodias())