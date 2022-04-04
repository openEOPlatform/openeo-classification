import time
from pathlib import Path

import requests

from openeo_classification.connection import connection,terrascope_dev
from openeo.util import deep_get
import os
import pandas as pd
import geopandas as gpd

def run_jobs(df:pd.DataFrame,start_job, outputFile:Path, parallel_jobs=2,connection_provider=terrascope_dev):
    """
    Runs jobs, specified in a dataframe, and tracks parameters.

    @param df: Job dataframe
    @param start_job: A callback which will be invoked with the row of the dataframe for which a job should be started.
    @param outputFile: A file on disk to track job statuses.
    @return:
    """

    df["status"]="not_started"
    df["id"]="None"
    df["cpu"]=0
    df["memory"]=0
    df["duration"]=0

    if outputFile.is_file():
        df = pd.read_csv(outputFile)
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    else:
        df.to_csv(outputFile,index=False)


    while len(df[(df["status"] != "finished")])>0:
        try:
            jobs_to_run = df[df.status == "not_started"]
            df = update_statuses(df, connection_provider)
            df.to_csv(outputFile, index=False)
            if jobs_to_run.empty:
                time.sleep(60)
                continue

            if len(df[(df["status"] == "running") | (df["status"] == "queued") | (df["status"] == "created") ]) < parallel_jobs:

                next_job = jobs_to_run.iloc[0]
                job = start_job(next_job)
                next_job["status"] = job.status()
                next_job["id"] = job.job_id
                print(next_job)
                df.loc[next_job.name] = next_job

                df.to_csv(outputFile, index=False)
            else:
                time.sleep(60)

        except requests.exceptions.ConnectionError as e:
            print(e)



def running_jobs(status_df):
    return status_df.loc[(status_df["status"] == "queued") | (status_df["status"] == "running") | (status_df["status"] == "created")].index

def update_statuses(status_df, connection_provider=connection):
    con = connection_provider()
    default_usage = {
        'cpu':{'value':0, 'unit':''},
        'memory':{'value':0, 'unit':''}
    }
    for i in running_jobs(status_df):
        job_id = status_df.loc[i, 'id']
        job = con.job(job_id).describe_job()
        usage = job.get('usage',default_usage)
        status_df.loc[i, "status"] = job["status"]
        status_df.loc[i, "cpu"] = f"{deep_get(usage,'cpu','value',default=0)} {deep_get(usage,'cpu','unit',default='')}"
        status_df.loc[i, "memory"] = f"{deep_get(usage,'memory','value',default=0)} {deep_get(usage,'memory','unit',default='')}"
        status_df.loc[i, "duration"] = deep_get(usage,'duration','value',default=0)
        print(time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()) + "\tCurrent status of job " + job_id
              + " is : " + job["status"])
    return status_df


# TODO: invalid path reference (https://github.com/openEOPlatform/openeo-classification/issues/2)
def create_or_load_job_statistics(path = "resources/training_data/job_statistics.csv"):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame({
            "fp": [],
            "status": [],
            "id": [],
            "cpu": [],
            "memory": [],
            "duration": []
        })
        df.to_csv(path,index=False)
    return df