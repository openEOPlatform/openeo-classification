import logging
import os
import time
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd
import requests
from openeo.util import deep_get

from openeo_classification.connection import connection, terrascope_dev


_log = logging.getLogger(__name__)


def run_jobs(
        df: pd.DataFrame,
        start_job: Callable,
        outputFile: Path,
        connection_provider: Callable,
        parallel_jobs=2,
):
    """
    Runs jobs, specified in a dataframe, and tracks parameters.

    @param df: Job dataframe
    @param start_job: A callback which will be invoked with the row of the dataframe for which a job should be started.
    @param outputFile: A file on disk to track job statuses.
    @return:
    """

    # TODO: original dataframe is completely discarded if `outputFile` exists, isn't that weird?
    #       E.g. New code changes will not be picked up as long as an old/outdated CSV exist.
    if outputFile.is_file():
        df = pd.read_csv(outputFile)
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    else:
        df["status"] = "not_started"
        df["id"] = "None"
        df["cpu"] = 0
        df["memory"] = 0
        df["duration"] = 0
        df.to_csv(outputFile,index=False)

    # TODO: this will never exit if there are failed/skipped jobs
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
                if job is not None:
                    next_job["status"] = job.status()
                    next_job["id"] = job.job_id
                else:
                    next_job["status"] = "skipped"
                print(next_job)
                df.loc[next_job.name] = next_job

                df.to_csv(outputFile, index=False)
            else:
                time.sleep(60)

        except requests.exceptions.ConnectionError as e:
            _log.warning(f"Skipping connection error: {e}")


def running_jobs(status_df):
    return status_df.loc[(status_df["status"] == "queued") | (status_df["status"] == "running") | (status_df["status"] == "created")].index


def update_statuses(status_df, connection_provider=connection):
    con = connection_provider()
    for i in running_jobs(status_df):
        job_id = status_df.loc[i, 'id']
        job = con.job(job_id).describe_job()
        usage = job.get('usage', {})
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