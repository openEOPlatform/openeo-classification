import time
from connection import connection
from openeo.util import deep_get
import os
import pandas as pd

def running_jobs(status_df):
    return status_df.loc[(status_df["status"] == "queued") | (status_df["status"] == "running")].index

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