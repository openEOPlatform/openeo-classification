import openeo
import pandas as pd
from openeo.rest.job import RESTJob

from openeo_classification.job_management import MultiBackendJobManager


class TestMultiBackendJobManager:

    def test_basic(self, tmp_path, requests_mock):
        requests_mock.get("http://foo.test/", json={"api_version": "1.1.0"})
        requests_mock.get("http://bar.test/", json={"api_version": "1.1.0"})

        def mock_job_status(job_id, queued=1, running=2):
            """Mock job status polling sequence"""
            response_list = sum([
                    [{"json": {"id": job_id, "status": "queued"}}] * queued,
                    [{"json": {"id": job_id, "status": "running"}}] * running,
                    [{"json": {"id": job_id, "status": "finished"}}],
            ], [])
            for backend in ["http://foo.test", "http://bar.test"]:
                requests_mock.get(f"{backend}/jobs/{job_id}", response_list)

        mock_job_status("job-2018", queued=1, running=2)
        mock_job_status("job-2019", queued=2, running=3)
        mock_job_status("job-2020", queued=3, running=4)
        mock_job_status("job-2021", queued=3, running=5)
        mock_job_status("job-2022", queued=5, running=6)

        manager = MultiBackendJobManager(poll_sleep=.2)
        manager.add_backend("foo", connection=openeo.connect("http://foo.test"))
        manager.add_backend("bar", connection=openeo.connect("http://bar.test"))

        df = pd.DataFrame({
            "year": [2018, 2019, 2020, 2021, 2022],
        })
        output_file = tmp_path / "jobs.csv"

        def start_job(row, connection, **kwargs):
            year = row["year"]
            return RESTJob(job_id=f"job-{year}", connection=connection)

        manager.run_jobs(df=df, start_job=start_job, output_file=output_file)

        result = pd.read_csv(output_file)
        assert len(result) == 5
        assert set(result.status) == {"finished"}
        assert set(result.backend_name) == {"foo", "bar"}
