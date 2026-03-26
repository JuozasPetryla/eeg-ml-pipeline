import os
import time

from sqlalchemy import select

from ml.db import get_db
from ml.job_repository import get_analysis_type_by_job_id
from ml.models import AnalysisJob
from ml.night_pipeline import process_night_analysis_job
from ml.statistics import process_analysis_job as process_day_analysis_job

POLL_INTERVAL_SECONDS = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "2"))


def get_next_queued_job_id() -> int | None:
    with get_db() as db:
        stmt = (
            select(AnalysisJob.id)
            .where(AnalysisJob.status == "queued")
            .order_by(AnalysisJob.queued_at.asc(), AnalysisJob.id.asc())
            .limit(1)
        )
        return db.execute(stmt).scalar_one_or_none()


def main() -> None:
    print("EEG ML worker started")

    while True:
        job_id = get_next_queued_job_id()

        if job_id is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        print(f"Processing analysis_job_id={job_id}")

        try:
            with get_db() as db:
                analysis_type = get_analysis_type_by_job_id(db, job_id)

            if analysis_type == "night":
                process_night_analysis_job(job_id)
            else:
                process_day_analysis_job(job_id)

            print(f"Completed analysis_job_id={job_id}")
        except Exception as e:
            print(f"Failed analysis_job_id={job_id}: {e}")


if __name__ == "__main__":
    main()
