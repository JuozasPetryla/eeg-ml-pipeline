import os
import time

import psycopg
from sqlalchemy.exc import OperationalError, ProgrammingError

from ml.db import get_db
from ml.job_repository import claim_next_queued_job, requeue_stale_processing_jobs
from ml.night_pipeline import process_night_analysis_job
from ml.statistics import process_analysis_job as process_day_analysis_job

POLL_INTERVAL_SECONDS = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "2"))
REQUEUE_STALE_PROCESSING_SECONDS = int(
    os.getenv("REQUEUE_STALE_PROCESSING_SECONDS", "1800")
)

def claim_next_job() -> tuple[int, str] | None:
    with get_db() as db:
        if REQUEUE_STALE_PROCESSING_SECONDS > 0:
            requeued_count = requeue_stale_processing_jobs(
                db,
                REQUEUE_STALE_PROCESSING_SECONDS,
            )
            if requeued_count:
                print(f"Re-queued {requeued_count} stale processing job(s)")
        return claim_next_queued_job(db)


def main() -> None:
    print("EEG ML worker started")

    while True:
        try:
            claimed_job = claim_next_job()
        except (OperationalError, ProgrammingError) as exc:
            # Docker can start this worker before Postgres schema initialization finishes.
            if isinstance(getattr(exc, "orig", None), psycopg.errors.UndefinedTable):
                print("analysis_jobs is not available yet; waiting for migrations to finish")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            print(f"Database is not ready yet: {exc}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if claimed_job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        job_id, analysis_type = claimed_job
        print(f"Processing analysis_job_id={job_id}")

        try:
            if analysis_type == "night":
                process_night_analysis_job(job_id)
            else:
                process_day_analysis_job(job_id)

            print(f"Completed analysis_job_id={job_id}")
        except Exception as e:
            print(f"Failed analysis_job_id={job_id}: {e}")


if __name__ == "__main__":
    main()
