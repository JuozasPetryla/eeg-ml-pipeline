from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ml.models import AnalysisJob, EEGFile


def get_object_storage_key_by_job_id(db: Session, analysis_job_id: int) -> str:
    stmt = (
        select(EEGFile.object_storage_key)
        .join(AnalysisJob, AnalysisJob.eeg_file_id == EEGFile.id)
        .where(AnalysisJob.id == analysis_job_id)
    )

    object_storage_key = db.execute(stmt).scalar_one_or_none()

    if object_storage_key is None:
        raise ValueError(f"No EEG file found for analysis_job_id={analysis_job_id}")

    return object_storage_key


def get_analysis_type_by_job_id(db: Session, analysis_job_id: int) -> str:
    stmt = (
        select(AnalysisJob.analysis_type)
        .where(AnalysisJob.id == analysis_job_id)
    )

    analysis_type = db.execute(stmt).scalar_one_or_none()

    if analysis_type is None:
        raise ValueError(f"No analysis job found for analysis_job_id={analysis_job_id}")

    return analysis_type


def claim_next_queued_job(
    db: Session,
) -> tuple[int, str] | None:
    queued_job_subquery = (
        select(AnalysisJob.id)
        .where(AnalysisJob.status == "queued")
        .order_by(AnalysisJob.queued_at.asc(), AnalysisJob.id.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )

    row = db.execute(
        update(AnalysisJob)
        .where(AnalysisJob.id == queued_job_subquery)
        .values(
            status="processing",
            error_message=None,
            started_at=datetime.now(timezone.utc),
            finished_at=None,
        )
        .returning(AnalysisJob.id, AnalysisJob.analysis_type)
    ).first()
    db.commit()

    if row is None:
        return None

    return row.id, row.analysis_type


def requeue_stale_processing_jobs(
    db: Session,
    stale_after_seconds: int,
) -> int:
    if stale_after_seconds <= 0:
        return 0

    stale_before = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    result = db.execute(
        update(AnalysisJob)
        .where(AnalysisJob.status == "processing")
        .where(AnalysisJob.started_at.is_not(None))
        .where(AnalysisJob.started_at < stale_before)
        .values(
            status="queued",
            error_message="Re-queued after stale processing timeout",
            started_at=None,
            finished_at=None,
        )
    )
    db.commit()
    return result.rowcount or 0
