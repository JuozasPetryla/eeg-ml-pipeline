from sqlalchemy import select
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
