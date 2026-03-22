from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ml.models import AnalysisJob, AnalysisResult


def store_analysis_result(
    db: Session,
    analysis_job_id: int,
    result_data: dict[str, Any],
    model_version: str | None = None,
) -> None:
    result_stmt = insert(AnalysisResult).values(
        analysis_job_id=analysis_job_id,
        result_json=result_data,
    )

    result_stmt = result_stmt.on_conflict_do_update(
        index_elements=[AnalysisResult.analysis_job_id],
        set_={
            "result_json": result_data,
        },
    )

    db.execute(result_stmt)

    job_update_values: dict[str, Any] = {
        "status": "completed",
        "error_message": None,
        "finished_at": datetime.now(timezone.utc),
    }

    if model_version is not None:
        job_update_values["model_version"] = model_version

    db.execute(
        update(AnalysisJob)
        .where(AnalysisJob.id == analysis_job_id)
        .values(**job_update_values)
    )

    db.commit()
