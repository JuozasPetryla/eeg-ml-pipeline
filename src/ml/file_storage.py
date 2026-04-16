import os
import threading
import time
from pathlib import Path

from minio import Minio
from minio.error import S3Error

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio12345")
S3_BUCKET = os.getenv("S3_BUCKET", "eeg")
MINIO_INIT_MAX_ATTEMPTS = int(os.getenv("MINIO_INIT_MAX_ATTEMPTS", "10"))
MINIO_INIT_RETRY_SECONDS = float(os.getenv("MINIO_INIT_RETRY_SECONDS", "1"))
MINIO_OPERATION_MAX_ATTEMPTS = int(os.getenv("MINIO_OPERATION_MAX_ATTEMPTS", "5"))
MINIO_OPERATION_RETRY_SECONDS = float(os.getenv("MINIO_OPERATION_RETRY_SECONDS", "1"))

endpoint = S3_ENDPOINT.replace("http://", "").replace("https://", "")
secure = S3_ENDPOINT.startswith("https://")

minio_client = Minio(
    endpoint,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    secure=secure,
)

_bucket_init_lock = threading.Lock()
_bucket_ready = False


def ensure_bucket_exists() -> None:
    global _bucket_ready

    if _bucket_ready:
        return

    with _bucket_init_lock:
        if _bucket_ready:
            return

        last_error: Exception | None = None
        for attempt in range(1, MINIO_INIT_MAX_ATTEMPTS + 1):
            try:
                if not minio_client.bucket_exists(S3_BUCKET):
                    minio_client.make_bucket(S3_BUCKET)
                _bucket_ready = True
                return
            except S3Error as e:
                if e.code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                    _bucket_ready = True
                    return
                last_error = e
            except Exception as e:
                last_error = e

            if attempt < MINIO_INIT_MAX_ATTEMPTS:
                time.sleep(MINIO_INIT_RETRY_SECONDS)

        raise RuntimeError(f"MinIO bucket init failed: {last_error}") from last_error


def _run_with_minio_retries(operation_name: str, operation):
    last_error: Exception | None = None

    for attempt in range(1, MINIO_OPERATION_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt < MINIO_OPERATION_MAX_ATTEMPTS:
                time.sleep(MINIO_OPERATION_RETRY_SECONDS)

    raise RuntimeError(f"MinIO {operation_name} failed: {last_error}") from last_error


def download_file(object_name: str, destination_path: str | Path) -> Path:
    destination_path = Path(destination_path)

    parent = destination_path.parent
    if parent.exists() and not parent.is_dir():
        raise RuntimeError(f"Expected directory but found file: {parent}")

    parent.mkdir(parents=True, exist_ok=True)

    return _run_with_minio_retries(
        f"download for '{object_name}'",
        lambda: (
            minio_client.fget_object(
            bucket_name=S3_BUCKET,
            object_name=object_name,
            file_path=str(destination_path),
            ),
            destination_path.resolve(),
        )[1],
    )


def upload_file(source_path: str | Path, object_name: str, content_type: str) -> str:
    source_path = Path(source_path)

    if not source_path.exists():
        raise RuntimeError(f"File does not exist: {source_path}")

    ensure_bucket_exists()

    return _run_with_minio_retries(
        f"upload for '{source_path}'",
        lambda: (
            minio_client.fput_object(
                bucket_name=S3_BUCKET,
                object_name=object_name,
                file_path=str(source_path),
                content_type=content_type,
            ),
            object_name,
        )[1],
    )
