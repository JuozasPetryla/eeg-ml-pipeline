import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio12345")
S3_BUCKET = os.getenv("S3_BUCKET", "eeg")

endpoint = S3_ENDPOINT.replace("http://", "").replace("https://", "")
secure = S3_ENDPOINT.startswith("https://")

minio_client = Minio(
    endpoint,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    secure=secure,
)


def ensure_bucket_exists() -> None:
    try:
        if not minio_client.bucket_exists(S3_BUCKET):
            minio_client.make_bucket(S3_BUCKET)
    except S3Error as e:
        raise RuntimeError(f"MinIO bucket init failed: {e}") from e


def download_file(object_name: str, destination_path: str | Path) -> Path:
    destination_path = Path(destination_path)

    parent = destination_path.parent
    if parent.exists() and not parent.is_dir():
        raise RuntimeError(f"Expected directory but found file: {parent}")

    parent.mkdir(parents=True, exist_ok=True)

    try:
        minio_client.fget_object(
            bucket_name=S3_BUCKET,
            object_name=object_name,
            file_path=str(destination_path),
        )
        return destination_path.resolve()
    except S3Error as e:
        raise RuntimeError(
            f"Failed to download '{object_name}' from bucket '{S3_BUCKET}': {e}"
        ) from e


def upload_file(source_path: str | Path, object_name: str, content_type: str) -> str:
    source_path = Path(source_path)

    if not source_path.exists():
        raise RuntimeError(f"File does not exist: {source_path}")

    try:
        minio_client.fput_object(
            bucket_name=S3_BUCKET,
            object_name=object_name,
            file_path=str(source_path),
            content_type=content_type,
        )
        return object_name
    except S3Error as e:
        raise RuntimeError(
            f"Failed to upload '{source_path}' to bucket '{S3_BUCKET}': {e}"
        ) from e
