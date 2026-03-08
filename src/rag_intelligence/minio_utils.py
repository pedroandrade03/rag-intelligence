from __future__ import annotations

import codecs
import io
import logging
from collections.abc import Iterator
from typing import Any

from minio import Minio
from minio.error import S3Error

LOGGER = logging.getLogger(__name__)


def clean_cell(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def ensure_bucket(client: Minio, bucket_name: str) -> None:
    if client.bucket_exists(bucket_name):
        return
    LOGGER.info("Creating bucket %s", bucket_name)
    client.make_bucket(bucket_name)


def stream_text_lines(response: Any, *, chunk_size: int = 64 * 1024) -> Iterator[str]:
    """Decode a streaming MinIO response into text lines.

    Uses rfind + single split per chunk to avoid O(n²) buffer growth.
    """
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    buf = io.StringIO()
    for chunk in response.stream(chunk_size):
        if not chunk:
            continue
        buf.write(decoder.decode(chunk))
        content = buf.getvalue()
        last_nl = content.rfind("\n")
        if last_nl == -1:
            continue
        yield from content[: last_nl + 1].splitlines(keepends=True)
        remainder = content[last_nl + 1 :]
        buf = io.StringIO()
        buf.write(remainder)
    buf.write(decoder.decode(b"", final=True))
    remainder = buf.getvalue()
    if remainder:
        yield remainder


def load_minio_object(client: Minio, bucket: str, object_key: str, *, label: str) -> Any:
    """Fetch a MinIO object, raising ``FileNotFoundError`` for missing keys."""
    try:
        return client.get_object(bucket, object_key)
    except KeyError as exc:
        raise FileNotFoundError(f"{label} not found: {bucket}/{object_key}") from exc
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise FileNotFoundError(f"{label} not found: {bucket}/{object_key}") from exc
        raise
