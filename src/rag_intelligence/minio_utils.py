from __future__ import annotations

import logging

from minio import Minio

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
