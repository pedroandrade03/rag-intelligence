from __future__ import annotations

import json
import logging
import mimetypes
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from minio import Minio

from rag_intelligence.config import Settings
from rag_intelligence.minio_utils import ensure_bucket

LOGGER = logging.getLogger(__name__)
EXTRACTED_SUFFIXES = {".csv", ".png"}


@dataclass(frozen=True)
class UploadItem:
    source_path: Path
    object_key: str


def infer_extracted_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix != ".csv":
        return "assets"

    name = path.name.lower()
    if "meta" in name:
        return "round_meta"
    if "kill" in name:
        return "kills"
    if "dmg" in name or "damage" in name:
        return "damage"
    if "grenade" in name or "nade" in name:
        return "grenades"
    return "other"


def build_object_key(prefix: str, run_id: str, relative_path: str, *, section: str) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    return f"{prefix}/{run_id}/{section}/{normalized}"


def iter_dataset_assets(extracted_dir: Path) -> Iterable[Path]:
    for path in sorted(extracted_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTRACTED_SUFFIXES:
            yield path


def resolve_downloaded_archive(download_dir: Path, dataset_slug: str) -> Path:
    expected_name = f"{dataset_slug.rsplit('/', maxsplit=1)[-1]}.zip"
    expected_path = download_dir / expected_name
    if expected_path.is_file():
        return expected_path

    zip_files = sorted(download_dir.glob("*.zip"))
    if len(zip_files) == 1:
        return zip_files[0]

    raise FileNotFoundError(f"Unable to locate the downloaded dataset archive in {download_dir}.")


def download_dataset_archive(dataset_slug: str, destination: Path, api_factory=None) -> Path:
    if api_factory is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api_factory = KaggleApi

    api = api_factory()
    api.authenticate()
    LOGGER.info("Downloading Kaggle dataset %s", dataset_slug)
    api.dataset_download_files(dataset_slug, path=str(destination), unzip=False, quiet=False)
    archive_path = resolve_downloaded_archive(destination, dataset_slug)
    LOGGER.info("Downloaded archive to %s", archive_path)
    return archive_path


def extract_archive(archive_path: Path, destination: Path) -> None:
    LOGGER.info("Extracting archive %s", archive_path.name)
    with zipfile.ZipFile(archive_path) as zip_file:
        zip_file.extractall(destination)


def upload_file(client: Minio, bucket_name: str, item: UploadItem) -> None:
    content_type = mimetypes.guess_type(item.source_path.name)[0] or "application/octet-stream"
    LOGGER.info("Uploading %s to %s", item.source_path.name, item.object_key)
    client.fput_object(
        bucket_name=bucket_name,
        object_name=item.object_key,
        file_path=str(item.source_path),
        content_type=content_type,
    )


def build_upload_manifest(
    settings: Settings, archive_path: Path, extracted_dir: Path
) -> list[UploadItem]:
    manifest = [
        UploadItem(
            source_path=archive_path,
            object_key=build_object_key(
                settings.dataset_prefix,
                settings.run_id,
                archive_path.name,
                section="raw",
            ),
        )
    ]

    manifest_payload: list[dict[str, object]] = []

    for asset_path in iter_dataset_assets(extracted_dir):
        relative_path = asset_path.relative_to(extracted_dir).as_posix()
        file_type = infer_extracted_file_type(asset_path)
        typed_relative_path = f"{file_type}/{relative_path}"
        object_key = build_object_key(
            settings.dataset_prefix,
            settings.run_id,
            typed_relative_path,
            section="extracted",
        )
        manifest.append(
            UploadItem(
                source_path=asset_path,
                object_key=object_key,
            )
        )
        manifest_payload.append(
            {
                "relative_path": relative_path,
                "typed_relative_path": typed_relative_path,
                "file_type": file_type,
                "object_key": object_key,
                "size_bytes": asset_path.stat().st_size,
            }
        )

    generated_at = datetime.now(UTC).isoformat()
    manifest_file = extracted_dir / "_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "dataset_prefix": settings.dataset_prefix,
                "run_id": settings.run_id,
                "files": manifest_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest.append(
        UploadItem(
            source_path=manifest_file,
            object_key=build_object_key(
                settings.dataset_prefix,
                settings.run_id,
                "manifest.json",
                section="extracted",
            ),
        )
    )

    return manifest


def run_import(settings: Settings, *, minio_factory=Minio, kaggle_api_factory=None) -> list[str]:
    client = minio_factory(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    ensure_bucket(client, settings.minio_bucket)

    with tempfile.TemporaryDirectory(prefix="bronze-import-") as temp_dir:
        working_dir = Path(temp_dir)
        archive_path = download_dataset_archive(
            settings.dataset_slug,
            working_dir,
            api_factory=kaggle_api_factory,
        )

        extracted_dir = working_dir / "extracted"
        extracted_dir.mkdir()
        extract_archive(archive_path, extracted_dir)

        extracted_assets = list(iter_dataset_assets(extracted_dir))
        extracted_count = len(extracted_assets)
        if extracted_count <= 0:
            raise FileNotFoundError(
                "No .csv or .png files were found after extracting the Kaggle dataset."
            )

        manifest = build_upload_manifest(settings, archive_path, extracted_dir)

        for item in manifest:
            upload_file(client, settings.minio_bucket, item)

    return [item.object_key for item in manifest]
