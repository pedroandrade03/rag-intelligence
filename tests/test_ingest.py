from __future__ import annotations

from pathlib import Path
import zipfile

from rag_intelligence.config import Settings
from rag_intelligence.ingest import iter_dataset_assets, run_import


class FakeKaggleApi:
    def authenticate(self) -> None:
        return None

    def dataset_download_files(self, dataset: str, path: str, unzip: bool, quiet: bool) -> None:
        archive_path = Path(path) / f"{dataset.rsplit('/', maxsplit=1)[-1]}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("mm_master_demos.csv", "tick,damage\n1,27\n")
            archive.writestr("maps/de_inferno.png", "png-bytes")
            archive.writestr("notes.txt", "ignore me")


class FakeMinio:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.bucket_created = False
        self.uploads: list[tuple[str, str, str, str]] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        return self.bucket_created

    def make_bucket(self, bucket_name: str) -> None:
        self.bucket_created = True

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str,
    ) -> None:
        self.uploads.append((bucket_name, object_name, file_path, content_type))


def build_settings() -> Settings:
    return Settings(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="bronze",
        minio_secure=False,
        dataset_slug="skihikingkevin/csgo-matchmaking-damage",
        dataset_prefix="csgo-matchmaking-damage",
        run_id="20260306T010203Z",
    )


def test_iter_dataset_assets_filters_only_csv_and_png(tmp_path) -> None:
    (tmp_path / "mm_master_demos.csv").write_text("tick,damage\n")
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / "de_inferno.png").write_text("png")
    (tmp_path / "notes.txt").write_text("ignore")

    files = [path.relative_to(tmp_path).as_posix() for path in iter_dataset_assets(tmp_path)]

    assert files == ["maps/de_inferno.png", "mm_master_demos.csv"]


def test_run_import_uploads_raw_archive_and_filtered_assets() -> None:
    fake_minio = FakeMinio("localhost:9000", "minioadmin", "minioadmin", False)

    uploaded = run_import(
        build_settings(),
        minio_factory=lambda **kwargs: fake_minio,
        kaggle_api_factory=FakeKaggleApi,
    )

    assert fake_minio.bucket_created is True
    assert uploaded == [
        "csgo-matchmaking-damage/20260306T010203Z/raw/csgo-matchmaking-damage.zip",
        "csgo-matchmaking-damage/20260306T010203Z/extracted/maps/de_inferno.png",
        "csgo-matchmaking-damage/20260306T010203Z/extracted/mm_master_demos.csv",
    ]
