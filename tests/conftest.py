from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FakeObject:
    object_name: str


class FakeMinio:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        *,
        initial_objects: dict[str, dict[str, bytes]] | None = None,
        existing_buckets: set[str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.objects = initial_objects or {}
        self.buckets = set(existing_buckets or set())

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.add(bucket_name)
        self.objects.setdefault(bucket_name, {})

    def list_objects(self, bucket_name: str, prefix: str, recursive: bool = True):
        del recursive
        for object_name in sorted(self.objects.get(bucket_name, {})):
            if object_name.startswith(prefix):
                yield FakeObject(object_name=object_name)

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        data = self.objects[bucket_name][object_name]
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        content_type: str,
    ) -> None:
        del content_type
        if bucket_name not in self.buckets:
            raise ValueError(f"Bucket not found: {bucket_name}")
        self.objects.setdefault(bucket_name, {})[object_name] = Path(file_path).read_bytes()
