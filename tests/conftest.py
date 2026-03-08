from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


class FakeCursor:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...]]] = []
        self._rows: list[tuple[Any, ...]] = []
        self.description: list[tuple[str, ...]] | None = None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.queries.append((query, params))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def close(self) -> None:
        pass

    def set_result(
        self,
        rows: list[tuple[Any, ...]],
        columns: list[str] | None = None,
    ) -> None:
        self._rows = rows
        if columns:
            self.description = [(col,) for col in columns]


class FakeConnection:
    def __init__(self, *, cursor: FakeCursor | None = None) -> None:
        self._cursor = cursor or FakeCursor()
        self.committed = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True
