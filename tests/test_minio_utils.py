from __future__ import annotations

from conftest import FakeMinio
from rag_intelligence.minio_utils import clean_cell, ensure_bucket


class TestCleanCell:
    def test_none_returns_none(self):
        assert clean_cell(None) is None

    def test_strips_whitespace(self):
        assert clean_cell("  hello  ") == "hello"

    def test_empty_string_returns_none(self):
        assert clean_cell("") is None

    def test_whitespace_only_returns_none(self):
        assert clean_cell("   ") is None

    def test_preserves_inner_whitespace(self):
        assert clean_cell(" a b ") == "a b"

    def test_already_clean(self):
        assert clean_cell("value") == "value"


class TestEnsureBucket:
    def test_creates_bucket_when_missing(self):
        client = FakeMinio("localhost:9000", "key", "secret", False)
        ensure_bucket(client, "bronze")  # type: ignore[arg-type]
        assert "bronze" in client.buckets

    def test_skips_when_bucket_exists(self):
        client = FakeMinio("localhost:9000", "key", "secret", False, existing_buckets={"bronze"})
        ensure_bucket(client, "bronze")  # type: ignore[arg-type]
        assert "bronze" in client.buckets
