from __future__ import annotations

import pytest

from conftest import FakeMinio, FakeObjectResponse
from rag_intelligence.minio_utils import (
    clean_cell,
    ensure_bucket,
    load_minio_object,
    stream_text_lines,
)


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


class TestStreamTextLines:
    def test_single_chunk_multiple_lines(self):
        response = FakeObjectResponse(b"line1\nline2\nline3\n")
        lines = list(stream_text_lines(response))
        assert lines == ["line1\n", "line2\n", "line3\n"]

    def test_no_trailing_newline(self):
        response = FakeObjectResponse(b"line1\nline2")
        lines = list(stream_text_lines(response))
        assert lines == ["line1\n", "line2"]

    def test_empty_response(self):
        response = FakeObjectResponse(b"")
        lines = list(stream_text_lines(response))
        assert lines == []

    def test_single_line_no_newline(self):
        response = FakeObjectResponse(b"hello")
        lines = list(stream_text_lines(response))
        assert lines == ["hello"]

    def test_utf8_bom_stripped(self):
        response = FakeObjectResponse(b"\xef\xbb\xbfheader\ndata\n")
        lines = list(stream_text_lines(response))
        assert lines == ["header\n", "data\n"]

    def test_small_chunk_size_splits_across_chunks(self):
        response = FakeObjectResponse(b"abcdef\nghijkl\n")
        lines = list(stream_text_lines(response, chunk_size=4))
        assert lines == ["abcdef\n", "ghijkl\n"]

    def test_blank_lines_preserved(self):
        response = FakeObjectResponse(b"a\n\nb\n")
        lines = list(stream_text_lines(response))
        assert lines == ["a\n", "\n", "b\n"]


class TestLoadMinioObject:
    def test_returns_response_for_existing_object(self):
        client = FakeMinio(
            "localhost:9000",
            "key",
            "secret",
            False,
            initial_objects={"bucket": {"key.txt": b"data"}},
            existing_buckets={"bucket"},
        )
        resp = load_minio_object(client, "bucket", "key.txt", label="test")  # type: ignore[arg-type]
        assert resp.read() == b"data"

    def test_raises_file_not_found_for_missing_key(self):
        client = FakeMinio(
            "localhost:9000",
            "key",
            "secret",
            False,
            initial_objects={"bucket": {}},
            existing_buckets={"bucket"},
        )
        with pytest.raises(FileNotFoundError, match="test not found"):
            load_minio_object(client, "bucket", "missing.txt", label="test")  # type: ignore[arg-type]

    def test_raises_file_not_found_for_missing_bucket(self):
        client = FakeMinio("localhost:9000", "key", "secret", False)
        with pytest.raises(FileNotFoundError, match="test not found"):
            load_minio_object(client, "no-bucket", "key.txt", label="test")  # type: ignore[arg-type]
