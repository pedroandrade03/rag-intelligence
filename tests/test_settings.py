from __future__ import annotations

import pytest

from rag_intelligence.settings import AppSettings, SettingsError


def base_env() -> dict[str, str]:
    return {
        "PG_HOST": "localhost",
        "PG_PORT": "5432",
        "PG_USER": "testuser",
        "PG_PASSWORD": "testpass",
        "PG_DATABASE": "testdb",
        "PG_TABLE_NAME": "test_vectors",
        "PG_EMBED_DIM": "768",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "minioadmin",
        "MINIO_SECRET_KEY": "minioadmin",
        "MINIO_BUCKET": "bronze",
        "MINIO_SECURE": "false",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "VOYAGE_API_KEY": "",
        "DEFAULT_LLM": "ollama/qwen2.5",
        "DEFAULT_EMBED_MODEL": "ollama/nomic-embed",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
    }


def test_from_env_with_all_values():
    env = base_env()
    settings = AppSettings.from_env(env)

    assert settings.pg_host == "localhost"
    assert settings.pg_port == 5432
    assert settings.pg_user == "testuser"
    assert settings.pg_database == "testdb"
    assert settings.pg_embed_dim == 768
    assert settings.minio_secure is False
    assert settings.default_llm == "ollama/qwen2.5"


def test_from_env_uses_defaults_when_keys_absent():
    settings = AppSettings.from_env({})

    assert settings.pg_host == "localhost"
    assert settings.pg_port == 54330
    assert settings.pg_user == "raguser"
    assert settings.pg_database == "ragdb"
    assert settings.pg_embed_dim == 768
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.default_llm == "ollama/qwen2.5"
    assert settings.api_port == 8000


def test_invalid_integer_raises_settings_error():
    env = base_env()
    env["PG_PORT"] = "not_a_number"

    with pytest.raises(SettingsError, match="Invalid integer"):
        AppSettings.from_env(env)


def test_invalid_bool_raises_settings_error():
    env = base_env()
    env["MINIO_SECURE"] = "maybe"

    with pytest.raises(SettingsError, match="Invalid boolean"):
        AppSettings.from_env(env)


def test_api_keys_are_stripped():
    env = base_env()
    env["OPENAI_API_KEY"] = "  sk-test-key  "
    settings = AppSettings.from_env(env)

    assert settings.openai_api_key == "sk-test-key"


def test_settings_is_frozen():
    settings = AppSettings.from_env(base_env())

    with pytest.raises(AttributeError):
        settings.pg_host = "other"  # type: ignore[misc]
