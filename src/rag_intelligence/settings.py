from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True)
class AppSettings:
    # TimescaleDB / pgvectorscale
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_database: str
    pg_table_name: str
    pg_embed_dim: int

    # Ollama
    ollama_base_url: str

    # MinIO
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool

    # API keys (empty string means unavailable)
    openai_api_key: str
    anthropic_api_key: str
    voyage_api_key: str

    # Provider defaults
    default_llm: str
    default_embed_model: str

    # API server
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]

    # Logging
    log_level: str
    log_json: bool | None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AppSettings:
        raw = dict(os.environ if env is None else env)

        return cls(
            pg_host=raw.get("PG_HOST", "localhost"),
            pg_port=_parse_int(raw, "PG_PORT", 54330),
            pg_user=raw.get("PG_USER", "raguser"),
            pg_password=raw.get("PG_PASSWORD", "ragpassword"),
            pg_database=raw.get("PG_DATABASE", "ragdb"),
            pg_table_name=raw.get("PG_TABLE_NAME", "rag_embeddings"),
            pg_embed_dim=_parse_int(raw, "PG_EMBED_DIM", 768),
            ollama_base_url=raw.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            minio_endpoint=raw.get("MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=raw.get("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=raw.get("MINIO_SECRET_KEY", "minioadmin"),
            minio_bucket=raw.get("MINIO_BUCKET", "bronze"),
            minio_secure=_parse_bool(raw.get("MINIO_SECURE", "false")),
            openai_api_key=raw.get("OPENAI_API_KEY", "").strip(),
            anthropic_api_key=raw.get("ANTHROPIC_API_KEY", "").strip(),
            voyage_api_key=raw.get("VOYAGE_API_KEY", "").strip(),
            default_llm=raw.get("DEFAULT_LLM", "ollama/qwen2.5"),
            default_embed_model=raw.get("DEFAULT_EMBED_MODEL", "ollama/nomic-embed"),
            api_host=raw.get("API_HOST", "0.0.0.0"),
            api_port=_parse_int(raw, "API_PORT", 8000),
            cors_origins=_parse_cors_origins(raw.get("CORS_ORIGINS", "*")),
            log_level=raw.get("LOG_LEVEL", "INFO").strip().upper(),
            log_json=_parse_optional_bool(raw.get("LOG_JSON", "")),
        )


def _parse_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as err:
        raise SettingsError(f"Invalid integer for {key}: {raw}") from err


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise SettingsError(f"Invalid boolean value: {value}")


def _parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"Invalid boolean value: {value}")


def _parse_cors_origins(value: str) -> tuple[str, ...]:
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())
