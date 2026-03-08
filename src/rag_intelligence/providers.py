from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM

if TYPE_CHECKING:
    from rag_intelligence.settings import AppSettings

LOGGER = logging.getLogger(__name__)

OLLAMA_LLM_FALLBACK = "ollama/qwen2.5"
OLLAMA_EMBED_FALLBACK = "ollama/nomic-embed-text"


class ProviderRegistry:
    """Lazy-initializing registry of LLM and embedding providers with Ollama fallback."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._llm_cache: dict[str, LLM] = {}
        self._embed_cache: dict[str, BaseEmbedding] = {}

    def get_llm(self, key: str | None = None) -> LLM:
        key = key or self._settings.default_llm
        if key in self._llm_cache:
            return self._llm_cache[key]

        try:
            llm = self._build_llm(key)
        except (ValueError, ImportError, ConnectionError, RuntimeError):
            LOGGER.warning("Failed to init LLM '%s', falling back to Ollama", key, exc_info=True)
            llm = self._build_llm(OLLAMA_LLM_FALLBACK)
            key = OLLAMA_LLM_FALLBACK

        self._llm_cache[key] = llm
        return llm

    def get_embed_model(self, key: str | None = None) -> BaseEmbedding:
        key = key or self._settings.default_embed_model
        if key in self._embed_cache:
            return self._embed_cache[key]

        try:
            model = self._build_embed(key)
        except (ValueError, ImportError, ConnectionError, RuntimeError):
            LOGGER.warning(
                "Failed to init embedding '%s', falling back to Ollama", key, exc_info=True
            )
            model = self._build_embed(OLLAMA_EMBED_FALLBACK)
            key = OLLAMA_EMBED_FALLBACK

        self._embed_cache[key] = model
        return model

    def available_llms(self) -> list[str]:
        keys = [OLLAMA_LLM_FALLBACK]
        if self._settings.openai_api_key:
            keys.append("gpt-4o")
        if self._settings.anthropic_api_key:
            keys.append("claude-sonnet")
        return keys

    def available_embeddings(self) -> list[str]:
        keys = [OLLAMA_EMBED_FALLBACK]
        if self._settings.openai_api_key:
            keys.append("text-embedding-3-small")
        if self._settings.voyage_api_key:
            keys.append("voyage-3")
        return keys

    def _build_llm(self, key: str) -> LLM:
        if key == "gpt-4o":
            from llama_index.llms.openai import OpenAI

            return OpenAI(model="gpt-4o", api_key=self._settings.openai_api_key)

        if key == "claude-sonnet":
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(
                model="claude-sonnet-4-20250514", api_key=self._settings.anthropic_api_key
            )

        if key.startswith("ollama/"):
            from llama_index.llms.ollama import Ollama

            model_name = key.removeprefix("ollama/")
            return Ollama(
                model=model_name, base_url=self._settings.ollama_base_url, request_timeout=120.0
            )

        raise ValueError(f"Unknown LLM provider: {key}")

    def _build_embed(self, key: str) -> BaseEmbedding:
        if key == "text-embedding-3-small":
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(
                model_name="text-embedding-3-small",
                api_key=self._settings.openai_api_key,
                dimensions=768,
            )

        if key == "voyage-3":
            from llama_index.embeddings.voyageai import VoyageEmbedding

            return VoyageEmbedding(
                model_name="voyage-3",
                voyage_api_key=self._settings.voyage_api_key,
            )

        if key.startswith("ollama/"):
            from llama_index.embeddings.ollama import OllamaEmbedding

            model_name = key.removeprefix("ollama/")
            return OllamaEmbedding(
                model_name=model_name,
                base_url=self._settings.ollama_base_url,
                embed_batch_size=self._settings.ollama_embed_batch_size,
            )

        raise ValueError(f"Unknown embedding provider: {key}")
