from __future__ import annotations

from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings


def _settings(**overrides: str) -> AppSettings:
    defaults = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "VOYAGE_API_KEY": "",
        "DEFAULT_LLM": "ollama/qwen2.5",
        "DEFAULT_EMBED_MODEL": "ollama/nomic-embed",
    }
    defaults.update(overrides)
    return AppSettings.from_env(defaults)


def test_available_llms_only_ollama_when_no_api_keys():
    registry = ProviderRegistry(_settings())
    llms = registry.available_llms()

    assert "ollama/qwen2.5" in llms
    assert "gpt-4o" not in llms
    assert "claude-sonnet" not in llms


def test_available_llms_includes_openai_when_key_present():
    registry = ProviderRegistry(_settings(OPENAI_API_KEY="sk-test"))
    llms = registry.available_llms()

    assert "gpt-4o" in llms


def test_available_llms_includes_anthropic_when_key_present():
    registry = ProviderRegistry(_settings(ANTHROPIC_API_KEY="sk-ant-test"))
    llms = registry.available_llms()

    assert "claude-sonnet" in llms


def test_available_embeddings_only_ollama_when_no_api_keys():
    registry = ProviderRegistry(_settings())
    embeds = registry.available_embeddings()

    assert "ollama/nomic-embed" in embeds
    assert "text-embedding-3-small" not in embeds
    assert "voyage-3" not in embeds


def test_available_embeddings_includes_voyage_when_key_present():
    registry = ProviderRegistry(_settings(VOYAGE_API_KEY="voy-test"))
    embeds = registry.available_embeddings()

    assert "voyage-3" in embeds


def test_get_llm_unknown_provider_falls_back_to_ollama():
    registry = ProviderRegistry(_settings())
    llm = registry.get_llm("nonexistent-provider")
    # Should fall back to Ollama and return a valid LLM object
    assert llm is not None
    assert "ollama/qwen2.5" in registry._llm_cache


def test_get_llm_caches_instance():
    registry = ProviderRegistry(_settings())
    llm1 = registry.get_llm("ollama/qwen2.5")
    llm2 = registry.get_llm("ollama/qwen2.5")
    assert llm1 is llm2


def test_get_embed_model_default():
    registry = ProviderRegistry(_settings())
    model = registry.get_embed_model()
    assert model is not None


def test_get_embed_model_unknown_falls_back_to_ollama():
    registry = ProviderRegistry(_settings())
    model = registry.get_embed_model("nonexistent-embed")
    assert model is not None
    assert "ollama/nomic-embed" in registry._embed_cache
