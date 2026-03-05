from functools import lru_cache

from src.core.config import get_settings
from src.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseVisionProvider,
)


def _get_provider_module():
    """Return the provider module based on config."""
    settings = get_settings()
    provider = settings.ai_provider.lower()

    if provider == "openai":
        from src.providers import openai_provider
        return openai_provider
    else:
        raise ValueError(
            f"Unsupported AI provider: '{provider}'. "
            f"Supported providers: openai"
        )


@lru_cache()
def get_llm_provider() -> BaseLLMProvider:
    """Return a cached LLM provider instance."""
    module = _get_provider_module()
    return module.OpenAILLMProvider()


@lru_cache()
def get_embedding_provider() -> BaseEmbeddingProvider:
    """Return a cached embedding provider instance."""
    module = _get_provider_module()
    return module.OpenAIEmbeddingProvider()


@lru_cache()
def get_vision_provider() -> BaseVisionProvider:
    """Return a cached vision provider instance."""
    module = _get_provider_module()
    return module.OpenAIVisionProvider()
