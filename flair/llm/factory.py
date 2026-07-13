"""Factory dei provider LLM."""

from __future__ import annotations

from .base import LLMProvider
from .deepseek import DeepSeekProvider
from .openai import OpenAIProvider

_REGISTRY = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
}


def create_provider(cfg) -> LLMProvider:
    cls = _REGISTRY.get(cfg.provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {cfg.provider}. Available: {list(_REGISTRY)}")
    return cls(cfg)
