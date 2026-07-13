"""Verified model-intelligence registry for Entroly."""

from .registry import (
    DiscoveryReport,
    ModelCapability,
    ModelRegistry,
    ModelResolution,
    RegistryTrust,
    discover_local_models,
    discover_ollama_models,
    discover_openai_compatible_models,
    discover_openrouter_models,
    discover_remote_models,
    get_model_registry,
    resolve_model,
)

__all__ = [
    "DiscoveryReport",
    "ModelCapability",
    "ModelRegistry",
    "ModelResolution",
    "RegistryTrust",
    "discover_local_models",
    "discover_ollama_models",
    "discover_openai_compatible_models",
    "discover_openrouter_models",
    "discover_remote_models",
    "get_model_registry",
    "resolve_model",
]
