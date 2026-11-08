"""Anthropic's undated aliases for Claude 4.x resolve to their dated records.

Dogfooding 1.0.88: a proxied request for `claude-sonnet-4-5`, the name most
SDK examples use, came back with `x-entroly-model-warning: unknown_model` and
a 128,000-token fallback window, although the registry held the same model as
`claude-sonnet-4-5-20250929` with its 200,000-token window.
"""

from __future__ import annotations

import pytest

from entroly.models.registry import get_model_registry
from entroly.proxy_config import context_window_for_model


@pytest.mark.parametrize(
    "alias, dated",
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        ("claude-sonnet-4-0", "claude-sonnet-4-20250514"),
        ("claude-opus-4-0", "claude-opus-4-20250514"),
    ],
)
def test_undated_alias_resolves_to_the_dated_model(alias: str, dated: str) -> None:
    registry = get_model_registry()
    by_alias = registry.resolve(alias)
    by_date = registry.resolve(dated)

    assert by_alias.capability is not None
    assert by_alias.capability.id == by_date.capability.id == f"anthropic/{dated}"
    assert by_alias.warning is None
    assert context_window_for_model(alias) == 200_000
