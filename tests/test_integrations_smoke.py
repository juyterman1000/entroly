"""
Smoke tests for documented integration adapters.

The README advertises entroly as having adapters for Hermes-format
tool calling, LangChain, AgentSkills, and three messenger gateways
(Discord, Slack, Telegram). Each adapter has a documented public
surface that downstream users wire into their code. These tests
verify:

  1. Each adapter module is importable (no top-level errors).
  2. Each documented public symbol exists with the right shape.

We deliberately do not exercise network or runtime behavior —
that's the integration owner's responsibility. We exercise the
**advertised contract**: if the README says "import the entroly
hermes adapter and call safe_compress_hermes", then importing and
finding that callable must work without surprises.

Catches regressions where an adapter is deleted or renamed but the
README still advertises it.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest


# Each tuple: (module path, symbol name, expected kind: 'function'|'class'|'constant')
ADAPTERS: list[tuple[str, str, str]] = [
    ("entroly.integrations.agentskills",      "export_promoted",            "function"),
    ("entroly.integrations.agentskills",      "SPEC_VERSION",               "constant"),
    ("entroly.integrations.hermes",           "safe_compress_hermes",       "function"),
    ("entroly.integrations.hermes",           "format_chatml",              "function"),
    ("entroly.integrations.hermes",           "HERMES_TOOL_SYSTEM_PROMPT",  "constant"),
    ("entroly.integrations.langchain",        "EntrolyCompressor",          "class"),
    ("entroly.integrations.discord_gateway",  "DiscordGateway",             "class"),
    ("entroly.integrations.slack_gateway",    "SlackGateway",               "class"),
    ("entroly.integrations.telegram_gateway", "TelegramGateway",            "class"),
    ("entroly.integrations.telegram_gateway", "API_BASE",                   "constant"),
]


def _resolve(module_path: str, symbol: str) -> Any:
    mod = importlib.import_module(module_path)
    if not hasattr(mod, symbol):
        raise AttributeError(
            f"module {module_path!r} does not expose {symbol!r} "
            f"(public surface drift: README advertises this symbol)"
        )
    return getattr(mod, symbol)


@pytest.mark.parametrize("module_path,symbol,kind", ADAPTERS)
def test_adapter_public_surface(module_path: str, symbol: str, kind: str):
    """Each documented adapter symbol must exist and have the right shape."""
    obj = _resolve(module_path, symbol)
    if kind == "function":
        assert inspect.isfunction(obj) or inspect.isbuiltin(obj), (
            f"{module_path}.{symbol} advertised as a function but is {type(obj).__name__}"
        )
    elif kind == "class":
        assert inspect.isclass(obj), (
            f"{module_path}.{symbol} advertised as a class but is {type(obj).__name__}"
        )
    elif kind == "constant":
        # Strings, ints, etc. — any non-callable, non-class top-level value
        assert not callable(obj) and not inspect.isclass(obj), (
            f"{module_path}.{symbol} advertised as a constant but is {type(obj).__name__}"
        )
    else:
        pytest.fail(f"unknown symbol kind in test parameter: {kind!r}")


@pytest.mark.parametrize("module_path", sorted({m for m, _, _ in ADAPTERS}))
def test_adapter_module_imports_cleanly(module_path: str):
    """Each adapter module must import without side effects that crash."""
    mod = importlib.import_module(module_path)
    assert mod is not None
    # Re-import: subsequent import must also succeed (catches one-shot
    # module-level errors that disappear after first load).
    mod2 = importlib.import_module(module_path)
    assert mod2 is mod


# ── Hermes-specific shape check (deeper smoke) ──────────────────────


def test_hermes_safe_compress_signature():
    """`safe_compress_hermes` is documented as the public entry point.
    Its signature must accept at minimum a `messages` argument so
    downstream users can wire it without runtime surprises."""
    from entroly.integrations.hermes import safe_compress_hermes
    sig = inspect.signature(safe_compress_hermes)
    params = list(sig.parameters)
    assert "messages" in params, (
        f"safe_compress_hermes signature missing `messages` parameter: "
        f"got {params}"
    )


def test_hermes_format_chatml_returns_str():
    """`format_chatml` must accept a list of dicts and return a string —
    the contract documented for ChatML serialization."""
    from entroly.integrations.hermes import format_chatml
    out = format_chatml([{"role": "user", "content": "hi"}])
    assert isinstance(out, str)
    assert "hi" in out


# ── AgentSkills-specific shape check ────────────────────────────────


def test_agentskills_spec_version_is_string():
    from entroly.integrations.agentskills import SPEC_VERSION
    assert isinstance(SPEC_VERSION, str)
    # SemVer-ish format
    parts = SPEC_VERSION.split(".")
    assert len(parts) >= 2
    assert all(p.lstrip("0123456789") == "" for p in parts[:2]), (
        f"SPEC_VERSION not numeric major.minor: {SPEC_VERSION!r}"
    )
