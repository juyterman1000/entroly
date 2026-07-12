"""
Smoke tests for documented integration adapters.

The README advertises entroly as having adapters for Hermes-format
tool calling, LangChain, AgentSkills, and three messenger gateways
(Discord, Slack, Telegram). Each adapter has a documented public
surface that downstream users wire into their code. These tests
verify:

  1. Each adapter module is importable (no top-level errors).
  2. Each documented public symbol exists with the right shape.

We deliberately do not exercise network or runtime behavior here.
Deeper durable-delivery behavior is covered in test_event_delivery.py
and test_gateway_delivery.py.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest


# Each tuple: (module path, symbol name, expected kind: 'function'|'class'|'constant')
ADAPTERS: list[tuple[str, str, str]] = [
    ("entroly.integrations.agentskills", "export_promoted", "function"),
    ("entroly.integrations.agentskills", "SPEC_VERSION", "constant"),
    ("entroly.integrations.hermes", "safe_compress_hermes", "function"),
    ("entroly.integrations.hermes", "format_chatml", "function"),
    ("entroly.integrations.hermes", "HERMES_TOOL_SYSTEM_PROMPT", "constant"),
    ("entroly.integrations.langchain", "EntrolyCompressor", "class"),
    ("entroly.integrations.discord_gateway", "DiscordGateway", "class"),
    ("entroly.integrations.slack_gateway", "SlackGateway", "class"),
    ("entroly.integrations.telegram_gateway", "TelegramGateway", "class"),
    ("entroly.integrations.telegram_gateway", "API_BASE", "constant"),
    ("entroly.integrations.event_delivery", "EventDeliveryStore", "class"),
    ("entroly.integrations.event_delivery", "ReliableEventDispatcher", "class"),
    ("entroly.integrations.event_delivery", "DeliveryReceipt", "class"),
    ("entroly.integrations.event_delivery", "DELIVERY_SCHEMA", "constant"),
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
        assert not callable(obj) and not inspect.isclass(obj), (
            f"{module_path}.{symbol} advertised as a constant but is {type(obj).__name__}"
        )
    else:
        pytest.fail(f"unknown symbol kind in test parameter: {kind!r}")


@pytest.mark.parametrize("module_path", sorted({module for module, _, _ in ADAPTERS}))
def test_adapter_module_imports_cleanly(module_path: str):
    """Each adapter module must import without side effects that crash."""
    mod = importlib.import_module(module_path)
    assert mod is not None
    mod2 = importlib.import_module(module_path)
    assert mod2 is mod


def test_hermes_safe_compress_signature():
    """The documented Hermes entry point must still accept messages."""
    from entroly.integrations.hermes import safe_compress_hermes

    signature = inspect.signature(safe_compress_hermes)
    params = list(signature.parameters)
    assert "messages" in params, (
        "safe_compress_hermes signature missing `messages` parameter: "
        f"got {params}"
    )


def test_hermes_format_chatml_returns_str():
    """ChatML serialization must return a string containing the content."""
    from entroly.integrations.hermes import format_chatml

    output = format_chatml([{"role": "user", "content": "hi"}])
    assert isinstance(output, str)
    assert "hi" in output


def test_agentskills_spec_version_is_string():
    from entroly.integrations.agentskills import SPEC_VERSION

    assert isinstance(SPEC_VERSION, str)
    parts = SPEC_VERSION.split(".")
    assert len(parts) >= 2
    assert all(part.lstrip("0123456789") == "" for part in parts[:2]), (
        f"SPEC_VERSION not numeric major.minor: {SPEC_VERSION!r}"
    )
