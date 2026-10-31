"""The streaming example must refuse WebSockets from other sites.

Browsers do not apply CORS to WebSockets, so before this check any page a user
visited while the example ran could open ws://localhost:8083/ws, start a Claude
run over their repository and read its output. The server also listened on every
interface and wrote model output into the page as raw HTML.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "stream_claude_server.py"


@pytest.fixture(scope="module")
def example():
    # The example depends on aiohttp, which is not a project dependency; the
    # policy under test is plain Python, so a stub is enough to import it.
    stub = types.ModuleType("aiohttp")
    web = types.SimpleNamespace(
        Application=lambda: types.SimpleNamespace(router=types.SimpleNamespace(add_get=lambda *a: None)),
        Response=object,
        WebSocketResponse=object,
        run_app=lambda *a, **k: None,
    )
    stub.web = web
    saved = sys.modules.get("aiohttp")
    sys.modules["aiohttp"] = stub
    try:
        spec = importlib.util.spec_from_file_location("stream_claude_server_example", EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if saved is None:
            sys.modules.pop("aiohttp", None)
        else:
            sys.modules["aiohttp"] = saved


@pytest.mark.parametrize(
    ("origin", "host"),
    [
        ("http://localhost:8083", "localhost:8083"),
        ("http://127.0.0.1:8083", "127.0.0.1:8083"),
    ],
)
def test_own_page_may_connect(example, origin, host) -> None:
    assert example.request_is_from_this_page(origin, host)


@pytest.mark.parametrize(
    ("origin", "host"),
    [
        ("https://attacker.example", "localhost:8083"),  # cross-site WebSocket
        (None, "localhost:8083"),  # non-browser or stripped Origin
        ("null", "localhost:8083"),  # sandboxed iframe or file:// page
        ("http://rebind.attacker.example:8083", "rebind.attacker.example:8083"),  # DNS rebinding
        ("http://localhost:8083", "rebind.attacker.example:8083"),
        ("http://localhost:9999", "localhost:8083"),  # another local app
    ],
)
def test_other_pages_are_refused(example, origin, host) -> None:
    assert not example.request_is_from_this_page(origin, host)


def test_listens_on_loopback_only(example) -> None:
    assert example.HOST == "127.0.0.1"


def test_model_output_is_escaped_before_rendering() -> None:
    page = EXAMPLE.read_text(encoding="utf-8")
    assert "let text = escapeHtml(event.data);" in page
