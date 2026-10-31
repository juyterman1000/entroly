"""The index must be able to reflect an edit that is not on disk yet.

Entroly indexes files from disk and re-indexes them when they change, so a
saved edit is picked up by `auto_index`. An unsaved buffer is invisible to
that: selection ranks and returns the version on disk while the developer is
asking about the version in front of them. Only the editor can see a dirty
buffer, so the editor pushes it here.

The endpoint writes into a trusted store, so these tests pin the containment
rules as hard as the behaviour: a path outside the workspace, a traversal, and
an oversized body are all refused, and clearing restores the file from disk
rather than leaving the index empty or still serving the buffer.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from entroly.proxy import create_proxy_app
from entroly.proxy_config import ProxyConfig

DISK = "def charge():\n    return 'ON_DISK'\n"
BUFFER = "def charge():\n    return 'UNSAVED_BUFFER'\n"


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "billing.py").write_text(DISK, encoding="utf-8")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_SOURCE", str(project))
    monkeypatch.setenv("ENTROLY_NO_SELF_HEAL", "1")
    return project


@pytest.fixture()
def app_and_engine(workspace):
    from entroly.engine import EntrolyEngine

    engine = EntrolyEngine()
    engine.ingest_fragment(DISK, str(workspace / "billing.py"), 40, False)
    app = create_proxy_app(
        engine, ProxyConfig(), start_dashboard=False, start_autotune=False
    )
    return app, engine


def _post(app, payload) -> httpx.Response:
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:9377"
        ) as client:
            return await client.post("/overlay", json=payload)

    return asyncio.run(_run())


def _served(engine) -> str:
    result = engine.optimize_context(4000, "charge")
    return " ".join(
        f.get("content", "") for f in (result.get("selected_fragments") or [])
    )


def test_overlay_replaces_what_selection_serves(app_and_engine) -> None:
    app, engine = app_and_engine
    assert "ON_DISK" in _served(engine)

    response = _post(app, {"path": "billing.py", "content": BUFFER})

    assert response.status_code == 200
    assert response.json()["status"] == "applied"
    served = _served(engine)
    assert "UNSAVED_BUFFER" in served
    assert "ON_DISK" not in served, "the overlay must replace, not accumulate"


def test_clearing_restores_the_file_from_disk(app_and_engine) -> None:
    """A save, a close, or a crashed editor must not strand the index."""
    app, engine = app_and_engine
    _post(app, {"path": "billing.py", "content": BUFFER})

    response = _post(app, {"path": "billing.py", "content": None})

    assert response.status_code == 200
    assert response.json()["restored_from_disk"] is True
    served = _served(engine)
    assert "ON_DISK" in served
    assert "UNSAVED_BUFFER" not in served


def test_overlay_is_idempotent(app_and_engine) -> None:
    app, engine = app_and_engine
    for _ in range(3):
        _post(app, {"path": "billing.py", "content": BUFFER})

    served = _served(engine)
    assert served.count("UNSAVED_BUFFER") == 1, "repeated pushes must not stack"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "../../etc/passwd",
        "/etc/passwd",
        "does-not-exist.py",
    ],
)
def test_paths_outside_the_workspace_are_refused(app_and_engine, path) -> None:
    app, _ = app_and_engine
    response = _post(app, {"path": path, "content": "x = 1\n"})
    assert response.status_code == 403
    assert response.json()["error"] == "path_outside_workspace"


def test_a_refused_path_does_not_reach_the_index(app_and_engine, tmp_path) -> None:
    app, engine = app_and_engine
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 'leaked'\n", encoding="utf-8")

    _post(app, {"path": str(outside), "content": "SECRET = 'leaked'\n"})

    assert "leaked" not in _served(engine)


def test_oversized_content_is_refused(app_and_engine) -> None:
    app, _ = app_and_engine
    response = _post(app, {"path": "billing.py", "content": "x" * (600 * 1024)})
    assert response.status_code == 413
    assert response.json()["error"] == "content_too_large"


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({}, 400),
        ({"path": ""}, 400),
        ({"path": "billing.py", "content": 42}, 400),
        ({"path": 7, "content": "x"}, 400),
    ],
)
def test_malformed_requests_are_rejected(app_and_engine, payload, status) -> None:
    app, _ = app_and_engine
    assert _post(app, payload).status_code == status


def test_tokens_are_measured_not_estimated(app_and_engine) -> None:
    """The reported count must match the tokenizer the budget is enforced in."""
    from entroly.tokens import count_tokens

    app, _ = app_and_engine
    content = "这是一个关于上下文压缩的技术文档。\n" * 5
    response = _post(app, {"path": "billing.py", "content": content})

    assert response.status_code == 200
    assert response.json()["tokens"] == count_tokens(content)
    assert response.json()["tokens"] != len(content) // 4
