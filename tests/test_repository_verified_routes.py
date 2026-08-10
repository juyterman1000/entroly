from __future__ import annotations

import copy
import io
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.verified_routes import (
    build_verified_routes,
    verify_routes_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_python_ast_resolves_router_mounts_handlers_and_collisions(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "api.py",
        """from fastapi import APIRouter, FastAPI
VERSION = "/v1"
app = FastAPI()
users = APIRouter(prefix="/users")

@users.get("/{user_id}")
def read_user(user_id: str):
    return user_id

@users.get("/:name")
def read_named(name: str):
    return name

app.include_router(users, prefix=VERSION)
""",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_routes(
        tmp_path,
        index,
        index_digest="sha256:test",
    )

    assert verify_routes_commitment(payload)
    assert [item["path"] for item in payload["routes"]] == [
        "/v1/users/:name",
        "/v1/users/{user_id}",
    ]
    assert {item["normalized_path"] for item in payload["routes"]} == {
        "/v1/users/{param}"
    }
    assert all(item["handler"]["status"] == "resolved" for item in payload["routes"])
    assert all(item["confidence"] == "exact-static" for item in payload["routes"])
    assert payload["conflicts"][0]["conclusion"] == "review-required-not-defect-proof"
    assert payload["receipt"]["omissions_by_reason"] == {}


def test_dynamic_python_paths_are_omitted_and_stale_sources_are_excluded(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "app.py",
        """from fastapi import FastAPI
app = FastAPI()
dynamic = input()
@app.get(dynamic)
def hidden():
    return None
""",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_routes(
        tmp_path, index, index_digest="sha256:dynamic"
    )
    assert payload["routes"] == []
    assert payload["receipt"]["omissions_by_reason"] == {"dynamic-path": 1}

    (tmp_path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    stale = build_verified_routes(
        tmp_path, index, index_digest="sha256:dynamic"
    )
    assert stale["routes"] == []
    assert stale["receipt"]["omissions_by_reason"] == {"stale-source": 1}


def test_cross_language_patterns_are_labelled_and_filterable(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "routes.ts",
        "import express from 'express';\napp.post('/items/:id', updateItem);\n",
    )
    _write(tmp_path, "server.rs", '.route("/health", get(health))\n')
    _write(tmp_path, "main.go", 'r.GET("/users/:id", getUser)\n')
    index = build_repository_index(tmp_path)
    payload = build_verified_routes(
        tmp_path,
        index,
        index_digest="sha256:polyglot",
        method="GET",
    )
    assert {item["framework"] for item in payload["routes"]} == {
        "axum", "go-router"
    }
    assert all(item["confidence"] == "heuristic-static" for item in payload["routes"])
    assert all(item["method"] == "GET" for item in payload["routes"])


def test_javascript_route_is_not_duplicated_when_framework_is_unknown(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "routes.ts", "app.get('/items', listItems);\n")
    index = build_repository_index(tmp_path)
    payload = build_verified_routes(tmp_path, index, index_digest="sha256:js")
    assert len(payload["routes"]) == 1
    assert payload["routes"][0]["framework"] == "javascript-router"
    assert payload["conflicts"] == []


def test_route_receipt_detects_tampering(tmp_path: Path) -> None:
    _write(tmp_path, "app.py", "from flask import Flask\napp = Flask(__name__)\n@app.get('/x')\ndef x():\n    pass\n")
    index = build_repository_index(tmp_path)
    payload = build_verified_routes(tmp_path, index, index_digest="sha256:test")
    tampered = copy.deepcopy(payload)
    tampered["routes"][0]["path"] = "/invented"
    assert not verify_routes_commitment(tampered)


def test_route_order_is_independent_of_line_endings(tmp_path: Path) -> None:
    """The same source must order identically whether stored LF or CRLF.

    Colliding routes tie on (normalized_path, method), so ordering fell through
    to route_id -- a digest computed over (start_byte, end_byte). CRLF shifts
    every offset, so a Windows checkout emitted ['/v1/users/:name', ...] while
    the identical file with LF emitted ['/v1/users/{user_id}', ...]. Receipt
    ordering must depend on the code, not on how git normalised the newlines.

    Written with io.open(newline=...) deliberately: Path.write_text translates
    "\n" to os.linesep, so it cannot express both cases on one platform.
    """
    source = (
        'from fastapi import APIRouter, FastAPI\n'
        'VERSION = "/v1"\n'
        'app = FastAPI()\n'
        'users = APIRouter(prefix="/users")\n'
        '\n'
        '@users.get("/{user_id}")\n'
        'def read_user(user_id: str):\n'
        '    return user_id\n'
        '\n'
        '@users.get("/:name")\n'
        'def read_named(name: str):\n'
        '    return name\n'
        '\n'
        'app.include_router(users, prefix=VERSION)\n'
    )

    observed = {}
    for label, newline in (("lf", ""), ("crlf", "\r\n")):
        root = tmp_path / label
        root.mkdir()
        with io.open(root / "api.py", "w", encoding="utf-8", newline=newline) as handle:
            handle.write(source)
        raw = (root / "api.py").read_bytes()
        assert (b"\r\n" in raw) == (label == "crlf"), "fixture did not store the intended newline"
        payload = build_verified_routes(
            root, build_repository_index(root), index_digest="sha256:test"
        )
        observed[label] = [item["path"] for item in payload["routes"]]

    assert observed["lf"] == observed["crlf"]
    assert observed["lf"] == ["/v1/users/:name", "/v1/users/{user_id}"]
