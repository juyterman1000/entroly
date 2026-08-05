from __future__ import annotations

import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from entroly.repository_intelligence import (
    InvalidChangedPaths,
    RepositoryIntelligenceService,
    build_repository_index,
)
from entroly.repository_intelligence.mcp import create_repository_mcp_server
from entroly.repository_intelligence.service import UnknownChangedPaths


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "pkg/source.py", "def execute():\n    return 1\n")
    _write(
        root,
        "pkg/api.py",
        "from pkg.source import execute\n"
        "def invoke():\n"
        "    return execute()\n",
    )
    _write(
        root,
        "tests/test_api.py",
        "from pkg.api import invoke\n"
        "def test_invoke():\n"
        "    assert invoke() == 1\n",
    )


def test_service_caches_one_deterministic_snapshot_until_refresh(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    first = service.summary()
    second = service.summary()
    assert first == second
    assert first["generation"] == 1
    assert first["index_digest"].startswith("sha256:")

    _write(tmp_path, "pkg/new.py", "def added():\n    return True\n")
    assert service.summary()["files"] == 3
    refreshed = service.refresh()
    assert refreshed["files"] == 4
    assert refreshed["generation"] == 2
    assert refreshed["index_digest"] != first["index_digest"]


def test_service_digest_is_independent_of_checkout_location(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _project(first_root)
    _project(second_root)

    first = RepositoryIntelligenceService(first_root).summary()
    second = RepositoryIntelligenceService(second_root).summary()

    assert first["root"] != second["root"]
    assert first["index_digest"] == second["index_digest"]


def test_service_impact_and_tests_share_snapshot_identity(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    impact = service.impact(["pkg/source.py"])
    tests = service.tests(["pkg/source.py"])
    assert impact["index_digest"] == tests["index_digest"]
    assert impact["generation"] == tests["generation"] == 1
    assert impact["report"]["impacted_paths"] == [
        "pkg/api.py",
        "pkg/source.py",
        "tests/test_api.py",
    ]
    assert tests["candidates"][0]["path"] == "tests/test_api.py"


def test_service_unknown_paths_are_structured(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    try:
        service.impact(["missing.py"])
    except UnknownChangedPaths as exc:
        assert exc.to_dict()["unknown"] == ["missing.py"]
    else:
        raise AssertionError("unknown paths must fail visibly")


def test_service_rejects_non_relative_paths_without_echoing_them(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    invalid = [
        "/pkg/source.py",
        "../pkg/source.py",
        "pkg/../source.py",
        "C:\\pkg\\source.py",
        "pkg/source.py\x00ignored",
        "",
    ]
    try:
        service.impact(invalid)
    except InvalidChangedPaths as exc:
        payload = exc.to_dict()
        assert payload["error"] == "invalid_changed_paths"
        assert payload["invalid_count"] == len(invalid)
        rendered = json.dumps(payload)
        for path in invalid:
            if path:
                assert path not in rendered
    else:
        raise AssertionError("non-relative paths must fail visibly")


def test_service_counts_input_paths_before_deduplication(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    repeated = ["pkg/source.py"] * 201
    try:
        service.impact(repeated)
    except ValueError as exc:
        assert "at most 200" in str(exc)
    else:
        raise AssertionError("request-size limit must precede path deduplication")


def test_service_snapshot_is_single_flight_under_concurrency(tmp_path: Path) -> None:
    _project(tmp_path)
    calls = 0
    calls_lock = threading.Lock()
    build_started = threading.Event()
    release_build = threading.Event()

    def builder(root: Path, *, limits):
        nonlocal calls
        with calls_lock:
            calls += 1
        build_started.set()
        assert release_build.wait(timeout=5)
        return build_repository_index(root, limits=limits)

    service = RepositoryIntelligenceService(tmp_path, builder=builder)
    barrier = threading.Barrier(8)

    def read_digest() -> str:
        barrier.wait()
        return str(service.summary()["index_digest"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_digest) for _ in range(8)]
        assert build_started.wait(timeout=5)
        release_build.set()
        digests = [future.result(timeout=5) for future in futures]

    assert calls == 1
    assert len(set(digests)) == 1
    assert service.summary()["generation"] == 1


class FakeFastMCP:
    def __init__(self, name: str, instructions: str = "") -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, object] = {}

    def tool(self):
        def decorate(function):
            self.tools[function.__name__] = function
            return function

        return decorate

    def run(self) -> None:
        raise AssertionError("tests do not launch a server")


def _install_fake_mcp(monkeypatch) -> None:
    mcp = types.ModuleType("mcp")
    server = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp)


def test_mcp_exposes_fixed_root_bounded_tools(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    _install_fake_mcp(monkeypatch)
    mcp = create_repository_mcp_server(tmp_path)
    assert set(mcp.tools) == {
        "refresh_repository_index",
        "repository_change_impact",
        "repository_summary",
        "repository_tests_for_changes",
    }
    summary = json.loads(mcp.tools["repository_summary"]())
    assert summary["files"] == 3
    assert summary["root"] == "."
    assert str(tmp_path) not in json.dumps(summary)
    impact = json.loads(
        mcp.tools["repository_change_impact"](["pkg/source.py"])
    )
    assert impact["report"]["impacted_paths"][-1] == "tests/test_api.py"
    tests = json.loads(
        mcp.tools["repository_tests_for_changes"](["pkg/source.py"])
    )
    assert tests["candidates"][0]["path"] == "tests/test_api.py"


def test_mcp_unknown_path_returns_machine_readable_error(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    _install_fake_mcp(monkeypatch)
    mcp = create_repository_mcp_server(tmp_path)
    payload = json.loads(mcp.tools["repository_change_impact"](["missing.py"]))
    assert payload["error"] == "unknown_changed_paths"
    assert payload["operation"] == "repository_change_impact"
    assert payload["unknown"] == ["missing.py"]
    assert str(tmp_path) not in json.dumps(payload)


def test_mcp_invalid_path_does_not_echo_local_path(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    _install_fake_mcp(monkeypatch)
    mcp = create_repository_mcp_server(tmp_path)
    leaked = str(tmp_path / "pkg" / "source.py")
    payload = json.loads(mcp.tools["repository_change_impact"]([leaked]))
    assert payload["error"] == "invalid_changed_paths"
    assert payload["invalid_count"] == 1
    assert leaked not in json.dumps(payload)


def test_mcp_refresh_atomically_changes_generation(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    _install_fake_mcp(monkeypatch)
    mcp = create_repository_mcp_server(tmp_path)
    before = json.loads(mcp.tools["repository_summary"]())
    _write(tmp_path, "pkg/new.py", "def added():\n    return True\n")
    cached = json.loads(mcp.tools["repository_summary"]())
    after = json.loads(mcp.tools["refresh_repository_index"]())
    assert cached["files"] == before["files"]
    assert after["files"] == before["files"] + 1
    assert after["generation"] == before["generation"] + 1
    assert after["root"] == "."
