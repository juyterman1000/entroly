from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.cli import main, run
from entroly.repository_intelligence.mcp import create_repository_mcp_server
from entroly.repository_intelligence.rename_context import build_rename_context, render_rename_context
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from test_repository_intelligence_service import _install_fake_mcp


def _project(root: Path, *, large: bool = False) -> None:
    (root / "source.py").write_bytes((
        "@decorate\n"
        "def execute(value):\n"
        + ("    # α漢🙂\\\"\t\n" * 650 if large else "")
        + "    return value + 1\n"
    ).encode("utf-8"))
    (root / "caller.py").write_bytes(
        b"from source import (\n    execute,\n)\n\ndef run():\n    return execute(1)\n"
    )


def _wire(page: dict, budget: int) -> None:
    assert len(render_rename_context(page).encode("utf-8")) == page["receipt"]["wire_bytes"] <= budget


def _pages(service: RepositoryIntelligenceService, budget: int = 4096) -> list[dict]:
    result, cursor = [], None
    for _ in range(100):
        page = service.rename_context("execute", "perform", byte_budget=budget, cursor=cursor)
        _wire(page, budget)
        result.append(page)
        cursor = page["delivery"]["next_cursor"]
        if cursor is None:
            return result
    pytest.fail("pagination did not terminate")


def test_exact_regions_include_import_caller_and_decorators_without_execution(tmp_path: Path) -> None:
    _project(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    pages = _pages(RepositoryIntelligenceService(tmp_path))
    fragments = [item for page in pages for item in page["fragments"]]
    assert len(fragments) == 3
    assert {item["text"] for item in fragments} == {
        "from source import (\n    execute,\n)\n",
        "def run():\n    return execute(1)\n",
        "@decorate\ndef execute(value):\n    return value + 1\n",
    }
    for fragment in fragments:
        raw = before[fragment["path"]]
        assert raw[fragment["start_byte"]:fragment["end_byte"]].decode() == fragment["text"]
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert pages[-1]["scope"]["reference_completeness"] == "not-proven"
    assert pages[-1]["scope"]["semantic_status"] == "not-checked"


def test_unicode_pagination_recovers_exact_bytes_and_region_hashes(tmp_path: Path) -> None:
    _project(tmp_path, large=True)
    pages = _pages(RepositoryIntelligenceService(tmp_path))
    assert len(pages) > 3
    recovered: dict[tuple, bytes] = {}
    for page in pages:
        assert page["receipt"]["source_bytes"] == sum(
            len(fragment["text"].encode()) for fragment in page["fragments"]
        )
        for fragment in page["fragments"]:
            key = (fragment["path"], fragment["start_byte"], fragment["end_byte"], fragment["region_sha256"])
            previous = recovered.get(key, b"")
            assert fragment["chunk_start_byte"] == fragment["start_byte"] + len(previous)
            piece = fragment["text"].encode()
            assert fragment["chunk_end_byte"] - fragment["chunk_start_byte"] == len(piece)
            recovered[key] = previous + piece
    for (path, start, end, digest), raw in recovered.items():
        assert raw == (tmp_path / path).read_bytes()[start:end]
        assert hashlib.sha256(raw).hexdigest() == digest
    assert sum(map(len, recovered.values())) == pages[0]["delivery"]["selected_source_bytes"]


def test_cached_index_does_not_hide_new_importers_or_stale_cursor(tmp_path: Path) -> None:
    _project(tmp_path, large=True)
    service = RepositoryIntelligenceService(tmp_path)
    assert service.summary()["files"] == 2
    first = service.rename_context("execute", "perform", byte_budget=4096)
    (tmp_path / "new.py").write_bytes(b"from source import execute\n")
    with pytest.raises(ValueError, match="source or task changed"):
        service.rename_context("execute", "perform", cursor=first["delivery"]["next_cursor"])
    fresh = _pages(service)
    assert any(item["path"] == "new.py" for page in fresh for item in page["fragments"])


@pytest.mark.parametrize("change", ["body", "unrelated", "task"])
def test_cursor_rejects_changed_source_or_task(tmp_path: Path, change: str) -> None:
    _project(tmp_path, large=True)
    service = RepositoryIntelligenceService(tmp_path)
    first = service.rename_context("execute", "perform", byte_budget=4096)
    if change == "body":
        with (tmp_path / "source.py").open("ab") as stream:
            stream.write(b"\n# changed\n")
    elif change == "unrelated":
        (tmp_path / "other.py").write_bytes(b"unused = 1\n")
    with pytest.raises(ValueError, match="source or task changed"):
        service.rename_context("execute", "changed" if change == "task" else "perform",
                               cursor=first["delivery"]["next_cursor"])


@pytest.mark.parametrize("cursor", ["{", "[]", "null", '{"path":"../secret"}', "x" * 513, 42])
def test_malformed_cursors_fail_closed(tmp_path: Path, cursor) -> None:
    _project(tmp_path)
    with pytest.raises(ValueError, match="cursor"):
        RepositoryIntelligenceService(tmp_path).rename_context("execute", "perform", cursor=cursor)


@pytest.mark.parametrize("position", [(True, 0), (0, True), (-1, 0), (999, 0), (0, -1), (0, 99999), (0, 1.5)])
def test_invalid_cursor_positions(tmp_path: Path, position) -> None:
    _project(tmp_path, large=True)
    service = RepositoryIntelligenceService(tmp_path)
    page = service.rename_context("execute", "perform", byte_budget=4096)
    cursor = json.loads(page["delivery"]["next_cursor"])
    cursor.update(region=position[0], offset=position[1])
    with pytest.raises(ValueError, match="cursor"):
        service.rename_context("execute", "perform", cursor=json.dumps(cursor))


def test_cursor_cannot_split_unicode(tmp_path: Path) -> None:
    _project(tmp_path, large=True)
    service = RepositoryIntelligenceService(tmp_path)
    page = service.rename_context("execute", "perform", byte_budget=4096)
    cursor = json.loads(page["delivery"]["next_cursor"])
    cursor.update(region=2, offset=(tmp_path / "source.py").read_bytes().index("α".encode()) + 1)
    with pytest.raises(ValueError, match="UTF-8 offset"):
        service.rename_context("execute", "perform", cursor=json.dumps(cursor))


@pytest.mark.parametrize("budget", [True, 4095, 262145, 8192.0, "8192", None])
def test_budget_validated_before_indexing(tmp_path: Path, budget) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("invalid budget must be rejected before indexing")
    with pytest.raises(ValueError, match="byte_budget"):
        RepositoryIntelligenceService(tmp_path, builder=forbidden).rename_context(
            "execute", "perform", byte_budget=budget,
        )


def test_ambiguous_query_requires_resolved_plan(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "other.py").write_bytes(b"def execute():\n    return 2\n")
    with pytest.raises(ValueError, match="resolved plan"):
        RepositoryIntelligenceService(tmp_path).rename_context("execute", "perform")


def test_source_change_during_selection_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    import entroly.repository_intelligence.rename_context as module
    original = module._regions

    def change_after_selection(*args):
        result = original(*args)
        (tmp_path / "source.py").write_bytes(b"def execute():\n    return 77\n")
        return result
    monkeypatch.setattr(module, "_regions", change_after_selection)
    with pytest.raises(ValueError, match="source changed during selection"):
        RepositoryIntelligenceService(tmp_path).rename_context("execute", "perform")


@pytest.mark.parametrize("raw", [
    b"@decorate(\n    'value'\n)\ndef execute():\n    return 1\n",
    b"@decorate\r\ndef execute():\r\n    return 1\r\n",
    b"def execute(): return 1;\n",
])
def test_physical_lines_preserve_decorators_and_crlf(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "source.py").write_bytes(raw)
    pages = _pages(RepositoryIntelligenceService(tmp_path))
    fragments = [fragment for page in pages for fragment in page["fragments"]]
    assert b"".join(fragment["text"].encode() for fragment in fragments) == raw


def test_unresolved_bom_file_is_rejected_without_fabricating_source_coverage(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_bytes(b"\xef\xbb\xbfdef execute():\n    return 1\n")
    with pytest.raises(ValueError, match="resolved plan"):
        RepositoryIntelligenceService(tmp_path).rename_context("execute", "perform")


def test_conflicting_binding_remains_visible_in_context(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_bytes(
        b"def execute():\n    return 1\ndef perform():\n    return 2\n"
    )
    page = RepositoryIntelligenceService(tmp_path).rename_context("execute", "perform")
    assert page["plan"]["binding_review"] == "conflict"
    assert page["plan"]["binding_conflict_count"] > 0
    assert page["scope"]["semantic_status"] == "not-checked"


def test_non_python_plan_is_not_silently_dropped(tmp_path: Path) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    plan = service.rename_preview("execute", "perform")
    plan = copy.deepcopy(plan)
    plan["changes"][0]["path"] = "caller.ts"
    from dataclasses import replace
    index.files["caller.ts"] = replace(index.files["caller.py"], language="typescript", path="caller.ts")
    with pytest.raises(ValueError, match="Python source only"):
        build_rename_context(tmp_path, index, plan, byte_budget=8192)


def test_final_cli_and_mcp_strings_share_exact_measured_formatter(tmp_path: Path, monkeypatch, capsys) -> None:
    _project(tmp_path, large=True)
    args = ["--root", str(tmp_path), "rename-context", "--symbol", "execute", "--new-name", "perform",
            "--byte-budget", "4096"]
    code, payload = run(args)
    assert code == 0
    assert main(args) == 0
    wire = capsys.readouterr().out
    assert wire == render_rename_context(payload)
    assert len(wire.encode()) == payload["receipt"]["wire_bytes"] <= 4096
    _install_fake_mcp(monkeypatch)
    mcp = create_repository_mcp_server(tmp_path)
    result = mcp.tools["repository_rename_context"]("execute", "perform", byte_budget=4096)
    assert result == wire
    next_wire = mcp.tools["repository_rename_context"](
        "execute", "perform", byte_budget=4096, cursor=payload["delivery"]["next_cursor"],
    )
    assert len(next_wire.encode()) == json.loads(next_wire)["receipt"]["wire_bytes"] <= 4096
    # This tool adds no write authority; destructive methods retain their gate.
    plan = json.loads(mcp.tools["repository_rename_preview"]("execute", "perform"))
    refused = json.loads(mcp.tools["repository_rename_apply"](
        plan, plan["receipt"]["plan_sha256"], acknowledge_incomplete=True,
    ))
    assert "error" in refused


def test_cli_and_mcp_report_invalid_inputs(tmp_path: Path, monkeypatch) -> None:
    _project(tmp_path)
    code, payload = run(["--root", str(tmp_path), "rename-context", "--symbol", "execute",
                         "--new-name", "perform", "--byte-budget", "1"])
    assert code == 2 and "error" in payload
    _install_fake_mcp(monkeypatch)
    mcp = create_repository_mcp_server(tmp_path)
    error = json.loads(mcp.tools["repository_rename_context"]("execute", "perform", byte_budget=True))
    assert "error" in error


def test_cli_actual_utf8_stdout_ignores_windows_locale_and_newline_translation(tmp_path: Path) -> None:
    _project(tmp_path, large=True)
    process = subprocess.run(
        [sys.executable, "-B", "-m", "entroly.repository_intelligence", "--root", str(tmp_path),
         "rename-context", "--symbol", "execute", "--new-name", "perform", "--byte-budget", "4096"],
        capture_output=True, timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "ascii", "ENTROLY_NO_SELF_HEAL": "1", "ENTROLY_AIR_GAP": "1"},
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert "α" in process.stdout.decode("utf-8")
    assert process.stdout == render_rename_context(payload).encode("utf-8")
    assert len(process.stdout) == payload["receipt"]["wire_bytes"] <= 4096
