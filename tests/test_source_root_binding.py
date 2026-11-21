"""ENTROLY_SOURCE must govern the corpus, not merely be reported.

Seven sites in `server.py`, plus `proxy.py`, `work_graph_mcp.py` and
`brain_execute.py`, resolved `ENTROLY_SOURCE` for themselves. The indexer did
not: `auto_index` defaults `project_dir` to `os.getcwd()` and `server.py`
called it with no root, so the variable was inert for the one decision it
names. Retrieval answered from the MCP host's working directory while guidance
reported `ENTROLY_SOURCE` as `resolved_source_root`, and the wrong-corpus guard
returned "not suspicious" because the operator had chosen a root that nothing
honoured. Following the documented remedy made detection *less* likely.

`start_incremental_watcher` defaulted the same way and was also called with no
root, so even a corrected first pass would have been reconciled against a
different tree.

These tests pin the binding itself and the two call sites, so a future change
cannot quietly reintroduce "configured here, indexed there".
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

from entroly.server_source_root import (
    _source_root_is_suspicious,
    reported_source_root,
    resolved_source_root,
    source_root_is_configured_but_unusable,
)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real directory to point ENTROLY_SOURCE at, with cwd elsewhere."""
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='p'\n", encoding="utf-8")
    elsewhere = tmp_path / "host_app"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return project, elsewhere


def test_configured_root_wins_over_the_inherited_one(repo, monkeypatch) -> None:
    project, elsewhere = repo
    monkeypatch.setenv("ENTROLY_SOURCE", str(project))
    assert Path(resolved_source_root()) == project.resolve()
    assert Path(resolved_source_root()) != elsewhere.resolve()


def test_without_the_variable_the_working_directory_is_used(repo, monkeypatch) -> None:
    _project, elsewhere = repo
    monkeypatch.delenv("ENTROLY_SOURCE", raising=False)
    assert Path(resolved_source_root()) == elsewhere.resolve()


def test_an_unusable_configured_root_falls_back_but_is_reported(repo, monkeypatch) -> None:
    """A typo must not stop the server, and must not pass silently either."""
    _project, elsewhere = repo
    monkeypatch.setenv("ENTROLY_SOURCE", str(elsewhere / "does_not_exist"))
    assert source_root_is_configured_but_unusable() is True
    assert Path(resolved_source_root()) == elsewhere.resolve()
    # The operator believes they pinned a root; they did not. Warn.
    assert _source_root_is_suspicious(resolved_source_root()) is True


def test_a_usable_configured_root_is_not_flagged(repo, monkeypatch) -> None:
    project, _elsewhere = repo
    monkeypatch.setenv("ENTROLY_SOURCE", str(project))
    assert source_root_is_configured_but_unusable() is False
    assert _source_root_is_suspicious(resolved_source_root()) is False


def test_guidance_names_the_root_that_was_actually_indexed(repo, monkeypatch) -> None:
    """Reporting must not re-derive a root the index was never built from."""
    project, elsewhere = repo
    monkeypatch.setenv("ENTROLY_SOURCE", str(project))

    class _Engine:
        _indexed_source_root = str(elsewhere)

    assert Path(reported_source_root(_Engine())) == elsewhere.resolve()
    assert Path(reported_source_root(None)) == project.resolve()


def _call_keywords(func, name: str) -> list[ast.Call]:
    """Every call to *name* inside *func*'s source."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == name
    ]


@pytest.mark.parametrize("callee", ["auto_index", "start_incremental_watcher"])
def test_the_server_passes_an_explicit_root(callee) -> None:
    """Both default project_dir to os.getcwd(); neither may be left implicit.

    Asserted structurally rather than by running a server: the defect was an
    omitted argument, and an omitted argument is exactly what a source check
    catches without needing an indexed corpus.
    """
    source = Path("entroly/server.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == callee
    ]
    assert calls, f"no call to {callee} found in entroly/server.py"
    for call in calls:
        assert len(call.args) >= 2 or any(
            kw.arg == "project_dir" for kw in call.keywords
        ), (
            f"{callee} called at line {call.lineno} without a root; it defaults "
            "to os.getcwd(), which is how ENTROLY_SOURCE became inert"
        )


def test_the_indexer_and_the_watcher_receive_the_same_root() -> None:
    """A corrected first pass reconciled against another tree is still wrong."""
    source = Path("entroly/server.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    roots = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", None))
        if name in {"auto_index", "start_incremental_watcher"} and len(node.args) >= 2:
            roots.setdefault(name, set()).add(ast.unparse(node.args[1]))
    assert roots.get("auto_index") == roots.get("start_incremental_watcher"), (
        f"indexer and watcher use different roots: {roots}"
    )


def test_resolved_root_is_absolute_and_normalised(repo, monkeypatch) -> None:
    project, _elsewhere = repo
    monkeypatch.setenv("ENTROLY_SOURCE", str(project) + os.sep + "." + os.sep)
    resolved = resolved_source_root()
    assert Path(resolved).is_absolute()
    assert Path(resolved) == project.resolve()
