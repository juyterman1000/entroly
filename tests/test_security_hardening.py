"""Exploit-oriented regression checks for local trust boundaries."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.auto_index import _resolve_project_file
from entroly.belief_compiler import BeliefCompiler
from entroly.daemon import EntrolyDaemon
from entroly.dashboard import DashboardHandler
from entroly.fast_path import FastPathRouter
from entroly.path_safety import resolve_file_within
from entroly.ravs.executors import PythonExecutor, SymPyExecutor
from entroly.skill_engine import (
    SkillSynthesizer,
    StructuralSynthesizer,
    promoted_skill_execution_enabled,
)
from entroly.verifiers.cache import CacheMeta, _load_from_cache
from entroly.verifiers.ngram_model import CharNGramModel


def test_dashboard_control_writes_reject_cross_origin_browser_requests():
    handler = object.__new__(DashboardHandler)
    handler.server = SimpleNamespace(server_port=9378)

    handler.headers = {}
    assert handler._is_trusted_write_origin() is True

    handler.headers = {"Origin": "http://localhost:9378"}
    assert handler._is_trusted_write_origin() is True

    handler.headers = {"Origin": "https://attacker.example"}
    assert handler._is_trusted_write_origin() is False

    handler.headers = {"Origin": "http://localhost:9999"}
    assert handler._is_trusted_write_origin() is False


def test_fast_path_parses_vault_tool_without_executing_it(tmp_path):
    marker = tmp_path / "executed"
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "tool.py").write_text(
        "import re\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('owned')\n"
        "TRIGGER_PATTERN = re.compile(r'\\b(auth)\\b', re.I)\n"
        "FRAGMENT_RECIPE = ['auth.py']\n"
        "WEIGHT_PROFILE = {'semantic': 0.8}\n",
        encoding="utf-8",
    )

    recipe, pattern, weights = FastPathRouter._extract_from_tool(str(skill_dir), "auth")
    assert not marker.exists()
    assert recipe == ["auth.py"]
    assert pattern is not None and pattern.search("fix auth")
    assert weights == {"semantic": 0.8}


def test_fast_path_rejects_quantified_trigger_patterns(tmp_path):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "tool.py").write_text(
        "import re\n"
        "TRIGGER_PATTERN = re.compile(r'(a+)+$')\n"
        "FRAGMENT_RECIPE = ['auth.py']\n"
        "WEIGHT_PROFILE = {}\n",
        encoding="utf-8",
    )

    _, pattern, _ = FastPathRouter._extract_from_tool(str(skill_dir), "auth")
    assert pattern is None


def test_generated_skill_treats_entity_as_data(tmp_path):
    marker = tmp_path / "executed"
    entity = f'x"; __import__("pathlib").Path({str(marker)!r}).write_text("owned"); #'
    spec = SkillSynthesizer().synthesize_from_gap(entity, ["fix entity"])
    namespace: dict = {}

    exec(compile(spec.tool_code, "<generated-skill>", "exec"), namespace)
    result = namespace["execute"]("fix entity", {})

    assert not marker.exists()
    assert result["entity"] == entity


def test_structural_skill_treats_entity_and_paths_as_data(tmp_path):
    marker = tmp_path / "executed"
    source = tmp_path / "quote'name.py"
    source.write_text("def hello(name: str) -> str:\n    return name\n", encoding="utf-8")
    entity = f'x"; __import__("pathlib").Path({str(marker)!r}).write_text("owned"); #'

    spec = StructuralSynthesizer().synthesize_structural(entity, [str(source)], ["hello"])
    assert spec is not None
    namespace: dict = {}
    exec(compile(spec.tool_code, "<generated-structural-skill>", "exec"), namespace)
    result = namespace["execute"]("hello", {})

    assert not marker.exists()
    assert result["entity"] == entity


def test_promoted_skill_execution_is_opt_in(monkeypatch):
    monkeypatch.delenv("ENTROLY_EXECUTE_PROMOTED_SKILLS", raising=False)
    assert promoted_skill_execution_enabled() is False
    monkeypatch.setenv("ENTROLY_EXECUTE_PROMOTED_SKILLS", "1")
    assert promoted_skill_execution_enabled() is True


class _PicklePayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "owned")


def test_verifier_cache_ignores_planted_pickle(tmp_path):
    marker = tmp_path / "executed"
    model = CharNGramModel()
    model.train_from_strings(["def auth(): return True"])

    (tmp_path / "meta.json").write_text(json.dumps(CacheMeta(
        version=2,
        repo_root=str(tmp_path),
        n_files=1,
        n_chars=model.total_chars,
        n_symbols=0,
        built_at=0.0,
        file_hash="fixture",
    ).__dict__), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "repo": [], "stdlib": [], "installed": [], "builtins": [],
    }), encoding="utf-8")
    (tmp_path / "ngram.json").write_text(json.dumps(model.to_dict()), encoding="utf-8")
    (tmp_path / "ngram.pkl").write_bytes(pickle.dumps(_PicklePayload(marker)))

    verifier, _ = _load_from_cache(tmp_path, 6.5)
    assert verifier.ngram_model is not None
    assert not marker.exists()


def test_repository_scoped_reads_reject_outside_symlinks(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("SECRET = 'do-not-index'\n", encoding="utf-8")
    link = project / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    assert resolve_file_within(project, link) is None
    assert _resolve_project_file(str(project), "linked.py") is None


def test_targeted_belief_compile_rejects_parent_traversal(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "outside.py").write_text("def leak(): pass\n", encoding="utf-8")

    compiler = BeliefCompiler(SimpleNamespace())
    result = compiler.compile_paths(str(project), ["../outside.py"])
    assert result.files_processed == 0
    assert result.beliefs_written == 0


def test_daemon_reindex_restores_cwd_after_failure(tmp_path, monkeypatch):
    start = tmp_path / "start"
    start.mkdir()
    monkeypatch.chdir(start)

    daemon = EntrolyDaemon.__new__(EntrolyDaemon)
    daemon.state = SimpleNamespace(repos=[SimpleNamespace(path=str(tmp_path / "missing"))])
    daemon._engine = object()
    daemon.reindex_repo()

    assert Path.cwd() == start


def test_daemon_reindex_rejects_unregistered_repo(tmp_path, monkeypatch):
    start = tmp_path / "start"
    start.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(start)

    daemon = EntrolyDaemon.__new__(EntrolyDaemon)
    daemon.state = SimpleNamespace(repos=[])
    daemon._engine = object()

    assert daemon.reindex_repo(str(outside)) is False
    assert Path.cwd() == start


def test_daemon_startup_index_restores_cwd_after_failure(tmp_path, monkeypatch):
    start = tmp_path / "start"
    start.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(start)

    def fail_index(_engine):
        raise RuntimeError("index failure")

    monkeypatch.setattr("entroly.auto_index.auto_index", fail_index)
    daemon = EntrolyDaemon.__new__(EntrolyDaemon)
    daemon.state = SimpleNamespace(repos=[])
    daemon._repo_paths = [str(repo)]
    daemon._engine = SimpleNamespace(optimize_context=lambda **_kwargs: None)
    daemon._index_repos()

    assert Path.cwd() == start


def test_daemon_federation_state_requires_restart(monkeypatch):
    monkeypatch.delenv("ENTROLY_FEDERATION", raising=False)
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    assert daemon.state.federation_enabled is False
    with pytest.raises(RuntimeError, match="restart the daemon"):
        daemon.set_federation_enabled(True)

    monkeypatch.setenv("ENTROLY_FEDERATION", "1")
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    assert daemon.state.federation_enabled is True
    with pytest.raises(RuntimeError, match="restart the daemon"):
        daemon.set_federation_enabled(False)


def test_python_executor_rejects_resource_heavy_expressions():
    executor = PythonExecutor()
    assert executor.execute("9**1000000").succeeded is False
    assert executor.execute("'x' * 1000000").succeeded is False


def test_sympy_executor_rejects_dynamic_python():
    executor = SymPyExecutor()
    if not executor.available:
        pytest.skip("sympy not installed")
    assert executor.execute("__import__('os').system('echo unsafe')").succeeded is False
