from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from entroly.context_check import (
    CONTEXT_CHECK_SCHEMA,
    assess_context_commit,
    context_check_markdown,
    create_context_check_from_path,
    git_changed_files,
)
from entroly.context_commit import create_context_commit, verify_context_commit

DOCUMENTS = [
    ("src/handler.py", "def handle_timeout(request):\n    return request.retry_after\n"),
    ("src/auth.py", "def rotate_secret(user):\n    return user.rotate_credentials()\n"),
    ("tests/test_handler.py", "def test_timeout():\n    assert handle_timeout(fake_request()) == 30\n"),
]


def _commit():
    return create_context_commit(DOCUMENTS, query="handle_timeout request retry_after", token_budget=18, prefer_rust=False)


def test_context_check_measures_changed_file_coverage():
    check = assess_context_commit(_commit(), changed_files=["src/handler.py", "src/auth.py"])
    assert check["schema_version"] == CONTEXT_CHECK_SCHEMA
    assert check["check_id"].startswith("cck_")
    assert check["comparison"]["status"] == "measured"
    assert "src/handler.py" in check["comparison"]["changed_files_selected"]
    assert check["metrics"]["changed_file_recall"] < 1.0
    assert check["risk"]["level"] == "high"
    assert any(flag["code"] == "sensitive_changed_files_not_selected" for flag in check["risk"]["flags"])


def test_context_check_is_content_addressed_and_deterministic():
    first = assess_context_commit(_commit(), changed_files=["src/auth.py", "src/handler.py"])
    second = assess_context_commit(_commit(), changed_files=["src/handler.py", "src/auth.py", "src/auth.py"])
    assert first == second


def test_missing_comparison_is_unknown_not_false_success():
    check = assess_context_commit(_commit())
    assert check["comparison"]["status"] == "not_measured"
    assert check["metrics"]["changed_file_recall"] is None
    assert check["risk"]["level"] == "unknown"


def test_empty_change_set_is_measured_full_recall():
    check = assess_context_commit(_commit(), changed_files=[])
    assert check["comparison"]["status"] == "measured"
    assert check["metrics"]["changed_file_recall"] == 1.0


def test_unindexed_changed_file_is_high_risk():
    check = assess_context_commit(_commit(), changed_files=["generated/unknown.bin"])
    assert check["risk"]["level"] == "high"
    assert check["comparison"]["changed_files_not_indexed"] == ["generated/unknown.bin"]


def test_truncated_corpus_is_explicit():
    check = assess_context_commit(_commit(), changed_files=[], corpus_total_files=10, corpus_included_files=3)
    assert any(flag["code"] == "corpus_truncated" for flag in check["risk"]["flags"])


def test_markdown_is_bounded_and_contains_key_evidence():
    report = context_check_markdown(assess_context_commit(_commit(), changed_files=["src/auth.py"]))
    assert "# Entroly Context Check" in report
    assert "Changed-file recall" in report
    assert "sensitive_changed_files_not_selected" in report
    assert len(report) < 20_000


def test_create_from_path_writes_replayable_commit(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "handler.py").write_text(DOCUMENTS[0][1], encoding="utf-8")
    (tmp_path / "src" / "auth.py").write_text(DOCUMENTS[1][1], encoding="utf-8")
    check, commit = create_context_check_from_path(
        tmp_path,
        task="handle_timeout request retry_after",
        token_budget=18,
        changed_files=["src/handler.py"],
        prefer_rust=False,
    )
    assert check["context_commit"]["commit_id"] == commit["commit_id"]
    assert verify_context_commit(commit).valid


def test_create_from_path_rejects_traversal(tmp_path: Path):
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the checked root"):
        create_context_check_from_path(tmp_path, task="main", changed_files=["../secret.py"], prefer_rust=False)


def test_git_changed_files_uses_merge_base_comparison(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "main.py"
    tracked.write_text("print('one')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    tracked.write_text("print('two')\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "change"], cwd=tmp_path, check=True, capture_output=True)
    assert git_changed_files(tmp_path, base=base) == ["main.py"]


def test_module_cli_emits_artifacts(tmp_path: Path):
    (tmp_path / "main.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "entroly.context_check", str(tmp_path), "--task", "answer", "--budget", "32", "--out", str(tmp_path / "check.json"), "--report", str(tmp_path / "check.md"), "--commit-out", str(tmp_path / "commit.json")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    check = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    assert check["risk"]["level"] == "unknown"
    assert (tmp_path / "check.md").exists()
    assert (tmp_path / "commit.json").exists()
