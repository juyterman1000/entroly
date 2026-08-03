from __future__ import annotations

from pathlib import Path

from entroly.repository_intelligence.cli import run


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


def test_summary_is_bounded_and_versioned(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(["--root", str(tmp_path), "summary"])
    assert code == 0
    assert payload["schema_version"] == "entroly.repository-cli.v1"
    assert payload["files"] == 3
    assert payload["tests"] == 1
    assert "symbols" in payload and "call_edges" in payload
    assert "files_payload" not in payload


def test_impact_reports_reverse_dependencies(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(
        ["--root", str(tmp_path), "impact", "--changed", "pkg/source.py"]
    )
    assert code == 0
    assert payload["report"]["impacted_paths"] == [
        "pkg/api.py",
        "pkg/source.py",
        "tests/test_api.py",
    ]


def test_tests_ranks_directly_related_test(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(
        ["--root", str(tmp_path), "tests", "--changed", "pkg/source.py"]
    )
    assert code == 0
    assert payload["candidates"][0]["path"] == "tests/test_api.py"


def test_unknown_changed_path_fails_visibly(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(
        ["--root", str(tmp_path), "impact", "--changed", "missing.py"]
    )
    assert code == 2
    assert payload["error"] == "unknown_changed_paths"
    assert payload["unknown"] == ["missing.py"]


def test_invalid_root_is_machine_readable(tmp_path: Path) -> None:
    code, payload = run(["--root", str(tmp_path / "absent"), "summary"])
    assert code == 2
    assert payload["error"] == "invalid_repository"
