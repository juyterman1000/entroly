"""A cached repository identity is not a current filesystem precondition."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from entroly.repository_intelligence.service import (
    RepositoryIntelligenceService,
    VerifiedRefactorError,
)
from entroly.repository_intelligence.write_authority import REPOSITORY_WRITES_ENV


OPERATIONS = ("rename", "safe-delete", "file-move")
BACKENDS = ("uncached", "cold-cache", "warm-cache")


def _service(tmp_path: Path, monkeypatch, backend: str):
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "source.py").write_text(
        "def execute(value):\n    return value + 1\n", encoding="utf-8"
    )
    (root / "observer.py").write_text(
        "def observe():\n    return 0\n", encoding="utf-8"
    )
    # Authority applies only to this test's service and disposable project.
    monkeypatch.setenv(REPOSITORY_WRITES_ENV, "1")
    kwargs = {} if backend == "uncached" else {"cache_dir": tmp_path / "cache"}
    service = RepositoryIntelligenceService(root, **kwargs)
    if backend == "warm-cache":
        service.summary()
        service.refresh()
    return root, service


def _preview(service: RepositoryIntelligenceService, operation: str):
    if operation == "rename":
        plan = service.rename_preview("execute", "perform")
    elif operation == "safe-delete":
        plan = service.safe_delete_preview("execute")
        assert plan["safe_to_apply"] is True
    else:
        plan = service.file_move_preview("pkg/source.py", "pkg/renamed.py")
        assert plan["safe_to_apply"] is True
    assert plan["changes"] or operation == "file-move"
    return plan


def _apply(service: RepositoryIntelligenceService, operation: str, plan):
    method = getattr(service, operation.replace("-", "_") + "_apply")
    return method(
        plan,
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )


def _project_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mutation", ("new-importer", "changed-nonparticipant", "removed-file"))
def test_apply_rejects_repository_changes_outside_planned_edits(
    tmp_path: Path, monkeypatch, operation: str, backend: str, mutation: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, backend)
    plan = _preview(service, operation)
    importer = "from pkg.source import execute\ndef observe():\n    return execute(1)\n"
    if mutation == "new-importer":
        (root / "late.py").write_text(importer, encoding="utf-8")
    elif mutation == "changed-nonparticipant":
        (root / "observer.py").write_text(importer, encoding="utf-8")
    else:
        (root / "observer.py").unlink()
    before = _project_bytes(root)

    with pytest.raises(VerifiedRefactorError, match="changed|stale|snapshot"):
        _apply(service, operation, plan)

    assert _project_bytes(root) == before
    assert service.summary()["index_digest"] != plan["index_digest"]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_unchanged_repository_plan_still_applies(
    tmp_path: Path, monkeypatch, operation: str, backend: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, backend)
    plan = _preview(service, operation)
    observer = (root / "observer.py").read_bytes()

    result = _apply(service, operation, plan)

    assert result["refresh"]["status"] == "refreshed"
    assert (root / "observer.py").read_bytes() == observer
    if operation == "safe-delete":
        assert "def execute" not in (root / "pkg" / "source.py").read_text(encoding="utf-8")
    else:
        target = "renamed.py" if operation == "file-move" else "source.py"
        source = root / "pkg" / target
        namespace = {}
        # Only the authored, trusted fixture is executed, never repository code.
        exec(compile(source.read_bytes(), str(source), "exec"), namespace)
        function = namespace["perform" if operation == "rename" else "execute"]
        assert [function(value) for value in (-2, 0, 3)] == [-1, 1, 4]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("error_type", (OSError, ValueError))
def test_snapshot_recheck_failure_never_mutates_sources(
    tmp_path: Path, monkeypatch, operation: str, error_type,
) -> None:
    root, service = _service(tmp_path, monkeypatch, "uncached")
    plan = _preview(service, operation)
    before = _project_bytes(root)

    def fail_build():
        raise error_type("injected preflight failure")

    monkeypatch.setattr(service, "_build", fail_build)
    with pytest.raises(VerifiedRefactorError, match="snapshot|verify"):
        _apply(service, operation, plan)
    assert _project_bytes(root) == before


@pytest.mark.parametrize("operation", OPERATIONS)
def test_new_preview_recovers_new_importer_without_ignoring_its_reference(
    tmp_path: Path, monkeypatch, operation: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, "warm-cache")
    old_plan = _preview(service, operation)
    (root / "late.py").write_text(
        "from pkg.source import execute\ndef observe():\n    return execute(1)\n",
        encoding="utf-8",
    )
    with pytest.raises(VerifiedRefactorError, match="changed"):
        _apply(service, operation, old_plan)

    if operation == "safe-delete":
        current = service.safe_delete_preview("execute")
        assert current["safe_to_apply"] is False
        assert current["index_digest"] != old_plan["index_digest"]
        assert "def execute" in (root / "pkg" / "source.py").read_text(encoding="utf-8")
        return

    current = _preview(service, operation)
    assert current["index_digest"] != old_plan["index_digest"]
    assert any(change["path"] == "late.py" for change in current["changes"])
    result = _apply(service, operation, current)
    assert result["refresh"]["status"] == "refreshed"
    # A fresh, isolated child prevents cached or ambient imports from turning
    # a broken refactor into a passing behavioral check.
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", """
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import late
assert Path(late.__file__).resolve(strict=True) == root / 'late.py'
print(late.observe())
""", str(root)],
        cwd=root, capture_output=True, text=True, check=True, timeout=10,
    )
    assert not child.stderr
    assert child.stdout.strip() == "2"
