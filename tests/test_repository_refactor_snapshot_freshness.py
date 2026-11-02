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


def _current_action_preview(service: RepositoryIntelligenceService, operation: str):
    """Unlike _preview, allow safe-delete to report newly discovered blockers."""
    if operation == "rename":
        return service.rename_preview("execute", "perform")
    if operation == "safe-delete":
        return service.safe_delete_preview("execute")
    return service.file_move_preview("pkg/source.py", "pkg/renamed.py")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mutation", ("new-importer", "same-size-nonparticipant"))
@pytest.mark.parametrize("after_context_page", (False, True), ids=("direct", "after-context"))
def test_action_preview_discovers_importer_without_apply_or_refresh(
    tmp_path: Path, monkeypatch, operation: str, backend: str,
    mutation: str, after_context_page: bool,
) -> None:
    import os

    # Finish authoring the fixture before warming any service/cache snapshot.
    root, service = _service(tmp_path, monkeypatch, "uncached")
    observer = root / "observer.py"
    unrelated = b"def observe():\n    return 0\n"
    importer = b"from pkg.source import execute\ndef observe():\n    return execute(1)\n"
    observer.write_bytes(unrelated + b"#" + b"." * (255 - len(unrelated)) + b"\n")
    if after_context_page:
        (root / "pkg" / "source.py").write_bytes(
            b"def execute(value):\n" + b"    # continuation evidence\n" * 500
            + b"    return value + 1\n"
        )
    if backend != "uncached":
        service = RepositoryIntelligenceService(root, cache_dir=tmp_path / "cache")
    if backend == "warm-cache":
        service.summary()
        service.refresh()
    original = _preview(service, operation)
    assert not any(change["path"] == "observer.py" for change in original["changes"])
    if mutation == "new-importer":
        changed_path = "late.py"
        (root / changed_path).write_bytes(importer)
    else:
        changed_path = "observer.py"
        previous_stat = observer.stat()
        observer.write_bytes(importer + b"#" + b"." * (255 - len(importer)) + b"\n")
        os.utime(observer, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
        assert observer.stat().st_size == previous_stat.st_size
        assert observer.stat().st_mtime_ns == previous_stat.st_mtime_ns
    before = _project_bytes(root)
    if after_context_page:
        page = service.rename_context("execute", "perform", byte_budget=4096)
        assert page["delivery"]["next_cursor"] is not None
        assert page["plan"]["index_digest"] != original["index_digest"]

    # No failed apply and no manual refresh should be needed to discover this.
    current = _current_action_preview(service, operation)

    assert current["index_digest"] != original["index_digest"]
    assert current["resolution"] == "resolved"
    if operation == "safe-delete":
        assert current["safe_to_apply"] is False
    else:
        assert any(change["path"] == changed_path for change in current["changes"])
    assert _project_bytes(root) == before


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_unchanged_action_previews_keep_digest_and_generation(
    tmp_path: Path, monkeypatch, operation: str, backend: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, backend)
    before = _project_bytes(root)
    original = _preview(service, operation)

    for _ in range(2):
        current = _preview(service, operation)
        assert current["index_digest"] == original["index_digest"]
        assert current["generation"] == original["generation"]
        assert current["receipt"]["plan_sha256"] == original["receipt"]["plan_sha256"]
    assert _project_bytes(root) == before


@pytest.mark.parametrize("backend", BACKENDS)
def test_ordinary_summary_stays_cached_until_explicit_refresh(
    tmp_path: Path, monkeypatch, backend: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, backend)
    original = service.summary()
    (root / "late.py").write_bytes(b"from pkg.source import execute\n")

    cached = service.summary()

    assert cached["index_digest"] == original["index_digest"]
    assert cached["generation"] == original["generation"]
    assert cached["files"] == original["files"]
    refreshed = service.refresh()
    assert refreshed["index_digest"] != original["index_digest"]
    assert refreshed["files"] == original["files"] + 1
    assert service.summary()["index_digest"] == refreshed["index_digest"]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_changed_action_preview_installs_current_generation_exactly_once(
    tmp_path: Path, monkeypatch, operation: str, backend: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, backend)
    original = _preview(service, operation)
    (root / "late.py").write_bytes(b"from pkg.source import execute\n")
    assert service.summary()["index_digest"] == original["index_digest"]

    current = _current_action_preview(service, operation)

    assert current["index_digest"] != original["index_digest"]
    assert current["generation"] == original["generation"] + 1
    installed = service.summary()
    assert installed["index_digest"] == current["index_digest"]
    assert installed["generation"] == current["generation"]
    for _ in range(2):
        repeated = _current_action_preview(service, operation)
        assert repeated["index_digest"] == current["index_digest"]
        assert repeated["generation"] == current["generation"]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("error_type", (OSError, ValueError))
def test_action_preview_build_failure_rejects_and_invalidates_cached_snapshot(
    tmp_path: Path, monkeypatch, operation: str, backend: str, error_type,
) -> None:
    root, service = _service(tmp_path, monkeypatch, backend)
    original = _preview(service, operation)
    (root / "late.py").write_bytes(b"from pkg.source import execute\n")
    before = _project_bytes(root)

    def fail_build():
        raise error_type("injected preview freshness failure")

    with monkeypatch.context() as failing:
        failing.setattr(service, "_build", fail_build)
        with pytest.raises(VerifiedRefactorError, match="snapshot|verify|current"):
            _current_action_preview(service, operation)
    assert _project_bytes(root) == before
    # A plain summary must rebuild after invalidation rather than return the
    # previously cached index. No refresh/second preview can mask this failure.
    recovered = service.summary()
    assert recovered["index_digest"] != original["index_digest"]
    assert recovered["files"] == 4


@pytest.mark.parametrize("phase", ("_digest_index", "_install"))
def test_preview_verification_failure_after_build_discards_cached_snapshot(
    tmp_path: Path, monkeypatch, phase: str,
) -> None:
    root, service = _service(tmp_path, monkeypatch, "uncached")
    original = service.rename_preview("execute", "perform")
    (root / "late.py").write_bytes(b"from pkg.source import execute\n")
    before = _project_bytes(root)

    def fail_verification(*_args):
        raise ValueError("injected snapshot verification failure")

    with monkeypatch.context() as failing:
        failing.setattr(service, phase, fail_verification)
        with pytest.raises(VerifiedRefactorError, match="cannot verify current"):
            service.rename_preview("execute", "perform")
    assert _project_bytes(root) == before
    assert service.summary()["index_digest"] != original["index_digest"]


def test_lsp_preview_validates_current_inputs_before_provider_work(tmp_path: Path, monkeypatch) -> None:
    from entroly.repository_intelligence import service_impl

    root, service = _service(tmp_path, monkeypatch, "uncached")
    old = service.summary()
    (root / "late.py").write_bytes(b"from pkg.source import execute\n")

    def inspect_provider_input(_root, index, *_args, **_kwargs):
        assert "late.py" in index.files
        raise ValueError("fixture stops before any provider process")

    monkeypatch.setattr(service_impl, "collect_lsp_references", inspect_provider_input)
    with pytest.raises(VerifiedRefactorError, match="fixture stops before"):
        service.lsp_rename_preview("execute", "perform", command=(), language_id="python")
    assert service.summary()["index_digest"] != old["index_digest"]

    def fail_build():
        raise OSError("cannot establish current source")

    def forbidden_provider(*_args, **_kwargs):
        pytest.fail("failed freshness verification must not start provider work")

    monkeypatch.setattr(service, "_build", fail_build)
    monkeypatch.setattr(service_impl, "collect_lsp_references", forbidden_provider)
    with pytest.raises(VerifiedRefactorError, match="cannot verify current"):
        service.lsp_rename_preview("execute", "perform", command=(), language_id="python")
