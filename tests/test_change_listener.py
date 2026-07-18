from entroly.belief_compiler import BeliefCompiler
from entroly.change_listener import WorkspaceChangeListener
from entroly.change_pipeline import ChangePipeline
from entroly.verification_engine import VerificationEngine
from entroly.vault import VaultConfig, VaultManager


def test_workspace_change_listener_syncs_changed_file(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    src = project / "auth.py"
    src.write_text(
        """class AuthService:
    def verify_token(self, token: str) -> bool:
        return bool(token)
""",
        encoding="utf-8",
    )

    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    compiler = BeliefCompiler(vault)
    verifier = VerificationEngine(vault)
    change_pipe = ChangePipeline(vault, verifier)
    listener = WorkspaceChangeListener(vault, compiler, verifier, change_pipe, str(project))

    first = listener.scan_once(force=True)
    assert first.status == "synced"
    assert first.beliefs_written >= 1
    assert first.action_path
    assert vault.read_belief("auth") is not None

    src.write_text(
        """class AuthService:
    def verify_token(self, token: str) -> bool:
        return bool(token)

    def rotate_keys(self) -> None:
        return None
""",
        encoding="utf-8",
    )

    second = listener.scan_once()
    assert second.status == "synced"
    assert "auth.py" in second.changed_files
    assert second.beliefs_written >= 1
    assert second.verification_summary["total_beliefs_checked"] >= 1


def test_workspace_change_listener_drains_backlog_without_losing_changes(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (project / f"{name}.py").write_text(
            f"class {name.title()}Service:\n    def run(self): return '{name}'\n",
            encoding="utf-8",
        )

    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    compiler = BeliefCompiler(vault)
    verifier = VerificationEngine(vault)
    listener = WorkspaceChangeListener(
        vault,
        compiler,
        verifier,
        ChangePipeline(vault, verifier),
        str(project),
    )

    batches = [listener.scan_once(force=True, max_files=1)]
    batches.extend(listener.scan_once(max_files=1) for _ in range(2))

    assert [len(batch.changed_files) for batch in batches] == [1, 1, 1]
    assert sum(batch.beliefs_written for batch in batches) == 3
    assert listener.scan_once(max_files=1).status == "no_changes"


def test_workspace_change_listener_retries_transient_compile_failure(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "auth.py").write_text(
        "class AuthService:\n    def verify(self): return True\n",
        encoding="utf-8",
    )

    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    compiler = BeliefCompiler(vault)
    verifier = VerificationEngine(vault)
    listener = WorkspaceChangeListener(
        vault,
        compiler,
        verifier,
        ChangePipeline(vault, verifier),
        str(project),
    )
    real_compile = compiler.compile_file
    attempts = 0

    def flaky_compile(file_path, content):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient compiler failure")
        return real_compile(file_path, content)

    monkeypatch.setattr(compiler, "compile_file", flaky_compile)

    first = listener.scan_once(force=True)
    second = listener.scan_once()

    assert first.errors
    assert second.changed_files == ["auth.py"]
    assert second.beliefs_written == 1
