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
