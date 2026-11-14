"""Regression controls for research-runner security and fixed task eligibility."""
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.real_commit_pilot import _check_patch_paths, _oracle, _local_model_digest


@pytest.mark.parametrize("line", [
    "new file mode 120000", "old mode 100644", "new mode 120000",
    "rename from a.py", "rename to b.py", "copy from a.py", "GIT binary patch",
])
def test_untrusted_diff_cannot_change_modes_or_link_to_credentials(line):
    patch = (
        "diff --git a/solver.py b/solver.py\n"
        f"{line}\n"
        "--- a/solver.py\n+++ b/solver.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    with pytest.raises(ValueError, match="unsupported"):
        _check_patch_paths(patch, ["solver.py"])


def test_model_url_rejects_userinfo_and_external_redirects():
    for url in ("http://127.0.0.1:11434@attacker.invalid:80",
                "http://attacker.invalid:11434",
                "https://127.0.0.1:11434",
                "http://127.0.0.1:11434/evil"):
        with pytest.raises(ValueError, match="local Ollama"):
            _local_model_digest(url, "model")


def test_oracle_requires_a_networkless_readonly_docker_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")
    captured = []

    class FakeResult:
        returncode = 0
        stdout = "1 passed"
        stderr = ""

    def run(command, **options):
        captured.append((command, options))
        return FakeResult()

    with patch("subprocess.run", side_effect=run):
        status, _ = _oracle(tmp_path, {"test_files": ["test_safe.py"]}, 5)
    assert status == "passed"
    command, options = captured[0]
    assert command[:3] == ["docker", "run", "--rm"]
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL",
                 "--security-opt=no-new-privileges", "--user=65534:65534"):
        assert flag in command
    assert f"--mount=type=bind,src={tmp_path},dst=/task,readonly" in command
    assert options["timeout"] == 5
    assert "--env=PYTHONPATH=/task" in command


def test_sandbox_infrastructure_errors_cannot_be_scored_as_wrong_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")

    class FakeResult:
        returncode = 125
        stdout = ""
        stderr = "docker container cannot start"

    with patch("subprocess.run", return_value=FakeResult()):
        status, _ = _oracle(tmp_path, {"test_files": ["test_safe.py"]}, 5)
    assert status == "infrastructure_error"


def test_unsafe_test_path_rejected_before_container_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")
    with pytest.raises(ValueError, match="invalid test path"):
        _oracle(tmp_path, {"test_files": ["../secrets.py"]}, 5)
