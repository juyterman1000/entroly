"""Check model-generated patch executor is isolated from CI host."""
from pathlib import Path

import pytest

from benchmarks.intensive_oracle_sandbox import IMAGE, docker_command


def test_oracle_docker_is_networkless_nonroot_readonly() -> None:
    cmd = docker_command(Path("/tmp/entroly-scratch"), "test_task.py")
    assert cmd[:3] == ["docker", "run", "--rm"]
    for mandatory in (
        "--pull=never", "--network=none", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--read-only",
        "--pids-limit=64", "--user=65534:65534", "--workdir=/task",
    ):
        assert mandatory in cmd
    assert "--mount=type=bind,src=/tmp/entroly-scratch,dst=/task,readonly" in cmd
    assert IMAGE in cmd
    assert cmd[-4:] == ["-q", "--no-header", "-p", "no:cacheprovider"]
    assert "GITHUB_TOKEN" not in " ".join(cmd)


@pytest.mark.parametrize("bad", ["../secret.py", "/etc/shadow", "a/../../credentials"])
def test_oracle_refuses_path_traversal(bad) -> None:
    with pytest.raises(ValueError, match="stay inside"):
        docker_command(Path("/tmp/entroly-scratch"), bad)


def test_oracle_requires_absolute_scratch_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        docker_command(Path("scratch"), "test_task.py")
