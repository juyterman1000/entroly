"""Sandbox untrusted generated Python code during actual model-task grading.

Never execute model-generated source on the GitHub Actions host. This module
assembles a minimal synthetic task and invokes pytest inside a networkless,
privilege-dropped Docker container with only a read-only scratch mount.
The Docker test-oracle image is built BEFORE any model output is generated.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.agentic_tasks_run import Task

IMAGE = "entroly-dogfood-oracle:py312"


def docker_command(project: Path, test_file: str, image: str = IMAGE) -> list[str]:
    if not project.is_absolute():
        raise ValueError("sandbox project path must be absolute")
    # A Windows root-relative path such as /etc/shadow has an anchor but
    # Path.is_absolute() can be false when no drive letter is supplied.
    if Path(test_file).anchor or ".." in Path(test_file).parts:
        raise ValueError("sandbox test path must stay inside task workspace")
    return [
        "docker", "run", "--rm", "--pull=never",
        "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--read-only", "--tmpfs=/tmp:rw,nosuid,nodev,size=32m",
        "--pids-limit=64", "--memory=512m", "--cpus=1",
        "--user=65534:65534", "--workdir=/task",
        f"--mount=type=bind,src={project},dst=/task,readonly",
        "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "--env=PYTHONPATH=/task", image,
        "python", "-m", "pytest", test_file,
        "-q", "--no-header", "-p", "no:cacheprovider",
    ]


def run_sandboxed_oracle(task: Task, patched_source: str, *, timeout: float = 120) -> tuple[bool, str]:
    if not isinstance(patched_source, str) or len(patched_source) > 1_000_000:
        raise ValueError("model patch must be bounded text")
    with tempfile.TemporaryDirectory(prefix="entroly-untrusted-oracle-") as tmp:
        root = Path(tmp).resolve()
        # TemporaryDirectory defaults to 0700, but the nonroot container
        # needs to traverse the scratch directory: only synthetic code lives
        # here, never git credentials or the original checkout.
        root.chmod(0o755)
        files = {task.target_file: patched_source, task.test_file: task.test_source}
        for name, body in task.distractors.items():
            if name in files or name == task.dependency_file:
                raise ValueError("distractor cannot override target, test or dependency")
            files[name] = body
        if task.dependency_file:
            if task.dependency_file in files:
                raise ValueError("dependency overlaps target or test")
            files[task.dependency_file] = task.dependency_source
        for name, body in files.items():
            path = root / name
            if (Path(name).anchor or ".." in Path(name).parts
                    or not path.resolve().is_relative_to(root)):
                raise ValueError("task file escapes the isolated project")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            path.chmod(0o644)
            path.parent.chmod(0o755)
            # Ensure Python can import packages from the scratch workspace.
            for ancestor in path.parent.relative_to(root).parents:
                if ancestor != Path("."):
                    (root / ancestor / "__init__.py").touch(exist_ok=True)
            if path.parent != root:
                (path.parent / "__init__.py").touch(exist_ok=True)
        try:
            completed = subprocess.run(
                docker_command(root, task.test_file), cwd=root,
                capture_output=True, text=True, timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("sandbox oracle is unavailable or timed out") from exc
        if completed.returncode in (125, 126, 127):
            raise RuntimeError(f"sandbox infrastructure failed: {completed.stderr[-300:]}")
        return completed.returncode == 0, (completed.stdout + "\n" + completed.stderr)[-600:]
