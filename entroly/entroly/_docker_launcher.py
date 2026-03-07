"""
entroly Docker launcher — cross-platform entry point.

When installed via `pip install entroly`, this is what runs.
It launches the actual MCP server inside a Docker container so it
works identically on Linux, macOS, and Windows without needing Rust.

The Docker image is built from Dockerfile.entroly and pushed to:
  ghcr.io/juyterman1000/entroly:latest

MCP stdio protocol is passed through transparently via stdin/stdout.
"""

from __future__ import annotations

import os
import subprocess
import sys


DOCKER_IMAGE = "ghcr.io/juyterman1000/entroly:latest"


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _pull_image() -> None:
    """Pull (or update) the entroly Docker image silently."""
    subprocess.run(
        ["docker", "pull", DOCKER_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,  # don't crash if offline and image is cached
    )


def _run_native() -> None:
    """Fall back to running local Python server (when inside Docker)."""
    from entroly.server import main  # noqa: PLC0415
    main()


def launch() -> None:
    """Main entry point — docker launch or native fallback."""

    # If already inside Docker (or user explicitly opts out), go native
    if os.environ.get("ENTROLY_NO_DOCKER") or os.path.exists("/.dockerenv"):
        _run_native()
        return

    # Check Docker is installed and running
    if not _docker_available():
        print(
            "[entroly] Docker is not running. "
            "Please start Docker Desktop (or the Docker daemon) and try again.\n"
            "Alternatively, set ENTROLY_NO_DOCKER=1 to run without Docker "
            "(requires entroly-core Rust wheel for your platform).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pull latest image (no-op if already cached; silent on failure)
    _pull_image()

    # Launch MCP server via Docker, binding stdio
    # --rm        → auto-remove container when done
    # -i          → keep stdin open (required for MCP stdio protocol)
    # --network=host on Linux for best performance; bridge on Mac/Win
    cmd = [
        "docker", "run", "--rm", "-i",
        # Pass through any entroly env vars prefixed with ENTROLY_
        *_env_passthrough(),
        DOCKER_IMAGE,
    ]

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


def _env_passthrough() -> list[str]:
    """Forward ENTROLY_* environment variables into the container."""
    args: list[str] = []
    for key, value in os.environ.items():
        if key.startswith("ENTROLY_"):
            args += ["-e", f"{key}={value}"]
    return args


if __name__ == "__main__":
    launch()
