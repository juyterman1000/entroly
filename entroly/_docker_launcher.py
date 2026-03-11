"""
entroly launcher — smart entry point with graceful fallback.

Priority order:
  1. If ENTROLY_NO_DOCKER=1 or inside Docker → run native Python server
  2. If entroly-core is installed locally → run native (no Docker needed)
  3. If Docker is available → pull and run the container
  4. Otherwise → show clear install instructions and exit

This ensures `pip install entroly && entroly serve` gives a useful
experience no matter what's installed.
"""

from __future__ import annotations

import os
import subprocess
import sys


DOCKER_IMAGE = "ghcr.io/juyterman1000/entroly:latest"


def _docker_available() -> bool:
    """Check if Docker is installed and the daemon is reachable."""
    try:
        # First check: can we reach the Docker daemon?
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return True

        # Fallback: Docker installed but daemon needs sudo?
        # Try 'docker version' which shows client info even without daemon.
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Client.Version}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _rust_engine_available() -> bool:
    """Check if entroly-core (Rust engine) is importable."""
    try:
        import entroly_core  # noqa: F401
        return True
    except ImportError:
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
    """Run the local Python MCP server (native mode)."""
    from entroly.server import main  # noqa: PLC0415
    main()


def launch() -> None:
    """Main entry point — tries native first, then Docker, then helpful error."""

    # If explicitly set to no-docker or already inside Docker, go native
    if os.environ.get("ENTROLY_NO_DOCKER") or os.path.exists("/.dockerenv"):
        _run_native()
        return

    # If Rust engine is installed locally, run native (best experience)
    if _rust_engine_available():
        _run_native()
        return

    # Try Docker
    if _docker_available():
        _pull_image()

        cmd = [
            "docker", "run", "--rm", "-i",
            *_env_passthrough(),
            DOCKER_IMAGE,
        ]

        try:
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        # No Rust engine, no Docker — give clear, beautiful instructions
        R = "\033[0m"     # Reset
        B = "\033[1m"     # Bold
        RED = "\033[91m"
        CYN = "\033[96m"
        GRY = "\033[90m"
        YLW = "\033[93m"

        print(f"""
  {RED}{B}✗ entroly cannot start{R}

  {GRY}The Rust engine (entroly-core) is not installed,{R}
  {GRY}and Docker is not available.{R}

  {B}Choose one:{R}

  {YLW}1.{R} Install Docker Desktop {GRY}(recommended for Mac/Windows){R}
     {GRY}Download from{R} {CYN}https://docker.com/products/docker-desktop{R}
     {GRY}Then just run:{R} {CYN}entroly serve{R}
     {GRY}(Entroly auto-detects Docker — zero build required){R}

  {YLW}2.{R} Install the prebuilt Rust engine
     {CYN}pip install entroly-core{R}

  {YLW}3.{R} Build from source
     {CYN}git clone https://github.com/juyterman1000/entroly{R}
     {CYN}cd entroly/entroly-core{R}
     {CYN}pip install maturin && maturin develop --release{R}
     {CYN}cd .. && pip install -e .{R}
""", file=sys.stderr)
        sys.exit(1)


def _env_passthrough() -> list[str]:
    """Forward ENTROLY_* environment variables into the container."""
    args: list[str] = []
    for key, value in os.environ.items():
        if key.startswith("ENTROLY_"):
            args += ["-e", f"{key}={value}"]
    return args


if __name__ == "__main__":
    launch()
