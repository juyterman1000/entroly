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
import time
from pathlib import Path

DOCKER_IMAGE = "ghcr.io/juyterman1000/entroly:latest"

# TTL-based pull caching: skip `docker pull` if last pull was recent.
_PULL_CACHE_FILE = Path.home() / ".entroly" / ".last_pull_ts"
_DEFAULT_PULL_TTL = 3600  # 1 hour


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


def _should_pull() -> bool:
    """Check if enough time has elapsed since the last successful pull."""
    ttl = int(os.environ.get("ENTROLY_PULL_TTL", str(_DEFAULT_PULL_TTL)))
    if ttl <= 0:
        return True  # TTL=0 means always pull
    try:
        if _PULL_CACHE_FILE.exists():
            last_pull = float(_PULL_CACHE_FILE.read_text().strip())
            if time.time() - last_pull < ttl:
                return False
    except (ValueError, OSError):
        pass
    return True


def _pull_image() -> None:
    """Pull (or update) the entroly Docker image with TTL caching and retry."""
    if not _should_pull():
        return

    for attempt in range(2):
        try:
            result = subprocess.run(
                ["docker", "pull", DOCKER_IMAGE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
            if result.returncode == 0:
                try:
                    _PULL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    _PULL_CACHE_FILE.write_text(str(time.time()))
                except OSError:
                    pass
                return
        except subprocess.TimeoutExpired:
            pass
        if attempt < 1:
            time.sleep(3)


def _run_native() -> None:
    """Fall back to running local Python server (when inside Docker)."""
    from entroly.server import main  # noqa: PLC0415
    main()


def launch() -> None:
    """Main entry point — docker launch or native fallback.

    Routes CLI subcommands (init, dashboard, health, autotune, benchmark,
    status, proxy) to the local CLI handler. Only `serve` and bare `entroly`
    go through Docker (or native fallback).
    """

    # Bare command or --help/--version → show help without Docker
    _help_flags = {"--help", "-h", "--version", "-V"}
    if len(sys.argv) <= 1 or (len(sys.argv) > 1 and sys.argv[1] in _help_flags):
        from entroly.cli import main as cli_main
        cli_main()
        return

    # CLI subcommands that don't need Docker or the MCP server
    _local_commands = {
        "init", "dashboard", "health", "autotune", "benchmark",
        "status", "config", "proxy", "completions", "clean", "telemetry",
        "export", "import", "drift", "profile", "batch",
        "demo", "doctor", "digest", "migrate", "role",
    }
    if len(sys.argv) > 1 and sys.argv[1] in _local_commands:
        from entroly.cli import main as cli_main
        cli_main()
        return

    # If already inside Docker (or user explicitly opts out), go native
    if os.environ.get("ENTROLY_NO_DOCKER") or os.path.exists("/.dockerenv"):
        _run_native()
        return

    # Check Docker is installed and running
    if not _docker_available():
        print(
            "[entroly] Docker is not running.\n"
            "\n"
            "Options:\n"
            "  1. Start Docker Desktop (or the Docker daemon) and try again\n"
            "  2. Install the native Rust engine:\n"
            "       pip install entroly[native]\n"
            "     Then run with:\n"
            "       ENTROLY_NO_DOCKER=1 entroly serve\n"
            "  3. Use the Python fallback engine (slower):\n"
            "       ENTROLY_NO_DOCKER=1 entroly serve\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pull latest image (with TTL caching and retry)
    _pull_image()

    # Detect proxy mode
    proxy_mode = "--proxy" in sys.argv or os.environ.get("ENTROLY_PROXY") == "1"
    port = os.environ.get("ENTROLY_PROXY_PORT", "9377")

    # Build docker run command
    cmd = ["docker", "run", "--rm"]

    if proxy_mode:
        # Proxy mode: expose port, no stdin needed
        cmd += ["-p", f"{port}:9377"]
        # Use host networking on Linux for best latency
        if sys.platform == "linux":
            cmd += ["--network=host"]
    else:
        # MCP mode: keep stdin open for stdio protocol
        cmd.append("-i")

    # Pass through ENTROLY_* env vars
    cmd += _env_passthrough()
    cmd.append(DOCKER_IMAGE)

    # Pass any remaining CLI args to the server
    server_args = [a for a in sys.argv[1:] if a != "--proxy"]
    if proxy_mode and "--proxy" not in server_args:
        server_args.append("--proxy")
    cmd += server_args

    # Configurable timeout for Docker run (default: None = no timeout for server)
    docker_timeout_str = os.environ.get("ENTROLY_DOCKER_TIMEOUT", "0")
    docker_timeout = int(docker_timeout_str) if docker_timeout_str != "0" else None

    try:
        result = subprocess.run(cmd, check=False, timeout=docker_timeout)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("[entroly] Docker container timed out.", file=sys.stderr)
        sys.exit(1)
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
