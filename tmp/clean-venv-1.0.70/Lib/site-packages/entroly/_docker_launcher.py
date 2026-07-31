"""
entroly Docker launcher — cross-platform entry point.

When installed via `pip install entroly`, this is what runs.
CLI commands run in the installed Python environment. Only ``entroly serve``
uses the Docker image by default so ordinary commands never trigger a remote
image pull or require a Docker daemon.

The Docker image is built from Dockerfile.entroly and pushed to versioned tags
under ``ghcr.io/juyterman1000/entroly``. The launcher defaults to the tag that
matches the installed Python package. Operators may explicitly override it with
``ENTROLY_DOCKER_IMAGE``.

MCP stdio protocol is passed through transparently via stdin/stdout.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from entroly import __version__

DEFAULT_DOCKER_IMAGE = f"ghcr.io/juyterman1000/entroly:{__version__}"
_DOCKER_IMAGE_RE = re.compile(
    r"^(?:[a-zA-Z0-9.-]+(?::[0-9]+)?/)?"
    r"[a-zA-Z0-9._/-]+"
    r"(?::[a-zA-Z0-9._-]+|@sha256:[a-fA-F0-9]{64})?$"
)


def _docker_image() -> str:
    """Return a validated explicit image override or the installed-version image.

    A package installed as ``entroly==X`` must not silently execute whatever
    code happens to be under the mutable ``latest`` tag. Reject values that can
    be interpreted as Docker CLI options or contain whitespace/control bytes.
    """
    override = os.environ.get("ENTROLY_DOCKER_IMAGE", "").strip()
    image = override or DEFAULT_DOCKER_IMAGE
    if (
        not image
        or image.startswith("-")
        or any(char.isspace() or ord(char) < 32 for char in image)
        or not _DOCKER_IMAGE_RE.fullmatch(image)
    ):
        raise RuntimeError(
            "ENTROLY_DOCKER_IMAGE is not a valid Docker image reference: "
            f"{image!r}"
        )
    return image


# TTL-based pull caching: skip `docker pull` only when this exact image was
# pulled recently and remains present locally.
def _runtime_dir() -> Path:
    explicit = os.environ.get("ENTROLY_DIR")
    candidates = [Path(explicit).expanduser()] if explicit else [
        Path.home() / ".entroly",
        Path(tempfile.gettempdir()) / "entroly",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=candidate,
                prefix=".write_probe-",
                delete=True,
            ):
                pass
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / "entroly"


_PULL_CACHE_FILE = _runtime_dir() / ".last_pull.json"
_DEFAULT_PULL_TTL = 3600  # 1 hour


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
        )
        return True
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return False


def _stderr_text(result: Any) -> str:
    value = getattr(result, "stderr", "")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _image_exists_locally(image: str) -> bool:
    """Return whether Docker has the exact selected reference locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _read_pull_cache() -> dict[str, Any] | None:
    """Read the image-scoped pull cache; legacy/corrupt state is a cache miss."""
    try:
        payload = json.loads(_PULL_CACHE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    image = payload.get("image")
    timestamp = payload.get("timestamp")
    if not isinstance(image, str) or not isinstance(timestamp, (int, float)):
        return None
    return {"image": image, "timestamp": float(timestamp)}


def _write_pull_cache(image: str) -> None:
    """Atomically record a successful pull for one exact image reference."""
    try:
        _PULL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            dir=_PULL_CACHE_FILE.parent,
            prefix=f".{_PULL_CACHE_FILE.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"image": image, "timestamp": time.time()},
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, _PULL_CACHE_FILE)
        finally:
            tmp_path.unlink(missing_ok=True)
    except OSError:
        # Cache persistence is an optimization. Pull success itself remains the
        # trust boundary, so a read-only HOME must not make serving impossible.
        return


def _should_pull(image: str | None = None) -> bool:
    """Return whether the exact selected image needs a fresh pull."""
    selected = image or _docker_image()
    ttl = _env_int("ENTROLY_PULL_TTL", _DEFAULT_PULL_TTL)
    if ttl <= 0:
        return True  # TTL=0 means always pull
    cached = _read_pull_cache()
    if cached is None or cached["image"] != selected:
        return True
    age = time.time() - cached["timestamp"]
    if age < 0 or age >= ttl:
        return True
    # Cache files can outlive Docker cleanup. Verify the exact image still exists.
    return not _image_exists_locally(selected)


def _pull_image() -> None:
    """Pull the selected image, or prove the exact local image exists.

    A pull failure is not silently deferred to ``docker run``. After bounded
    retries, offline execution is allowed only when Docker can inspect the exact
    selected reference locally; otherwise an actionable RuntimeError is raised.
    """
    image = _docker_image()
    if not _should_pull(image):
        return

    errors: list[str] = []
    for attempt in range(2):
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
            if result.returncode == 0:
                _write_pull_cache(image)
                return
            detail = _stderr_text(result).strip()
            errors.append(
                f"attempt {attempt + 1}: exit {result.returncode}"
                + (f" ({detail[-500:]})" if detail else "")
            )
        except subprocess.TimeoutExpired:
            errors.append(f"attempt {attempt + 1}: timed out after 60 seconds")
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
            break
        if attempt < 1:
            time.sleep(3)

    if _image_exists_locally(image):
        # The exact version is present, so an offline user can continue without
        # silently crossing into another tag. Do not refresh the pull timestamp:
        # the next launch should retry the registry rather than masking outage.
        print(
            f"[entroly] Could not refresh {image}; using the exact local image.",
            file=sys.stderr,
        )
        return

    detail = "; ".join(errors) or "unknown Docker pull failure"
    raise RuntimeError(
        f"Unable to pull Docker image {image!r}, and that exact image is not "
        f"available locally. {detail}"
    )


def _run_native() -> None:
    """Fall back to running local Python server (when inside Docker)."""
    from entroly.server import main  # noqa: PLC0415

    try:
        main()
    except RuntimeError as exc:
        print(f"[entroly] {exc}", file=sys.stderr)
        sys.exit(1)


def _run_memory_cli() -> None:
    """Route `entroly memory ...` to the local MemoryOS CLI without Docker."""
    from entroly.memory_cli import main as memory_main  # noqa: PLC0415

    # `memory_cli` expects argv like `remember ...`, not `memory remember ...`.
    rc = memory_main(sys.argv[2:])
    sys.exit(int(rc or 0))


def launch() -> None:
    """Main entry point — docker launch or native fallback.

    Route every CLI subcommand locally. Only ``serve`` goes through Docker (or
    the native server fallback when ``ENTROLY_NO_DOCKER`` is set).
    """

    # Bare `entroly` is the documented MCP command for Claude Code and other
    # stdio clients. In a terminal it remains friendly help; under an MCP host
    # stdin is a pipe, so start the server and keep stdout protocol-clean.
    if len(sys.argv) <= 1:
        if sys.stdin.isatty():
            from entroly.cli import main as cli_main

            cli_main()
            return
        _run_native()
        return

    # --help/--version → show help without Docker
    _help_flags = {"--help", "-h", "--version", "-V"}
    if len(sys.argv) > 1 and sys.argv[1] in _help_flags:
        from entroly.cli import main as cli_main

        cli_main()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "memory":
        _run_memory_cli()
        return

    # The canonical CLI owns command discovery and validation. Routing every
    # non-server token through it prevents newly added commands (and typos)
    # from accidentally becoming Docker invocations.
    if sys.argv[1] != "serve":
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

    # Pull the image matching the installed package (with TTL caching + retry)
    try:
        _pull_image()
    except RuntimeError as exc:
        print(f"[entroly] {exc}", file=sys.stderr)
        sys.exit(1)

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
    cmd.append(_docker_image())

    # Pass any remaining CLI args to the server
    server_args = [a for a in sys.argv[1:] if a != "--proxy"]
    if proxy_mode and "--proxy" not in server_args:
        server_args.append("--proxy")
    cmd += server_args

    # Configurable timeout for Docker run (default: None = no timeout for server)
    docker_timeout_seconds = _env_int("ENTROLY_DOCKER_TIMEOUT", 0)
    docker_timeout = docker_timeout_seconds if docker_timeout_seconds > 0 else None

    try:
        result = subprocess.run(cmd, check=False, timeout=docker_timeout)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("[entroly] Docker container timed out.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


def _env_passthrough() -> list[str]:
    """Forward ENTROLY_* names without exposing their values in process argv.

    Docker's ``-e NAME`` form copies the value from the inherited host
    environment. Keeping ``NAME=value`` out of the command line prevents API
    keys and tokens from appearing in process listings, CI diagnostics, or
    crash reports. Sorting also makes the generated command deterministic.
    """
    args: list[str] = []
    for key in sorted(os.environ):
        if key.startswith("ENTROLY_"):
            args += ["-e", key]
    return args


if __name__ == "__main__":
    launch()
