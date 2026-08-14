"""Cross-process serialization for ``entroly wrap`` proxy startup.

Two wrappers can observe a free port before either child binds it. The startup
lease makes that transition linearizable: acquire, re-check health, start or
reuse, release. OS-owned locks disappear on process crash, so no stale PID file
is trusted.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from typing import Any, Callable


class ProxyStartLockTimeout(TimeoutError):
    pass


class InterprocessProxyStartLock:
    def __init__(
        self,
        path: Path,
        *,
        timeout_s: float = 35.0,
        poll_s: float = 0.05,
    ) -> None:
        if timeout_s <= 0 or poll_s <= 0:
            raise ValueError("startup lock timing must be positive")
        self.path = path
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._handle: Any = None

    def __enter__(self) -> "InterprocessProxyStartLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        self._handle = handle
        deadline = time.monotonic() + self.timeout_s
        delay = min(self.poll_s, 0.05)

        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    return self
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        handle.close()
                        self._handle = None
                        raise
                    if time.monotonic() >= deadline:
                        handle.close()
                        self._handle = None
                        raise ProxyStartLockTimeout(
                            f"timed out acquiring proxy startup lease {self.path}"
                        ) from exc
                    time.sleep(delay)
                    delay = min(0.25, delay * 1.5)

        import fcntl

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    handle.close()
                    self._handle = None
                    raise ProxyStartLockTimeout(
                        f"timed out acquiring proxy startup lease {self.path}"
                    ) from exc
                time.sleep(delay)
                delay = min(0.25, delay * 1.5)

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def proxy_start_lock_path(runtime_dir: Path, port: int) -> Path:
    value = int(port)
    if not 1 <= value <= 65535:
        raise ValueError("proxy port must be between 1 and 65535")
    return runtime_dir / "locks" / f"proxy-start-{value}.lock"


def serialized_proxy_start(
    *,
    port: int,
    runtime_dir: Path,
    is_running: Callable[[int], bool],
    start: Callable[[int], bool],
    timeout_s: float = 35.0,
) -> tuple[bool, str]:
    """Perform exactly one startup decision under a per-port lease."""
    lock_path = proxy_start_lock_path(runtime_dir, port)
    with InterprocessProxyStartLock(lock_path, timeout_s=timeout_s):
        if is_running(port):
            return True, "reused"
        return bool(start(port)), "started"


def _start_assured_cli_proxy(cli: Any, port: int) -> bool:
    """Mirror the existing CLI startup UX through the assurance bootstrap."""
    import subprocess
    import sys

    if cli._is_entroly_proxy_running(port):
        print(
            f"  {cli.C.GREEN}Proxy already running at "
            f"http://localhost:{port}{cli.C.RESET}"
        )
        return True
    if not cli._free_port(port):
        print(
            f"  {cli.C.RED}Port {port} is occupied by a non-Entroly service."
            f"{cli.C.RESET}"
        )
        print(
            f"  {cli.C.GRAY}Choose another port with: "
            f"entroly wrap <agent> --port <other-port>{cli.C.RESET}\n"
        )
        return False

    print(f"  {cli.C.GRAY}Starting proxy on port {port}...{cli.C.RESET}")
    command = [sys.executable, "-m", "entroly.proxy_cli_entry", "--port", str(port)]
    log_path: Path | None = cli._ENTROLY_DIR / f"wrap-proxy-{port}.log"
    try:
        cli._ENTROLY_DIR.mkdir(parents=True, exist_ok=True)
        log_handle: Any = log_path.open("wb")
    except OSError:
        log_handle = subprocess.DEVNULL
        log_path = None

    try:
        if hasattr(log_handle, "write"):
            log_handle.write(
                f"\n--- entroly wrap proxy start port={port} ---\n".encode("utf-8")
            )
            log_handle.flush()
        try:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            print(f"  {cli.C.RED}Could not start proxy process:{cli.C.RESET} {exc}")
            if log_path is not None:
                print(f"  {cli.C.GRAY}Proxy log: {log_path}{cli.C.RESET}")
            return False
    finally:
        if hasattr(log_handle, "close"):
            log_handle.close()

    for _ in range(150):
        time.sleep(0.2)
        if cli._is_entroly_proxy_running(port):
            print(
                f"  {cli.C.GREEN}Proxy running at "
                f"http://localhost:{port}{cli.C.RESET}"
            )
            return True
        if process.poll() is not None:
            print(
                f"  {cli.C.RED}Proxy exited before becoming healthy "
                f"(exit {process.returncode}).{cli.C.RESET}"
            )
            if log_path is not None:
                print(f"  {cli.C.GRAY}Proxy log: {log_path}{cli.C.RESET}")
                tail = cli._tail_text_file(log_path)
                if tail:
                    print(f"\n{tail}\n")
            return False

    print(
        f"  {cli.C.RED}Proxy failed to start on port {port} within 30s."
        f"{cli.C.RESET}"
    )
    if log_path is not None:
        print(f"  {cli.C.GRAY}Proxy log: {log_path}{cli.C.RESET}")
        tail = cli._tail_text_file(log_path)
        if tail:
            print(f"\n{tail}\n")
    cli._stop_process_best_effort(process)
    return False


def install_cli_startup_serialization() -> None:
    """Install serialized wrap startup without changing the CLI contract."""
    from . import cli

    current = cli._start_proxy_if_needed
    if hasattr(current, "__entroly_serialized_start_original__"):
        return

    def serialized_start(port: int) -> bool:
        try:
            ok, outcome = serialized_proxy_start(
                port=port,
                runtime_dir=cli._ENTROLY_DIR,
                is_running=cli._is_entroly_proxy_running,
                start=lambda selected: _start_assured_cli_proxy(cli, selected),
            )
        except ProxyStartLockTimeout as exc:
            print(
                f"  {cli.C.RED}Proxy startup lease timed out for port {port}."
                f"{cli.C.RESET}"
            )
            print(f"  {cli.C.GRAY}{exc}{cli.C.RESET}")
            return False
        if outcome == "reused":
            print(
                f"  {cli.C.GREEN}Proxy became ready while waiting; reusing "
                f"http://localhost:{port}{cli.C.RESET}"
            )
        return ok

    serialized_start.__entroly_serialized_start_original__ = current
    cli._start_proxy_if_needed = serialized_start


__all__ = [
    "InterprocessProxyStartLock",
    "ProxyStartLockTimeout",
    "install_cli_startup_serialization",
    "proxy_start_lock_path",
    "serialized_proxy_start",
]
