"""`entroly serve` must stop when it is told to.

The stdio transport reads stdin on a worker thread, and interpreter shutdown
waits for that thread. While stdin stays open -- a terminal, or a host that
signals before it closes the pipe -- nothing ends that read. Published 1.0.88:

* on a terminal ignored Ctrl-C until Enter was pressed, then printed three
  tracebacks (KeyboardInterrupt through anyio's CancelledError chain);
* on SIGTERM logged "Shutdown signal received -- persisting state..." and kept
  running, holding its dashboard port and file watchers, until stdin closed
  (20 s later in the reproduction) or the host sent SIGKILL.

Closing stdin, the shutdown the MCP specification asks hosts to try first,
already worked; it is covered here so the signal path cannot regress it.

The kernel may hand a process-directed signal to any thread that does not
block it. Python runs signal handlers only on the main thread, and an idle
server's main thread sleeps in its event loop until a file descriptor is
ready, so a SIGTERM that landed on a worker thread was never acted on: the
SIGTERM test failed that way once on CI (Python 3.11, "still running 15s
after SIGTERM", with no "Shutdown signal received" logged), and sending the
signal to a worker thread directly reproduces it every time.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")

ROOT = Path(__file__).resolve().parents[1]
DEADLINE_S = 15


def _serve(tmp_path: Path) -> subprocess.Popen[bytes]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def fix():\n    return 1\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT),
        "HOME": str(home),
        "NO_COLOR": "1",
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_MCP_PASSIVE": "1",  # no indexing, watchers or dashboard port
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "entroly", "serve"],
        cwd=project, env=env, start_new_session=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    initialize = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "signal-test", "version": "0"},
        },
    }
    proc.stdin.write((json.dumps(initialize) + "\n").encode())
    proc.stdin.flush()
    ready, _, _ = select.select([proc.stdout], [], [], 120)
    if not ready:
        proc.kill()
        pytest.fail("server never answered initialize: " + proc.stderr.read().decode()[-2000:])
    assert json.loads(proc.stdout.readline())["id"] == 1  # the transport is serving
    return proc


def _stopped(proc: subprocess.Popen[bytes]) -> tuple[int | None, str]:
    """Wait for exit while stdin stays open; kill it if the deadline passes."""
    try:
        proc.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return None, proc.stderr.read().decode("utf-8", "replace")
    return proc.returncode, proc.stderr.read().decode("utf-8", "replace")


def test_sigterm_stops_the_server_while_the_host_still_holds_stdin(tmp_path: Path) -> None:
    proc = _serve(tmp_path)

    os.kill(proc.pid, signal.SIGTERM)
    code, stderr = _stopped(proc)

    assert code is not None, f"still running {DEADLINE_S}s after SIGTERM:\n{stderr[-1500:]}"
    assert code == 0, stderr[-1500:]
    assert "State persisted successfully" in stderr
    assert "Traceback" not in stderr, stderr[-3000:]


def test_ctrl_c_stops_the_server_quietly(tmp_path: Path) -> None:
    proc = _serve(tmp_path)

    os.killpg(proc.pid, signal.SIGINT)  # what the terminal sends on Ctrl-C
    code, stderr = _stopped(proc)

    assert code is not None, f"still running {DEADLINE_S}s after Ctrl-C:\n{stderr[-1500:]}"
    assert code in (130, -signal.SIGINT), (code, stderr[-1500:])
    assert "State persisted successfully" in stderr
    assert "Traceback" not in stderr, stderr[-3000:]


def test_closing_stdin_still_shuts_down_cleanly(tmp_path: Path) -> None:
    proc = _serve(tmp_path)

    proc.stdin.close()
    code, stderr = _stopped(proc)

    assert code == 0, (code, stderr[-1500:])
    assert "State persisted successfully" in stderr
    assert "Traceback" not in stderr, stderr[-3000:]


def _signal_thread(pid: int, tid: int, signum: int) -> None:
    """tgkill(2): deliver a signal to one thread of another process."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "tgkill"):
        result = libc.tgkill(pid, tid, signum)
    else:  # glibc before 2.30
        import platform

        number = {"x86_64": 234, "aarch64": 131}.get(platform.machine())
        if number is None:
            pytest.skip("tgkill is not available on this platform")
        result = libc.syscall(number, pid, tid, signum)
    assert result == 0, f"tgkill failed: errno {ctypes.get_errno()}"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="thread-directed signals need Linux")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_delivered_to_a_worker_thread_still_stops_the_server(tmp_path: Path, signum) -> None:
    proc = _serve(tmp_path)
    workers = [int(t) for t in os.listdir(f"/proc/{proc.pid}/task") if int(t) != proc.pid]
    assert workers, "the server is expected to run worker threads"

    _signal_thread(proc.pid, workers[0], signum)
    code, stderr = _stopped(proc)

    assert code is not None, f"still running {DEADLINE_S}s after the signal reached a worker thread:\n{stderr[-1500:]}"
    assert "State persisted successfully" in stderr
    assert "Traceback" not in stderr, stderr[-3000:]
