"""`entroly learn --json` must print JSON.

`--json` was only read by `learn --history`. Without `--history` the command
printed its human report regardless, so `entroly learn --json` exited 0 with
"No feedback data available. Run the proxy and make some requests first."
on stdout, which no JSON consumer can parse.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_learn_json_without_a_proxy_is_a_json_no_data_report(tmp_path) -> None:
    env = {**os.environ, "HOME": str(tmp_path), "NO_COLOR": "1", "PYTHONPATH": str(ROOT),
           "ENTROLY_NO_DOCKER": "1", "ENTROLY_DISABLE_UPDATE_CHECK": "1"}
    done = subprocess.run(
        [sys.executable, "-m", "entroly", "learn", "--json", "--port", str(_free_port())],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )

    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["status"] == "no_data"
    assert "proxy" in report["detail"]
