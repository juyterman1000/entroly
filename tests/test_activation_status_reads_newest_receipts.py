"""`entroly activation status` must judge a project by its newest hook runs.

Status reads at most 500 receipts. Receipt files are named by a random
activation id, and they were chosen by sorting those names, so once a project
had more than 500 prompts the newest run could be left out. A hook that had
just started running degraded (for example after an upgrade broke the native
engine) was then reported as "active" from older receipts.

The receipts here are written with the hook's own writer; each file's
modification time is set to its recorded time, as successive prompts would.
"""

from __future__ import annotations

import os
import time

from entroly import agent_activation as aa


def _receipt(project, activation_id, recorded_at, status, native):
    return {
        "schema_version": aa.SCHEMA_VERSION,
        "activation_id": activation_id,
        "recorded_at_unix": recorded_at,
        "host": "claude-code",
        "event": "UserPromptSubmit",
        "enforcement": "host_hook",
        "status": status,
        "project_fingerprint": aa._project_fingerprint(project),
        "source_root": str(project),
        "native_engine": native,
        "selected_sources": [],
        "selected_context_tokens_estimate": 0,
    }


def test_the_newest_receipt_decides_state_beyond_500_receipts(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "activation"
    now = time.time()

    for index in range(500):
        recorded_at = now - 3_600 + index
        path = aa._write_receipt(
            _receipt(project, f"f{index:031x}", recorded_at, "activated", True),
            state_dir,
        )
        os.utime(path, (recorded_at, recorded_at))
    newest_at = now - 60
    newest = aa._write_receipt(
        _receipt(project, "0" * 32, newest_at, "degraded", False), state_dir
    )
    os.utime(newest, (newest_at, newest_at))

    report = aa.activation_status(project, state_dir=state_dir)

    assert report["latest"]["activation_id"] == "0" * 32
    assert report["latest"]["status"] == "degraded"
    assert report["state"] == "observed_degraded"
