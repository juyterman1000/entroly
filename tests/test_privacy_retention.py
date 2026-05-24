from __future__ import annotations

import importlib
import json
import time
from pathlib import Path


def test_retention_days_parsing(monkeypatch):
    import entroly.privacy as priv
    importlib.reload(priv)

    monkeypatch.delenv("ENTROLY_RETENTION_DAYS", raising=False)
    assert priv.retention_days() == priv.DEFAULT_RETENTION_DAYS

    monkeypatch.setenv("ENTROLY_RETENTION_DAYS", "nonsense")
    assert priv.retention_days() == priv.DEFAULT_RETENTION_DAYS

    monkeypatch.setenv("ENTROLY_RETENTION_DAYS", "0")
    assert priv.retention_days() == 0

    monkeypatch.setenv("ENTROLY_RETENTION_DAYS", "-1")
    assert priv.retention_days() == 0

    monkeypatch.setenv("ENTROLY_RETENTION_DAYS", "30.9")
    assert priv.retention_days() == 30


def test_activity_jsonl_pruned_by_retention(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    monkeypatch.setenv("ENTROLY_RETENTION_DAYS", "7")

    now = time.time()
    old = {"ts": now - (10 * 24 * 60 * 60), "kind": "compress", "summary": "old"}
    new = {"ts": now - 60, "kind": "compress", "summary": "new"}
    (tmp_path / "activity.jsonl").write_text(
        json.dumps(old) + "\n" + json.dumps(new) + "\n",
        encoding="utf-8",
    )

    import entroly.value_tracker as vt
    importlib.reload(vt)
    t = vt.ValueTracker(tmp_path)
    rows = t.get_activity(50)
    assert len(rows) == 1
    assert rows[0]["summary"] == "new"

