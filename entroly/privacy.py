"""Privacy + retention utilities for local telemetry.

Entroly runs as a user-configured localhost proxy/supervisor. To keep this
low-risk by default, we enforce a bounded retention window for JSONL telemetry
files (activity feed, feedback journal) and expose the effective policy via the
Control API.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RETENTION_DAYS = 14
_MAX_RETENTION_DAYS = 3650  # 10 years hard cap as a safety guard


def retention_days() -> int:
    """Return retention window for local telemetry (days).

    Controlled by ENTROLY_RETENTION_DAYS. Values <= 0 disable pruning.
    """
    raw = os.environ.get("ENTROLY_RETENTION_DAYS", "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        days = int(float(raw))
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    if days <= 0:
        return 0
    return min(days, _MAX_RETENTION_DAYS)


def retention_cutoff_ts(now: float | None = None) -> float | None:
    """Epoch cutoff timestamp for pruning, or None when pruning disabled."""
    days = retention_days()
    if days <= 0:
        return None
    n = time.time() if now is None else float(now)
    return n - (days * 24 * 60 * 60)


@dataclass(frozen=True)
class PruneResult:
    kept: int
    removed: int
    changed: bool


def prune_jsonl_by_ts(path: Path, *, ts_key: str, cutoff_ts: float) -> PruneResult:
    """Prune a JSONL file in-place, keeping rows where row[ts_key] >= cutoff_ts.

    Fail-open: any parse errors keep the original line (so telemetry never
    breaks the product). Returns counts best-effort.
    """
    try:
        if not path.exists():
            return PruneResult(kept=0, removed=0, changed=False)
    except OSError:
        return PruneResult(kept=0, removed=0, changed=False)

    kept_lines: list[str] = []
    kept = 0
    removed = 0

    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                ts = float(obj.get(ts_key, 0) or 0)
                if ts >= cutoff_ts:
                    kept_lines.append(json.dumps(obj, separators=(",", ":")))
                    kept += 1
                else:
                    removed += 1
            except Exception:
                # Keep unparseable lines rather than risk data loss.
                kept_lines.append(s)
                kept += 1
    except OSError:
        return PruneResult(kept=0, removed=0, changed=False)

    # Only rewrite when something was actually removed.
    if removed <= 0:
        return PruneResult(kept=kept, removed=0, changed=False)

    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        tmp.replace(path)
        return PruneResult(kept=kept, removed=removed, changed=True)
    except OSError:
        return PruneResult(kept=kept, removed=removed, changed=False)


def file_info(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "bytes": int(st.st_size),
            "mtime": float(st.st_mtime),
        }
    except OSError:
        return {"path": str(path), "exists": False, "bytes": 0, "mtime": None}

