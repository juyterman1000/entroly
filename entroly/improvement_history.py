"""Bounded local memory of executed tuning experiments.

Records describe fixed retrieval benchmarks, not agent-answer correctness or
activation. Checksums detect edits, not fabricated observations. History may
guide proposals; it never supplies promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from .local_file_lock import exclusive_file_lock
from .privacy import retention_cutoff_ts

MAX_RECORDS = 128
MAX_BYTES = 1024 * 1024
_SCHEMA = "entroly.improvement-history.v1"
_SCOPE_KEYS = {"cases_sha256", "evaluator_sha256", "native_sha256", "seed", "objective"}
_METRIC_KEYS = {"recall_accuracy", "context_efficiency", "avg_wall_time_ms", "total_tokens_used", "timed_out"}
_ROW_KEYS = {"timestamp", "scope", "scope_sha256", "incumbent_sha256", "candidate_sha256",
             "incumbent_metrics", "candidate_metrics", "decision", "reason", "kind", "prediction", "record_id"}


class ImprovementHistoryError(RuntimeError):
    """History is invalid or unavailable; do not use it to guide proposals."""


def _encoded(value: Any) -> bytes:
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImprovementHistoryError("history requires finite JSON values") from exc
    if len(data) > 4 * MAX_BYTES:
        raise ImprovementHistoryError("history input exceeds the size bound")
    return data


def digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ImprovementHistoryError("expected a SHA-256 fingerprint")
    return value


def _tag(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
        raise ImprovementHistoryError("history labels must be bounded identifiers")
    return value


def _integer(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ImprovementHistoryError("expected a bounded nonnegative integer")
    return value


def _number(value: Any, low: float = 0, high: float = 1e12) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ImprovementHistoryError("expected a finite bounded measurement")
    return float(value)


def _scope(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != _SCOPE_KEYS:
        raise ImprovementHistoryError("history scope is incomplete")
    return {"cases_sha256": _hash(value["cases_sha256"]),
            "evaluator_sha256": _hash(value["evaluator_sha256"]),
            "native_sha256": _hash(value["native_sha256"]),
            "seed": _integer(value["seed"]), "objective": _tag(value["objective"])}


def _metrics(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != _METRIC_KEYS or type(value["timed_out"]) is not bool:
        raise ImprovementHistoryError("history measurements are incomplete")
    return {"recall_accuracy": _number(value["recall_accuracy"], high=1),
            "context_efficiency": _number(value["context_efficiency"]),
            "avg_wall_time_ms": _number(value["avg_wall_time_ms"]),
            "total_tokens_used": _integer(value["total_tokens_used"]),
            "timed_out": value["timed_out"]}


def _prediction(value: Any) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value or set(value) - {"reward", "confidence", "uncertainty", "model_version"}:
        raise ImprovementHistoryError("unsupported prediction fields")
    result = {}
    for key, item in value.items():
        if key == "model_version":
            result[key] = _tag(item)
        elif key == "reward":
            result[key] = _number(item, -1, 1)
        else:
            result[key] = _number(item, high=1 if key == "confidence" else 1e12)
    return result


def _validate_row(row: Any) -> None:
    if not isinstance(row, dict) or set(row) != _ROW_KEYS:
        raise ImprovementHistoryError("invalid experiment record")
    _number(row["timestamp"])
    scope = _scope(row["scope"])
    if row["scope_sha256"] != digest(scope):
        raise ImprovementHistoryError("experiment scope fingerprint changed")
    for key in ("record_id", "incumbent_sha256", "candidate_sha256"):
        _hash(row[key])
    _metrics(row["incumbent_metrics"])
    if row["decision"] not in {"improved", "rejected", "evaluation_error"}:
        raise ImprovementHistoryError("invalid experiment decision")
    if row["candidate_metrics"] is None:
        if row["decision"] != "evaluation_error":
            raise ImprovementHistoryError("an observed decision requires candidate measurements")
    else:
        _metrics(row["candidate_metrics"])
        if row["decision"] == "evaluation_error":
            raise ImprovementHistoryError("failed evaluations cannot claim measurements")
    _tag(row["reason"])
    _tag(row["kind"])
    _prediction(row["prediction"])
    if row["record_id"] != digest({k: v for k, v in row.items() if k != "record_id"}):
        raise ImprovementHistoryError("experiment fingerprint changed")


class ImprovementHistory:
    """Private, bounded snapshots coordinated across local processes."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.path = self.directory / "improvement_history.json"
        self.lock_path = self.directory / ".improvement_history.lock"

    def _guard(self) -> None:
        if any(path.is_symlink() for path in (self.directory, self.path, self.lock_path)):
            raise ImprovementHistoryError("symlinked history storage is unsupported")

    def _read(self) -> dict:
        self._guard()
        try:
            with self.path.open("rb") as handle:
                data = handle.read(MAX_BYTES + 1)
        except FileNotFoundError:
            return {"schema": _SCHEMA, "pruned_records": 0, "records": []}
        except OSError as exc:
            raise ImprovementHistoryError("unable to read improvement history") from exc
        if len(data) > MAX_BYTES:
            raise ImprovementHistoryError("improvement history exceeds its read bound")
        try:
            state = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise ImprovementHistoryError("improvement history is not valid JSON") from exc
        if (not isinstance(state, dict)
                or set(state) != {"schema", "pruned_records", "records", "store_sha256"}
                or state["schema"] != _SCHEMA or not isinstance(state["records"], list)
                or len(state["records"]) > MAX_RECORDS):
            raise ImprovementHistoryError("unsupported improvement history schema")
        _integer(state["pruned_records"])
        if state["store_sha256"] != digest({k: v for k, v in state.items() if k != "store_sha256"}):
            raise ImprovementHistoryError("improvement history fingerprint changed")
        seen = set()
        for row in state["records"]:
            _validate_row(row)
            if row["record_id"] in seen:
                raise ImprovementHistoryError("duplicate experiment record")
            seen.add(row["record_id"])
        return state

    @staticmethod
    def _unexpired(records: list[dict]) -> list[dict]:
        cutoff = retention_cutoff_ts()
        return records if cutoff is None else [row for row in records if row["timestamp"] >= cutoff]

    def _write(self, state: dict) -> None:
        payload = {key: state[key] for key in ("schema", "pruned_records", "records")}
        payload["store_sha256"] = digest(payload)
        data = _encoded(payload)
        if len(data) > MAX_BYTES:
            raise ImprovementHistoryError("improvement history exceeds its write bound")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.directory, prefix=".improvement-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            self._guard()
            temporary.replace(self.path)
        except OSError as exc:
            raise ImprovementHistoryError("unable to persist improvement history") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def record(
        self, *, scope: dict, incumbent: dict, candidate: dict,
        incumbent_metrics: dict, candidate_metrics: dict | None,
        decision: str, reason: str, prediction: dict | None = None, kind: str = "mutation",
    ) -> str:
        scope = _scope(scope)
        row = {"timestamp": time.time(), "scope": scope, "scope_sha256": digest(scope),
               "incumbent_sha256": digest(incumbent), "candidate_sha256": digest(candidate),
               "incumbent_metrics": _metrics(incumbent_metrics),
               "candidate_metrics": _metrics(candidate_metrics) if candidate_metrics is not None else None,
               "decision": decision, "reason": _tag(reason), "kind": _tag(kind),
               "prediction": _prediction(prediction)}
        row["record_id"] = digest(row)
        _validate_row(row)
        self._guard()
        try:
            with exclusive_file_lock(self.lock_path, timeout=2):
                state = self._read()
                kept = list(self._unexpired(state["records"]))
                kept.append(row)
                retained = kept[-MAX_RECORDS:]
                state["pruned_records"] += len(state["records"]) + 1 - len(retained)
                state["records"] = retained
                self._write(state)
        except (OSError, TimeoutError) as exc:
            raise ImprovementHistoryError("improvement history is unavailable") from exc
        return row["record_id"]

    def summary(self, *, limit: int = 20, scope: dict | None = None, incumbent: dict | None = None) -> dict:
        if type(limit) is not int or not 1 <= limit <= MAX_RECORDS:
            raise ImprovementHistoryError(f"history limit must be between 1 and {MAX_RECORDS}")
        scope_id = digest(_scope(scope)) if scope is not None else None
        incumbent_id = digest(incumbent) if incumbent is not None else None
        # Writers replace complete snapshots atomically. Reading one does not
        # need a lock file or create any storage.
        state = self._read()
        visible = self._unexpired(state["records"])
        matched = [row for row in visible if (scope_id is None or row["scope_sha256"] == scope_id)
                   and (incumbent_id is None or row["incumbent_sha256"] == incumbent_id)]
        return {"schema": _SCHEMA, "records": list(reversed(matched[-limit:])),
                "retained_records": len(state["records"]), "returned_records": min(limit, len(matched)),
                "omitted_records": max(0, len(matched) - limit),
                "filtered_records": len(visible) - len(matched),
                "expired_records": len(state["records"]) - len(visible),
                "pruned_records": state["pruned_records"], "promotion_authority": False,
                "quality_claim_allowed": False, "evidence_scope": "fixed_retrieval_benchmark"}

    def rejected_candidates(self, *, scope: dict, incumbent: dict) -> set[str]:
        rows = self.summary(limit=MAX_RECORDS, scope=scope, incumbent=incumbent)["records"]
        return {row["candidate_sha256"] for row in rows if row["decision"] == "rejected"
                and row["candidate_metrics"] is not None and not row["candidate_metrics"]["timed_out"]
                and not row["incumbent_metrics"]["timed_out"]}

    def prune(self) -> dict:
        self._guard()
        if not self.path.exists():
            return {"kept": 0, "removed": 0}
        try:
            with exclusive_file_lock(self.lock_path, timeout=2):
                state = self._read()
                kept = self._unexpired(state["records"])
                removed = len(state["records"]) - len(kept)
                if removed:
                    state["pruned_records"] += removed
                    state["records"] = kept
                    self._write(state)
                return {"kept": len(kept), "removed": removed}
        except (OSError, TimeoutError) as exc:
            raise ImprovementHistoryError("improvement history is unavailable") from exc
