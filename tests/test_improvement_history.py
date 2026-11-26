"""Durability, privacy and scope boundaries of local experiment memory."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from entroly import improvement_history as history


SCOPE = {"cases_sha256": "a" * 64, "evaluator_sha256": "b" * 64,
         "native_sha256": "c" * 64, "seed": 17, "objective": "pareto.v1"}
METRICS = {"recall_accuracy": 0.8, "context_efficiency": 0.01,
           "avg_wall_time_ms": 2.0, "total_tokens_used": 500, "timed_out": False}


def record(store, **overrides):
    values = dict(scope=SCOPE, incumbent={"w_r": 0.3}, candidate={"w_r": 0.2},
                  incumbent_metrics=METRICS, candidate_metrics=METRICS,
                  decision="rejected", reason="no_pareto_gain")
    values.update(overrides)
    return store.record(**values)


def test_empty_inspection_does_not_create_state(tmp_path):
    directory = tmp_path / "absent"
    result = history.ImprovementHistory(directory).summary()
    assert result["records"] == []
    assert result["returned_records"] == result["omitted_records"] == 0
    assert result["promotion_authority"] is result["quality_claim_allowed"] is False
    assert not directory.exists()


def test_restart_filters_exact_scope_and_incumbent_without_raw_content(tmp_path):
    store = history.ImprovementHistory(tmp_path)
    incumbent = {"w_r": 0.3, "secret": "PRIVATE_INCUMBENT"}
    candidate = {"w_r": 0.2, "query": "PRIVATE_QUERY"}
    rid = record(store, incumbent=incumbent, candidate=candidate,
                 prediction={"reward": 0.2, "confidence": 0.7, "uncertainty": 0.1, "model_version": "v1"})
    restored = history.ImprovementHistory(tmp_path)
    assert restored.rejected_candidates(scope=SCOPE, incumbent=incumbent) == {history.digest(candidate)}
    assert restored.rejected_candidates(scope=SCOPE, incumbent={"w_r": 0.31}) == set()
    for key, value in (("seed", 18), ("objective", "other.v1"),
                       ("cases_sha256", "d" * 64), ("native_sha256", "e" * 64),
                       ("evaluator_sha256", "f" * 64)):
        assert restored.rejected_candidates(scope={**SCOPE, key: value}, incumbent=incumbent) == set()
    row = restored.summary()["records"][0]
    assert row["record_id"] == rid
    assert row["prediction"]["reward"] == 0.2
    assert row["candidate_metrics"] == METRICS
    raw = store.path.read_text()
    assert "PRIVATE_INCUMBENT" not in raw and "PRIVATE_QUERY" not in raw


def test_improvements_errors_and_timeout_results_never_exclude_proposals(tmp_path):
    store = history.ImprovementHistory(tmp_path)
    record(store, decision="improved")
    record(store, candidate_metrics=None, decision="evaluation_error")
    record(store, candidate_metrics={**METRICS, "timed_out": True})
    record(store, incumbent_metrics={**METRICS, "timed_out": True})
    assert store.rejected_candidates(scope=SCOPE, incumbent={"w_r": 0.3}) == set()
    assert len(store.summary()["records"]) == 4


def test_size_limit_and_omissions_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "MAX_RECORDS", 3)
    store = history.ImprovementHistory(tmp_path)
    for i in range(5):
        record(store, candidate={"trial": i})
    result = store.summary(limit=2)
    assert result["pruned_records"] == 2
    assert result["retained_records"] == 3
    assert result["returned_records"] == 2
    assert result["omitted_records"] == 1
    assert result["records"][0]["candidate_sha256"] == history.digest({"trial": 4})
    filtered = store.summary(limit=2, scope={**SCOPE, "seed": 99})
    assert filtered["filtered_records"] == 3


def test_retention_applies_to_reads_and_durable_pruning(tmp_path, monkeypatch):
    cutoff = [None]
    monkeypatch.setattr(history, "retention_cutoff_ts", lambda: cutoff[0])
    monkeypatch.setattr(history.time, "time", lambda: 100.0)
    store = history.ImprovementHistory(tmp_path)
    record(store)
    cutoff[0] = 101.0
    assert store.summary()["expired_records"] == 1
    assert store.summary()["records"] == []
    assert store.prune() == {"kept": 0, "removed": 1}
    assert history.ImprovementHistory(tmp_path).summary()["pruned_records"] == 1


def test_atomic_write_failure_preserves_previous_evidence(tmp_path, monkeypatch):
    store = history.ImprovementHistory(tmp_path)
    record(store)
    previous = store.path.read_bytes()
    monkeypatch.setattr(Path, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(history.ImprovementHistoryError):
        record(store, candidate={"w_r": 0.4})
    assert store.path.read_bytes() == previous
    assert list(tmp_path.glob(".improvement-*")) == []


@pytest.mark.parametrize("case", ["json", "checksum", "record_checksum", "duplicate", "oversize", "schema"])
def test_corruption_is_visible_and_not_overwritten(tmp_path, case):
    store = history.ImprovementHistory(tmp_path)
    record(store)
    state = json.loads(store.path.read_text())
    if case == "json":
        store.path.write_text("not json")
    elif case == "oversize":
        store.path.write_bytes(b"x" * (history.MAX_BYTES + 1))
    else:
        if case in {"checksum", "record_checksum"}:
            state["records"][0]["decision"] = "improved"
        elif case == "duplicate":
            state["records"].append(deepcopy(state["records"][0]))
        else:
            state["schema"] = "future"
        if case != "checksum":
            state["store_sha256"] = history.digest({k: v for k, v in state.items() if k != "store_sha256"})
        store.path.write_text(json.dumps(state))
    corrupted = store.path.read_bytes()
    with pytest.raises(history.ImprovementHistoryError):
        store.summary()
    with pytest.raises(history.ImprovementHistoryError):
        record(store)
    assert store.path.read_bytes() == corrupted


@pytest.mark.parametrize("overrides", [
    {"scope": {**SCOPE, "seed": True}}, {"scope": {**SCOPE, "cases_sha256": "bad"}},
    {"candidate_metrics": {**METRICS, "recall_accuracy": float("nan")}},
    {"candidate_metrics": {**METRICS, "recall_accuracy": 1.01}},
    {"candidate_metrics": {**METRICS, "total_tokens_used": True}},
    {"candidate_metrics": {**METRICS, "query": "private"}},
    {"candidate_metrics": None}, {"decision": "passed"}, {"reason": "raw private text"},
    {"prediction": {"reward": 2.0}}, {"prediction": {"raw_answer": "private"}},
    {"decision": "evaluation_error", "candidate_metrics": METRICS},
])
def test_invalid_observations_do_not_create_storage(tmp_path, overrides):
    directory = tmp_path / "new"
    with pytest.raises(history.ImprovementHistoryError):
        record(history.ImprovementHistory(directory), **overrides)
    assert not directory.exists()


@pytest.mark.parametrize("limit", [0, -1, 129, True, 1.5])
def test_inspection_limit_is_bounded(tmp_path, limit):
    with pytest.raises(history.ImprovementHistoryError):
        history.ImprovementHistory(tmp_path).summary(limit=limit)


def test_threaded_writers_preserve_every_attempt(tmp_path):
    def append(i):
        return record(history.ImprovementHistory(tmp_path), candidate={"trial": i})
    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(append, range(16)))
    restored = history.ImprovementHistory(tmp_path).summary(limit=20)
    assert len(restored["records"]) == len(set(ids)) == 16


def test_independent_processes_share_history_without_lost_writes(tmp_path):
    code = (
        "import json,sys; from entroly.improvement_history import ImprovementHistory; "
        "s=ImprovementHistory(sys.argv[1]); a=json.loads(sys.argv[2]); "
        "[s.record(**dict(a,candidate={'worker':sys.argv[3],'trial':i})) for i in range(4)]"
    )
    args = dict(scope=SCOPE, incumbent={"w_r": 0.3}, incumbent_metrics=METRICS,
                candidate_metrics=METRICS, decision="rejected", reason="no_gain")
    processes = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path), json.dumps(args), str(i)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                 for i in range(2)]
    try:
        for process in processes:
            out, err = process.communicate(timeout=30)
            assert process.returncode == 0, (out, err)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    assert history.ImprovementHistory(tmp_path).summary()["returned_records"] == 8


def test_symlinked_storage_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("private")
    directory = tmp_path / "store"
    directory.mkdir()
    try:
        (directory / "improvement_history.json").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(history.ImprovementHistoryError):
        record(history.ImprovementHistory(directory))
    assert target.read_text() == "private"
