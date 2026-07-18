"""Dashboard cross-mode wiring — regression guard.

The "blank dashboard" bug had four root causes; this file pins a test
to each so it cannot silently come back:

  1. value_tracker froze its in-memory copy at startup → a standalone
     `entroly dashboard` process never saw proxy/MCP/npm writes.
     Guard: test_reader_goes_live_on_external_write.
  2. Python ignored ENTROLY_DIR while npm honored it → split files.
     Guard: test_entroly_dir_is_honored.
  3. SDK / npm producers never recorded.
     Guard: test_sdk_compress_records_value (+ node test, gated).
  4. Dashboard rendered blank when no in-process engine.
     Guard: test_snapshot_not_blank_without_engine.

Plus the cross-runtime schema contract (Python <-> npm) and the
v2->v4 migration that must never drop or overstate a long-time user's history.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def fresh_dir(monkeypatch):
    """Isolated ENTROLY_DIR + reset module singletons so tests don't
    bleed into each other or the developer's real ~/.entroly."""
    d = Path(tempfile.mkdtemp())
    monkeypatch.setenv("ENTROLY_DIR", str(d))
    import entroly.value_tracker as vt
    importlib.reload(vt)
    yield d, vt
    import entroly.value_tracker as vt2
    vt2._tracker = None
    shutil.rmtree(d, ignore_errors=True)


def test_entroly_dir_is_honored(fresh_dir):
    d, vt = fresh_dir
    t = vt.ValueTracker()
    t.record(tokens_saved=10, model="gpt-4o")
    assert (d / "value_tracker.json").exists(), \
        "tracker must write under ENTROLY_DIR (else py/npm paths split)"


def test_reader_goes_live_on_external_write(fresh_dir):
    """THE core fix: a second process (the dashboard) must observe a
    writer process's data after reload_if_changed()."""
    d, vt = fresh_dir
    writer = vt.ValueTracker(d)
    reader = vt.ValueTracker(d)        # simulates the dashboard process
    assert reader.get_lifetime()["tokens_saved"] == 0
    writer.record(tokens_saved=1234, model="gpt-4o-mini")
    assert reader.get_lifetime()["tokens_saved"] == 0, "stale before reload"
    changed = reader.reload_if_changed()
    assert changed is True
    assert reader.get_lifetime()["tokens_saved"] == 1234
    # No-op when nothing changed (cheap-stat path).
    assert reader.reload_if_changed() is False


def test_activity_feed_persisted_and_bounded(fresh_dir):
    d, vt = fresh_dir
    t = vt.ValueTracker(d)
    for i in range(vt.ValueTracker._MAX_ACTIVITY + 25):
        t.record_event("compress", f"e{i}", source="sdk", tokens_saved=i)
    assert (d / "activity.jsonl").exists()
    fresh = vt.ValueTracker(d)
    act = fresh.get_activity(10_000)
    assert len(act) == vt.ValueTracker._MAX_ACTIVITY, "ring not bounded"
    assert act[0]["summary"].startswith("e"), "newest-first ordering"


def test_v2_migration_preserves_history(fresh_dir):
    d, vt = fresh_dir
    (d / "value_tracker.json").write_text(json.dumps({
        "version": 2,
        "lifetime": {"tokens_saved": 999, "cost_saved_usd": 1.23,
                     "requests_optimized": 7, "requests_total": 7,
                     "duplicates_caught": 2, "first_seen": 1.0,
                     "last_seen": 2.0, "evolution_spent_usd": 0.0,
                     "evolution_attempts": 0, "evolution_successes": 0},
        "daily": {"2026-01-01": {"tokens_saved": 999, "cost_saved": 1.23,
                                 "requests": 7}},
        "weekly": {}, "monthly": {},
    }), encoding="utf-8")
    t = vt.ValueTracker(d)
    lt = t.get_lifetime()
    assert lt["tokens_saved"] == 999, "must not drop existing history"
    assert lt["hallucinations_blocked"] == 0, "v3 field back-filled"
    assert lt["provider_tokens_saved"] == 0
    assert lt["provider_cost_avoided_usd"] == 0.0
    assert lt["unclassified_tokens_reduced"] == 999
    assert lt["unclassified_cost_estimate_usd"] == pytest.approx(1.23)
    assert lt["unclassified_operations"] == 7
    assert lt["cost_saved_usd"] == 0.0
    day = t.get_daily()[0]
    assert day["unclassified_tokens_reduced"] == 999
    assert day["provider_tokens_saved"] == 0
    assert day["cost_saved"] == 0.0
    assert t._data["version"] == vt.ValueTracker._SCHEMA_VERSION


def test_sdk_compress_records_value(fresh_dir):
    d, vt = fresh_dir
    import entroly.sdk as sdk
    importlib.reload(sdk)
    big = ("The quick brown fox jumps over the lazy dog. " * 400)
    sdk.compress(big, budget=200)
    lt = vt.ValueTracker(d).get_lifetime()
    assert lt["tokens_saved"] > 0, "SDK producer must feed the sink"
    assert lt["requests_optimized"] >= 1
    assert lt["local_tokens_reduced"] == lt["tokens_saved"]
    assert lt["local_operations"] >= 1
    assert lt["provider_tokens_saved"] == 0
    assert lt["provider_cost_avoided_usd"] == 0.0
    assert lt["cost_saved_usd"] == 0.0

    receipt = vt.ValueTracker(d).get_value_receipt()
    assert receipt["local_operations"]["tokens_reduced"] > 0
    assert receipt["local_operations"]["dollar_claimed_usd"] == 0.0
    assert receipt["provider_path"]["modeled_input_cost_avoided_usd"] == 0.0


def test_value_receipt_separates_provider_and_local_evidence(fresh_dir):
    d, vt = fresh_dir
    tracker = vt.ValueTracker(d)
    tracker.record(tokens_saved=1_000, model="gpt-4o", source="proxy")
    tracker.record(tokens_saved=700, model="gpt-4o", source="sdk")

    receipt = tracker.get_value_receipt()
    assert receipt["provider_path"]["requests_observed"] == 1
    assert receipt["provider_path"]["input_tokens_reduced"] == 1_000
    assert receipt["provider_path"]["modeled_input_cost_avoided_usd"] == pytest.approx(0.0025)
    assert receipt["local_operations"]["operations"] == 1
    assert receipt["local_operations"]["tokens_reduced"] == 700
    assert receipt["local_operations"]["dollar_claimed_usd"] == 0.0


def test_unknown_provider_model_remains_unpriced(fresh_dir):
    d, vt = fresh_dir
    tracker = vt.ValueTracker(d)
    tracker.record(
        tokens_saved=900,
        model="private-router-model",
        source="proxy",
    )

    receipt = tracker.get_value_receipt()
    assert receipt["provider_path"]["input_tokens_reduced"] == 900
    assert receipt["provider_path"]["modeled_input_cost_avoided_usd"] == 0.0
    assert receipt["provider_path"]["unpriced_requests"] == 1
    assert receipt["provider_path"]["unpriced_input_tokens"] == 900
    assert tracker.get_evolution_budget()["available_usd"] == 0.0


def test_snapshot_not_blank_without_engine(fresh_dir):
    """engine=None (npm / SDK-only / standalone dashboard) must still
    yield a populated, error-free snapshot — never the old blank."""
    d, vt = fresh_dir
    writer = vt.ValueTracker(d)
    writer.record(
        tokens_saved=3400,
        model="claude-sonnet-4",
        duplicates=5,
        source="proxy",
    )
    writer.record_hallucination_blocked(2)
    writer.record_routing_saving(0.012, chosen_model="claude-haiku-4")

    import entroly.dashboard as dash
    importlib.reload(dash)
    dash._engine = None
    vt._tracker = None
    snap = dash._get_full_snapshot()

    assert snap["engine_available"] is False
    assert snap["errors"] == []
    lt = snap["value_trends"]["lifetime"]
    assert lt["tokens_saved"] == 3400
    assert lt["provider_tokens_saved"] == 3400
    assert lt["provider_requests_optimized"] == 1
    assert lt["hallucinations_blocked"] == 2
    assert lt["routing_saved_usd"] == pytest.approx(0.012)
    assert len(snap["activity"]) >= 3
    assert len(snap["recent_requests"]) >= 3   # mapped for existing UI
    assert snap["engine_state"]["has_value_data"] is True


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_node_unknown_provider_model_remains_unpriced(fresh_dir):
    d, _vt = fresh_dir
    js = (
        Path(__file__).resolve().parent.parent
        / "entroly-wasm"
        / "js"
        / "value_tracker.js"
    )
    env = {**os.environ, "ENTROLY_DIR": str(d)}
    out = subprocess.run(
        [
            _NODE,
            "-e",
            f"const{{getTracker}}=require({json.dumps(str(js))});"
            "const t=getTracker();"
            "t.record({tokensSaved:900,model:'private-router-model',source:'proxy'});"
            "process.stdout.write(JSON.stringify(t.getValueReceipt()));",
        ],
        check=True,
        env=env,
        capture_output=True,
        timeout=30,
    )
    receipt = json.loads(out.stdout)
    provider = receipt["provider_path"]
    assert provider["input_tokens_reduced"] == 900
    assert provider["modeled_input_cost_avoided_usd"] == 0
    assert provider["unpriced_requests"] == 1
    assert provider["unpriced_input_tokens"] == 900


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_cross_runtime_schema_contract(fresh_dir):
    """npm (Node) writes -> Python reads the SAME file/schema/dir, and
    Python writes -> Node reads. Diverging this is the npm blank bug."""
    d, vt = fresh_dir
    js = (Path(__file__).resolve().parent.parent
          / "entroly-wasm" / "js" / "value_tracker.js")
    env = {**os.environ, "ENTROLY_DIR": str(d)}
    subprocess.run(
        [_NODE, "-e",
         f"const{{getTracker}}=require({json.dumps(str(js))});"
         "const t=getTracker();"
         "t.record({tokensSaved:2200,model:'gpt-4o-mini',duplicates:3});"
         "t.recordHallucinationBlocked(4);"],
        check=True, env=env, capture_output=True, timeout=30,
    )
    lt = vt.ValueTracker(d).get_lifetime()
    assert lt["tokens_saved"] == 2200
    assert lt["local_tokens_reduced"] == 2200
    assert lt["provider_tokens_saved"] == 0
    assert lt["provider_cost_avoided_usd"] == 0.0
    assert lt["hallucinations_blocked"] == 4
    assert lt["requests_optimized"] == 1

    # Reverse: Python writes, Node reads it back.
    vt._tracker = None
    vt.ValueTracker(d).record(tokens_saved=500, model="gpt-4o", source="proxy")
    out = subprocess.run(
        [_NODE, "-e",
         f"const{{getTracker}}=require({json.dumps(str(js))});"
         "const tr=getTracker().getTrends();"
         "process.stdout.write(String(tr.lifetime.tokens_saved));"],
        check=True, env=env, capture_output=True, timeout=30,
    )
    assert out.stdout.decode().strip() == "2700", \
        "Node must read Python's write (2200 + 500)"

    provider_out = subprocess.run(
        [_NODE, "-e",
         f"const{{getTracker}}=require({json.dumps(str(js))});"
         "const tr=getTracker().getTrends();"
         "process.stdout.write(String(tr.lifetime.provider_tokens_saved));"],
        check=True, env=env, capture_output=True, timeout=30,
    )
    assert provider_out.stdout.decode().strip() == "500"

    receipt_out = subprocess.run(
        [_NODE, "-e",
         f"const{{getTracker}}=require({json.dumps(str(js))});"
         "process.stdout.write(JSON.stringify(getTracker().getValueReceipt()));"],
        check=True, env=env, capture_output=True, timeout=30,
    )
    receipt = json.loads(receipt_out.stdout)
    assert receipt["schema_version"] == "entroly.value-receipt.v1"
    assert receipt["provider_path"]["input_tokens_reduced"] == 500
    assert receipt["local_operations"]["tokens_reduced"] == 2200
    assert receipt["local_operations"]["dollar_claimed_usd"] == 0
