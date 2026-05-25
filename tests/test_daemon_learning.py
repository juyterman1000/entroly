"""Regression tests for daemon learning-loop integration."""

import json
import sys
import time as _time
from types import SimpleNamespace

from entroly.autotune import (
    DreamingLoop,
    FeedbackJournal,
    TaskProfileOptimizer,
    reward_weighted_optimize,
)
from entroly.daemon import EntrolyDaemon, EntrolyDaemonState
from entroly.online_learner import OnlinePrism


def _weights() -> dict[str, float]:
    return {"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2}


def test_daemon_state_exposes_learning_fields():
    state = EntrolyDaemonState()
    data = state.to_dict()

    assert data["learning"]["local_enabled"] is True
    assert data["learning"]["autotune_enabled"] is True
    assert data["learning"]["dreaming_active"] is True


def test_learning_interval_decision_explains_cadence():
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)

    interval, reason = daemon._next_learning_interval(
        current_interval_s=60.0,
        saw_new_feedback=True,
        dreamed=True,
        optimized_profiles=True,
    )
    assert interval == 10.0
    assert reason == "new_feedback"

    interval, reason = daemon._next_learning_interval(
        current_interval_s=60.0,
        saw_new_feedback=False,
        dreamed=True,
        optimized_profiles=True,
    )
    assert interval == 30.0
    assert reason == "dreamed"

    interval, reason = daemon._next_learning_interval(
        current_interval_s=100.0,
        saw_new_feedback=False,
        dreamed=False,
        optimized_profiles=False,
    )
    assert interval == 120.0
    assert reason == "idle_backoff"


def test_learning_interval_decision_respects_configured_bounds():
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)

    interval, reason = daemon._next_learning_interval(
        current_interval_s=30.0,
        saw_new_feedback=False,
        dreamed=True,
        optimized_profiles=False,
        min_interval_s=5.0,
        max_interval_s=20.0,
    )
    assert interval == 20.0
    assert reason == "dreamed"

    interval, reason = daemon._next_learning_interval(
        current_interval_s=1.0,
        saw_new_feedback=False,
        dreamed=False,
        optimized_profiles=False,
        min_interval_s=10.0,
        max_interval_s=120.0,
    )
    assert interval == 10.0
    assert reason == "idle_backoff"


def test_learning_stats_include_next_cadence_reason():
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._learning_next_interval_s = 10.0
    daemon._learning_last_interval_reason = "new_feedback"

    stats = daemon.get_learning_stats()

    assert stats["learning_loop"]["next_interval_s"] == 10.0
    assert stats["learning_loop"]["interval_reason"] == "new_feedback"
    assert stats["learning_loop"]["journal_callback"] == {
        "attempts": 0,
        "failures": 0,
        "last_status": "not_started",
        "last_error": None,
    }


def test_online_prism_observes_and_enters_learning_phase():
    prism = OnlinePrism(
        prior_weights={
            "w_recency": 0.30,
            "w_frequency": 0.25,
            "w_semantic": 0.25,
            "w_entropy": 0.20,
        }
    )
    initial = prism.weights()

    assert abs(sum(initial.values()) - 1.0) < 0.01

    prism.observe(
        0.8,
        {
            "w_recency": 0.5,
            "w_frequency": 0.2,
            "w_semantic": 0.2,
            "w_entropy": 0.1,
        },
    )
    assert prism.stats()["n_observations"] == 1
    assert prism.stats()["phase"] == "warmup"

    for _ in range(5):
        prism.observe(
            0.9,
            {
                "w_recency": 0.6,
                "w_frequency": 0.1,
                "w_semantic": 0.2,
                "w_entropy": 0.1,
            },
        )

    updated = prism.weights()
    assert updated["w_recency"] > initial["w_recency"]
    assert prism.stats()["phase"] == "learning"


def test_feedback_journal_stats_report_last_appended_reward(tmp_path):
    journal = FeedbackJournal(str(tmp_path))

    journal.log(weights=_weights(), reward=0.7)
    journal.log(weights=_weights(), reward=0.8)
    journal.log(weights=_weights(), reward=0.6)

    stats = journal.stats()
    assert journal.count() == 3
    assert stats["episodes"] == 3
    assert stats["successes"] == 3
    assert stats["avg_reward"] == 0.7
    assert stats["last_reward"] == 0.6


def test_task_profile_optimizer_uses_feedback_journal(tmp_path):
    journal = FeedbackJournal(str(tmp_path))
    journal.log(weights=_weights(), reward=0.7)
    journal.log(weights=_weights(), reward=0.8)
    journal.log(weights=_weights(), reward=0.6)

    profiles = TaskProfileOptimizer(journal).optimize_all()

    assert len(profiles) >= 1


def test_reward_weighted_optimize_prefers_rust_core(monkeypatch):
    calls = {"n": 0}

    def native_optimizer(episodes_json: str, current_weights_json: str) -> str:
        calls["n"] += 1
        assert json.loads(episodes_json)[0]["r"] == 0.7
        assert json.loads(current_weights_json)["w_r"] == 0.3
        return json.dumps({
            "blended": {"w_r": 0.4, "w_f": 0.2, "w_s": 0.2, "w_e": 0.2},
            "confidence": 0.5,
            "success_count": 3,
            "failure_count": 0,
            "total_episodes": 3,
        })

    monkeypatch.setitem(
        sys.modules,
        "entroly_core",
        SimpleNamespace(py_reward_weighted_optimize=native_optimizer),
    )
    result = reward_weighted_optimize(
        [
            {"t": 1, "w": _weights(), "r": 0.7},
            {"t": 2, "w": _weights(), "r": 0.8},
            {"t": 3, "w": _weights(), "r": 0.6},
        ],
        _weights(),
    )

    assert calls["n"] == 1
    assert result["blended"]["w_r"] == 0.4


def test_task_profile_optimizer_prefers_rust_core(tmp_path, monkeypatch):
    calls = {"n": 0}

    def native_profiles(episodes_json: str) -> str:
        calls["n"] += 1
        assert len(json.loads(episodes_json)) == 3
        return json.dumps({
            "Debugging": {
                "weights": {"w_r": 0.5, "w_f": 0.1, "w_s": 0.3, "w_e": 0.1},
                "confidence": 0.75,
                "episodes": 3,
            }
        })

    monkeypatch.setitem(
        sys.modules,
        "entroly_core",
        SimpleNamespace(py_optimize_task_profiles=native_profiles),
    )
    journal = FeedbackJournal(str(tmp_path))
    journal.log(weights=_weights(), reward=0.7, query="fix auth bug")
    journal.log(weights=_weights(), reward=0.8, query="debug auth error")
    journal.log(weights=_weights(), reward=0.6, query="wrong login behavior")

    profiles = TaskProfileOptimizer(journal).optimize_all()

    assert calls["n"] == 1
    assert profiles["Debugging"]["confidence"] == 0.75


def test_dreaming_loop_idle_detection_and_stats(tmp_path):
    journal = FeedbackJournal(str(tmp_path))
    journal.log(weights=_weights(), reward=0.7)
    journal.log(weights=_weights(), reward=0.8)
    journal.log(weights=_weights(), reward=0.6)

    dreaming = DreamingLoop(journal=journal, max_iterations=3)

    assert not dreaming.should_dream()
    pre = dreaming.stats()
    assert pre["last_dream_at"] is None
    assert pre["last_check_at"] is None

    result = dreaming.run_dream_cycle()
    assert result["status"] == "not_idle"

    mid = dreaming.stats()
    assert mid["last_dream_at"] is None
    assert mid["last_check_at"] is not None
    assert mid["next_dream_in_s"] > 0

    dreaming._last_activity = _time.time() - 120
    assert dreaming.should_dream()
    idle_stats = dreaming.stats()
    assert idle_stats["will_dream"] is True
    assert idle_stats["idle_seconds"] >= 100


def test_daemon_learning_callback_resets_idle_before_logging(tmp_path):
    journal = FeedbackJournal(str(tmp_path))
    journal.log(weights=_weights(), reward=0.7)
    journal.log(weights=_weights(), reward=0.8)
    journal.log(weights=_weights(), reward=0.6)

    dreaming = DreamingLoop(journal=journal, max_iterations=3)
    dreaming._last_activity = _time.time() - 120
    assert dreaming.should_dream()

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._feedback_journal = journal
    daemon._dreaming_loop = dreaming

    daemon._log_learning_episode(weights=_weights(), reward=0.9)

    assert journal.count() == 4
    assert journal.stats()["last_reward"] == 0.9
    assert not dreaming.should_dream()

    callback = daemon.get_learning_stats()["learning_loop"]["journal_callback"]
    assert callback["attempts"] == 1
    assert callback["failures"] == 0
    assert callback["last_status"] == "ok"
    assert callback["last_error"] is None


def test_daemon_learning_callback_reports_journal_failures():
    class BrokenJournal:
        def log(self, **episode):
            raise RuntimeError("disk full")

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._feedback_journal = BrokenJournal()

    daemon._log_learning_episode(weights=_weights(), reward=0.2)

    stats = daemon.get_learning_stats()
    callback = stats["learning_loop"]["journal_callback"]
    assert callback["attempts"] == 1
    assert callback["failures"] == 1
    assert callback["last_status"] == "error"
    assert callback["last_error"] == "disk full"
    assert stats["journal"]["status"] == "error"
    assert daemon.state.last_feedback_at is not None
