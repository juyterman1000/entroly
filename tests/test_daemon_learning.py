"""Regression tests for daemon learning-loop integration."""

import time as _time

from entroly.autotune import DreamingLoop, FeedbackJournal, TaskProfileOptimizer
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
