"""Regression tests for feedback-journal retention safety."""

import threading
import time
from pathlib import Path

from entroly.autotune import FeedbackJournal


def _weights() -> dict[str, float]:
    return {"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2}


def test_prune_preserves_journal_when_atomic_replace_fails(tmp_path, monkeypatch):
    journal = FeedbackJournal(str(tmp_path))
    assert journal.log(weights=_weights(), reward=0.7)
    before = journal.journal_path.read_text()

    def fail_replace(_self, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    assert journal.prune(max_age=0) is False
    assert journal.journal_path.read_text() == before
    assert list(tmp_path.glob(".feedback_journal.jsonl.*.tmp")) == []


def test_prune_serializes_append_with_atomic_replace(tmp_path, monkeypatch):
    journal = FeedbackJournal(str(tmp_path))
    assert journal.log(weights=_weights(), reward=0.7)
    replace_started = threading.Event()
    allow_replace = threading.Event()
    real_replace = Path.replace

    def paused_replace(self, target):
        replace_started.set()
        assert allow_replace.wait(timeout=1)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", paused_replace)

    prune_thread = threading.Thread(target=lambda: journal.prune(max_age=0))
    prune_thread.start()
    assert replace_started.wait(timeout=1)

    append_thread = threading.Thread(
        target=lambda: journal.log(weights=_weights(), reward=0.9)
    )
    append_thread.start()
    time.sleep(0.05)
    assert append_thread.is_alive()

    allow_replace.set()
    prune_thread.join(timeout=1)
    append_thread.join(timeout=1)

    assert not prune_thread.is_alive()
    assert not append_thread.is_alive()
    assert journal.stats()["episodes"] == 1
    assert journal.stats()["last_reward"] == 0.9
