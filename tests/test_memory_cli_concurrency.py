"""Concurrent `entroly-memory` commands must not silently lose memories.

Every mutating CLI command loads the whole memory file, changes it in process
memory and replaces the file.  Without a cross-process lock, a second command
that runs inside that window has its write overwritten while both commands
report success.  `recall` is a mutating command too: it persists recall
reinforcement, so a read-only-looking recall could erase a concurrent write.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from entroly.memory import MemoryOS
from entroly.memory_cli import main

_PAUSE_S = 1.5


def _contents(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [entry["content"] for entry in data["entries"]]


def _run_cli(argv: list[str], results: dict[str, int], key: str) -> None:
    results[key] = main(argv)


def test_recall_does_not_erase_a_concurrent_remember(tmp_path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    assert main(["remember", "seed memory about auth", "--file", str(path)]) == 0

    recall_loaded = threading.Event()
    remember_done = threading.Event()
    original_recall = MemoryOS.recall

    def recall_then_pause(self, *args, **kwargs):
        # Hold the recall command between its load and its save, which is
        # exactly where a concurrent writer used to be overwritten.
        ctx = original_recall(self, *args, **kwargs)
        recall_loaded.set()
        remember_done.wait(_PAUSE_S)
        return ctx

    monkeypatch.setattr(MemoryOS, "recall", recall_then_pause)

    results: dict[str, int] = {}
    reader = threading.Thread(
        target=_run_cli, args=(["recall", "auth", "--file", str(path)], results, "recall")
    )
    reader.start()
    assert recall_loaded.wait(10)

    def write() -> None:
        _run_cli(["remember", "writer fact during recall", "--file", str(path)], results, "remember")
        remember_done.set()

    writer = threading.Thread(target=write)
    writer.start()
    reader.join(30)
    writer.join(30)

    assert results == {"recall": 0, "remember": 0}
    assert "writer fact during recall" in _contents(path)


def test_two_remembers_keep_both_memories(tmp_path, monkeypatch) -> None:
    path = tmp_path / "memory.json"
    first_loaded = threading.Event()
    second_done = threading.Event()
    original_remember = MemoryOS.remember

    def remember_then_pause(self, content, **kwargs):
        mem_id = original_remember(self, content, **kwargs)
        if content == "first writer fact":
            first_loaded.set()
            second_done.wait(_PAUSE_S)
        return mem_id

    monkeypatch.setattr(MemoryOS, "remember", remember_then_pause)

    results: dict[str, int] = {}
    first = threading.Thread(
        target=_run_cli,
        args=(["remember", "first writer fact", "--file", str(path)], results, "first"),
    )
    first.start()
    assert first_loaded.wait(10)

    def second_write() -> None:
        _run_cli(["remember", "second writer fact", "--file", str(path)], results, "second")
        second_done.set()

    second = threading.Thread(target=second_write)
    second.start()
    first.join(30)
    second.join(30)

    assert results == {"first": 0, "second": 0}
    assert sorted(_contents(path)) == ["first writer fact", "second writer fact"]


def test_parallel_cli_processes_store_every_memory(tmp_path) -> None:
    path = tmp_path / "memory.json"
    count = 8
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "entroly.memory_cli", "remember", f"parallel fact {i}", "--file", str(path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(count)
    ]
    for proc in procs:
        _out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err

    assert sorted(_contents(path)) == sorted(f"parallel fact {i}" for i in range(count))
