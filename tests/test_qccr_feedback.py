from entroly.server import EntrolyEngine


class _FakeRust:
    def __init__(self):
        self.success_calls = []
        self.failure_calls = []
        self.reward_calls = []

    def export_fragments(self):
        return [
            {"fragment_id": "native-1", "source": "src/a.py"},
            {"fragment_id": "native-2", "source": "src/a.py"},
            {"fragment_id": "native-3", "source": "src/b.py"},
        ]

    def record_success(self, fragment_ids):
        self.success_calls.append(fragment_ids)

    def record_failure(self, fragment_ids):
        self.failure_calls.append(fragment_ids)

    def record_reward(self, fragment_ids, reward):
        self.reward_calls.append((fragment_ids, reward))


class _FakePruner:
    def __init__(self):
        self.calls = []

    def apply_feedback(self, fragment_id, reward):
        self.calls.append((fragment_id, reward))


def _engine_with_fake_rust():
    engine = EntrolyEngine.__new__(EntrolyEngine)
    engine._use_rust = True
    engine._rust = _FakeRust()
    engine._pruner = _FakePruner()
    engine._journal_callback = None
    return engine


def test_qccr_feedback_resolves_all_native_fragments_for_source():
    engine = _engine_with_fake_rust()

    engine.record_success(["qccr::src/a.py"])
    engine.record_failure(["qccr::src/b.py"])
    engine.record_reward(["qccr::src/a.py"], 0.5)

    assert engine._rust.success_calls == [["native-1", "native-2"]]
    assert engine._rust.failure_calls == [["native-3"]]
    assert engine._rust.reward_calls == [(["native-1", "native-2"], 0.5)]
    assert engine._pruner.calls == [
        ("qccr::src/a.py", 1.0),
        ("qccr::src/b.py", -1.0),
        ("qccr::src/a.py", 0.5),
    ]


def test_native_and_unknown_feedback_ids_are_preserved_and_deduplicated():
    engine = _engine_with_fake_rust()

    resolved = engine._resolve_native_feedback_ids(
        ["native-1", "qccr::src/a.py", "unknown", "native-1"]
    )

    assert resolved == ["native-1", "native-2", "unknown"]
