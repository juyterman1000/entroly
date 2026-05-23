"""Focused test: Control API learning snapshot uses daemon aggregated stats."""

from entroly.dashboard import _control_learning_snapshot


class _State:
    learning_enabled = False
    autotune_enabled = True


class _DaemonWithStats:
    state = _State()

    def get_learning_stats(self) -> dict:
        return {
            "learning_enabled": self.state.learning_enabled,
            "autotune_enabled": self.state.autotune_enabled,
            "dreaming": {"total_dreams": 1, "last_status": "completed"},
        }


class _DaemonLegacy:
    state = _State()

    def get_learning_weights(self) -> dict:
        return {"source": "defaults", "weights": {"recency": 0.3}}


snap = _control_learning_snapshot(_DaemonWithStats())
assert snap["dreaming"]["last_status"] == "completed"
assert snap["learning_enabled"] is False

legacy = _control_learning_snapshot(_DaemonLegacy())
assert legacy["local_enabled"] is False
assert legacy["autotune_enabled"] is True
assert legacy["weights"]["source"] == "defaults"

print("[PASS] Control API learning snapshot prefers aggregated daemon stats")

