"""Verified model-based learning for Entroly.

This module implements the evidence boundary required for safe Dyna-style
learning:

* real, externally verified ``(state, action, reward, next_state)``
  transitions train the world model;
* model-generated transitions are stored in a separate dream ledger;
* uncertainty is penalized and limits rollout depth; and
* dream evidence can propose experiments, but cannot promote a policy.

The default model is a dependency-free, locally weighted empirical dynamics
model.  ``EbbiforgeWorldModelAdapter`` can use Ebbiforge's Rust
``AutoregressivePredictor`` for next-state prediction while retaining the same
Entroly evidence and promotion gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence


_SCHEMA_VERSION = 1
_REAL_EVIDENCE = "real_verified"
_DREAM_EVIDENCE = "synthetic_dream"
_VALID_STRENGTHS = frozenset({"strong", "medium", "weak", "synthetic"})
_UNVERIFIED_VERIFIERS = frozenset(
    {"", "agent_self_report", "model_prediction_not_ground_truth"}
)
_REAL_INFLUENCE_SCOPE = "real_evidence"
_DREAM_INFLUENCE_SCOPE = "proposal_only"


@contextmanager
def _exclusive_ledger_lock(path: Path):
    """Hold a cross-process lock while validating or appending a ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows is the primary development host
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class WorldModelError(RuntimeError):
    """Base class for verified-world-model failures."""


class TransitionIntegrityError(WorldModelError):
    """The transition ledger failed hash-chain or schema validation."""


class InsufficientWorldModelData(WorldModelError):
    """The model does not yet have enough verified support to predict."""


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{name} cannot be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite numbers")
    return vector


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True, slots=True)
class VerifiedTransition:
    """One real or explicitly synthetic state transition.

    ``strength="strong"`` means a deterministic/external verifier observed the
    outcome.  Synthetic transitions always use ``strength="synthetic"`` and
    are structurally ineligible for real-model fitting or promotion.
    """

    state: tuple[float, ...]
    action: tuple[float, ...]
    next_state: tuple[float, ...]
    reward: float
    environment: str
    source: str
    verifier: str
    strength: str = "strong"
    synthetic: bool = False
    transition_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    parent_transition_ids: tuple[str, ...] = ()
    parent_receipt_hashes: tuple[str, ...] = ()
    prediction_confidence: float | None = None
    uncertainty: float | None = None
    policy_version: str = ""
    model_version: str = ""
    model_backend: str = ""
    influence_scope: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _finite_vector(self.state, "state"))
        object.__setattr__(self, "action", _finite_vector(self.action, "action"))
        object.__setattr__(
            self, "next_state", _finite_vector(self.next_state, "next_state")
        )
        reward = float(self.reward)
        if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be finite and within [-1, 1]")
        object.__setattr__(self, "reward", reward)
        if len(self.state) != len(self.next_state):
            raise ValueError("state and next_state dimensions must match")
        if self.strength not in _VALID_STRENGTHS:
            raise ValueError(f"unknown transition strength: {self.strength}")
        if not self.transition_id:
            raise ValueError("transition_id cannot be empty")
        if not self.environment.strip():
            raise ValueError("environment cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if self.synthetic:
            if self.strength != "synthetic":
                raise ValueError("synthetic transitions must use synthetic strength")
            if not self.parent_transition_ids:
                raise ValueError("synthetic transitions require real parent evidence")
        elif self.strength == "synthetic":
            raise ValueError("real transitions cannot use synthetic strength")
        if self.parent_receipt_hashes:
            if len(self.parent_receipt_hashes) != len(self.parent_transition_ids):
                raise ValueError(
                    "parent receipt hashes must align with parent transition IDs"
                )
            if not all(_is_sha256(value) for value in self.parent_receipt_hashes):
                raise ValueError("parent receipt hashes must be lowercase SHA-256 values")
        influence_scope = self.influence_scope or (
            _DREAM_INFLUENCE_SCOPE if self.synthetic else _REAL_INFLUENCE_SCOPE
        )
        expected_scope = (
            _DREAM_INFLUENCE_SCOPE if self.synthetic else _REAL_INFLUENCE_SCOPE
        )
        if influence_scope != expected_scope:
            raise ValueError(
                f"{'synthetic' if self.synthetic else 'real'} transitions require "
                f"influence_scope={expected_scope!r}"
            )
        object.__setattr__(self, "influence_scope", influence_scope)
        for name in ("prediction_confidence", "uncertainty"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be within [0, 1]")

    @property
    def is_real_verified(self) -> bool:
        verifier = self.verifier.strip().lower()
        return (
            not self.synthetic
            and self.strength == "strong"
            and verifier not in _UNVERIFIED_VERIFIERS
            and not self.source.strip().lower().startswith("world_model:")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "timestamp": self.timestamp,
            "state": list(self.state),
            "action": list(self.action),
            "next_state": list(self.next_state),
            "reward": self.reward,
            "environment": self.environment,
            "source": self.source,
            "verifier": self.verifier,
            "strength": self.strength,
            "synthetic": self.synthetic,
            "parent_transition_ids": list(self.parent_transition_ids),
            "parent_receipt_hashes": list(self.parent_receipt_hashes),
            "prediction_confidence": self.prediction_confidence,
            "uncertainty": self.uncertainty,
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "model_backend": self.model_backend,
            "influence_scope": self.influence_scope,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VerifiedTransition":
        return cls(
            transition_id=str(payload["transition_id"]),
            timestamp=float(payload["timestamp"]),
            state=tuple(payload["state"]),
            action=tuple(payload["action"]),
            next_state=tuple(payload["next_state"]),
            reward=float(payload["reward"]),
            environment=str(payload["environment"]),
            source=str(payload["source"]),
            verifier=str(payload.get("verifier", "")),
            strength=str(payload.get("strength", "weak")),
            synthetic=bool(payload.get("synthetic", False)),
            parent_transition_ids=tuple(payload.get("parent_transition_ids", ())),
            parent_receipt_hashes=tuple(payload.get("parent_receipt_hashes", ())),
            prediction_confidence=payload.get("prediction_confidence"),
            uncertainty=payload.get("uncertainty"),
            policy_version=str(payload.get("policy_version", "")),
            model_version=str(payload.get("model_version", "")),
            model_backend=str(payload.get("model_backend", "")),
            influence_scope=str(payload.get("influence_scope", "")),
        )


@dataclass(frozen=True, slots=True)
class TransitionReceipt:
    transition_id: str
    evidence_class: str
    receipt_hash: str
    previous_hash: str
    ledger_path: str


class TransitionLedger:
    """Two append-only, hash-chained ledgers: real evidence and dreams."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.real_path = self.root / "world_model_real.jsonl"
        self.dream_path = self.root / "world_model_dream.jsonl"
        self.lock_path = self.root / ".world_model.lock"
        self._lock = threading.RLock()
        self._last_hash: dict[str, str] = {}
        self._ids: dict[str, set[str]] = {}
        with _exclusive_ledger_lock(self.lock_path):
            for evidence_class, path in (
                (_REAL_EVIDENCE, self.real_path),
                (_DREAM_EVIDENCE, self.dream_path),
            ):
                events = self._read_events(path, evidence_class)
                self._refresh_cache(evidence_class, events)

    def record_real(self, transition: VerifiedTransition) -> TransitionReceipt:
        if not transition.is_real_verified:
            raise ValueError(
                "real world-model evidence requires a strong external verifier"
            )
        return self._append(self.real_path, _REAL_EVIDENCE, transition)

    def record_dream(self, transition: VerifiedTransition) -> TransitionReceipt:
        if not transition.synthetic:
            raise ValueError("dream ledger accepts synthetic transitions only")
        if transition.influence_scope != _DREAM_INFLUENCE_SCOPE:
            raise ValueError("dream evidence must be proposal-only")
        if not transition.model_version or not transition.model_backend:
            raise ValueError("dream evidence requires a committed model identity")
        self._validate_dream_lineage(transition, require_receipt_hashes=True)
        return self._append(self.dream_path, _DREAM_EVIDENCE, transition)

    def read_real(self) -> list[VerifiedTransition]:
        return [
            VerifiedTransition.from_dict(event["payload"])
            for event in self._validated_events(self.real_path, _REAL_EVIDENCE)
        ]

    def read_dreams(self) -> list[VerifiedTransition]:
        transitions = [
            VerifiedTransition.from_dict(event["payload"])
            for event in self._validated_events(self.dream_path, _DREAM_EVIDENCE)
        ]
        for transition in transitions:
            # Pre-v2 experimental ledgers did not commit parent receipt hashes.
            # They remain readable for recovery, but new writes must be closed.
            self._validate_dream_lineage(
                transition,
                require_receipt_hashes=bool(transition.parent_receipt_hashes),
            )
        return transitions

    def real_receipt_hashes(self) -> dict[str, str]:
        """Return the validated real-transition ID -> receipt commitment map."""
        return {
            str(event["payload"]["transition_id"]): str(event["receipt_hash"])
            for event in self._validated_events(self.real_path, _REAL_EVIDENCE)
        }

    def real_receipt(self, transition_id: str) -> TransitionReceipt | None:
        """Return the validated receipt for one real transition, if present.

        This makes interrupted writes safely retryable: a caller can prove that
        the exact transition was already committed instead of appending a
        duplicate or treating an earlier successful write as a failure.
        """
        for event in self._validated_events(self.real_path, _REAL_EVIDENCE):
            if str(event["payload"]["transition_id"]) != transition_id:
                continue
            return TransitionReceipt(
                transition_id=transition_id,
                evidence_class=_REAL_EVIDENCE,
                receipt_hash=str(event["receipt_hash"]),
                previous_hash=str(event["previous_hash"]),
                ledger_path=str(self.real_path),
            )
        return None

    def contains_exact_real(self, transition: VerifiedTransition) -> bool:
        """Whether the exact real transition payload is committed to this ledger."""
        if not transition.is_real_verified:
            return False
        committed = self.real_payload_commitments()
        return committed.get(transition.transition_id) == _sha256(transition.as_dict())

    def real_payload_commitments(self) -> dict[str, str]:
        """Return validated transition IDs bound to their canonical payload hash."""
        return {
            str(event["payload"]["transition_id"]): _sha256(
                VerifiedTransition.from_dict(event["payload"]).as_dict()
            )
            for event in self._validated_events(self.real_path, _REAL_EVIDENCE)
        }

    def _validated_events(
        self, path: Path, evidence_class: str
    ) -> list[dict[str, Any]]:
        with self._lock, _exclusive_ledger_lock(self.lock_path):
            events = self._read_events(path, evidence_class)
            self._refresh_cache(evidence_class, events)
            return events

    def _refresh_cache(
        self, evidence_class: str, events: list[dict[str, Any]]
    ) -> None:
        self._last_hash[evidence_class] = (
            str(events[-1]["receipt_hash"]) if events else ""
        )
        self._ids[evidence_class] = {
            str(event["payload"]["transition_id"]) for event in events
        }

    def _validate_dream_lineage(
        self,
        transition: VerifiedTransition,
        *,
        require_receipt_hashes: bool,
    ) -> None:
        real_receipts = self.real_receipt_hashes()
        missing = [
            transition_id
            for transition_id in transition.parent_transition_ids
            if transition_id not in real_receipts
        ]
        if missing:
            raise TransitionIntegrityError(
                "dream lineage references real evidence absent from the ledger: "
                + ", ".join(missing[:3])
            )
        if require_receipt_hashes and not transition.parent_receipt_hashes:
            raise TransitionIntegrityError(
                "dream lineage does not commit its parent receipt hashes"
            )
        if transition.parent_receipt_hashes:
            expected_hashes = tuple(
                real_receipts[transition_id]
                for transition_id in transition.parent_transition_ids
            )
            if transition.parent_receipt_hashes != expected_hashes:
                raise TransitionIntegrityError(
                    "dream lineage parent receipt commitment mismatch"
                )

    def _append(
        self,
        path: Path,
        evidence_class: str,
        transition: VerifiedTransition,
    ) -> TransitionReceipt:
        with self._lock, _exclusive_ledger_lock(self.lock_path):
            events = self._read_events(path, evidence_class)
            self._refresh_cache(evidence_class, events)
            if transition.transition_id in self._ids[evidence_class]:
                raise ValueError(f"duplicate transition_id: {transition.transition_id}")
            previous_hash = self._last_hash[evidence_class]
            event_without_hash = {
                "schema_version": _SCHEMA_VERSION,
                "kind": "world_transition",
                "evidence_class": evidence_class,
                "previous_hash": previous_hash,
                "payload": transition.as_dict(),
            }
            receipt_hash = _sha256(event_without_hash)
            event = {**event_without_hash, "receipt_hash": receipt_hash}
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(_canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._last_hash[evidence_class] = receipt_hash
            self._ids[evidence_class].add(transition.transition_id)
            return TransitionReceipt(
                transition_id=transition.transition_id,
                evidence_class=evidence_class,
                receipt_hash=receipt_hash,
                previous_hash=previous_hash,
                ledger_path=str(path),
            )

    @staticmethod
    def _read_events(path: Path, expected_class: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        previous_hash = ""
        seen_ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TransitionIntegrityError(
                        f"malformed transition ledger at {path}:{line_number}"
                    ) from exc
                if event.get("schema_version") != _SCHEMA_VERSION:
                    raise TransitionIntegrityError(
                        f"unsupported transition schema at {path}:{line_number}"
                    )
                if event.get("evidence_class") != expected_class:
                    raise TransitionIntegrityError(
                        f"evidence-class mismatch at {path}:{line_number}"
                    )
                if event.get("previous_hash", "") != previous_hash:
                    raise TransitionIntegrityError(
                        f"broken transition hash chain at {path}:{line_number}"
                    )
                supplied_hash = str(event.get("receipt_hash", ""))
                unsigned = {key: value for key, value in event.items() if key != "receipt_hash"}
                if not supplied_hash or _sha256(unsigned) != supplied_hash:
                    raise TransitionIntegrityError(
                        f"transition receipt mismatch at {path}:{line_number}"
                    )
                transition = VerifiedTransition.from_dict(event["payload"])
                transition_id = transition.transition_id
                if transition_id in seen_ids:
                    raise TransitionIntegrityError(
                        f"duplicate transition_id at {path}:{line_number}"
                    )
                if expected_class == _REAL_EVIDENCE and not transition.is_real_verified:
                    raise TransitionIntegrityError(
                        f"unverified evidence in real ledger at {path}:{line_number}"
                    )
                if expected_class == _DREAM_EVIDENCE and not transition.synthetic:
                    raise TransitionIntegrityError(
                        f"real evidence in dream ledger at {path}:{line_number}"
                    )
                seen_ids.add(transition_id)
                previous_hash = supplied_hash
                events.append(event)
        return events


def transition_from_ravs(
    trace: Any,
    outcome: Any,
    *,
    state: Sequence[float],
    action: Sequence[float],
    next_state: Sequence[float],
    environment: str,
) -> VerifiedTransition:
    """Convert a strong RAVS outcome into world-model training evidence.

    The caller supplies the environment's numeric state representation; RAVS
    supplies the identity, executed policy, and verifiable reward. Weak agent
    self-reports and medium behavioral inferences are rejected.
    """
    from .events import HONEST_OUTCOME_TYPES

    def read(obj: Any, key: str, default: Any = None) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    event_type = str(read(outcome, "event_type", "") or "")
    strength = str(read(outcome, "strength", "") or "")
    include = bool(read(outcome, "include_in_default_training", False))
    if event_type not in HONEST_OUTCOME_TYPES or strength != "strong" or not include:
        raise ValueError("RAVS transition requires strong default-training evidence")

    value = str(read(outcome, "value", "") or "").strip().lower()
    if value in {"success", "passed", "pass", "accepted", "accept", "0"}:
        reward = 1.0
    elif value in {"failure", "failed", "fail", "rejected", "reject", "1"}:
        reward = -1.0
    else:
        raise ValueError(f"RAVS outcome value is not a verifiable binary reward: {value}")

    request_id = str(read(outcome, "request_id", "") or read(trace, "request_id", ""))
    if not request_id:
        raise ValueError("RAVS transition requires a request_id")
    outcome_timestamp = float(read(outcome, "timestamp", time.time()))
    collector = str(read(outcome, "source", "ravs") or "ravs")
    transition_id = _sha256(
        {
            "request_id": request_id,
            "event_type": event_type,
            "timestamp": outcome_timestamp,
            "collector": collector,
        }
    )
    return VerifiedTransition(
        transition_id=transition_id,
        timestamp=outcome_timestamp,
        state=tuple(state),
        action=tuple(action),
        next_state=tuple(next_state),
        reward=reward,
        environment=environment,
        source=f"ravs:{collector}",
        verifier=event_type,
        strength="strong",
        policy_version=str(read(trace, "policy_decision", "") or ""),
    )


@dataclass(frozen=True, slots=True)
class WorldModelPrediction:
    next_state: tuple[float, ...]
    reward: float
    confidence: float
    uncertainty: float
    support_count: int
    supporting_transition_ids: tuple[str, ...]
    model_version: str
    backend: str


class WorldModelBackend(Protocol):
    def fit(self, transitions: Iterable[VerifiedTransition]) -> None: ...

    def predict(
        self, state: Sequence[float], action: Sequence[float], *, horizon: int = 1
    ) -> WorldModelPrediction: ...

    def stats(self) -> dict[str, Any]: ...


class EmpiricalWorldModel:
    """A deterministic, locally weighted dynamics-and-reward model.

    It is intentionally conservative: predictions need nearby verified
    support, expose their support IDs, and carry an uncertainty score.  This
    backend is useful before an optional neural Ebbiforge model has enough
    trajectories to train reliably.
    """

    def __init__(
        self,
        *,
        neighbors: int = 8,
        min_samples: int = 4,
        max_samples: int = 10_000,
    ) -> None:
        if neighbors <= 0 or min_samples <= 0 or max_samples < min_samples:
            raise ValueError("invalid empirical world-model sample limits")
        self.neighbors = neighbors
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._samples: list[VerifiedTransition] = []
        self._state_scale: tuple[float, ...] = ()
        self._action_scale: tuple[float, ...] = ()
        self._model_version = "untrained"

    def fit(self, transitions: Iterable[VerifiedTransition]) -> None:
        samples = list(transitions)
        rejected = [sample for sample in samples if not sample.is_real_verified]
        if rejected:
            raise ValueError("world models may fit real verified transitions only")
        samples = samples[-self.max_samples :]
        if samples:
            state_dim = len(samples[0].state)
            action_dim = len(samples[0].action)
            if any(
                len(sample.state) != state_dim
                or len(sample.next_state) != state_dim
                or len(sample.action) != action_dim
                for sample in samples
            ):
                raise ValueError("all world-model transitions must share dimensions")
        self._samples = samples
        self._recompute_scales()
        self._model_version = (
            _sha256([sample.transition_id for sample in samples])[:16]
            if samples
            else "untrained"
        )

    def update(self, transition: VerifiedTransition) -> None:
        if not transition.is_real_verified:
            raise ValueError("synthetic or weak transitions cannot update the model")
        self.fit([*self._samples, transition])

    @property
    def ready(self) -> bool:
        return len(self._samples) >= self.min_samples

    def predict(
        self,
        state: Sequence[float],
        action: Sequence[float],
        *,
        horizon: int = 1,
    ) -> WorldModelPrediction:
        if horizon != 1:
            raise ValueError("EmpiricalWorldModel predicts one transition at a time")
        if not self.ready:
            raise InsufficientWorldModelData(
                f"need {self.min_samples} real transitions; have {len(self._samples)}"
            )
        state_v = _finite_vector(state, "state")
        action_v = _finite_vector(action, "action")
        if len(state_v) != len(self._samples[0].state):
            raise ValueError("state dimension does not match fitted world model")
        if len(action_v) != len(self._samples[0].action):
            raise ValueError("action dimension does not match fitted world model")

        ranked = sorted(
            (
                (self._distance(state_v, action_v, sample), sample)
                for sample in self._samples
            ),
            key=lambda item: item[0],
        )[: self.neighbors]
        weights = [1.0 / max(distance, 1e-6) for distance, _ in ranked]
        total_weight = sum(weights)
        norm_weights = [weight / total_weight for weight in weights]

        next_state = tuple(
            sum(weight * sample.next_state[index] for weight, (_, sample) in zip(norm_weights, ranked))
            for index in range(len(state_v))
        )
        reward = sum(
            weight * sample.reward
            for weight, (_, sample) in zip(norm_weights, ranked)
        )
        mean_distance = sum(
            weight * distance
            for weight, (distance, _) in zip(norm_weights, ranked)
        )
        reward_variance = sum(
            weight * (sample.reward - reward) ** 2
            for weight, (_, sample) in zip(norm_weights, ranked)
        )
        support = len(ranked)
        # `min_samples` is the declared amount of support required before the
        # model is usable.  Treat reaching that threshold as full support and
        # let proximity/agreement express the remaining uncertainty.  The
        # previous 1-exp(-support/min_samples) formula capped support at 0.632
        # when `neighbors == min_samples`, making the default 0.55 confidence
        # gate nearly impossible to satisfy even for a perfectly local,
        # unanimous neighborhood.
        support_factor = min(1.0, support / self.min_samples)
        proximity = math.exp(-mean_distance)
        agreement = math.exp(-math.sqrt(max(0.0, reward_variance)))
        confidence = max(0.0, min(1.0, support_factor * proximity * agreement))
        uncertainty = 1.0 - confidence
        return WorldModelPrediction(
            next_state=next_state,
            reward=max(-1.0, min(1.0, reward)),
            confidence=confidence,
            uncertainty=uncertainty,
            support_count=support,
            supporting_transition_ids=tuple(
                sample.transition_id for _, sample in ranked
            ),
            model_version=self._model_version,
            backend="entroly_empirical",
        )

    def rank_actions(
        self,
        state: Sequence[float],
        actions: Iterable[Sequence[float]],
        *,
        uncertainty_penalty: float = 0.75,
    ) -> list[tuple[tuple[float, ...], float, WorldModelPrediction]]:
        if uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty cannot be negative")
        ranked = []
        for action in actions:
            action_v = _finite_vector(action, "action")
            prediction = self.predict(state, action_v)
            pessimistic_value = prediction.reward - (
                uncertainty_penalty * prediction.uncertainty
            )
            ranked.append((action_v, pessimistic_value, prediction))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "entroly_empirical",
            "ready": self.ready,
            "real_training_transitions": len(self._samples),
            "min_samples": self.min_samples,
            "neighbors": self.neighbors,
            "model_version": self._model_version,
            "synthetic_training_transitions": 0,
        }

    def _recompute_scales(self) -> None:
        if not self._samples:
            self._state_scale = ()
            self._action_scale = ()
            return

        def scales(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
            return tuple(
                max(1e-6, max(column) - min(column))
                for column in zip(*vectors)
            )

        self._state_scale = scales([sample.state for sample in self._samples])
        self._action_scale = scales([sample.action for sample in self._samples])

    def _distance(
        self,
        state: tuple[float, ...],
        action: tuple[float, ...],
        sample: VerifiedTransition,
    ) -> float:
        state_error = sum(
            ((left - right) / scale) ** 2
            for left, right, scale in zip(state, sample.state, self._state_scale)
        ) / len(state)
        action_error = sum(
            ((left - right) / scale) ** 2
            for left, right, scale in zip(action, sample.action, self._action_scale)
        ) / len(action)
        return math.sqrt(max(0.0, state_error + action_error))


class EbbiforgeWorldModelAdapter:
    """Duck-typed adapter for Ebbiforge's Rust autoregressive predictor.

    Ebbiforge predicts the next latent state.  Entroly's empirical reward head
    supplies the verified reward estimate and support IDs.  Confidence is the
    minimum of reward-head confidence, Ebbiforge rollout confidence, and a
    validation-loss calibration term.
    """

    def __init__(
        self,
        predictor: Any,
        *,
        state_factory: Callable[[Sequence[float]], Any],
        reward_model: EmpiricalWorldModel | None = None,
        min_train_steps: int = 1,
        max_validation_loss: float = 1.0,
        retrain_interval: int = 8,
        train_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.predictor = predictor
        self.state_factory = state_factory
        self.reward_model = reward_model or EmpiricalWorldModel()
        self.min_train_steps = max(1, int(min_train_steps))
        self.max_validation_loss = float(max_validation_loss)
        self.retrain_interval = max(1, int(retrain_interval))
        self.train_kwargs = {
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 128,
            "val_split": 0.1,
            **(train_kwargs or {}),
        }
        self._model_version = "untrained"
        self._samples: list[VerifiedTransition] = []

    def fit(self, transitions: Iterable[VerifiedTransition]) -> None:
        samples = list(transitions)
        self.reward_model.fit(samples)
        self._samples = samples[-self.reward_model.max_samples :]
        if not self.reward_model.ready:
            return
        self._train_dynamics()

    def update(self, transition: VerifiedTransition) -> None:
        if not transition.is_real_verified:
            raise ValueError("synthetic or weak transitions cannot update Ebbiforge")
        self._samples = [*self._samples, transition][-self.reward_model.max_samples :]
        self.reward_model.update(transition)
        if not self.reward_model.ready:
            return
        untrained = (
            self._read_metric("get_total_train_steps", 0.0) < self.min_train_steps
        )
        if untrained or len(self._samples) % self.retrain_interval == 0:
            self._train_dynamics()

    def _train_dynamics(self) -> None:
        payload = [
            {
                "state": list(sample.state),
                "action": list(sample.action),
                "next_state": list(sample.next_state),
            }
            for sample in self._samples
        ]
        train = getattr(self.predictor, "train", None)
        if not callable(train):
            raise TypeError("Ebbiforge predictor must expose train(...)")
        stats = train(
            _canonical_json(payload),
            int(self.train_kwargs["epochs"]),
            float(self.train_kwargs["learning_rate"]),
            int(self.train_kwargs["batch_size"]),
            float(self.train_kwargs["val_split"]),
        )
        self._model_version = _sha256(
            {
                "transitions": [sample.transition_id for sample in self._samples],
                "stats": [str(item) for item in (stats or [])[-3:]],
            }
        )[:16]

    def predict(
        self,
        state: Sequence[float],
        action: Sequence[float],
        *,
        horizon: int = 1,
    ) -> WorldModelPrediction:
        reward_prediction = self.reward_model.predict(state, action)
        train_steps = self._read_metric("get_total_train_steps", 0.0)
        validation_loss = self._read_metric("get_validation_loss", math.inf)
        if train_steps < self.min_train_steps or not math.isfinite(validation_loss):
            raise InsufficientWorldModelData("Ebbiforge dynamics model is not trained")
        if validation_loss > self.max_validation_loss:
            raise InsufficientWorldModelData(
                "Ebbiforge validation loss exceeds the configured safety gate"
            )
        initial = self.state_factory(_finite_vector(state, "state"))
        predict_sequence = getattr(self.predictor, "predict_sequence", None)
        if not callable(predict_sequence):
            raise TypeError("Ebbiforge predictor must expose predict_sequence(...)")
        action_v = list(_finite_vector(action, "action"))
        prediction = predict_sequence(initial, [action_v for _ in range(horizon)])
        future_states = list(getattr(prediction, "future_states", ()) or ())
        if not future_states:
            raise WorldModelError("Ebbiforge returned no predicted future states")
        next_vector = _finite_vector(
            getattr(future_states[-1], "vector", ()), "Ebbiforge next_state"
        )
        rollout_confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        calibration = math.exp(-max(0.0, validation_loss))
        confidence = max(
            0.0,
            min(
                1.0,
                reward_prediction.confidence,
                rollout_confidence,
                calibration,
            ),
        )
        return WorldModelPrediction(
            next_state=next_vector,
            reward=reward_prediction.reward,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            support_count=reward_prediction.support_count,
            supporting_transition_ids=reward_prediction.supporting_transition_ids,
            model_version=self._model_version,
            backend="ebbiforge_autoregressive",
        )

    def stats(self) -> dict[str, Any]:
        reward_stats = self.reward_model.stats()
        training_steps = self._read_metric("get_total_train_steps", 0.0)
        validation_loss = self._read_metric("get_validation_loss", math.inf)
        return {
            "backend": "ebbiforge_autoregressive",
            "ready": reward_stats["ready"]
            and training_steps >= self.min_train_steps
            and math.isfinite(validation_loss)
            and validation_loss <= self.max_validation_loss,
            "real_training_transitions": reward_stats["real_training_transitions"],
            "synthetic_training_transitions": 0,
            "training_steps": int(training_steps),
            "validation_loss": validation_loss,
            "model_version": self._model_version,
        }

    def _read_metric(self, method_name: str, default: float) -> float:
        method = getattr(self.predictor, method_name, None)
        if not callable(method):
            return default
        try:
            return float(method())
        except (TypeError, ValueError, RuntimeError):
            return default


@dataclass(frozen=True, slots=True)
class DreamRollout:
    transitions: tuple[VerifiedTransition, ...]
    predicted_return: float
    pessimistic_return: float
    stopped_reason: str
    promotion_status: str = "proposal_only"


def anytime_hoeffding_radius(count: int, confidence_delta: float) -> float:
    """A time-uniform one-sided radius for rewards in ``[-1, 1]``.

    The per-time error budget is ``3 * delta / (pi^2 * n^2)``. Summing it
    over every positive ``n`` spends ``delta / 2`` on one evidence stream.
    Calling the bound for both candidate-lower and incumbent-upper streams
    therefore costs at most ``delta`` in total. This deliberately simple
    stitched Hoeffding confidence sequence remains valid at data-dependent
    stopping times, provided each policy's real benchmark rewards have a
    stable conditional mean and stay in the declared range.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("confidence_delta must be within (0, 1)")
    per_time_delta = (3.0 * confidence_delta) / (math.pi**2 * count**2)
    return min(2.0, math.sqrt(2.0 * math.log(1.0 / per_time_delta) / count))


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promote: bool
    reason: str
    candidate_real_samples: int
    incumbent_real_samples: int
    candidate_mean_reward: float | None = None
    incumbent_mean_reward: float | None = None
    candidate_lower_bound: float | None = None
    incumbent_upper_bound: float | None = None
    boundary_type: str | None = None
    anytime_valid: bool = False
    confidence_delta: float | None = None


class VerifiedDreamController:
    """Coordinates real learning, uncertainty-bounded dreams, and promotion."""

    def __init__(
        self,
        ledger: TransitionLedger,
        model: WorldModelBackend | None = None,
        *,
        min_confidence: float = 0.55,
        uncertainty_penalty: float = 0.75,
        experiment_exploration_bonus: float = 0.50,
        max_horizon: int = 5,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty cannot be negative")
        if experiment_exploration_bonus < 0:
            raise ValueError("experiment_exploration_bonus cannot be negative")
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        self.ledger = ledger
        self.model = model or EmpiricalWorldModel()
        self.min_confidence = min_confidence
        self.uncertainty_penalty = uncertainty_penalty
        self.experiment_exploration_bonus = experiment_exploration_bonus
        self.max_horizon = max_horizon
        self.model.fit(self.ledger.read_real())

    def observe_real(self, transition: VerifiedTransition) -> TransitionReceipt:
        receipt = self.ledger.record_real(transition)
        update = getattr(self.model, "update", None)
        if callable(update):
            update(transition)
        else:
            self.model.fit(self.ledger.read_real())
        return receipt

    def rank_actions(
        self,
        state: Sequence[float],
        actions: Iterable[Sequence[float]],
    ) -> list[tuple[tuple[float, ...], float, WorldModelPrediction]]:
        ranked = []
        for action in actions:
            action_v = _finite_vector(action, "action")
            prediction = self.model.predict(state, action_v)
            value = prediction.reward - (
                self.uncertainty_penalty * prediction.uncertainty
            )
            ranked.append((action_v, value, prediction))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def propose_experiment(
        self,
        state: Sequence[float],
        actions: Iterable[Sequence[float]],
        *,
        environment: str,
        policy_version: str = "",
    ) -> DreamRollout:
        """Choose one benchmark experiment with confidence-gated UCB.

        Deployed rollouts use pessimistic value, while a reversible real
        experiment should also value information. The acquisition is
        ``predicted_reward + beta * uncertainty`` after the confidence gate.
        Its output remains a proposal-only dream; only real execution can
        create promotion evidence.
        """
        state_v = _finite_vector(state, "state")
        action_list = [_finite_vector(action, "action") for action in actions]
        if not action_list:
            return DreamRollout((), 0.0, 0.0, "no_actions")

        eligible: list[tuple[int, float, tuple[float, ...]]] = []
        for index, action in enumerate(action_list):
            prediction = self.model.predict(state_v, action)
            if prediction.confidence < self.min_confidence:
                continue
            acquisition = prediction.reward + (
                self.experiment_exploration_bonus * prediction.uncertainty
            )
            eligible.append((index, acquisition, action))
        if not eligible:
            return DreamRollout((), 0.0, 0.0, "uncertainty_gate")

        # Preserve candidate order as the deterministic tie-break.
        _index, _acquisition, chosen = max(
            eligible,
            key=lambda item: (item[1], -item[0]),
        )
        return self.dream(
            state_v,
            lambda _state: (chosen,),
            horizon=1,
            environment=environment,
            policy_version=policy_version,
        )

    def dream(
        self,
        initial_state: Sequence[float],
        action_candidates: Callable[[tuple[float, ...]], Iterable[Sequence[float]]],
        *,
        horizon: int | None = None,
        environment: str,
        policy_version: str = "",
    ) -> DreamRollout:
        limit = min(self.max_horizon, horizon or self.max_horizon)
        if limit <= 0:
            raise ValueError("horizon must be positive")
        state = _finite_vector(initial_state, "initial_state")
        transitions: list[VerifiedTransition] = []
        predicted_return = 0.0
        pessimistic_return = 0.0
        discount = 1.0
        stopped_reason = "horizon_reached"

        for _step in range(limit):
            actions = list(action_candidates(state))
            if not actions:
                stopped_reason = "no_actions"
                break
            ranked = self.rank_actions(state, actions)
            action, pessimistic_reward, prediction = ranked[0]
            if prediction.confidence < self.min_confidence:
                stopped_reason = "uncertainty_gate"
                break
            real_receipts = self.ledger.real_receipt_hashes()
            try:
                parent_receipt_hashes = tuple(
                    real_receipts[transition_id]
                    for transition_id in prediction.supporting_transition_ids
                )
            except KeyError as exc:
                raise TransitionIntegrityError(
                    "world-model support is not committed to the real ledger"
                ) from exc
            dream_transition = VerifiedTransition(
                state=state,
                action=action,
                next_state=prediction.next_state,
                reward=prediction.reward,
                environment=environment,
                source=f"world_model:{prediction.backend}",
                verifier="model_prediction_not_ground_truth",
                strength="synthetic",
                synthetic=True,
                parent_transition_ids=prediction.supporting_transition_ids,
                parent_receipt_hashes=parent_receipt_hashes,
                prediction_confidence=prediction.confidence,
                uncertainty=prediction.uncertainty,
                policy_version=policy_version,
                model_version=prediction.model_version,
                model_backend=prediction.backend,
                influence_scope=_DREAM_INFLUENCE_SCOPE,
            )
            self.ledger.record_dream(dream_transition)
            transitions.append(dream_transition)
            predicted_return += discount * prediction.reward
            pessimistic_return += discount * pessimistic_reward
            discount *= 0.99
            state = prediction.next_state

        return DreamRollout(
            transitions=tuple(transitions),
            predicted_return=predicted_return,
            pessimistic_return=pessimistic_return,
            stopped_reason=stopped_reason,
        )

    def assess_promotion(
        self,
        candidate: Sequence[VerifiedTransition],
        incumbent: Sequence[VerifiedTransition],
        *,
        min_real_samples: int = 10,
        confidence_delta: float = 0.05,
        noninferiority_margin: float = 0.0,
    ) -> PromotionDecision:
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_delta must be within (0, 1)")
        if min_real_samples <= 0:
            raise ValueError("min_real_samples must be positive")
        if noninferiority_margin < 0:
            raise ValueError("noninferiority_margin cannot be negative")
        if any(not transition.is_real_verified for transition in candidate):
            return PromotionDecision(
                False,
                "candidate contains synthetic or unverified evidence",
                sum(t.is_real_verified for t in candidate),
                sum(t.is_real_verified for t in incumbent),
            )
        if any(not transition.is_real_verified for transition in incumbent):
            return PromotionDecision(
                False,
                "incumbent contains synthetic or unverified evidence",
                len(candidate),
                sum(t.is_real_verified for t in incumbent),
            )
        real_payloads = self.ledger.real_payload_commitments()
        if any(
            real_payloads.get(t.transition_id) != _sha256(t.as_dict())
            for t in candidate
        ):
            return PromotionDecision(
                False,
                "candidate evidence is not exactly committed to the real ledger",
                len(candidate),
                len(incumbent),
            )
        if any(
            real_payloads.get(t.transition_id) != _sha256(t.as_dict())
            for t in incumbent
        ):
            return PromotionDecision(
                False,
                "incumbent evidence is not exactly committed to the real ledger",
                len(candidate),
                len(incumbent),
            )
        candidate_ids = {transition.transition_id for transition in candidate}
        incumbent_ids = {transition.transition_id for transition in incumbent}
        if len(candidate_ids) != len(candidate) or len(incumbent_ids) != len(incumbent):
            return PromotionDecision(
                False,
                "promotion evidence contains duplicate transition IDs",
                len(candidate_ids),
                len(incumbent_ids),
            )
        if candidate_ids & incumbent_ids:
            return PromotionDecision(
                False,
                "candidate and incumbent evidence must be disjoint",
                len(candidate),
                len(incumbent),
            )
        candidate_versions = {transition.policy_version for transition in candidate}
        incumbent_versions = {transition.policy_version for transition in incumbent}
        if "" in candidate_versions or len(candidate_versions) != 1:
            return PromotionDecision(
                False,
                "candidate evidence must bind one non-empty policy version",
                len(candidate),
                len(incumbent),
            )
        if "" in incumbent_versions or len(incumbent_versions) != 1:
            return PromotionDecision(
                False,
                "incumbent evidence must bind one non-empty policy version",
                len(candidate),
                len(incumbent),
            )
        if candidate_versions == incumbent_versions:
            return PromotionDecision(
                False,
                "candidate and incumbent must bind different policy versions",
                len(candidate),
                len(incumbent),
            )
        if len(candidate) < min_real_samples or len(incumbent) < min_real_samples:
            return PromotionDecision(
                False,
                "insufficient real holdout evidence",
                len(candidate),
                len(incumbent),
            )
        environments = {transition.environment for transition in [*candidate, *incumbent]}
        if len(environments) != 1:
            return PromotionDecision(
                False,
                "candidate and incumbent evidence must share one environment",
                len(candidate),
                len(incumbent),
            )
        candidate_mean = sum(t.reward for t in candidate) / len(candidate)
        incumbent_mean = sum(t.reward for t in incumbent) / len(incumbent)

        # Dream-driven candidate generation and repeated checks make a
        # fixed-sample interval unsafe: an optimizer could keep peeking until
        # noise crosses the release line. These confidence-sequence bounds
        # hold simultaneously over every sample count, while using only real,
        # ledger-closed outcomes. The world model may choose the next real
        # experiment; it cannot enter the promotion statistic.
        candidate_lower = max(
            -1.0,
            candidate_mean
            - anytime_hoeffding_radius(len(candidate), confidence_delta),
        )
        incumbent_upper = min(
            1.0,
            incumbent_mean
            + anytime_hoeffding_radius(len(incumbent), confidence_delta),
        )
        promote = candidate_lower >= incumbent_upper - noninferiority_margin
        return PromotionDecision(
            promote=promote,
            reason=(
                "real holdout non-inferiority gate passed"
                if promote
                else "real holdout non-inferiority gate failed"
            ),
            candidate_real_samples=len(candidate),
            incumbent_real_samples=len(incumbent),
            candidate_mean_reward=candidate_mean,
            incumbent_mean_reward=incumbent_mean,
            candidate_lower_bound=candidate_lower,
            incumbent_upper_bound=incumbent_upper,
            boundary_type="stitched_hoeffding_cs",
            anytime_valid=True,
            confidence_delta=confidence_delta,
        )

    def stats(self) -> dict[str, Any]:
        real_count = len(self.ledger.read_real())
        dream_count = len(self.ledger.read_dreams())
        model_stats = self.model.stats()
        return {
            **model_stats,
            "real_transitions": real_count,
            "dream_transitions": dream_count,
            "dream_to_real_ratio": dream_count / real_count if real_count else 0.0,
            "min_confidence": self.min_confidence,
            "uncertainty_penalty": self.uncertainty_penalty,
            "experiment_exploration_bonus": self.experiment_exploration_bonus,
            "promotion_requires_real_holdout": True,
            "promotion_requires_ledger_closure": True,
            "promotion_boundary": "stitched_hoeffding_cs",
            "promotion_is_anytime_valid": True,
        }


__all__ = [
    "anytime_hoeffding_radius",
    "DreamRollout",
    "EbbiforgeWorldModelAdapter",
    "EmpiricalWorldModel",
    "InsufficientWorldModelData",
    "PromotionDecision",
    "TransitionIntegrityError",
    "TransitionLedger",
    "TransitionReceipt",
    "transition_from_ravs",
    "VerifiedDreamController",
    "VerifiedTransition",
    "WorldModelBackend",
    "WorldModelError",
    "WorldModelPrediction",
]
