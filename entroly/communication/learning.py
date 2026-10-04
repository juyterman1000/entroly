"""Self-improving communication-taste selection.

This module deliberately reuses Entroly's existing PRISM 5D mathematics with
its original dimension meanings:

    recency, frequency, semantic, entropy, resonance

The weights decide which *historical communication examples* should influence
an inferred taste profile.  They never decide whether an external action is
authorized.

Learning has two paths, mirroring Entroly's existing architecture:

1. Inline verified feedback -> PRISM 5D spectral update.
2. Daemon shadow autotune -> time-split next-message style benchmark.

Only user-grounded outcomes may enter the journal. Model self-reports and mere
delivery success are not taste rewards.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    CommunicationEvent,
    CommunicationStateError,
    canonical_json,
    sha256_text,
)
from .preferences import infer_taste_from_outbound
from .store import default_store_path

PRISM_DIMS = (
    "w_recency",
    "w_frequency",
    "w_semantic",
    "w_entropy",
    "w_resonance",
)
DEFAULT_PRISM_WEIGHTS = {
    "w_recency": 0.30,
    "w_frequency": 0.20,
    "w_semantic": 0.25,
    "w_entropy": 0.15,
    "w_resonance": 0.10,
}
MIN_WEIGHT = 0.03
MAX_WEIGHT = 0.70
DEFAULT_LEARNING_RATE = 0.01
FEEDBACK_SCHEMA = "entroly.communication.learning.v1"
STATE_SCHEMA = "entroly.communication.prism5d.v1"
MAX_JOURNAL_AGE_S = 180 * 24 * 60 * 60
SAFE_FEEDBACK_SOURCES = {
    "explicit_approval",
    "explicit_rejection",
    "explicit_correction",
    "owner_rewrite",
}
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def default_learning_state_path() -> Path:
    return default_store_path().parent / "taste-prism.json"


def default_learning_journal_path() -> Path:
    return default_store_path().parent / "taste-feedback.jsonl"


def _resolve_absolute(
    value: str | os.PathLike[str] | None,
    default: Path,
    *,
    label: str,
) -> Path:
    if value is None or not str(value).strip():
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise CommunicationStateError(f"{label} must be an absolute path")
    return path.absolute()


def _scope_key(scope_type: str, scope_id: str) -> str:
    if scope_type not in {"owner", "contact", "group", "conversation"}:
        raise CommunicationStateError("invalid communication learning scope_type")
    if not str(scope_id).strip():
        raise CommunicationStateError("communication learning scope_id is required")
    return f"{scope_type}:{sha256_text(str(scope_id))[:24]}"


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    raw = {
        name: max(MIN_WEIGHT, min(MAX_WEIGHT, float(weights.get(name, 0.0))))
        for name in PRISM_DIMS
    }
    total = sum(raw.values())
    if total <= 0:
        return dict(DEFAULT_PRISM_WEIGHTS)
    return {name: value / total for name, value in raw.items()}


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.parent.chmod(0o700)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            tmp_path.chmod(0o600)
        os.replace(tmp_path, path)
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) >= 2
    }


def _jaccard(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _entropy_score(text: str) -> float:
    chars = [char for char in (text or "") if not char.isspace()]
    if len(chars) <= 1:
        return 0.0
    counts = Counter(chars)
    n = len(chars)
    entropy = -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )
    maximum = math.log2(max(2, len(counts)))
    return max(0.0, min(1.0, entropy / maximum if maximum > 0 else 0.0))


def _style_signature(event: CommunicationEvent) -> tuple[str, str]:
    length = len(event.content.strip())
    if length <= 24:
        length_bucket = "very_short"
    elif length <= 96:
        length_bucket = "short"
    else:
        length_bucket = "medium"
    emoji = "emoji" if _EMOJI_PATTERN.search(event.content) else "no_emoji"
    return length_bucket, emoji


def _feature_rows(
    events: Sequence[CommunicationEvent],
    *,
    query: str,
) -> dict[str, dict[str, float]]:
    usable = [
        event
        for event in events
        if event.direction == "outbound"
        and event.event_type == "message"
        and event.content.strip()
    ]
    if not usable:
        return {}
    newest = max(
        (event.timestamp or 0.0) for event in usable
    )
    oldest = min(
        (event.timestamp or newest) for event in usable
    )
    span = max(1.0, newest - oldest)

    signatures = Counter(_style_signature(event) for event in usable)
    max_frequency = max(signatures.values(), default=1)

    # Resonance is consensus with other examples' low-risk surface style.
    signature_counts = Counter(_style_signature(event) for event in usable)
    rows: dict[str, dict[str, float]] = {}
    for event in usable:
        timestamp = event.timestamp if event.timestamp is not None else oldest
        recency = max(0.0, min(1.0, 1.0 - ((newest - timestamp) / span)))
        signature = _style_signature(event)
        frequency = signature_counts[signature] / max_frequency
        semantic = _jaccard(query, event.content) if query.strip() else 0.5
        entropy = _entropy_score(event.content)
        resonance = (
            (signature_counts[signature] - 1) / max(1, len(usable) - 1)
            if len(usable) > 1
            else 0.0
        )
        rows[event.event_id] = {
            "w_recency": recency,
            "w_frequency": max(0.0, min(1.0, frequency)),
            "w_semantic": max(0.0, min(1.0, semantic)),
            "w_entropy": max(0.0, min(1.0, entropy)),
            "w_resonance": max(0.0, min(1.0, resonance)),
        }
    return rows


@dataclass(frozen=True, slots=True)
class SelectedTasteEvidence:
    event_ids: tuple[str, ...]
    feature_mean: dict[str, float]
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_ids": list(self.event_ids),
            "feature_mean": dict(self.feature_mean),
            "weights": dict(self.weights),
        }


@dataclass(frozen=True, slots=True)
class CommunicationFeedback:
    feedback_id: str
    scope_type: str
    scope_id_hash: str
    reward: float
    feature_vector: tuple[float, float, float, float, float]
    source: str
    evidence_event_ids: tuple[str, ...]
    created_at: float
    schema: str = FEEDBACK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_vector"] = list(self.feature_vector)
        result["evidence_event_ids"] = list(self.evidence_event_ids)
        return result


class CommunicationTasteOptimizer:
    """Per-scope PRISM 5D selector with verified-feedback learning."""

    def __init__(
        self,
        state_path: str | os.PathLike[str] | None = None,
        journal_path: str | os.PathLike[str] | None = None,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        self.state_path = _resolve_absolute(
            state_path, default_learning_state_path(), label="communication learning state"
        )
        self.journal_path = _resolve_absolute(
            journal_path,
            default_learning_journal_path(),
            label="communication learning journal",
        )
        self.learning_rate = max(1e-5, min(1.0, float(learning_rate)))
        self._lock = threading.RLock()
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema": STATE_SCHEMA,
                "scopes": {},
                "processed_feedback_ids": [],
                "last_autotune_event_count": 0,
            }
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {
                "schema": STATE_SCHEMA,
                "scopes": {},
                "processed_feedback_ids": [],
                "last_autotune_event_count": 0,
            }
        if not isinstance(raw, dict) or raw.get("schema") != STATE_SCHEMA:
            raise CommunicationStateError("unsupported communication PRISM state")
        raw.setdefault("scopes", {})
        raw.setdefault("processed_feedback_ids", [])
        raw.setdefault("last_autotune_event_count", 0)
        return raw

    def _save_state(self) -> None:
        _atomic_json_write(self.state_path, self._state)

    def weights(self, *, scope_type: str, scope_id: str) -> dict[str, float]:
        key = _scope_key(scope_type, scope_id)
        with self._lock:
            scope = self._state["scopes"].get(key, {})
            return _normalize_weights(scope.get("weights", DEFAULT_PRISM_WEIGHTS))

    def select_examples(
        self,
        events: Sequence[CommunicationEvent],
        *,
        scope_type: str,
        scope_id: str,
        query: str = "",
        top_k: int = 12,
        override_weights: Mapping[str, float] | None = None,
    ) -> SelectedTasteEvidence:
        weights = _normalize_weights(
            override_weights
            if override_weights is not None
            else self.weights(scope_type=scope_type, scope_id=scope_id)
        )
        rows = _feature_rows(events, query=query)
        scored: list[tuple[float, str]] = []
        for event_id, features in rows.items():
            score = sum(weights[name] * features[name] for name in PRISM_DIMS)
            scored.append((score, event_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_ids = tuple(
            event_id for _, event_id in scored[: max(3, min(int(top_k), 64))]
        )
        if not selected_ids:
            return SelectedTasteEvidence((), {name: 0.0 for name in PRISM_DIMS}, weights)
        feature_mean = {
            name: sum(rows[event_id][name] for event_id in selected_ids)
            / len(selected_ids)
            for name in PRISM_DIMS
        }
        return SelectedTasteEvidence(selected_ids, feature_mean, weights)

    def append_feedback(
        self,
        *,
        scope_type: str,
        scope_id: str,
        reward: float,
        selection: SelectedTasteEvidence,
        source: str,
        evidence_event_ids: Sequence[str] = (),
    ) -> CommunicationFeedback:
        if source not in SAFE_FEEDBACK_SOURCES:
            raise CommunicationStateError(
                "communication taste feedback must be explicitly user-grounded"
            )
        if not selection.event_ids:
            raise CommunicationStateError(
                "communication taste feedback requires selected historical evidence"
            )
        bounded_reward = max(-1.0, min(1.0, float(reward)))
        if source == "explicit_approval" and bounded_reward <= 0:
            raise CommunicationStateError("explicit approval reward must be positive")
        if source in {"explicit_rejection", "explicit_correction"} and bounded_reward >= 0:
            raise CommunicationStateError("negative feedback source requires negative reward")

        scope_hash = _scope_key(scope_type, scope_id)
        feature_vector = tuple(float(selection.feature_mean[name]) for name in PRISM_DIMS)
        evidence = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in (*selection.event_ids, *evidence_event_ids)
                    if str(item).strip()
                }
            )
        )
        material = {
            "scope": scope_hash,
            "reward": bounded_reward,
            "feature_vector": feature_vector,
            "source": source,
            "evidence": evidence,
        }
        feedback = CommunicationFeedback(
            feedback_id="commfb_" + sha256_text(canonical_json(material))[:40],
            scope_type=scope_type,
            scope_id_hash=scope_hash,
            reward=bounded_reward,
            feature_vector=feature_vector,  # type: ignore[arg-type]
            source=source,
            evidence_event_ids=evidence,
            created_at=time.time(),
        )
        self.journal_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            self.journal_path.parent.chmod(0o700)
        with self._lock:
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(feedback.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "posix":
                self.journal_path.chmod(0o600)
        return feedback

    def _load_feedback(self) -> list[dict[str, Any]]:
        cutoff = time.time() - MAX_JOURNAL_AGE_S
        items: list[dict[str, Any]] = []
        try:
            with self.journal_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    if (
                        isinstance(item, dict)
                        and item.get("schema") == FEEDBACK_SCHEMA
                        and float(item.get("created_at", 0.0) or 0.0) >= cutoff
                    ):
                        items.append(item)
        except FileNotFoundError:
            pass
        return items

    @staticmethod
    def _gradient(item: Mapping[str, Any]) -> list[float]:
        reward = max(-1.0, min(1.0, float(item.get("reward", 0.0))))
        values = item.get("feature_vector")
        if not isinstance(values, list) or len(values) != 5:
            raise CommunicationStateError("invalid communication feedback feature vector")
        # Center features at neutral 0.5 so positive reward reinforces features
        # that were strong in the selected examples; rejection reverses it.
        return [
            reward * (max(0.0, min(1.0, float(value))) - 0.5)
            for value in values
        ]

    @staticmethod
    def _spectral_step(
        gradient: Sequence[float],
        *,
        state_json: str | None,
        learning_rate: float,
    ) -> dict[str, Any]:
        try:
            from entroly_core import py_prism5d_step  # type: ignore

            result = json.loads(
                py_prism5d_step(list(gradient), state_json, learning_rate)
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        # Pure-Python fallback: bounded isotropic step. It intentionally does
        # not pretend to expose spectral diagnostics.
        return {
            "update": [learning_rate * float(value) for value in gradient],
            "state_json": state_json,
            "condition_number": None,
            "effective_rank": None,
            "regret_bound": None,
            "phase": "python_fallback",
            "steps": None,
            "eigenvalues": [],
            "spectral_energy": [],
        }

    def process_pending(self, *, max_items: int = 100) -> dict[str, Any]:
        """Apply unprocessed verified feedback through PRISM 5D."""
        with self._lock:
            processed = set(self._state.get("processed_feedback_ids", []))
            episodes = [
                item
                for item in self._load_feedback()
                if str(item.get("feedback_id") or "") not in processed
            ][: max(1, min(int(max_items), 1000))]
            updates = 0
            last_diagnostic: dict[str, Any] | None = None
            for item in episodes:
                feedback_id = str(item.get("feedback_id") or "")
                scope_key = str(item.get("scope_id_hash") or "")
                if not feedback_id or not scope_key:
                    continue
                scope = self._state["scopes"].setdefault(
                    scope_key,
                    {
                        "weights": dict(DEFAULT_PRISM_WEIGHTS),
                        "prism_state_json": None,
                        "updates": 0,
                        "last_updated": 0.0,
                    },
                )
                gradient = self._gradient(item)
                step = self._spectral_step(
                    gradient,
                    state_json=scope.get("prism_state_json"),
                    learning_rate=self.learning_rate,
                )
                update = step.get("update")
                if not isinstance(update, list) or len(update) != 5:
                    continue
                current = _normalize_weights(scope.get("weights", DEFAULT_PRISM_WEIGHTS))
                candidate = {
                    name: current[name]
                    + max(-0.05, min(0.05, float(update[index])))
                    for index, name in enumerate(PRISM_DIMS)
                }
                scope["weights"] = _normalize_weights(candidate)
                scope["prism_state_json"] = step.get("state_json")
                scope["updates"] = int(scope.get("updates", 0)) + 1
                scope["last_updated"] = time.time()
                scope["last_feedback_id"] = feedback_id
                scope["diagnostic"] = {
                    key: step.get(key)
                    for key in (
                        "condition_number",
                        "effective_rank",
                        "regret_bound",
                        "phase",
                        "steps",
                        "spectral_energy",
                    )
                }
                processed.add(feedback_id)
                updates += 1
                last_diagnostic = scope["diagnostic"]

            self._state["processed_feedback_ids"] = sorted(processed)[-10_000:]
            if updates:
                self._save_state()
            return {
                "processed": updates,
                "remaining": max(0, len(self._load_feedback()) - len(processed)),
                "last_diagnostic": last_diagnostic,
            }

    @staticmethod
    def _target_style(event: CommunicationEvent) -> tuple[str, str]:
        return _style_signature(event)

    def _benchmark_weights(
        self,
        events: Sequence[CommunicationEvent],
        weights: Mapping[str, float],
        *,
        scope_type: str,
        scope_id: str,
    ) -> tuple[float, int]:
        ordered = sorted(
            [
                event
                for event in events
                if event.direction == "outbound"
                and event.event_type == "message"
                and event.content.strip()
                and event.timestamp is not None
            ],
            key=lambda event: (float(event.timestamp or 0), event.event_id),
        )
        if len(ordered) < 12:
            return 0.0, 0
        start = max(6, int(len(ordered) * 0.7))
        scores: list[float] = []
        for index in range(start, len(ordered)):
            history = ordered[:index]
            target = ordered[index]
            selection = self.select_examples(
                history,
                scope_type=scope_type,
                scope_id=scope_id,
                query="",
                top_k=min(12, len(history)),
                override_weights=weights,
            )
            selected_set = set(selection.event_ids)
            selected_events = [event for event in history if event.event_id in selected_set]
            taste = infer_taste_from_outbound(
                selected_events,
                scope_type=scope_type,  # type: ignore[arg-type]
                scope_id=scope_id,
                minimum_samples=3,
            )
            if taste is None:
                continue
            target_length, target_emoji = self._target_style(target)
            score = 0.0
            score += 0.5 if taste.response_length == target_length else 0.0
            predicted_emoji = (
                "no_emoji" if taste.emoji_level == "none" else "emoji"
            )
            score += 0.5 if predicted_emoji == target_emoji else 0.0
            scores.append(score)
        return (
            sum(scores) / len(scores) if scores else 0.0,
            len(scores),
        )

    def shadow_autotune(
        self,
        events: Sequence[CommunicationEvent],
        *,
        scope_type: str = "owner",
        scope_id: str = "owner-global",
        min_cases: int = 8,
    ) -> dict[str, Any]:
        """Time-split local autotune; only measured improvement is promoted."""
        current = self.weights(scope_type=scope_type, scope_id=scope_id)
        baseline, cases = self._benchmark_weights(
            events, current, scope_type=scope_type, scope_id=scope_id
        )
        if cases < min_cases:
            return {
                "status": "insufficient_holdout",
                "cases": cases,
                "baseline": baseline,
                "promoted": False,
            }

        candidates: list[dict[str, float]] = []
        step = 0.04
        for name in PRISM_DIMS:
            for direction in (-1.0, 1.0):
                candidate = dict(current)
                candidate[name] += direction * step
                candidates.append(_normalize_weights(candidate))

        best = current
        best_score = baseline
        for candidate in candidates:
            score, candidate_cases = self._benchmark_weights(
                events,
                candidate,
                scope_type=scope_type,
                scope_id=scope_id,
            )
            if candidate_cases == cases and score > best_score + 0.02:
                best = candidate
                best_score = score

        promoted = best != current
        if promoted:
            key = _scope_key(scope_type, scope_id)
            with self._lock:
                scope = self._state["scopes"].setdefault(
                    key,
                    {
                        "weights": dict(DEFAULT_PRISM_WEIGHTS),
                        "prism_state_json": None,
                        "updates": 0,
                        "last_updated": 0.0,
                    },
                )
                scope["weights"] = best
                scope["last_autotune"] = {
                    "baseline": baseline,
                    "score": best_score,
                    "cases": cases,
                    "promoted_at": time.time(),
                }
                self._state["last_autotune_event_count"] = len(events)
                self._save_state()

        return {
            "status": "completed",
            "cases": cases,
            "baseline": baseline,
            "best_score": best_score,
            "promoted": promoted,
            "weights": best,
        }

    def daemon_cycle(
        self,
        events: Sequence[CommunicationEvent],
        *,
        scope_type: str = "owner",
        scope_id: str = "owner-global",
    ) -> dict[str, Any]:
        online = self.process_pending()
        last_count = int(self._state.get("last_autotune_event_count", 0) or 0)
        outbound_count = sum(
            event.direction == "outbound" and event.event_type == "message"
            for event in events
        )
        autotune: dict[str, Any] = {
            "status": "unchanged",
            "promoted": False,
        }
        if outbound_count >= 20 and outbound_count >= last_count + 5:
            autotune = self.shadow_autotune(
                events,
                scope_type=scope_type,
                scope_id=scope_id,
            )
            with self._lock:
                self._state["last_autotune_event_count"] = outbound_count
                self._save_state()
        return {"online_prism": online, "autotune": autotune}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": STATE_SCHEMA,
                "scopes": {
                    key: {
                        "weights": _normalize_weights(value.get("weights", {})),
                        "updates": int(value.get("updates", 0)),
                        "last_updated": value.get("last_updated"),
                        "diagnostic": value.get("diagnostic"),
                        "last_autotune": value.get("last_autotune"),
                    }
                    for key, value in self._state.get("scopes", {}).items()
                },
                "feedback_episodes": len(self._load_feedback()),
                "processed_feedback": len(
                    self._state.get("processed_feedback_ids", [])
                ),
                "learning_rate": self.learning_rate,
                "authority_surface": "none",
            }
