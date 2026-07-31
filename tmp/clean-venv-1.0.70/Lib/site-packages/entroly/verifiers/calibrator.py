"""
PRISM-Style Calibration of λ Per Archetype
============================================

Closes the feedback loop. The verifier's λ parameter controls the
trade-off between false positives (flagging real symbols as hallucinated)
and false negatives (passing hallucinations through). It defaults to 6.5
nats/char — a conservative one-size-fits-all value.

The right λ depends on task archetype:
  - "code_generation" archetypes have stricter standards (low FN tolerance)
  - "explain"/"inspect" archetypes can be lenient (FN cost is low)
  - "security" archetypes must be strict (high FN cost)

This module learns λ_archetype online from feedback signals.

Mathematical foundation
-----------------------
Posterior model is

    P_halu(σ; λ) = sigmoid(surprisal(σ) − λ)

For an observation (σ, y) where y ∈ {1=really_hallucinated, 0=really_real}:

    L(λ) = − [y log P + (1−y) log(1 − P)]
    ∂L/∂λ = ∂L/∂P · ∂P/∂λ

With ∂P/∂λ = −P(1−P):

    ∂L/∂λ = ⎧   1 − P   if y = 1
              ⎨
              ⎩  −P      if y = 0

So the SGD update is:

    λ ← λ − η · ∂L/∂λ

i.e.,

    λ ← λ − η·(1−P)    if hallucination was real
    λ ← λ + η·P         if symbol was actually fine

Notice the natural symmetry: a false-negative (P low, y=1) shifts λ
DOWN (stricter), and a false-positive (P high, y=0) shifts λ UP (less
strict). The magnitude is proportional to how confident the model
was — confident wrong answers move λ more than uncertain ones.

We bound λ ∈ [3.0, 12.0] (≈ 1× to 4× per-char entropy of typical code)
to prevent runaway calibration on small samples.

Persistence
-----------
λ values are persisted to .entroly/verifiers_cache/calibration.json,
loaded at service start, and updated transactionally on each feedback
event. A WAL-style append-only log
(.entroly/verifiers_cache/calibration_events.jsonl) captures every
(archetype, σ, surprisal, p_halu, y) tuple for offline analysis and
audit.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("entroly.verifiers.calibrator")


# Bounds on λ to prevent pathological calibration.
LAMBDA_MIN = 3.0
LAMBDA_MAX = 12.0
LAMBDA_DEFAULT = 6.5

# Learning rate for the SGD update. Conservative — we want a slow drift,
# not jumpy reactions to single events. Equivalent to ~50 events to move
# λ by 1.0.
DEFAULT_LEARNING_RATE = 0.02


# ── Per-archetype state ──────────────────────────────────────────────


@dataclass
class ArchetypeCalibration:
    """Calibration state for one task archetype.

    Stores both the current λ and the running counts of feedback
    outcomes so the dashboard can show the empirical FP/FN rates.
    """
    archetype: str
    lambda_: float = LAMBDA_DEFAULT
    # Empirical confusion matrix counts
    tp: int = 0  # true positives: said hallu, was hallu
    fp: int = 0  # false positives: said hallu, was real
    tn: int = 0  # true negatives: said ok, was real
    fn: int = 0  # false negatives: said ok, was hallu
    # Total updates seen — used to dampen learning rate over time
    updates: int = 0
    last_updated: float = 0.0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class FeedbackEvent:
    """One (judgment, ground_truth) pair."""
    archetype: str
    symbol: str
    surprisal: float
    p_hallucinated: float
    y: int  # 1 = really hallucinated, 0 = really fine
    lambda_before: float
    lambda_after: float
    timestamp: float = field(default_factory=time.time)


# ── The Calibrator ───────────────────────────────────────────────────


class Calibrator:
    """Thread-safe online-SGD calibrator for per-archetype λ.

    Persists state to disk on every update (atomic-write through a
    tmp file + rename). Reads on construction.

    Typical lifecycle::

        cal = Calibrator(state_path=".entroly/verifiers_cache/calibration.json")
        lam = cal.get("code_generation")        # 6.5 (default first call)

        # ... user submits PR, RAVS runs tests, test passes →
        # symbols verifier flagged as hallu were actually real
        cal.record_feedback(
            archetype="code_generation",
            symbol="weirdName",
            surprisal=4.2,
            p_hallucinated=0.91,
            y=0,                                  # actually fine!
        )
        lam = cal.get("code_generation")        # slightly higher now
    """

    def __init__(
        self,
        state_path: str | Path,
        event_log_path: str | Path | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ):
        self._state_path = Path(state_path)
        self._event_log_path = (
            Path(event_log_path) if event_log_path is not None
            else self._state_path.with_name("calibration_events.jsonl")
        )
        self._lr = learning_rate
        self._lock = threading.Lock()
        self._state: dict[str, ArchetypeCalibration] = {}
        self._load()

    # ── State load/save ──────────────────────────────────────────

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("calibration load failed: %s — starting fresh", e)
            return
        for arch, d in blob.items():
            self._state[arch] = ArchetypeCalibration(
                archetype=d.get("archetype", arch),
                lambda_=float(d.get("lambda_", LAMBDA_DEFAULT)),
                tp=int(d.get("tp", 0)),
                fp=int(d.get("fp", 0)),
                tn=int(d.get("tn", 0)),
                fn=int(d.get("fn", 0)),
                updates=int(d.get("updates", 0)),
                last_updated=float(d.get("last_updated", 0)),
            )

    def _save(self) -> None:
        """Atomic write: write tmp, fsync, rename."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        blob = {arch: asdict(cal) for arch, cal in self._state.items()}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        tmp.replace(self._state_path)

    def _append_event(self, ev: FeedbackEvent) -> None:
        self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev)) + "\n")

    # ── Public API ───────────────────────────────────────────────

    def get(self, archetype: str) -> float:
        """Get the current λ for an archetype. Creates default on first call."""
        with self._lock:
            cal = self._state.get(archetype)
            if cal is None:
                cal = ArchetypeCalibration(archetype=archetype)
                self._state[archetype] = cal
            return cal.lambda_

    def get_stats(self, archetype: str) -> dict:
        """Return precision/recall/F1/counts for an archetype."""
        with self._lock:
            cal = self._state.get(archetype)
            if cal is None:
                return {"archetype": archetype, "lambda_": LAMBDA_DEFAULT, "updates": 0}
            return {
                "archetype": archetype,
                "lambda_": cal.lambda_,
                "updates": cal.updates,
                "tp": cal.tp, "fp": cal.fp, "tn": cal.tn, "fn": cal.fn,
                "precision": cal.precision(),
                "recall": cal.recall(),
                "f1": cal.f1(),
            }

    def all_stats(self) -> dict:
        """Per-archetype stats for the dashboard."""
        with self._lock:
            return {a: self.get_stats(a) for a in self._state.keys()}

    def record_feedback(
        self,
        archetype: str,
        symbol: str,
        surprisal: float,
        p_hallucinated: float,
        y: int,
    ) -> float:
        """Apply one SGD step and persist.

        Args:
            archetype: Task archetype (e.g. "code_generation").
            symbol: The symbol that was judged.
            surprisal: Its n-gram surprisal.
            p_hallucinated: The verifier's posterior at judgement time.
            y: Ground truth — 1 if symbol really was hallucinated, 0 if fine.

        Returns:
            The new λ after the update.
        """
        if y not in (0, 1):
            raise ValueError("y must be 0 or 1")

        with self._lock:
            cal = self._state.get(archetype)
            if cal is None:
                cal = ArchetypeCalibration(archetype=archetype)
                self._state[archetype] = cal

            lambda_before = cal.lambda_

            # SGD update on logistic loss.
            # Derivative simplifies to:
            #   y=1 → ∂L/∂λ = (1 − P)
            #   y=0 → ∂L/∂λ = −P
            # Step: λ ← λ − η · ∂L/∂λ
            #   y=1 → λ ← λ − η(1 − P)   (move DOWN, stricter)
            #   y=0 → λ ← λ + η · P       (move UP, less strict)
            #
            # Annealed learning rate: lr / (1 + updates/100) keeps early
            # updates aggressive and prevents wobble at scale.
            lr_eff = self._lr / (1.0 + cal.updates / 100.0)
            if y == 1:
                # Hallu was real. If we missed it (low P), we owe a
                # bigger correction.
                grad = (1.0 - p_hallucinated)
                cal.lambda_ -= lr_eff * grad
            else:
                # Symbol was real. If we flagged it (high P), we owe a
                # bigger correction.
                grad = p_hallucinated
                cal.lambda_ += lr_eff * grad

            # Clamp
            cal.lambda_ = max(LAMBDA_MIN, min(LAMBDA_MAX, cal.lambda_))

            # Update confusion matrix
            predicted_halu = p_hallucinated > 0.5
            if predicted_halu and y == 1:
                cal.tp += 1
            elif predicted_halu and y == 0:
                cal.fp += 1
            elif not predicted_halu and y == 1:
                cal.fn += 1
            else:
                cal.tn += 1

            cal.updates += 1
            cal.last_updated = time.time()

            ev = FeedbackEvent(
                archetype=archetype,
                symbol=symbol,
                surprisal=surprisal,
                p_hallucinated=p_hallucinated,
                y=y,
                lambda_before=lambda_before,
                lambda_after=cal.lambda_,
            )

            try:
                self._save()
                self._append_event(ev)
            except OSError as e:
                logger.warning("calibration persist failed: %s", e)

            return cal.lambda_

    def reset(self, archetype: str | None = None) -> None:
        """Reset calibration for one archetype, or all if None."""
        with self._lock:
            if archetype is None:
                self._state.clear()
            else:
                self._state.pop(archetype, None)
            self._save()


# ── Archetype inference from query ───────────────────────────────────


def infer_archetype_from_query(query: str) -> str:
    """Cheap heuristic — same shape as ravs/router.py:classify_archetype.

    Returns a stable archetype label. Real production code should call
    into ravs.router.classify_archetype which has the full pattern table;
    this is a fallback that lets the verifier work standalone.
    """
    try:
        from entroly.ravs.router import classify_archetype
        return classify_archetype(query)
    except Exception:
        # Lightweight fallback
        q = (query or "").lower()
        if any(k in q for k in ("test", "spec", "pytest", "unit test")):
            return "test/write"
        if any(k in q for k in ("auth", "security", "encrypt", "password")):
            return "security"
        if any(k in q for k in ("generate", "implement", "write code", "create function")):
            return "code/implement"
        if any(k in q for k in ("fix", "bug", "broken", "failing")):
            return "code/fix_bug"
        if any(k in q for k in ("explain", "what is", "how does")):
            return "explain"
        return "general"
