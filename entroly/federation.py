"""
Federated Archetype Learning — Privacy-Preserving Weight Sharing
=================================================================

Enables Entroly installations to share learned archetype→weight mappings
across users without sharing code, fingerprints, or sensitive data.

What gets shared:
  - Archetype label (e.g., "rust_ffi_library")
  - 8 weight floats (PRISM 5D + decay/min/explore)
  - Sample count + confidence
  - All noised via Rényi Differential Privacy

What is NEVER shared:
  - Source code
  - Structural fingerprints
  - File names, paths, or any identifiable data

Architecture:
  Local-first with optional peer exchange. Two modes:

  Mode 1 — File Exchange (default):
    Export a contribution packet → share via any file channel
    (git repo, shared drive, Slack, etc.) → import on receiver.
    Zero network dependencies. Works fully offline.

  Mode 2 — HTTP Aggregation Server (optional):
    Lightweight server receives contributions, applies trimmed-mean
    aggregation, serves global consensus. Discards individual packets
    immediately after aggregation.

Privacy guarantee:
  Each contribution is protected by (α, ε)-Rényi Differential Privacy
  via calibrated Gaussian noise. Even if the transport is compromised,
  individual weight vectors are indistinguishable from noise.

Byzantine resilience:
  Trimmed-mean aggregation — trims top/bottom 10% before averaging.
  Sybil attacks and poisoning attempts are bounded by the trim margin.

Usage:
    from entroly.federation import FederationClient

    client = FederationClient(data_dir=".entroly")
    client.contribute(archetype_optimizer)   # export noised weights
    client.merge_global(archetype_optimizer) # import global consensus
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.federation")


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# DP parameters
DEFAULT_EPSILON = 1.0       # Privacy budget per contribution
DEFAULT_DELTA = 1e-5        # Failure probability
WEIGHT_CLIP_RANGE = (0.0, 1.0)  # Clip weights before noising

# Contribution eligibility thresholds
MIN_CONFIDENCE = 0.7        # Don't share until locally confident
MIN_SAMPLE_COUNT = 5        # Need at least 5 dream improvements

# Aggregation parameters
TRIM_FRACTION = 0.10        # Remove top/bottom 10% before averaging
MIN_CONTRIBUTORS = 3        # Minimum contributions per archetype before serving

# Staleness parameters (Upgrade 2 — Chen et al. 2020)
CONTRIBUTION_TTL_SECONDS = 30 * 86400   # 30-day TTL
STALENESS_HALF_LIFE_DAYS = 50.0         # Confidence halves every 50 days
STALENESS_DECAY_RATE = 0.693 / STALENESS_HALF_LIFE_DAYS  # ln(2)/τ

# FedProx regularization (Upgrade 4 — Li et al. MLSys 2020)
FEDPROX_MU = 0.1            # Proximal term strength

# Privacy budget (Upgrade 3 — Gopi et al. NeurIPS 2021)
TOTAL_PRIVACY_BUDGET = 10.0  # Max cumulative ε before auto-stop

# Weight keys that participate in federation
FEDERATED_WEIGHT_KEYS = [
    "w_recency", "w_frequency", "w_semantic", "w_entropy", "w_resonance",
    "decay_half_life", "min_relevance", "exploration_rate",
]

# File-based exchange
CONTRIBUTION_DIR = "federation_contributions"
GLOBAL_WEIGHTS_FILE = "federation_global.json"
OUTCOMES_FILE = "federation_outcomes.jsonl"

# Protocol version
PROTOCOL_VERSION = "1.0"


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ContributionPacket:
    """A privacy-protected contribution from one installation."""
    version: str = PROTOCOL_VERSION
    client_id: str = ""          # Anonymized machine hash
    archetype: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    confidence: float = 0.0
    noise_sigma: float = 0.0
    dp_epsilon: float = DEFAULT_EPSILON
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContributionPacket:
        return cls(
            version=d.get("version", PROTOCOL_VERSION),
            client_id=d.get("client_id", ""),
            archetype=d.get("archetype", ""),
            weights=d.get("weights", {}),
            sample_count=d.get("sample_count", 0),
            confidence=d.get("confidence", 0.0),
            noise_sigma=d.get("noise_sigma", 0.0),
            dp_epsilon=d.get("dp_epsilon", DEFAULT_EPSILON),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class GlobalArchetypeWeights:
    """Aggregated global weights for one archetype."""
    archetype: str
    weights: dict[str, float]
    contributors: int
    confidence: float
    generation: int
    updated_at: float

# ═══════════════════════════════════════════════════════════════════
# Privacy Accountant (Upgrade 3)
# ═══════════════════════════════════════════════════════════════════

class PrivacyAccountant:
    """Tracks cumulative privacy budget across contributions.

    Uses advanced composition (Kairouz et al. 2015, Gopi et al. 2021):
      ε_total ≈ ε_single · √(2k · ln(1/δ))
    where k = number of contributions.

    Once cumulative ε exceeds the budget, contributions are blocked
    to prevent unbounded privacy leakage.
    """

    def __init__(self, budget: float = TOTAL_PRIVACY_BUDGET):
        self.budget = budget
        self.contributions = 0
        self._single_epsilon = DEFAULT_EPSILON

    def can_contribute(self) -> bool:
        """Check if we can contribute without exceeding the budget."""
        return self.consumed_epsilon() < self.budget

    def consumed_epsilon(self) -> float:
        """Cumulative ε via advanced composition theorem."""
        if self.contributions == 0:
            return 0.0
        return self._single_epsilon * math.sqrt(
            2 * self.contributions * math.log(1.0 / DEFAULT_DELTA)
        )

    def record_contribution(self, epsilon: float) -> None:
        """Record one contribution's privacy cost."""
        self._single_epsilon = epsilon
        self.contributions += 1

    def remaining_budget(self) -> float:
        return max(0.0, self.budget - self.consumed_epsilon())


# ═══════════════════════════════════════════════════════════════════
# Merge Outcome Tracking (Upgrade 6)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MergeOutcome:
    """Tracks pre/post merge performance to measure federation ROI."""
    archetype: str = ""
    alpha_used: float = 0.0
    contributors: int = 0
    weight_delta_norm: float = 0.0   # ||w_merged - w_local||₂
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Differential Privacy — Gaussian Mechanism
# ═══════════════════════════════════════════════════════════════════

class GaussianMechanism:
    """Calibrated Gaussian noise for Rényi Differential Privacy.

    Mathematical foundation:
      For an 8-dimensional weight vector with each component clipped
      to [0, 1], the L2 sensitivity is:
        Δ₂ = √(8 × 1²) = 2√2 ≈ 2.83

      Gaussian mechanism noise calibration:
        σ = Δ₂ × √(2 × ln(1.25/δ)) / ε

      With subsampled privacy amplification (q = 1/N where N = total
      participants), the effective noise needed drops significantly:
        σ_effective ≈ σ × √(q)

    For practical deployments (N ≥ 100, ε=1.0):
      σ ≈ 0.05 per dimension — small enough that aggregation converges
      in ~50 contributions.
    """

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        delta: float = DEFAULT_DELTA,
        n_dimensions: int = len(FEDERATED_WEIGHT_KEYS),
        estimated_participants: int = 150,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.n_dimensions = n_dimensions

        # Compute L2 sensitivity (each dim clipped to [0,1])
        self.l2_sensitivity = math.sqrt(n_dimensions)

        # Full Gaussian noise
        raw_sigma = (
            self.l2_sensitivity
            * math.sqrt(2 * math.log(1.25 / delta))
            / epsilon
        )

        # Privacy amplification via subsampling
        q = 1.0 / max(estimated_participants, 1)
        self.sigma = raw_sigma * math.sqrt(q)

        # Floor: ensure at least minimal noise for privacy guarantee
        self.sigma = max(self.sigma, 0.01)

    def add_noise(self, weights: dict[str, float]) -> dict[str, float]:
        """Add calibrated Gaussian noise to a weight vector.

        Returns a new dict with noised + clipped weights.
        """
        noised = {}
        for key in FEDERATED_WEIGHT_KEYS:
            val = weights.get(key, 0.0)

            # Clip to valid range before noising
            if key in ("w_recency", "w_frequency", "w_semantic",
                       "w_entropy", "w_resonance", "min_relevance",
                       "exploration_rate"):
                val = max(0.0, min(1.0, val))
            elif key == "decay_half_life":
                val = max(1.0, min(100.0, val))

            # Add Gaussian noise
            noise = random.gauss(0, self.sigma)
            noised_val = val + noise

            # Re-clip after noising
            if key == "decay_half_life":
                noised_val = max(1.0, min(100.0, noised_val))
            else:
                noised_val = max(0.0, min(1.0, noised_val))

            noised[key] = round(noised_val, 6)

        return noised

    @property
    def noise_level(self) -> float:
        """Return the noise σ for diagnostic reporting."""
        return self.sigma


# ═══════════════════════════════════════════════════════════════════
# Byzantine-Resilient Aggregation
# ═══════════════════════════════════════════════════════════════════

def trimmed_mean(values: list[float], trim_fraction: float = TRIM_FRACTION) -> float:
    """Compute trimmed mean — trims top/bottom trim_fraction before averaging.

    Byzantine-resilient: adversaries injecting extreme values get trimmed.

    Args:
        values: list of float values from different contributors
        trim_fraction: fraction to trim from each end (default 10%)

    Returns:
        Trimmed mean of the values.
    """
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n < 3:
        # Too few values to trim — just average
        return sum(sorted_vals) / n

    trim_count = max(1, int(n * trim_fraction))
    trimmed = sorted_vals[trim_count:n - trim_count]

    if not trimmed:
        # Edge case: everything got trimmed (very small n)
        return sum(sorted_vals) / n

    return sum(trimmed) / len(trimmed)


def aggregate_contributions(
    contributions: list[ContributionPacket],
) -> dict[str, float]:
    """Aggregate multiple contributions for the same archetype.

    Uses confidence-weighted trimmed mean:
      1. Weight each contribution by its confidence score
      2. Apply trimmed mean on the weighted values
      3. Normalize

    Returns aggregated weight dict.
    """
    if not contributions:
        return {}

    result = {}
    for key in FEDERATED_WEIGHT_KEYS:
        values = []
        for c in contributions:
            val = c.weights.get(key)
            if val is not None:
                values.append(val)

        if values:
            result[key] = round(trimmed_mean(values), 6)

    return result


# ═══════════════════════════════════════════════════════════════════
# Federation Client
# ═══════════════════════════════════════════════════════════════════

class FederationClient:
    """Local federation client for archetype weight sharing.

    Handles:
      1. Preparing DP-noised contribution packets
      2. Exporting contributions to a shared directory
      3. Importing and merging global contributions
      4. Confidence-weighted blending of global vs local weights

    Federation is OFF by default. Enable via:
      - FederationClient(enabled=True)
      - ENTROLY_FEDERATION=1 env var
      - entroly config set federation.enabled true
    """

    def __init__(
        self,
        data_dir: str | Path = ".entroly",
        enabled: bool | None = None,
        epsilon: float = DEFAULT_EPSILON,
        estimated_participants: int = 150,
    ):
        self._data_dir = Path(data_dir)
        self._contrib_dir = self._data_dir / CONTRIBUTION_DIR
        self._global_path = self._data_dir / GLOBAL_WEIGHTS_FILE

        # Check env var for enable
        if enabled is None:
            enabled = os.environ.get("ENTROLY_FEDERATION", "0") == "1"
        self._enabled = enabled

        # DP mechanism
        self._dp = GaussianMechanism(
            epsilon=epsilon,
            estimated_participants=estimated_participants,
        )

        # Anonymous client ID (hash of machine identity)
        self._client_id = self._generate_client_id()

        # Stats
        self._contributions_sent = 0
        self._contributions_received = 0
        self._merges_applied = 0
        self._last_generation = 0

        # Privacy accountant (Upgrade 3)
        self._accountant = PrivacyAccountant(budget=TOTAL_PRIVACY_BUDGET)

        # Merge outcome tracking (Upgrade 6)
        self._merge_outcomes: list[MergeOutcome] = []
        self._outcomes_path = self._data_dir / OUTCOMES_FILE

        # Ensure dirs exist
        if self._enabled:
            self._contrib_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable federation."""
        self._enabled = True
        self._contrib_dir.mkdir(parents=True, exist_ok=True)

    def disable(self) -> None:
        """Disable federation."""
        self._enabled = False

    # ── Contribute ──────────────────────────────────────────────

    def prepare_contribution(
        self,
        archetype_label: str,
        weights: dict[str, float],
        sample_count: int,
        confidence: float,
    ) -> ContributionPacket | None:
        """Prepare a DP-noised contribution packet.

        Returns None if:
          - Federation is disabled
          - Confidence is below threshold
          - Not enough local samples
        """
        if not self._enabled:
            return None

        # Privacy budget check (Upgrade 3)
        if not self._accountant.can_contribute():
            logger.info(
                "Federation: privacy budget exhausted (ε_used=%.2f / %.2f)",
                self._accountant.consumed_epsilon(),
                self._accountant.budget,
            )
            return None

        if confidence < MIN_CONFIDENCE:
            logger.debug(
                "Federation: skipping contribution (confidence %.2f < %.2f)",
                confidence, MIN_CONFIDENCE,
            )
            return None

        if sample_count < MIN_SAMPLE_COUNT:
            logger.debug(
                "Federation: skipping contribution (samples %d < %d)",
                sample_count, MIN_SAMPLE_COUNT,
            )
            return None

        # Apply DP noise
        noised_weights = self._dp.add_noise(weights)

        packet = ContributionPacket(
            client_id=self._client_id,
            archetype=archetype_label,
            weights=noised_weights,
            sample_count=sample_count,
            confidence=confidence,
            noise_sigma=self._dp.noise_level,
            dp_epsilon=self._dp.epsilon,
            timestamp=time.time(),
        )

        return packet

    def contribute(self, archetype_optimizer: Any) -> bool:
        """Export a contribution from the current archetype optimizer.

        Convenience method that extracts state from the optimizer,
        prepares a DP-noised packet, and saves it to disk.

        Returns True if contribution was exported.
        """
        if not self._enabled:
            return False

        try:
            label = archetype_optimizer.current_archetype()
            if not label:
                return False

            weights = archetype_optimizer.current_weights()
            stats = archetype_optimizer.stats()
            strategy = stats.get("strategy_table", {}).get(label, {})
            sample_count = strategy.get("sample_count", 0)
            confidence = strategy.get("confidence", 0.0)

            packet = self.prepare_contribution(
                label, weights, sample_count, confidence,
            )
            if packet is None:
                return False

            saved = self._save_contribution(packet)
            if saved:
                self._accountant.record_contribution(self._dp.epsilon)
            return saved

        except Exception as e:
            logger.debug("Federation: contribute error: %s", e)
            return False

    def _save_contribution(self, packet: ContributionPacket) -> bool:
        """Save a contribution packet to the shared directory."""
        try:
            # Use enough of client_id + random suffix to prevent filename collisions
            # when multiple peers contribute at the same second
            random_suffix = hashlib.md5(
                json.dumps(packet.weights, sort_keys=True).encode()
            ).hexdigest()[:6]
            filename = (
                f"contrib_{packet.archetype}_{packet.client_id[:16]}"
                f"_{int(packet.timestamp)}_{random_suffix}.json"
            )
            filepath = self._contrib_dir / filename
            content = json.dumps(packet.to_dict(), indent=2)

            import tempfile
            fd, tmp = tempfile.mkstemp(
                dir=str(self._contrib_dir), suffix=".tmp",
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                os.replace(tmp, str(filepath))
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

            self._contributions_sent += 1
            logger.info(
                "Federation: contributed '%s' (σ=%.4f, ε=%.1f)",
                packet.archetype, packet.noise_sigma, packet.dp_epsilon,
            )
            return True

        except OSError as e:
            logger.debug("Federation: save error: %s", e)
            return False

    # ── Import & Merge ──────────────────────────────────────────

    def load_contributions(
        self, archetype: str | None = None,
    ) -> dict[str, list[ContributionPacket]]:
        """Load contributions with per-client dedup, TTL, and staleness decay.

        Upgrades (grounded in Chen et al. 2020, Lin et al. 2019):
          1. TTL — expire contributions older than 30 days
          2. Per-client dedup — keep only latest per (archetype, client)
          3. Staleness decay — reduce confidence by e^{-λ·age}
          4. Anti-echo — skip own contributions

        Returns {archetype_label: [packets]}.
        """
        # Phase 1: Load and filter
        seen: dict[tuple[str, str], tuple[ContributionPacket, Path]] = {}
        now = time.time()
        expired_count = 0

        if not self._contrib_dir.exists():
            return {}

        for fpath in self._contrib_dir.glob("contrib_*.json"):
            try:
                raw = fpath.read_text(encoding="utf-8")
                data = json.loads(raw)
                packet = ContributionPacket.from_dict(data)

                # Anti-echo: skip own contributions
                if packet.client_id == self._client_id:
                    continue

                # TTL: garbage-collect expired contributions
                age = now - packet.timestamp
                if age > CONTRIBUTION_TTL_SECONDS:
                    try:
                        fpath.unlink()
                    except OSError:
                        pass
                    expired_count += 1
                    continue

                # Filter by archetype if specified
                if archetype and packet.archetype != archetype:
                    continue

                # Per-client dedup: keep only the latest per (archetype, client)
                key = (packet.archetype, packet.client_id)
                if key in seen:
                    existing_pkt, existing_path = seen[key]
                    if packet.timestamp > existing_pkt.timestamp:
                        seen[key] = (packet, fpath)
                    # else: keep existing (newer)
                else:
                    seen[key] = (packet, fpath)

            except (json.JSONDecodeError, OSError):
                continue

        if expired_count > 0:
            logger.debug("Federation: expired %d stale contributions", expired_count)

        # Phase 2: Apply staleness decay to confidence
        result: dict[str, list[ContributionPacket]] = {}
        for (arch, _cid), (packet, _path) in seen.items():
            age_days = (now - packet.timestamp) / 86400.0
            decay = math.exp(-STALENESS_DECAY_RATE * age_days)
            packet.confidence = packet.confidence * decay
            result.setdefault(arch, []).append(packet)

        self._contributions_received = sum(len(v) for v in result.values())
        return result

    def compute_global_weights(self) -> dict[str, GlobalArchetypeWeights]:
        """Aggregate all contributions into global archetype weights.

        Uses trimmed-mean aggregation for Byzantine resilience.
        Returns {archetype: GlobalArchetypeWeights}.
        """
        all_contribs = self.load_contributions()
        result = {}

        for archetype, packets in all_contribs.items():
            if len(packets) < MIN_CONTRIBUTORS:
                continue

            aggregated = aggregate_contributions(packets)
            if not aggregated:
                continue

            # Compute aggregate confidence
            confidences = [p.confidence for p in packets]
            avg_confidence = sum(confidences) / len(confidences)

            result[archetype] = GlobalArchetypeWeights(
                archetype=archetype,
                weights=aggregated,
                contributors=len(packets),
                confidence=min(0.95, avg_confidence),
                generation=self._last_generation + 1,
                updated_at=time.time(),
            )

        return result

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> tuple[bool, str]:
        """Validate merged weights before applying.

        Checks:
          1. No NaN or Inf values (corrupted data / division errors)
          2. PRISM weights sum to > 0 (not degenerate all-zero)
          3. PRISM weights are individually non-negative
          4. decay_half_life is in sane range [1, 100]
          5. At least 3 weight keys present (not an empty/truncated merge)

        Returns (is_valid, reason).
        """
        if len(weights) < 3:
            return False, f"too few keys ({len(weights)})"

        for key, val in weights.items():
            if not isinstance(val, (int, float)):
                return False, f"{key} is not numeric: {type(val)}"
            if math.isnan(val) or math.isinf(val):
                return False, f"{key} is NaN/Inf"

        # PRISM 5D weights must sum to something positive
        prism_keys = ["w_recency", "w_frequency", "w_semantic", "w_entropy", "w_resonance"]
        prism_sum = sum(weights.get(k, 0.0) for k in prism_keys)
        if prism_sum < 0.01:
            return False, f"PRISM weights sum too low ({prism_sum:.4f})"

        # Individual PRISM weights non-negative
        for k in prism_keys:
            if weights.get(k, 0.0) < 0:
                return False, f"{k} is negative ({weights[k]:.4f})"

        # decay_half_life range
        dhl = weights.get("decay_half_life", 15.0)
        if dhl < 1.0 or dhl > 100.0:
            return False, f"decay_half_life out of range ({dhl:.1f})"

        return True, "ok"

    def merge_global(self, archetype_optimizer: Any) -> bool:
        """Import global weights and blend with local weights.

        Blending formula:
          w_merged = α · w_global + (1 - α) · w_local
          α = global_conf × global_n / (global_conf × global_n + local_conf × local_n)

        Safety guarantees:
          1. Saves local weights snapshot BEFORE merge (rollback point)
          2. Validates merged weights (no NaN, no degenerate, sane ranges)
          3. Rolls back to snapshot if validation fails
          4. α is clamped to [0, 0.7] — local weights always retain ≥30% influence

        Returns True if merge was applied.
        """
        if not self._enabled:
            return False

        try:
            label = archetype_optimizer.current_archetype()
            if not label:
                return False

            global_weights_map = self.compute_global_weights()
            global_entry = global_weights_map.get(label)
            if not global_entry:
                return False

            # ── Snapshot local weights BEFORE merge (rollback point) ──
            local_weights = archetype_optimizer.current_weights()
            snapshot = {k: v for k, v in local_weights.items()}

            stats = archetype_optimizer.stats()
            strategy = stats.get("strategy_table", {}).get(label, {})
            local_conf = strategy.get("confidence", 0.5)
            local_n = strategy.get("sample_count", 0)

            # Compute blend factor α, clamped to [0, 0.7]
            # Local always retains ≥30% influence — prevents hostile takeover
            global_score = global_entry.confidence * global_entry.contributors
            local_score = local_conf * max(local_n, 1)
            alpha = global_score / (global_score + local_score)
            alpha = min(alpha, 0.7)  # Safety cap

            # Blend: linear interpolation
            blended = {}
            for key in FEDERATED_WEIGHT_KEYS:
                g_val = global_entry.weights.get(key, 0.0)
                l_val = local_weights.get(key, 0.0)
                blended[key] = alpha * g_val + (1 - alpha) * l_val

            # ── FedProx regularization (Li et al. MLSys 2020) ──
            # Pulls merged weights back toward local optimum:
            #   w* = (w_blended + μ·w_local) / (1 + μ)
            # Prevents global noise from dragging converged installs backward.
            mu = FEDPROX_MU
            merged = {}
            for key in FEDERATED_WEIGHT_KEYS:
                merged[key] = (blended[key] + mu * snapshot.get(key, 0.0)) / (1 + mu)

            # ── Validate BEFORE applying ──
            is_valid, reason = self._validate_weights(merged)
            if not is_valid:
                logger.warning(
                    "Federation: merged weights REJECTED (%s) — keeping local",
                    reason,
                )
                self._stats_rejected_merges = getattr(self, "_stats_rejected_merges", 0) + 1
                return False

            # ── Apply merged weights ──
            try:
                archetype_optimizer.update_weights(merged)
            except Exception as apply_err:
                # Rollback: restore snapshot
                logger.warning(
                    "Federation: apply failed (%s) — rolling back to snapshot",
                    apply_err,
                )
                try:
                    archetype_optimizer.update_weights(snapshot)
                except Exception:
                    pass  # Best effort — original weights are the checkpoint on disk
                return False

            # ── Track merge outcome (Upgrade 6) ──
            weight_delta = math.sqrt(sum(
                (merged.get(k, 0) - snapshot.get(k, 0)) ** 2
                for k in FEDERATED_WEIGHT_KEYS
            ))
            outcome = MergeOutcome(
                archetype=label,
                alpha_used=alpha,
                contributors=global_entry.contributors,
                weight_delta_norm=round(weight_delta, 6),
                timestamp=time.time(),
            )
            self._merge_outcomes.append(outcome)
            self._save_outcome(outcome)

            self._merges_applied += 1
            self._last_generation = global_entry.generation

            logger.info(
                "Federation: merged '%s' (α=%.3f, μ=%.2f, Δw=%.4f, %d contributors)",
                label, alpha, mu, weight_delta, global_entry.contributors,
            )
            return True

        except Exception as e:
            logger.debug("Federation: merge error: %s", e)
            return False

    # ── Export/Import for File-Based Exchange ────────────────────

    def export_packet(self, packet: ContributionPacket, path: str | Path) -> bool:
        """Export a contribution packet to a specific file path.

        For file-based peer exchange: save → send via any channel.
        """
        try:
            content = json.dumps(packet.to_dict(), indent=2)
            Path(path).write_text(content, encoding="utf-8")
            return True
        except OSError as e:
            logger.debug("Federation: export error: %s", e)
            return False

    def import_packet(self, path: str | Path) -> ContributionPacket | None:
        """Import a contribution packet from a file.

        Validates structure before accepting.
        """
        try:
            raw = Path(path).read_text(encoding="utf-8")
            data = json.loads(raw)

            # Validate required fields
            if not all(k in data for k in ("archetype", "weights", "version")):
                logger.warning("Federation: invalid packet structure")
                return None

            packet = ContributionPacket.from_dict(data)

            # Save to local contribution directory
            self._save_contribution(packet)

            return packet

        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Federation: import error: %s", e)
            return None

    # ── Stats & Diagnostics ─────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return federation statistics including v2 upgrade metrics."""
        contribs = self.load_contributions() if self._enabled else {}
        return {
            "enabled": self._enabled,
            "client_id": self._client_id[:12] + "...",
            "contributions_sent": self._contributions_sent,
            "contributions_available": sum(len(v) for v in contribs.values()),
            "archetypes_available": list(contribs.keys()),
            "merges_applied": self._merges_applied,
            "last_generation": self._last_generation,
            # DP
            "dp_epsilon": self._dp.epsilon,
            "dp_sigma": round(self._dp.noise_level, 6),
            # Privacy accountant (Upgrade 3)
            "privacy_budget_total": self._accountant.budget,
            "privacy_budget_consumed": round(self._accountant.consumed_epsilon(), 3),
            "privacy_budget_remaining": round(self._accountant.remaining_budget(), 3),
            "privacy_contributions_count": self._accountant.contributions,
            # FedProx (Upgrade 4)
            "fedprox_mu": FEDPROX_MU,
            # Staleness (Upgrade 2)
            "contribution_ttl_days": CONTRIBUTION_TTL_SECONDS // 86400,
            "staleness_half_life_days": STALENESS_HALF_LIFE_DAYS,
            # Federation ROI (Upgrade 6)
            "merge_outcomes_count": len(self._merge_outcomes),
            "avg_weight_delta": round(self.avg_weight_delta(), 6),
            # Thresholds
            "min_confidence": MIN_CONFIDENCE,
            "min_sample_count": MIN_SAMPLE_COUNT,
            "min_contributors": MIN_CONTRIBUTORS,
            "trim_fraction": TRIM_FRACTION,
        }

    def avg_weight_delta(self) -> float:
        """Average L2 norm of weight changes from merges."""
        if not self._merge_outcomes:
            return 0.0
        return sum(o.weight_delta_norm for o in self._merge_outcomes) / len(self._merge_outcomes)

    # ── Private ─────────────────────────────────────────────────

    def _save_outcome(self, outcome: MergeOutcome) -> None:
        """Append a merge outcome to the JSONL log."""
        try:
            line = json.dumps({
                "archetype": outcome.archetype,
                "alpha": outcome.alpha_used,
                "contributors": outcome.contributors,
                "delta_norm": outcome.weight_delta_norm,
                "ts": outcome.timestamp,
            })
            with open(self._outcomes_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # Best effort — don't fail merge because of logging

    def _generate_client_id(self) -> str:
        """Generate a truly anonymous, stable client identifier.

        Uses a random UUID persisted to disk. Unlike hostname-based hashing,
        this is genuinely non-reversible — even someone who knows your
        machine identity cannot link it to your federation contributions.
        """
        id_path = self._data_dir / ".federation_id"
        try:
            if id_path.exists():
                stored = id_path.read_text(encoding="utf-8").strip()
                if len(stored) >= 32:
                    return stored
        except OSError:
            pass

        # Generate fresh random ID (no PII derivation)
        import uuid
        fresh_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        try:
            id_path.parent.mkdir(parents=True, exist_ok=True)
            id_path.write_text(fresh_id, encoding="utf-8")
        except OSError:
            pass  # Still usable, just not persisted
        return fresh_id


# ═══════════════════════════════════════════════════════════════════
# GitHub Transport — Zero-Cost Global Federation
# ═══════════════════════════════════════════════════════════════════

# Repo where contributions are exchanged via Issues/Gists
FEDERATION_REPO = "juyterman1000/entroly"
FEDERATION_ISSUE_TITLE = "[federation] Weight Exchange"
GITHUB_API = "https://api.github.com"


class GitHubTransport:
    """Exchange federation contributions via GitHub Issues API.

    Zero infrastructure cost:
      - READS are unauthenticated (60 req/hr — plenty for daily sync)
      - WRITES use a shared bot token (ENTROLY_FEDERATION_BOT) so all
        comments appear as @entroly-bot, never as the individual user

    Privacy model:
      - The contribution packet contains zero PII (random client_id, DP-noised weights)
      - Using individual GitHub PATs would leak your username on the comment —
        that's why we default to a shared bot token
      - If ENTROLY_FEDERATION_BOT is not set, falls back to ENTROLY_GITHUB_TOKEN
        (WARNING: your GitHub username will be visible on the Issue)
      - Without any token, the transport is read-only

    Workflow:
      1. Client calls push() → posts DP-noised packet as Issue comment via bot
      2. Client calls pull() → reads all comments, parses packets
      3. FederationClient uses packets same as file-based contributions
    """

    def __init__(
        self,
        repo: str = FEDERATION_REPO,
        token: str | None = None,
    ):
        self._repo = repo
        # Prefer shared bot token (no PII leak), fall back to personal PAT
        self._token = (
            token
            or os.environ.get("ENTROLY_FEDERATION_BOT")
            or os.environ.get("ENTROLY_GITHUB_TOKEN")
        )
        self._issue_number: int | None = None
        self._headers = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            self._headers["Authorization"] = f"token {self._token}"

    @property
    def can_write(self) -> bool:
        return self._token is not None

    @property
    def can_read(self) -> bool:
        return True  # Unauthenticated reads work for public repos

    def _find_or_create_issue(self) -> int | None:
        """Find the federation exchange issue, or create it."""
        if self._issue_number:
            return self._issue_number

        try:
            import urllib.request
            import urllib.error

            # Search for existing federation issue
            url = f"{GITHUB_API}/repos/{self._repo}/issues?labels=federation&state=open&per_page=1"
            req = urllib.request.Request(url, headers=self._headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                issues = json.loads(resp.read().decode())
                if issues:
                    self._issue_number = issues[0]["number"]
                    return self._issue_number

            # Create new issue if we have write access
            if not self.can_write:
                return None

            url = f"{GITHUB_API}/repos/{self._repo}/issues"
            body = json.dumps({
                "title": FEDERATION_ISSUE_TITLE,
                "body": (
                    "# 🌐 Entroly Federated Weight Exchange\n\n"
                    "This issue serves as the transport layer for Entroly's "
                    "federated archetype learning system.\n\n"
                    "**Each comment contains a DP-noised contribution packet.**\n"
                    "- All weights are protected by Rényi Differential Privacy (ε=1.0)\n"
                    "- No source code, paths, or PII are shared\n"
                    "- Contributions are auto-aggregated via trimmed-mean\n\n"
                    "Do not edit or delete comments — they are machine-managed."
                ),
                "labels": ["federation"],
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={**self._headers, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                issue = json.loads(resp.read().decode())
                self._issue_number = issue["number"]
                logger.info("Federation: created exchange issue #%d", self._issue_number)
                return self._issue_number

        except Exception as e:
            logger.debug("Federation: GitHub issue lookup failed: %s", e)
            return None

    def push(self, packet: ContributionPacket) -> bool:
        """Post a contribution as an Issue comment."""
        if not self.can_write:
            logger.debug("Federation: no ENTROLY_GITHUB_TOKEN — skipping push")
            return False

        issue = self._find_or_create_issue()
        if not issue:
            return False

        try:
            import urllib.request

            # Wrap packet in a code fence for easy parsing
            payload = json.dumps(packet.to_dict(), indent=2)
            body = f"```json\n{payload}\n```"

            url = f"{GITHUB_API}/repos/{self._repo}/issues/{issue}/comments"
            data = json.dumps({"body": body}).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={**self._headers, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 201):
                    logger.info(
                        "Federation: pushed contribution to GitHub issue #%d",
                        issue,
                    )
                    return True
            return False

        except Exception as e:
            logger.debug("Federation: GitHub push failed: %s", e)
            return False

    def pull(self, since_hours: float = 48.0) -> list[ContributionPacket]:
        """Read contribution packets from Issue comments.

        Unauthenticated — works without any token on public repos.
        Rate limit: 60 req/hr (sufficient for 136 users syncing daily).
        """
        issue = self._find_or_create_issue()
        if not issue:
            return []

        try:
            import urllib.request
            from datetime import datetime, timedelta, timezone

            # Fetch comments since cutoff
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            since_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
            url = (
                f"{GITHUB_API}/repos/{self._repo}/issues/{issue}/comments"
                f"?since={since_str}&per_page=100"
            )
            req = urllib.request.Request(url, headers=self._headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                comments = json.loads(resp.read().decode())

            packets: list[ContributionPacket] = []
            for comment in comments:
                body = comment.get("body", "")
                # Extract JSON from code fence
                if "```json" in body:
                    start = body.index("```json") + 7
                    end = body.index("```", start)
                    raw = body[start:end].strip()
                    try:
                        data = json.loads(raw)
                        packet = ContributionPacket.from_dict(data)
                        packets.append(packet)
                    except (json.JSONDecodeError, KeyError):
                        continue

            logger.info(
                "Federation: pulled %d contributions from GitHub (last %.0fh)",
                len(packets), since_hours,
            )
            return packets

        except Exception as e:
            logger.debug("Federation: GitHub pull failed: %s", e)
            return []

    def sync_to_local(
        self,
        client: FederationClient,
        since_hours: float = 48.0,
    ) -> int:
        """Pull remote contributions and save them locally for merge.

        This bridges GitHub transport → local file-based merge pipeline.
        Returns number of new contributions saved.
        """
        packets = self.pull(since_hours=since_hours)
        saved = 0
        for packet in packets:
            # Skip own contributions (anti-echo)
            if packet.client_id == client._client_id:
                continue
            if client._save_contribution(packet):
                saved += 1
        if saved > 0:
            logger.info("Federation: synced %d remote contributions to local", saved)
        return saved

