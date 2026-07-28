"""
VerifierService — Daemon-Resident Singleton
============================================

Holds the manifest, n-gram model, reverse index, and calibrator in
process memory. Subsequent verifications skip the 2.6s repo walk.

Lifecycle::

    svc = VerifierService.for_repo("/path/to/repo")
    result = svc.verify(generated_code, archetype="code/implement")

    # Closing the feedback loop after RAVS observed ground truth
    svc.record_outcome(result, ground_truth_per_symbol={"foo": 0, "bar": 1},
                       archetype="code/implement")

Architecture
------------
A process-wide registry maps `repo_root → ServiceInstance`. Each instance
holds:
    - SymbolManifest         (4 layers)
    - CharNGramModel         (codebase-conditioned)
    - ReverseIndex           (symbol → defining module)
    - Calibrator             (per-archetype λ)
    - File watcher           (invalidate on source changes)

Thread safety: each instance has its own RLock. Concurrent verifies are
safe because verifier state is read-only post-construction; the
calibrator has its own lock; the manifest invalidation rebuilds atomically
and swaps via a single field-write.

Daemon integration: import this module from entroly/daemon.py and pin
one instance per repo at daemon startup. The MCP server route handler
then calls `verify()` with <30ms latency per request.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import _compute_file_hash, _enumerate_files, load_or_build_verifier
from .calibrator import Calibrator, infer_archetype_from_query
from .scope_analyzer import (
    ReverseIndex,
    build_reverse_index,
    compute_scope,
    judge_reachability,
)
from .symbol_resolution import (
    ALWAYS_GROUND,
    DUNDER_RE,
    SymbolJudgment,
    SymbolVerifier,
)

logger = logging.getLogger("entroly.verifiers.service")


# Re-verification budget — if more time has passed than this, rebuild
# the manifest before serving (safety net for missed file events).
DEFAULT_STALENESS_SECONDS = 300


@dataclass
class ExtendedJudgment:
    """A SymbolJudgment augmented with the reachability verdict.

    state: 0 = hallucinated, 1 = grounded, 2 = unreachable
    """

    base: SymbolJudgment
    state: int
    suggested_import: str | None = None
    scope_source: str | None = None
    type_errors: list[str] | None = None


@dataclass
class ExtendedResult:
    """Daemon-mode verification result with auditable degradation state."""

    code: str
    archetype: str
    lambda_used: float
    judgments: list[ExtendedJudgment]
    h_strict: float  # 1 - Π(1 - p(θ≠1))^w — flags halu+unreach
    h_lenient: float  # 1 - Π(1 - p(θ=0))^w — flags halu only
    n_grounded: int
    n_unreachable: int
    n_hallucinated: int
    manifest_size: int
    type_errors: list[Any]
    type_check_status: str = "not_requested"
    type_check_detail: str = ""

    def _type_check_allows_pass(self) -> bool:
        """A requested type check must complete cleanly to permit PASS."""
        return self.type_check_status in {"not_requested", "passed"}

    def passed_strict(self, threshold: float = 0.5) -> bool:
        return self.h_strict < threshold and self._type_check_allows_pass()

    def passed_lenient(self, threshold: float = 0.5) -> bool:
        return self.h_lenient < threshold and self._type_check_allows_pass()

    def explain(self, max_items: int = 30) -> str:
        verdict = "PASS" if self.passed_strict() else "REJECTED"
        lines = []
        lines.append(
            f"=== Verifier verdict: {verdict} "
            f"(archetype={self.archetype}, lambda={self.lambda_used:.2f}) ==="
        )
        lines.append(
            f"H_strict={self.h_strict:.4f}  H_lenient={self.h_lenient:.4f}  "
            f"grounded={self.n_grounded}  unreachable={self.n_unreachable}  "
            f"hallucinated={self.n_hallucinated}  manifest={self.manifest_size:,}"
        )
        if self.type_check_status == "passed":
            lines.append("type-check: PASS")
        elif self.type_check_status == "issues_found":
            lines.append(f"type-check: REJECTED ({len(self.type_errors)} issue(s))")
        elif self.type_check_status != "not_requested":
            detail = f" — {self.type_check_detail}" if self.type_check_detail else ""
            lines.append(f"type-check: DEGRADED ({self.type_check_status}){detail}")
        lines.append("")

        # Sort: hallucinated > unreachable > grounded-with-high-p
        def key(judgment: ExtendedJudgment):
            return (judgment.state, -judgment.base.p_hallucinated)

        for judgment in sorted(self.judgments, key=key)[:max_items]:
            if judgment.state == 0:
                tag = "[X] HALLU      "
            elif judgment.state == 2:
                tag = "[!] UNREACHABLE"
            else:
                tag = "[ok]           "
            lines.append(
                f"  {tag}  line {judgment.base.ref.line:4d}  "
                f"P_halu={judgment.base.p_hallucinated:.3f}  "
                f"surp={judgment.base.surprisal:5.2f}  "
                f"{judgment.base.ref.name}  ({judgment.base.ref.kind})"
            )
            if judgment.suggested_import:
                lines.append(f"        fix: {judgment.suggested_import}")
            if judgment.type_errors:
                for type_error in judgment.type_errors[:3]:
                    lines.append(f"        type: {type_error}")
        if len(self.judgments) > max_items:
            lines.append(f"  ... and {len(self.judgments) - max_items} more")
        return "\n".join(lines)


# ── Service Instance ─────────────────────────────────────────────────


class _ServiceInstance:
    """One repo's verifier state. Long-lived. Thread-safe."""

    def __init__(
        self,
        repo_root: str,
        run_type_check: bool = False,
        calibration_dir: str | None = None,
    ):
        self._repo_root = str(Path(repo_root).resolve())
        self._run_type_check = run_type_check
        self._lock = threading.RLock()
        self._last_validated_at = 0.0
        self._last_file_hash = ""

        # Lazy-built; first verify() call constructs them
        self._verifier: SymbolVerifier | None = None
        self._reverse_index: ReverseIndex | None = None

        # Calibrator persists in .entroly/verifiers_cache/calibration.json
        cal_dir = (
            Path(calibration_dir)
            if calibration_dir
            else Path(repo_root) / ".entroly" / "verifiers_cache"
        )
        cal_dir.mkdir(parents=True, exist_ok=True)
        self._calibrator = Calibrator(state_path=cal_dir / "calibration.json")

    # ── Lazy build / staleness check ────────────────────────────

    def _ensure_ready(self, force_rebuild: bool = False) -> None:
        """Build verifier on first use; rebuild if files changed."""
        with self._lock:
            now = time.time()
            stale = (now - self._last_validated_at) > DEFAULT_STALENESS_SECONDS

            if self._verifier is None or force_rebuild or stale:
                files = _enumerate_files(self._repo_root)
                file_hash = _compute_file_hash(files)

                if (
                    self._verifier is None
                    or file_hash != self._last_file_hash
                    or force_rebuild
                ):
                    logger.info(
                        "VerifierService: %s manifest for %s",
                        "rebuilding" if self._verifier else "building",
                        self._repo_root,
                    )
                    verifier, _meta = load_or_build_verifier(
                        repo_root=self._repo_root,
                        force_rebuild=force_rebuild,
                    )
                    self._verifier = verifier
                    self._reverse_index = build_reverse_index(self._repo_root)
                    self._last_file_hash = file_hash

                self._last_validated_at = now

    # ── Public API ───────────────────────────────────────────────

    def verify(
        self,
        code: str,
        archetype: str | None = None,
        query: str | None = None,
        run_type_check: bool | None = None,
    ) -> ExtendedResult:
        """Run the full verification pipeline.

        Args:
            code: Generated source to verify.
            archetype: Task archetype. If None, inferred from `query`.
            query: User query — used to infer archetype if not given.
            run_type_check: Override the instance default.
        """
        self._ensure_ready()
        assert self._verifier is not None
        assert self._reverse_index is not None

        if archetype is None:
            archetype = infer_archetype_from_query(query or "")

        lambda_ = self._calibrator.get(archetype)
        verifier = self._verifier
        old_lambda = verifier.lambda_
        try:
            verifier.lambda_ = lambda_

            # Phase 1: symbol-resolution Bayesian pass
            base_result = verifier.verify(code)

            # Phase 2: scope/reachability pass — assigns 3-state verdict.
            manifest = self._verifier.manifest
            scope = compute_scope(
                code,
                self._reverse_index,
                import_source_valid=lambda module: module in manifest,
            )
            extended_judgments: list[ExtendedJudgment] = []
            for base_judgment in base_result.judgments:
                if (
                    base_judgment.ref.name in ALWAYS_GROUND
                    or DUNDER_RE.match(base_judgment.ref.name)
                ):
                    extended_judgments.append(
                        ExtendedJudgment(base=base_judgment, state=1)
                    )
                    continue
                verdict = judge_reachability(
                    symbol=base_judgment.ref.name,
                    manifest_contains=base_judgment.resolved,
                    scope=scope,
                    reverse_index=self._reverse_index,
                )
                extended_judgments.append(
                    ExtendedJudgment(
                        base=base_judgment,
                        state=verdict.state,
                        suggested_import=verdict.suggested_import,
                        scope_source=verdict.scope_source,
                    )
                )

            # Phase 3 (optional): Pyright type check. Degraded execution is
            # explicitly recorded and blocks PASS instead of looking like zero errors.
            type_errors: list[Any] = []
            type_check_status = "not_requested"
            type_check_detail = ""
            should_typecheck = (
                run_type_check
                if run_type_check is not None
                else self._run_type_check
            )
            if should_typecheck:
                from .type_check import check_snippet_result

                type_result = check_snippet_result(code, timeout_s=5.0)
                type_errors = type_result.diagnostics
                type_check_status = type_result.status
                type_check_detail = type_result.detail

                by_line: dict[int, list[str]] = {}
                for type_error in type_errors:
                    by_line.setdefault(type_error.line, []).append(type_error.message)
                for judgment in extended_judgments:
                    if judgment.base.ref.line in by_line:
                        judgment.type_errors = by_line[judgment.base.ref.line]

            # Phase 4: aggregate scores under the 3-state model
            log_grounded_strict = 0.0
            log_grounded_lenient = 0.0
            for judgment in extended_judgments:
                weight = judgment.base.ref.weight
                p_halu_soft = (
                    judgment.base.p_hallucinated if judgment.state == 1 else 0.0
                )
                p_not_grounded = 1.0 if judgment.state != 1 else p_halu_soft
                p_not_grounded = min(max(p_not_grounded, 0.0), 1.0 - 1e-12)
                log_grounded_strict += weight * math.log(1.0 - p_not_grounded)

                p_hard_halu = 1.0 if judgment.state == 0 else p_halu_soft
                p_hard_halu = min(max(p_hard_halu, 0.0), 1.0 - 1e-12)
                log_grounded_lenient += weight * math.log(1.0 - p_hard_halu)

            h_strict = (
                1.0 - math.exp(log_grounded_strict)
                if extended_judgments
                else 0.0
            )
            h_lenient = (
                1.0 - math.exp(log_grounded_lenient)
                if extended_judgments
                else 0.0
            )

            return ExtendedResult(
                code=code,
                archetype=archetype,
                lambda_used=lambda_,
                judgments=extended_judgments,
                h_strict=h_strict,
                h_lenient=h_lenient,
                n_grounded=sum(1 for j in extended_judgments if j.state == 1),
                n_unreachable=sum(1 for j in extended_judgments if j.state == 2),
                n_hallucinated=sum(1 for j in extended_judgments if j.state == 0),
                manifest_size=base_result.manifest_size,
                type_errors=type_errors,
                type_check_status=type_check_status,
                type_check_detail=type_check_detail,
            )
        finally:
            verifier.lambda_ = old_lambda

    def record_outcome(
        self,
        result: ExtendedResult,
        ground_truth: dict[str, int],
    ) -> None:
        """Feed observed outcomes back to the calibrator.

        Args:
            result: The ExtendedResult returned by a previous verify().
            ground_truth: Map symbol_name → 1 if really hallucinated,
                          0 if actually valid. Symbols not in the map
                          are skipped (no observation).
        """
        for judgment in result.judgments:
            outcome = ground_truth.get(judgment.base.ref.name)
            if outcome is None:
                continue
            self._calibrator.record_feedback(
                archetype=result.archetype,
                symbol=judgment.base.ref.name,
                surprisal=judgment.base.surprisal,
                p_hallucinated=judgment.base.p_hallucinated,
                y=outcome,
            )

    def invalidate(self) -> None:
        """Force a rebuild on next verify()."""
        with self._lock:
            self._verifier = None
            self._reverse_index = None
            self._last_file_hash = ""

    def calibration_stats(self) -> dict:
        return self._calibrator.all_stats()


# ── Global registry ──────────────────────────────────────────────────


class VerifierService:
    """Process-wide registry of per-repo verifier instances."""

    _instances: dict[str, _ServiceInstance] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def for_repo(
        cls,
        repo_root: str,
        run_type_check: bool = False,
    ) -> _ServiceInstance:
        key = str(Path(repo_root).resolve())
        with cls._registry_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = _ServiceInstance(
                    repo_root=key,
                    run_type_check=run_type_check,
                )
                cls._instances[key] = instance
            return instance

    @classmethod
    def shutdown_all(cls) -> None:
        with cls._registry_lock:
            cls._instances.clear()
