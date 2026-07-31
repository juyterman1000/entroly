"""Vault hygiene — belief-vs-belief maintenance scan (report-only).

The claim-vs-evidence analyzers (ESG/EICV) check an answer against supplied
context. Nothing previously checked the vault's beliefs *against each other*.
This module adds that pass:

- **contradictions**: pairwise ESG on belief bodies. The discriminating
  signal is ``contradiction_fraction`` — NOT ``tension``, which saturates at
  1.0 for merely-unrelated beliefs (measured: unrelated pair scored
  contra=0.0 / tension=1.0; negated pair scored contra=1.0).
- **near-duplicates**: token-set Jaccard between bodies; merge suggestions.
- **staleness**: ``last_checked`` older than ``max_age_days`` or an explicit
  stale status.
- **confidence flapping** (needs the belief ledger): entities whose recorded
  confidence reverses direction repeatedly across versions — a signal that
  verification keeps disagreeing with itself.

Report-only by design: the scan never rewrites or deletes a belief. Acting on
findings stays a caller decision, so a scoring bug cannot destroy knowledge.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HYGIENE_SCHEMA = "entroly.vault-hygiene.v1"

_WORD_RE = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    "the a an is are was were be been being to of in on for with and or not "
    "no it its this that these those uses use used by at as from".split()
)


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class VaultHygiene:
    """Scans current beliefs (and the ledger, when present) for decay."""

    def __init__(
        self,
        vault_base: str | Path,
        *,
        contradiction_threshold: float = 0.5,
        duplicate_threshold: float = 0.8,
        overlap_prefilter: float = 0.1,
        max_age_days: int = 30,
        max_pairs: int = 2000,
    ):
        self._base = Path(vault_base)
        self.contradiction_threshold = contradiction_threshold
        self.duplicate_threshold = duplicate_threshold
        self.overlap_prefilter = overlap_prefilter
        self.max_age_days = max_age_days
        self.max_pairs = max_pairs

    def _load_beliefs(self) -> list[dict[str, Any]]:
        from .vault import _extract_body, _parse_frontmatter

        beliefs = []
        beliefs_dir = self._base / "beliefs"
        if not beliefs_dir.exists():
            return beliefs
        for md in sorted(beliefs_dir.rglob("*.md")):
            content = md.read_text(encoding="utf-8", errors="replace")
            fm = _parse_frontmatter(content) or {}
            body = _extract_body(content)
            beliefs.append({
                "entity": fm.get("entity", md.stem),
                "claim_id": fm.get("claim_id", ""),
                "status": fm.get("status", "unknown"),
                "confidence": float(fm.get("confidence", 0.0) or 0.0),
                "last_checked": fm.get("last_checked", ""),
                "body": body,
                "tokens": _content_tokens(body),
            })
        return beliefs

    def scan(self) -> dict[str, Any]:
        from .esg import ESGAnalyzer

        beliefs = self._load_beliefs()
        esg = ESGAnalyzer()

        contradictions: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        pairs_scored = 0
        truncated = False
        for a, b in itertools.combinations(beliefs, 2):
            overlap = _jaccard(a["tokens"], b["tokens"])
            if overlap >= self.duplicate_threshold:
                duplicates.append({
                    "entities": [a["entity"], b["entity"]],
                    "claim_ids": [a["claim_id"], b["claim_id"]],
                    "jaccard": round(overlap, 4),
                    "suggestion": "merge",
                })
                continue
            if overlap < self.overlap_prefilter:
                continue  # unrelated; ESG contradiction is 0 for these anyway
            if pairs_scored >= self.max_pairs:
                truncated = True
                break
            pairs_scored += 1
            contra = max(
                esg.score(a["body"], b["body"]).contradiction_fraction,
                esg.score(b["body"], a["body"]).contradiction_fraction,
            )
            if contra >= self.contradiction_threshold:
                contradictions.append({
                    "entities": [a["entity"], b["entity"]],
                    "claim_ids": [a["claim_id"], b["claim_id"]],
                    "contradiction_fraction": round(contra, 4),
                    "suggestion": "verify_and_retire_one",
                })

        now = datetime.now(timezone.utc)
        stale: list[dict[str, Any]] = []
        for belief in beliefs:
            age_days: float | None = None
            checked = belief["last_checked"]
            if checked:
                try:
                    dt = datetime.fromisoformat(checked)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_days = (now - dt).total_seconds() / 86400.0
                except ValueError:
                    age_days = None
            if belief["status"] == "stale" or age_days is None or age_days > self.max_age_days:
                stale.append({
                    "entity": belief["entity"],
                    "claim_id": belief["claim_id"],
                    "status": belief["status"],
                    "age_days": round(age_days, 1) if age_days is not None else None,
                    "suggestion": "refresh_beliefs",
                })

        return {
            "schema": HYGIENE_SCHEMA,
            "scanned_at": now.isoformat(),
            "n_beliefs": len(beliefs),
            "pairs_scored": pairs_scored,
            "pairs_truncated": truncated,
            "contradictions": contradictions,
            "duplicates": duplicates,
            "stale": stale,
            "confidence_flapping": self._flapping(),
            "healthy": not (contradictions or duplicates),
        }

    def _flapping(self, min_reversals: int = 3) -> list[dict[str, Any]]:
        """Entities whose ledger confidence keeps reversing direction."""
        from .vault_time import BeliefLedger, LedgerIntegrityError

        ledger = BeliefLedger(self._base)
        try:
            versions = list(ledger._iter_versions())
        except LedgerIntegrityError:
            return [{"entity": "<ledger>", "error": "ledger unreadable — run verify_chain"}]
        by_entity: dict[str, list[float]] = {}
        for v in sorted(versions, key=lambda v: v.seq):
            by_entity.setdefault(v.entity, []).append(v.confidence)
        flapping = []
        for entity, confs in by_entity.items():
            deltas = [b - a for a, b in zip(confs, confs[1:]) if b != a]
            reversals = sum(
                1 for d1, d2 in zip(deltas, deltas[1:]) if (d1 > 0) != (d2 > 0)
            )
            if reversals >= min_reversals:
                flapping.append({
                    "entity": entity,
                    "versions": len(confs),
                    "reversals": reversals,
                    "confidence_path": [round(c, 3) for c in confs],
                    "suggestion": "escalate_verification",
                })
        return flapping
