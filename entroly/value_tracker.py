"""
Entroly Value Tracker — Persistent lifetime savings across sessions
====================================================================

Tracks cumulative value delivered by Entroly across all sessions:
  - Total tokens saved (lifetime)
  - Estimated cost saved (USD, per-model pricing)
  - Requests optimized
  - Daily/weekly/monthly aggregates for trend charts
  - Context confidence score (real-time)

Data persists to ~/.entroly/value_tracker.json and survives proxy restarts.
The dashboard reads this for trend charts; the /confidence endpoint reads
it for IDE status bar widgets.

Thread-safe: all writes go through a lock + atomic file write.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.value_tracker")

# ── Per-model cost estimates (USD per 1K tokens, input pricing) ──────────

_MODEL_COSTS_PER_1K = {
    # OpenAI
    "gpt-4o": 0.0025,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.01,
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.0005,
    "o1": 0.015,
    "o1-mini": 0.003,
    "o1-pro": 0.015,
    "o3": 0.01,
    "o3-mini": 0.0011,
    "o4-mini": 0.0011,
    # Anthropic
    "claude-opus-4": 0.015,
    "claude-sonnet-4": 0.003,
    "claude-haiku-4": 0.0008,
    "claude-3-5-sonnet": 0.003,
    "claude-3-5-haiku": 0.0008,
    # Google
    "gemini-2.5-pro": 0.00125,
    "gemini-2.5-flash": 0.000075,
    "gemini-2.0-flash": 0.0001,
    "gemini-1.5-pro": 0.00125,
    "gemini-1.5-flash": 0.000075,
}

_DEFAULT_COST_PER_1K = 0.003  # Conservative default

# Aliases so old/new Anthropic names map to the same rate as the JS runtime.
_MODEL_ALIASES = {
    "claude-3-opus": "claude-opus-4",
    "claude-3-sonnet": "claude-sonnet-4",
    "claude-3-haiku": "claude-haiku-4",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
    "claude-3.5-haiku": "claude-3-5-haiku",
}

# ── Evolution Budget Guardrail ──────────────────────────────────────────
# The evolution daemon may only spend τ% of lifetime savings on LLM synthesis.
# Budget(t) = τ · S(t) − C_spent(t)  →  guaranteed token-negative.
EVOLUTION_TAX_RATE = 0.05  # 5% of lifetime savings


def estimate_cost(tokens_saved: int, model: str = "") -> float:
    """Estimate USD saved for a given number of tokens and model.

    Longest-prefix match to avoid 'gpt-4o' eating 'gpt-4o-mini'. Aliases let
    both old ('claude-3-opus') and new ('claude-opus-4') naming hit the same
    rate. Unknown models log a warning so the budget invariant stays honest.
    """
    cost = _DEFAULT_COST_PER_1K
    matched = False
    if model:
        m = model.lower()
        for alias, canonical in _MODEL_ALIASES.items():
            if m.startswith(alias):
                m = canonical + m[len(alias):]
                break
        # Longest prefix wins.
        for prefix in sorted(_MODEL_COSTS_PER_1K.keys(), key=len, reverse=True):
            if m.startswith(prefix):
                cost = _MODEL_COSTS_PER_1K[prefix]
                matched = True
                break
        if not matched:
            logger.warning("unknown model %r; falling back to default $%s/1K", model, _DEFAULT_COST_PER_1K)
    return (tokens_saved / 1000.0) * cost


def _day_key(ts: float | None = None) -> str:
    """Return YYYY-MM-DD for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"


def _week_key(ts: float | None = None) -> str:
    """Return YYYY-WNN for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    # ISO week number
    import datetime
    dt = datetime.date(t.tm_year, t.tm_mon, t.tm_mday)
    iso = dt.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def _month_key(ts: float | None = None) -> str:
    """Return YYYY-MM for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}"


class ValueTracker:
    """Persistent, thread-safe tracker for lifetime Entroly value.

    Stores cumulative stats + daily/weekly/monthly breakdowns.
    Survives proxy restarts via atomic JSON file writes.
    """

    _FILE_NAME = "value_tracker.json"
    _ACTIVITY_NAME = "activity.jsonl"
    _MAX_DAILY_ENTRIES = 90    # ~3 months of daily data
    _MAX_WEEKLY_ENTRIES = 52   # ~1 year
    _MAX_MONTHLY_ENTRIES = 24  # ~2 years
    _MAX_ACTIVITY = 200        # bounded cross-process live feed
    _SCHEMA_VERSION = 3

    @staticmethod
    def _default_dir() -> Path:
        """The shared telemetry directory. Honors ENTROLY_DIR so the
        Python and Node (npm) runtimes write to the SAME place — without
        this they diverge (~/.entroly vs cwd/.entroly) and no cross-mode
        dashboard ever sees data."""
        env = os.environ.get("ENTROLY_DIR")
        return Path(env) if env else (Path.home() / ".entroly")

    def __init__(self, data_dir: Path | None = None):
        self._dir = data_dir or self._default_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self._FILE_NAME
        self._activity_path = self._dir / self._ACTIVITY_NAME
        self._lock = threading.Lock()
        self._data = self._load()
        self._activity: list[dict[str, Any]] = self._load_activity()
        # mtime fingerprints so reader processes (the dashboard) can go
        # live on writes made by OTHER processes (proxy/MCP/npm).
        self._data_mtime: float = self._mtime(self._path)
        self._activity_mtime: float = self._mtime(self._activity_path)

        # In-memory snapshot for fast reads (updated on every record)
        self._last_confidence: float = 0.0
        self._last_coverage_pct: float = 0.0
        self._session_requests: int = 0
        self._session_tokens_saved: int = 0
        self._session_cost_saved: float = 0.0

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    def _load(self) -> dict[str, Any]:
        """Load tracker data from disk, or return fresh defaults.

        Migrates older (v2) files forward by back-filling any missing
        keys so a long-lived install never loses history on upgrade."""
        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict) and "version" in data:
                    return self._migrate(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Value tracker load failed, starting fresh: %s", e)
        return self._defaults()

    @classmethod
    def _migrate(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Backward-compatible forward migration. Adds v3 value fields
        (hallucinations blocked, model-routing $ saved) without touching
        existing counters."""
        base = cls._defaults()
        lt = data.setdefault("lifetime", {})
        for k, v in base["lifetime"].items():
            lt.setdefault(k, v)
        for bucket in ("daily", "weekly", "monthly"):
            data.setdefault(bucket, {})
        data["version"] = cls._SCHEMA_VERSION
        return data

    @classmethod
    def _defaults(cls) -> dict[str, Any]:
        return {
            "version": cls._SCHEMA_VERSION,
            "lifetime": {
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "requests_optimized": 0,
                "requests_total": 0,
                "duplicates_caught": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
                # Evolution budget accounting (Pillar 1)
                "evolution_spent_usd": 0.0,
                "evolution_attempts": 0,
                "evolution_successes": 0,
                # v3: hallucination + model-routing value (WITNESS / RAVS)
                "hallucinations_blocked": 0,
                "routing_saved_usd": 0.0,
                "routing_decisions": 0,
                # Belief-conditioned compression (H(X | beliefs))
                "beliefs_conditioned_fragments": 0,
                "belief_conditioning_passes": 0,
            },
            "daily": {},    # "YYYY-MM-DD" -> {tokens_saved, cost_saved, requests}
            "weekly": {},   # "YYYY-WNN" -> {tokens_saved, cost_saved, requests}
            "monthly": {},  # "YYYY-MM" -> {tokens_saved, cost_saved, requests}
        }

    def _load_activity(self) -> list[dict[str, Any]]:
        """Load the bounded cross-process activity feed (JSONL)."""
        if not self._activity_path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in self._activity_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as e:
            logger.debug("activity load failed: %s", e)
        # Enforce time-based retention on disk telemetry (privacy-by-default).
        try:
            from entroly.privacy import retention_cutoff_ts

            cutoff = retention_cutoff_ts()
            if cutoff is not None:
                out = [
                    row for row in out
                    if float(row.get("ts", 0) or 0) >= cutoff
                ]
        except Exception:
            pass

        return out[-self._MAX_ACTIVITY:]

    def _save(self) -> None:
        """Atomic write: write to temp file then rename (no partial writes)."""
        try:
            content = json.dumps(self._data, indent=2)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._dir), suffix=".tmp", prefix="vt_"
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                # Atomic rename (POSIX) or replace (Windows)
                if os.name == "nt":
                    # Windows: os.replace is atomic
                    os.replace(tmp_path, str(self._path))
                else:
                    os.rename(tmp_path, str(self._path))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except OSError as e:
            logger.debug("Value tracker save failed: %s", e)
        self._data_mtime = self._mtime(self._path)

    def _save_activity(self) -> None:
        """Atomically rewrite the bounded activity JSONL (≤200 lines, so
        whole-file rewrite is cheap and crash-safe via temp+rename)."""
        try:
            # Enforce time-based retention before size bounding.
            try:
                from entroly.privacy import retention_cutoff_ts

                cutoff = retention_cutoff_ts()
                if cutoff is not None:
                    self._activity = [
                        row for row in self._activity
                        if float(row.get("ts", 0) or 0) >= cutoff
                    ]
            except Exception:
                pass

            self._activity = self._activity[-self._MAX_ACTIVITY:]
            content = "\n".join(json.dumps(e, separators=(",", ":"))
                                for e in self._activity)
            if content:
                content += "\n"
            fd, tmp = tempfile.mkstemp(dir=str(self._dir),
                                       suffix=".tmp", prefix="act_")
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                os.replace(tmp, str(self._activity_path))
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except OSError as e:
            logger.debug("activity save failed: %s", e)
        self._activity_mtime = self._mtime(self._activity_path)

    def reload_if_changed(self) -> bool:
        """Re-read disk state IF another process advanced the files.

        This is the fix for the cross-process dashboard: the writer
        (proxy/MCP/npm) and the reader (`entroly dashboard`) are usually
        different processes, and the singleton otherwise froze its
        in-memory copy at startup. Readers call this each poll; it is a
        no-op (one cheap stat) when nothing changed. Per-process session
        counters are intentionally preserved (they are not on disk)."""
        changed = False
        with self._lock:
            dm = self._mtime(self._path)
            if dm > self._data_mtime:
                self._data = self._load()
                self._data_mtime = dm
                changed = True
            am = self._mtime(self._activity_path)
            if am > self._activity_mtime:
                self._activity = self._load_activity()
                self._activity_mtime = am
                changed = True
        return changed

    def record_event(
        self,
        kind: str,
        summary: str,
        *,
        source: str = "",
        tokens_saved: int = 0,
        cost_saved_usd: float = 0.0,
        model: str = "",
        **extra: Any,
    ) -> None:
        """Append one row to the persistent, cross-process activity feed.

        `kind` is a short tag the dashboard groups on: "optimize",
        "hallucination", "routing", "compress". Fail-open: telemetry must
        never break the caller."""
        try:
            row: dict[str, Any] = {
                "ts": round(time.time(), 3),
                "kind": str(kind),
                "summary": str(summary)[:240],
            }
            if source:
                row["source"] = source
            if tokens_saved:
                row["tokens_saved"] = int(tokens_saved)
            if cost_saved_usd:
                row["cost_saved_usd"] = round(float(cost_saved_usd), 6)
            if model:
                row["model"] = model
            for k, v in extra.items():
                if isinstance(v, (str, int, float, bool)):
                    row[k] = v
            with self._lock:
                self._activity.append(row)
                self._save_activity()
        except Exception as e:  # noqa: BLE001
            logger.debug("record_event failed (non-fatal): %s", e)

    def record_hallucination_blocked(
        self, n: int = 1, *, source: str = "", detail: str = ""
    ) -> None:
        """WITNESS suppressed `n` unsupported claims before they reached
        the user. Fail-open."""
        try:
            with self._lock:
                self._data["lifetime"]["hallucinations_blocked"] = (
                    self._data["lifetime"].get("hallucinations_blocked", 0)
                    + int(n)
                )
                self._save()
            self.record_event(
                "hallucination",
                detail or f"Blocked {n} unsupported claim(s)",
                source=source, blocked=int(n),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("record_hallucination_blocked failed: %s", e)

    def record_routing_saving(
        self, cost_saved_usd: float, *, source: str = "",
        chosen_model: str = "", detail: str = "",
    ) -> None:
        """RAVS routed to a cheaper capable model; record the $ avoided.
        Fail-open."""
        try:
            with self._lock:
                lt = self._data["lifetime"]
                lt["routing_saved_usd"] = round(
                    lt.get("routing_saved_usd", 0.0)
                    + float(cost_saved_usd), 6)
                lt["routing_decisions"] = lt.get("routing_decisions", 0) + 1
                self._save()
            self.record_event(
                "routing",
                detail or f"Routed to {chosen_model or 'cheaper model'}",
                source=source, cost_saved_usd=float(cost_saved_usd),
                model=chosen_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("record_routing_saving failed: %s", e)

    def record_belief_conditioning(
        self, n_fragments: int, *, source: str = "", detail: str = ""
    ) -> None:
        """Belief-conditioned compression discounted `n_fragments` candidate
        fragments that merely restated already-known vault beliefs
        (H(X | beliefs)), freeing budget for novel content. Fail-open."""
        try:
            if n_fragments <= 0:
                return
            with self._lock:
                lt = self._data["lifetime"]
                lt["beliefs_conditioned_fragments"] = (
                    lt.get("beliefs_conditioned_fragments", 0) + int(n_fragments)
                )
                lt["belief_conditioning_passes"] = (
                    lt.get("belief_conditioning_passes", 0) + 1
                )
                self._save()
            self.record_event(
                "belief_conditioning",
                detail or f"Discounted {n_fragments} fragment(s) restating known beliefs",
                source=source, fragments_discounted=int(n_fragments),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("record_belief_conditioning failed: %s", e)

    def get_activity(self, last_n: int = 50) -> list[dict[str, Any]]:
        """Most-recent-first slice of the cross-process activity feed."""
        with self._lock:
            return list(reversed(self._activity[-last_n:]))

    def _trim_history(self) -> None:
        """Keep history within size limits."""
        for key, limit in [
            ("daily", self._MAX_DAILY_ENTRIES),
            ("weekly", self._MAX_WEEKLY_ENTRIES),
            ("monthly", self._MAX_MONTHLY_ENTRIES),
        ]:
            bucket = self._data.get(key, {})
            if len(bucket) > limit:
                sorted_keys = sorted(bucket.keys())
                for old_key in sorted_keys[: len(sorted_keys) - limit]:
                    del bucket[old_key]

    def record(
        self,
        tokens_saved: int,
        model: str = "",
        duplicates: int = 0,
        optimized: bool = True,
        coverage_pct: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        """Record a single optimized request's value.

        Called from proxy.py after each successful optimization.
        Thread-safe. Persists to disk on every call (atomic write).
        """
        cost = estimate_cost(tokens_saved, model)
        now = time.time()
        day = _day_key(now)
        week = _week_key(now)
        month = _month_key(now)

        with self._lock:
            lt = self._data["lifetime"]
            lt["tokens_saved"] += tokens_saved
            lt["cost_saved_usd"] = round(lt["cost_saved_usd"] + cost, 6)
            lt["requests_total"] += 1
            if optimized:
                lt["requests_optimized"] += 1
            lt["duplicates_caught"] += duplicates
            lt["last_seen"] = now

            # Daily
            d = self._data.setdefault("daily", {})
            if day not in d:
                d[day] = {"tokens_saved": 0, "cost_saved": 0.0, "requests": 0}
            d[day]["tokens_saved"] += tokens_saved
            d[day]["cost_saved"] = round(d[day]["cost_saved"] + cost, 6)
            d[day]["requests"] += 1

            # Weekly
            w = self._data.setdefault("weekly", {})
            if week not in w:
                w[week] = {"tokens_saved": 0, "cost_saved": 0.0, "requests": 0}
            w[week]["tokens_saved"] += tokens_saved
            w[week]["cost_saved"] = round(w[week]["cost_saved"] + cost, 6)
            w[week]["requests"] += 1

            # Monthly
            m = self._data.setdefault("monthly", {})
            if month not in m:
                m[month] = {"tokens_saved": 0, "cost_saved": 0.0, "requests": 0}
            m[month]["tokens_saved"] += tokens_saved
            m[month]["cost_saved"] = round(m[month]["cost_saved"] + cost, 6)
            m[month]["requests"] += 1

            # Session counters (in-memory only)
            self._session_requests += 1
            self._session_tokens_saved += tokens_saved
            self._session_cost_saved += cost
            self._last_confidence = confidence
            self._last_coverage_pct = coverage_pct

            self._trim_history()
            self._save()

            # Light up the cross-process live feed for free. Append
            # in-lock to keep ordering; persist outside the heavy path.
            self._activity.append({
                "ts": round(now, 3),
                "kind": "optimize",
                "summary": (f"Optimized request: saved {tokens_saved:,} "
                            f"tokens" + (f" ({model})" if model else "")),
                "tokens_saved": int(tokens_saved),
                "cost_saved_usd": round(cost, 6),
                "model": model or "",
                "duplicates": int(duplicates),
            })
            self._save_activity()

    def get_lifetime(self) -> dict[str, Any]:
        """Return lifetime cumulative stats."""
        with self._lock:
            return dict(self._data.get("lifetime", {}))

    def get_daily(self, last_n: int = 30) -> list[dict[str, Any]]:
        """Return last N days of daily stats, sorted ascending."""
        with self._lock:
            d = self._data.get("daily", {})
            keys = sorted(d.keys())[-last_n:]
            return [{"date": k, **d[k]} for k in keys]

    def get_weekly(self, last_n: int = 12) -> list[dict[str, Any]]:
        """Return last N weeks of stats, sorted ascending."""
        with self._lock:
            w = self._data.get("weekly", {})
            keys = sorted(w.keys())[-last_n:]
            return [{"week": k, **w[k]} for k in keys]

    def get_monthly(self, last_n: int = 12) -> list[dict[str, Any]]:
        """Return last N months of stats, sorted ascending."""
        with self._lock:
            m = self._data.get("monthly", {})
            keys = sorted(m.keys())[-last_n:]
            return [{"month": k, **m[k]} for k in keys]

    def get_session(self) -> dict[str, Any]:
        """Return current session stats (since proxy start)."""
        with self._lock:
            return {
                "requests": self._session_requests,
                "tokens_saved": self._session_tokens_saved,
                "cost_saved_usd": round(self._session_cost_saved, 4),
            }

    def get_confidence(self) -> dict[str, Any]:
        """Return real-time confidence snapshot for IDE widgets.

        This is the single endpoint an IDE status bar polls.
        """
        with self._lock:
            lt = self._data.get("lifetime", {})
            today = _day_key()
            today_data = self._data.get("daily", {}).get(today, {})
            return {
                "confidence": round(self._last_confidence, 4),
                "coverage_pct": round(self._last_coverage_pct, 2),
                "session": {
                    "requests": self._session_requests,
                    "tokens_saved": self._session_tokens_saved,
                    "cost_saved_usd": round(self._session_cost_saved, 4),
                },
                "today": {
                    "tokens_saved": today_data.get("tokens_saved", 0),
                    "cost_saved_usd": today_data.get("cost_saved", 0.0),
                    "requests": today_data.get("requests", 0),
                },
                "lifetime": {
                    "tokens_saved": lt.get("tokens_saved", 0),
                    "cost_saved_usd": lt.get("cost_saved_usd", 0.0),
                    "requests_optimized": lt.get("requests_optimized", 0),
                    "hallucinations_blocked": lt.get(
                        "hallucinations_blocked", 0),
                    "routing_saved_usd": lt.get("routing_saved_usd", 0.0),
                },
                "status": "active" if self._session_requests > 0 else "idle",
            }

    def get_trends(self) -> dict[str, Any]:
        """Return all trend data for dashboard charts."""
        return {
            "daily": self.get_daily(30),
            "weekly": self.get_weekly(12),
            "monthly": self.get_monthly(12),
            "lifetime": self.get_lifetime(),
            "session": self.get_session(),
            "activity": self.get_activity(50),
        }

    # ── Evolution Budget Guardrail (Pillar 1) ─────────────────────────────

    def get_evolution_budget(self) -> dict[str, Any]:
        """Return the available evolution budget.

        Budget = τ · lifetime_savings − total_spent
        The system can only spend τ% (5%) of its lifetime savings on
        LLM-based skill synthesis. This guarantees token-negativity.

        Returns:
            {
                "available_usd": float,   # remaining evolution budget
                "total_earned_usd": float, # τ · lifetime savings
                "total_spent_usd": float,  # already debited
                "can_evolve": bool,        # available > 0
                "tax_rate": float,         # τ
            }
        """
        with self._lock:
            lt = self._data.get("lifetime", {})
            lifetime_saved = lt.get("cost_saved_usd", 0.0)
            total_spent = lt.get("evolution_spent_usd", 0.0)
            total_earned = lifetime_saved * EVOLUTION_TAX_RATE
            available = max(0.0, total_earned - total_spent)

            return {
                "available_usd": round(available, 6),
                "total_earned_usd": round(total_earned, 6),
                "total_spent_usd": round(total_spent, 6),
                "can_evolve": available > 0.001,  # > 0.1 cent floor
                "tax_rate": EVOLUTION_TAX_RATE,
            }

    def record_evolution_spend(
        self,
        cost_usd: float,
        success: bool = False,
    ) -> dict[str, Any]:
        """Debit an evolution attempt from the budget.

        Called by the evolution daemon after an LLM-based synthesis attempt.
        Only succeeds if the budget allows it (fail-safe: checks again here).

        Args:
            cost_usd: Cost of this evolution attempt.
            success: Whether the synthesized skill passed benchmarks.

        Returns:
            {"status": "recorded" | "rejected", "remaining_usd": float}
        """
        with self._lock:
            lt = self._data.get("lifetime", {})
            lifetime_saved = lt.get("cost_saved_usd", 0.0)
            current_spent = lt.get("evolution_spent_usd", 0.0)
            total_earned = lifetime_saved * EVOLUTION_TAX_RATE
            available = total_earned - current_spent

            if cost_usd > available + 0.001:  # 0.1 cent tolerance
                logger.warning(
                    "Evolution spend rejected: $%.4f requested, $%.4f available",
                    cost_usd, available,
                )
                return {
                    "status": "rejected",
                    "remaining_usd": round(max(0.0, available), 6),
                }

            lt["evolution_spent_usd"] = round(current_spent + cost_usd, 6)
            lt["evolution_attempts"] = lt.get("evolution_attempts", 0) + 1
            if success:
                lt["evolution_successes"] = lt.get("evolution_successes", 0) + 1

            self._save()

            remaining = max(0.0, total_earned - lt["evolution_spent_usd"])
            logger.info(
                "Evolution spend recorded: $%.4f (remaining: $%.4f, success=%s)",
                cost_usd, remaining, success,
            )
            return {
                "status": "recorded",
                "remaining_usd": round(remaining, 6),
            }


# ── Module-level singleton (lazy-init) ───────────────────────────────────

_tracker: ValueTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> ValueTracker:
    """Get or create the global ValueTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ValueTracker()
    return _tracker
