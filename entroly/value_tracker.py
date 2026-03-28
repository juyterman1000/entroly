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
from typing import Any, Dict, List, Optional

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


def estimate_cost(tokens_saved: int, model: str = "") -> float:
    """Estimate USD saved for a given number of tokens and model."""
    cost = _DEFAULT_COST_PER_1K
    if model:
        model_lower = model.lower()
        for prefix, c in _MODEL_COSTS_PER_1K.items():
            if model_lower.startswith(prefix):
                cost = c
                break
    return (tokens_saved / 1000.0) * cost


def _day_key(ts: Optional[float] = None) -> str:
    """Return YYYY-MM-DD for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"


def _week_key(ts: Optional[float] = None) -> str:
    """Return YYYY-WNN for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    # ISO week number
    import datetime
    dt = datetime.date(t.tm_year, t.tm_mon, t.tm_mday)
    iso = dt.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def _month_key(ts: Optional[float] = None) -> str:
    """Return YYYY-MM for a timestamp (or now)."""
    t = time.gmtime(ts or time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}"


class ValueTracker:
    """Persistent, thread-safe tracker for lifetime Entroly value.

    Stores cumulative stats + daily/weekly/monthly breakdowns.
    Survives proxy restarts via atomic JSON file writes.
    """

    _FILE_NAME = "value_tracker.json"
    _MAX_DAILY_ENTRIES = 90    # ~3 months of daily data
    _MAX_WEEKLY_ENTRIES = 52   # ~1 year
    _MAX_MONTHLY_ENTRIES = 24  # ~2 years

    def __init__(self, data_dir: Optional[Path] = None):
        self._dir = data_dir or (Path.home() / ".entroly")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self._FILE_NAME
        self._lock = threading.Lock()
        self._data = self._load()

        # In-memory snapshot for fast reads (updated on every record)
        self._last_confidence: float = 0.0
        self._last_coverage_pct: float = 0.0
        self._session_requests: int = 0
        self._session_tokens_saved: int = 0
        self._session_cost_saved: float = 0.0

    def _load(self) -> Dict[str, Any]:
        """Load tracker data from disk, or return fresh defaults."""
        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict) and "version" in data:
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Value tracker load failed, starting fresh: %s", e)
        return self._defaults()

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "version": 1,
            "lifetime": {
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "requests_optimized": 0,
                "requests_total": 0,
                "duplicates_caught": 0,
                "first_seen": time.time(),
                "last_seen": time.time(),
            },
            "daily": {},    # "YYYY-MM-DD" -> {tokens_saved, cost_saved, requests}
            "weekly": {},   # "YYYY-WNN" -> {tokens_saved, cost_saved, requests}
            "monthly": {},  # "YYYY-MM" -> {tokens_saved, cost_saved, requests}
        }

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

    def get_lifetime(self) -> Dict[str, Any]:
        """Return lifetime cumulative stats."""
        with self._lock:
            return dict(self._data.get("lifetime", {}))

    def get_daily(self, last_n: int = 30) -> List[Dict[str, Any]]:
        """Return last N days of daily stats, sorted ascending."""
        with self._lock:
            d = self._data.get("daily", {})
            keys = sorted(d.keys())[-last_n:]
            return [{"date": k, **d[k]} for k in keys]

    def get_weekly(self, last_n: int = 12) -> List[Dict[str, Any]]:
        """Return last N weeks of stats, sorted ascending."""
        with self._lock:
            w = self._data.get("weekly", {})
            keys = sorted(w.keys())[-last_n:]
            return [{"week": k, **w[k]} for k in keys]

    def get_monthly(self, last_n: int = 12) -> List[Dict[str, Any]]:
        """Return last N months of stats, sorted ascending."""
        with self._lock:
            m = self._data.get("monthly", {})
            keys = sorted(m.keys())[-last_n:]
            return [{"month": k, **m[k]} for k in keys]

    def get_session(self) -> Dict[str, Any]:
        """Return current session stats (since proxy start)."""
        with self._lock:
            return {
                "requests": self._session_requests,
                "tokens_saved": self._session_tokens_saved,
                "cost_saved_usd": round(self._session_cost_saved, 4),
            }

    def get_confidence(self) -> Dict[str, Any]:
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
                },
                "status": "active" if self._session_requests > 0 else "idle",
            }

    def get_trends(self) -> Dict[str, Any]:
        """Return all trend data for dashboard charts."""
        return {
            "daily": self.get_daily(30),
            "weekly": self.get_weekly(12),
            "monthly": self.get_monthly(12),
            "lifetime": self.get_lifetime(),
            "session": self.get_session(),
        }


# ── Module-level singleton (lazy-init) ───────────────────────────────────

_tracker: Optional[ValueTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> ValueTracker:
    """Get or create the global ValueTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ValueTracker()
    return _tracker
