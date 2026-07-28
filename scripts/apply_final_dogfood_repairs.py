from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Every optimize selector must expose counts that agree with the actual payload.
replace_once(
    "entroly/server.py",
    '''            # ── engine_s6 edit-target reordering (post-selection) ──
''',
    '''            # Normalize the public selection contract across QCCR, native,
            # pure-Python, fast-path, and oversize-excerpt selectors. Several
            # fallback paths returned real selected fragments while omitting
            # ``selected_count`` (or leaving it at zero), which made agents and
            # dashboards report an empty result despite receiving context.
            selected_payload = result.get(
                "selected_fragments", result.get("selected", [])
            )
            if isinstance(selected_payload, list):
                result["selected_count"] = len(selected_payload)
                optimization_stats = result.get("optimization_stats")
                if isinstance(optimization_stats, dict):
                    optimization_stats["selected_count"] = len(selected_payload)

            # ── engine_s6 edit-target reordering (post-selection) ──
''',
)

# Bounded, cross-platform lock acquisition. Telemetry is not allowed to freeze
# an MCP/proxy request because another process is writing the value ledger.
replace_once(
    "entroly/value_tracker.py",
    '''    @contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        """Serialize value/activity mutations across independent processes."""
        self._process_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._process_lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + 30.0
                delay = 0.001
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring value-tracker lock "
                                f"{self._process_lock_path}"
                            ) from error
                        time.sleep(delay)
                        delay = min(0.05, delay * 1.5)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
''',
    '''    @staticmethod
    def _lock_timeout_seconds() -> float:
        raw = os.environ.get("ENTROLY_VALUE_LOCK_TIMEOUT", "2.0")
        try:
            timeout = float(raw)
        except (TypeError, ValueError, OverflowError):
            return 2.0
        if not math.isfinite(timeout):
            return 2.0
        return min(30.0, max(0.01, timeout))

    @contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        """Serialize mutations without allowing telemetry to block forever."""
        self._process_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._process_lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\\0")
                handle.flush()
            handle.seek(0)
            deadline = time.monotonic() + self._lock_timeout_seconds()
            delay = 0.001

            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring value-tracker lock "
                                f"{self._process_lock_path}"
                            ) from error
                        time.sleep(delay)
                        delay = min(0.05, delay * 1.5)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                        break
                    except OSError as error:
                        if error.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring value-tracker lock "
                                f"{self._process_lock_path}"
                            ) from error
                        time.sleep(delay)
                        delay = min(0.05, delay * 1.5)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
''',
)

replace_once(
    "entroly/value_tracker.py",
    '''    @staticmethod
    def _mtime(p: Path) -> float:
''',
    '''    @classmethod
    def _activity_value(cls, value: Any) -> Any:
        """Return a strict-JSON-safe activity value without changing finite signs."""
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else 0.0
        if isinstance(value, dict):
            return {
                str(key)[:120]: cls._activity_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._activity_value(item) for item in value]
        return str(value)[:500]

    @staticmethod
    def _mtime(p: Path) -> float:
''',
)

replace_once(
    "entroly/value_tracker.py",
    '''            if source:
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
''',
    '''            safe_tokens = self._nonnegative_int(tokens_saved)
            safe_cost = self._finite_float(cost_saved_usd)
            if source:
                row["source"] = str(source)[:240]
            if safe_tokens:
                row["tokens_saved"] = safe_tokens
            if safe_cost:
                row["cost_saved_usd"] = round(safe_cost, 6)
            if model:
                row["model"] = str(model)[:240]
            reserved = {
                "ts", "kind", "summary", "source", "tokens_saved",
                "cost_saved_usd", "model",
            }
            for key, value in extra.items():
                normalized_key = str(key)[:120]
                if normalized_key in reserved:
                    continue
                row[normalized_key] = self._activity_value(value)
''',
)

# Keep the public record() path fail-open on lock contention while retaining the
# process-safe implementation in a private method.
replace_once(
    "entroly/value_tracker.py",
    '''    def record(
        self,
        tokens_saved: int,
        model: str = "",
        duplicates: int = 0,
        optimized: bool = True,
        coverage_pct: float = 0.0,
        confidence: float = 0.0,
        source: str = "unclassified",
    ) -> None:
        """Record an optimization without overstating its economic evidence.

        ``source="proxy"`` records a provider-bound request whose pre/post
        token counts may support modeled API input-cost avoidance. SDK, npm,
        MCP, and local operations record token reduction only because the
        tracker cannot prove their output was sent to a paid provider.
        """
''',
    '''    def record(
        self,
        tokens_saved: int,
        model: str = "",
        duplicates: int = 0,
        optimized: bool = True,
        coverage_pct: float = 0.0,
        confidence: float = 0.0,
        source: str = "unclassified",
    ) -> None:
        """Record value without letting optional telemetry block the caller."""
        try:
            self._record_with_lock(
                tokens_saved=tokens_saved,
                model=model,
                duplicates=duplicates,
                optimized=optimized,
                coverage_pct=coverage_pct,
                confidence=confidence,
                source=source,
            )
        except TimeoutError as error:
            logger.warning("Value tracker busy; telemetry event dropped: %s", error)

    def _record_with_lock(
        self,
        tokens_saved: int,
        model: str = "",
        duplicates: int = 0,
        optimized: bool = True,
        coverage_pct: float = 0.0,
        confidence: float = 0.0,
        source: str = "unclassified",
    ) -> None:
        """Process-safe record implementation.

        ``source="proxy"`` records a provider-bound request whose pre/post
        token counts may support modeled API input-cost avoidance. SDK, npm,
        MCP, and local operations record token reduction only because the
        tracker cannot prove their output was sent to a paid provider.
        """
''',
)

# Strengthen the existing activity test against reserved-key forgery.
replace_once(
    "tests/test_value_tracker_dogfood.py",
    '''        confidence=float("inf"),
        finite_negative_delta=-12.5,
    )
''',
    '''        confidence=float("inf"),
        finite_negative_delta=-12.5,
        ts=0,
    )
''',
)
replace_once(
    "tests/test_value_tracker_dogfood.py",
    '''    assert row["finite_negative_delta"] == -12.5
    json.dumps(row, allow_nan=False)


def test_corrupt_tracker_state_fails_safe_without_claiming_old_value(
''',
    '''    assert row["finite_negative_delta"] == -12.5
    assert float(row["ts"]) > 0.0
    json.dumps(row, allow_nan=False)


def test_lock_contention_drops_optional_telemetry_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = r"""
import sys
import time
from pathlib import Path
from entroly.value_tracker import ValueTracker

root = Path(sys.argv[1])
tracker = ValueTracker(root)
with tracker._interprocess_lock():
    (root / "lock-held").write_text("held", encoding="utf-8")
    time.sleep(3.0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", holder, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "ENTROLY_DISABLE_UPDATE_CHECK": "1"},
    )
    deadline = time.monotonic() + 15
    while not (tmp_path / "lock-held").exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert (tmp_path / "lock-held").exists()

    monkeypatch.setenv("ENTROLY_VALUE_LOCK_TIMEOUT", "0.15")
    tracker = ValueTracker(tmp_path)
    started = time.monotonic()
    tracker.record(10, source="sdk")
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"optional telemetry blocked for {elapsed:.3f}s"
    assert tracker.get_lifetime()["requests_total"] == 0

    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, f"stdout={stdout}\\nstderr={stderr}"
    tracker.record(10, source="sdk")
    assert ValueTracker(tmp_path).get_lifetime()["requests_total"] == 1


def test_corrupt_tracker_state_fails_safe_without_claiming_old_value(
''',
)

for relative in (
    "scripts/apply_final_dogfood_repairs.py",
    ".github/workflows/apply-final-dogfood-repairs.yml",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()

print("Applied final MCP and telemetry dogfood repairs")
