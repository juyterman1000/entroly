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


# ---------------------------------------------------------------------------
# Multimodal: multilingual region inference, Mermaid edge order, exact file
# references, multi-file diff boundaries/path preservation, and security intent.
# ---------------------------------------------------------------------------
replace_once(
    "entroly/multimodal.py",
    '''def _infer_ui_regions(text: str) -> list[str]:
    """Infer UI regions from text content heuristics."""
    regions = []
    lower = text.lower()
    if any(w in lower for w in ["login", "sign in", "username", "password", "email"]):
        regions.append("authentication-form")
    if any(w in lower for w in ["nav", "menu", "header", "breadcrumb"]):
        regions.append("navigation")
    if any(w in lower for w in ["button", "submit", "click", "action"]):
        regions.append("interactive-controls")
    if any(w in lower for w in ["table", "list", "row", "column", "grid"]):
        regions.append("data-table")
    if any(w in lower for w in ["error", "warning", "alert", "modal", "dialog"]):
        regions.append("alert-dialog")
    if any(w in lower for w in ["chart", "graph", "visualization", "axis"]):
        regions.append("data-visualization")
    return regions or ["general-ui"]
''',
    '''def _infer_ui_regions(text: str) -> list[str]:
    """Infer common UI regions from Unicode text without requiring English."""
    regions = []
    normalized = text.casefold()
    authentication_terms = (
        "login", "log in", "sign in", "username", "password", "email", "auth",
        "تسجيل الدخول", "الدخول", "התחברות", "כניסה", "लॉगिन", "साइन इन",
        "iniciar sesión", "connexion", "anmelden", "登录", "登入", "ログイン",
        "로그인",
    )
    if any(term in normalized for term in authentication_terms):
        regions.append("authentication-form")
    if any(w in normalized for w in ["nav", "menu", "header", "breadcrumb"]):
        regions.append("navigation")
    if any(w in normalized for w in ["button", "submit", "click", "action"]):
        regions.append("interactive-controls")
    if any(w in normalized for w in ["table", "list", "row", "column", "grid"]):
        regions.append("data-table")
    if any(w in normalized for w in ["error", "warning", "alert", "modal", "dialog"]):
        regions.append("alert-dialog")
    if any(w in normalized for w in ["chart", "graph", "visualization", "axis"]):
        regions.append("data-visualization")
    return regions or ["general-ui"]
''',
)

replace_once(
    "entroly/multimodal.py",
    '''        # Extract nodes with labels
        nm = node_label_re.match(stripped)
        if nm:
            node_name = nm.group(1)
            node_label = nm.group(2).strip()
            if node_name not in nodes:
                nodes.append(f"{node_name}: {node_label}")
            continue

        # Extract arrows (flowchart)
        am = arrow_re.match(stripped)
        if am:
            src = am.group(1).strip().split("[")[0].split("(")[0].strip('"\\' ')
            dst = am.group(3).strip().split("[")[0].split("(")[0].strip('"\\' ')
            label = am.group(2) or ""
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, label.strip()))
            continue
''',
    '''        # Extract arrows before standalone node declarations. A line such as
        # ``A[Client] --> B[API]`` otherwise matches the broad node-label regex
        # from the first ``[`` to the final ``]`` and silently loses the edge.
        am = arrow_re.match(stripped)
        if am:
            src = am.group(1).strip().split("[")[0].split("(")[0].strip('"\\' ')
            dst = am.group(3).strip().split("[")[0].split("(")[0].strip('"\\' ')
            label = am.group(2) or ""
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, label.strip()))
            continue

        # Extract standalone nodes with labels.
        nm = node_label_re.match(stripped)
        if nm:
            node_name = nm.group(1)
            node_label = nm.group(2).strip()
            if node_name not in nodes:
                nodes.append(f"{node_name}: {node_label}")
            continue
''',
)

replace_once(
    "entroly/multimodal.py",
    '''    # File extensions mentioned
    file_refs = re.findall(r'\\b[\\w]+\\.(py|rs|ts|js|go|java|sql|yaml|json|toml)\\b', text.lower())
    vocab.extend([f[0] + "." + f[1] for f in file_refs[:10]])
''',
    '''    # Complete file references. A capturing group previously returned only
    # the extension (``py``) and reconstructed it as ``p.y``.
    file_refs = re.findall(
        r'\\b[\\w.-]+\\.(?:py|rs|ts|js|go|java|sql|yaml|yml|json|toml)\\b',
        text,
        re.IGNORECASE,
    )
    vocab.extend(file_refs[:10])
''',
)

replace_once(
    "entroly/multimodal.py",
    '''def _parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current_path = ""
    current_added: list[str] = []
    current_removed: list[str] = []
    current_ctx: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("--- ") or line.startswith("diff --git"):
            if current_path:
                hunks.append(DiffHunk(current_path, current_added[:], current_removed[:], current_ctx[:]))
                current_added, current_removed, current_ctx = [], [], []
            continue
        if line.startswith("+++ "):
            path = line[4:].strip().lstrip("b/")
            current_path = path
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current_added.append(line[1:].rstrip())
        elif line.startswith("-") and not line.startswith("---"):
            current_removed.append(line[1:].rstrip())
        elif line.startswith(" ") and current_path:
            current_ctx.append(line[1:].rstrip())

    if current_path:
        hunks.append(DiffHunk(current_path, current_added, current_removed, current_ctx))

    return hunks
''',
    '''def _parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current_path = ""
    current_added: list[str] = []
    current_removed: list[str] = []
    current_ctx: list[str] = []

    def flush() -> None:
        nonlocal current_path, current_added, current_removed, current_ctx
        if current_path and current_path not in {"/dev/null", "dev/null"}:
            hunks.append(
                DiffHunk(
                    current_path,
                    current_added[:],
                    current_removed[:],
                    current_ctx[:],
                )
            )
        current_path = ""
        current_added = []
        current_removed = []
        current_ctx = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            flush()
            continue
        if line.startswith("--- "):
            # ``+++`` is the authoritative destination path. Flushing here used
            # to emit an empty duplicate of the previous file.
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith(("a/", "b/")):
                path = path[2:]
            current_path = path
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current_added.append(line[1:].rstrip())
        elif line.startswith("-") and not line.startswith("---"):
            current_removed.append(line[1:].rstrip())
        elif line.startswith(" ") and current_path:
            current_ctx.append(line[1:].rstrip())

    flush()
    return hunks
''',
)

replace_once(
    "entroly/multimodal.py",
    '''def _classify_diff_intent(diff_text: str, commit_msg: str) -> str:
    text = (diff_text + " " + commit_msg).lower()
    if any(w in text for w in ["fix", "bug", "error", "broken", "crash", "fail", "patch"]):
        return "bug-fix"
    if any(w in text for w in ["test", "spec", "assert", "mock", "stub"]):
        return "test"
    if any(w in text for w in ["refactor", "clean", "rename", "move", "extract", "reorganize"]):
        return "refactor"
    if any(w in text for w in ["feat", "feature", "add", "implement", "new", "create"]):
        return "feature"
    if any(w in text for w in ["doc", "readme", "comment", "docstring"]):
        return "docs"
    if any(w in text for w in ["perf", "optim", "speed", "fast", "slow", "latency", "benchmark"]):
        return "performance"
    if any(w in text for w in ["security", "vuln", "cve", "auth", "xss", "injection"]):
        return "security"
    return "other"
''',
    '''def _classify_diff_intent(diff_text: str, commit_msg: str) -> str:
    text = (diff_text + " " + commit_msg).lower()
    # Security must outrank generic words such as "fix" and "patch".
    if any(w in text for w in [
        "security", "vuln", "cve", "xss", "csrf", "ssrf", "injection",
        "path traversal", "privilege escalation", "secret leak",
    ]):
        return "security"
    if any(w in text for w in ["fix", "bug", "error", "broken", "crash", "fail", "patch"]):
        return "bug-fix"
    if any(w in text for w in ["test", "spec", "assert", "mock", "stub"]):
        return "test"
    if any(w in text for w in ["refactor", "clean", "rename", "move", "extract", "reorganize"]):
        return "refactor"
    if any(w in text for w in ["feat", "feature", "add", "implement", "new", "create"]):
        return "feature"
    if any(w in text for w in ["doc", "readme", "comment", "docstring"]):
        return "docs"
    if any(w in text for w in ["perf", "optim", "speed", "fast", "slow", "latency", "benchmark"]):
        return "performance"
    return "other"
''',
)

# Image dimensions are an accounting and allocation boundary.
replace_once(
    "entroly/image_optimizer.py",
    '''    provider = provider if provider in {"openai", "anthropic", "gemini"} else "unknown"
    if provider == "openai":
''',
    '''    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if isinstance(height, bool) or not isinstance(height, int):
        raise TypeError("height must be an integer")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if detail not in {"low", "high"}:
        raise ValueError("detail must be 'low' or 'high'")
    provider = provider if provider in {"openai", "anthropic", "gemini"} else "unknown"
    if provider == "openai":
''',
)

# The payload-bound test must use the exact indexed identifier; a valid no-match
# is not a payload bug.
replace_once(
    "tests/test_mcp_entrypoint_dogfood.py",
    '''        {"query": "needle root wiring probe", "token_budget": 8000},
''',
    '''        {"query": "root_wiring_probe needle-root-wiring", "token_budget": 8000},
''',
)

# The dependency audit found PYSEC-2026-1845 in pytest 8.4.2.
replace_once(
    "pyproject.toml",
    '''    "pytest>=8,<9",
''',
    '''    "pytest>=9.0.3,<10",
''',
)

# ---------------------------------------------------------------------------
# Value accounting: process-safe read/modify/write and hostile-number fencing.
# ---------------------------------------------------------------------------
replace_once(
    "entroly/value_tracker.py",
    '''import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
''',
    '''import errno
import json
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''Thread-safe: all writes go through a lock + atomic file write.
''',
    '''Process-safe: all read/modify/write operations use an interprocess lock and
atomic file replacement.
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''        self._path = self._dir / self._FILE_NAME
        self._activity_path = self._dir / self._ACTIVITY_NAME
        self._lock = threading.Lock()
''',
    '''        self._path = self._dir / self._FILE_NAME
        self._activity_path = self._dir / self._ACTIVITY_NAME
        self._process_lock_path = self._dir / f"{self._FILE_NAME}.lock"
        self._lock = threading.RLock()
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''    @staticmethod
    def _mtime(p: Path) -> float:
''',
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

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        """Reload current disk state under both thread and process locks."""
        with self._lock:
            with self._interprocess_lock():
                self._data = self._load()
                self._activity = self._load_activity()
                yield

    @staticmethod
    def _nonnegative_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _finite_float(
        value: Any,
        *,
        minimum: float = 0.0,
        maximum: float | None = None,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return minimum
        if not math.isfinite(number):
            return minimum
        number = max(minimum, number)
        return min(number, maximum) if maximum is not None else number

    @staticmethod
    def _mtime(p: Path) -> float:
''',
)

replace_once(
    "entroly/value_tracker.py",
    '''            with self._lock:
                self._activity.append(row)
                self._save_activity()
''',
    '''            with self._mutation():
                self._activity.append(row)
                self._save_activity()
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''        try:
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
''',
    '''        try:
            count = self._nonnegative_int(n)
            if count == 0:
                return
            with self._mutation():
                self._data["lifetime"]["hallucinations_blocked"] = (
                    self._data["lifetime"].get("hallucinations_blocked", 0)
                    + count
                )
                self._save()
            self.record_event(
                "hallucination",
                detail or f"Blocked {count} unsupported claim(s)",
                source=source, blocked=count,
            )
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''        try:
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
''',
    '''        try:
            amount = self._finite_float(cost_saved_usd)
            if amount == 0.0:
                return
            with self._mutation():
                lt = self._data["lifetime"]
                lt["routing_saved_usd"] = round(
                    lt.get("routing_saved_usd", 0.0) + amount, 6)
                lt["routing_decisions"] = lt.get("routing_decisions", 0) + 1
                self._save()
            self.record_event(
                "routing",
                detail or f"Routed to {chosen_model or 'cheaper model'}",
                source=source, cost_saved_usd=amount,
                model=chosen_model,
            )
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''            with self._lock:
                lt = self._data["lifetime"]
                lt["beliefs_conditioned_fragments"] = (
''',
    '''            with self._mutation():
                lt = self._data["lifetime"]
                lt["beliefs_conditioned_fragments"] = (
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''        tokens_saved = max(0, int(tokens_saved))
        channel = self._channel(source)
''',
    '''        tokens_saved = self._nonnegative_int(tokens_saved)
        duplicates = self._nonnegative_int(duplicates)
        confidence = self._finite_float(confidence, maximum=1.0)
        coverage_pct = self._finite_float(coverage_pct, maximum=100.0)
        channel = self._channel(source)
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''        with self._lock:
            lt = self._data["lifetime"]
            lt["tokens_saved"] += tokens_saved
''',
    '''        with self._mutation():
            lt = self._data["lifetime"]
            lt["tokens_saved"] += tokens_saved
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''    def record_evolution_spend(
        self,
        cost_usd: float,
        success: bool = False,
    ) -> dict[str, Any]:
''',
    '''    def record_evolution_spend(
        self,
        cost_usd: float,
        success: bool = False,
    ) -> dict[str, Any]:
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''        with self._lock:
            lt = self._data.get("lifetime", {})
            lifetime_saved = lt.get("provider_cost_avoided_usd", 0.0)
            current_spent = lt.get("evolution_spent_usd", 0.0)
''',
    '''        amount = self._finite_float(cost_usd)
        if amount == 0.0:
            return {
                "status": "rejected",
                "remaining_usd": self.get_evolution_budget()["available_usd"],
            }
        with self._mutation():
            lt = self._data.get("lifetime", {})
            lifetime_saved = lt.get("provider_cost_avoided_usd", 0.0)
            current_spent = lt.get("evolution_spent_usd", 0.0)
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''            if cost_usd > available + 0.001:  # 0.1 cent tolerance
                logger.warning(
                    "Evolution spend rejected: $%.4f requested, $%.4f available",
                    cost_usd, available,
''',
    '''            if amount > available + 0.001:  # 0.1 cent tolerance
                logger.warning(
                    "Evolution spend rejected: $%.4f requested, $%.4f available",
                    amount, available,
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''            lt["evolution_spent_usd"] = round(current_spent + cost_usd, 6)
''',
    '''            lt["evolution_spent_usd"] = round(current_spent + amount, 6)
''',
)
replace_once(
    "entroly/value_tracker.py",
    '''                cost_usd, remaining, success,
''',
    '''                amount, remaining, success,
''',
)

# One-shot repair machinery must never remain in the product branch.
for relative in (
    "scripts/apply_deep_dogfood_fixes.py",
    ".github/workflows/apply-deep-dogfood-fixes.yml",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()

print("Applied exact deep-dogfood repairs")
