#!/usr/bin/env python3
"""
Entroly Trust Benchmark
=======================
Zero API keys. Zero network. Runs in < 60s on any machine.

Proves four independent claims:
  A. Compression  — real token reduction on the entroly source files themselves
  B. Classifier   — archetype precision/recall on 40 labeled prompts
  C. Hook cover   — 100% coverage over all 50 tool patterns (no unknown leakage)
  D. Router logic — Bayesian gate opens when evidence passes, stays closed when it fails
  E. Determinism  — same input → bit-identical output (SHA-256 verified)

Usage:
    python bench/trust_bench.py
    python bench/trust_bench.py --json          # machine-readable JSON report
    python bench/trust_bench.py --quiet         # summary line only
    python bench/trust_bench.py --corpus N      # use N real source files (default 15)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── ANSI ───────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _c(color: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


# ══════════════════════════════════════════════════════════════════════
# A — COMPRESSION
# ══════════════════════════════════════════════════════════════════════

def _tokens(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_compression(n_files: int = 15) -> dict:
    from entroly.universal_compress import universal_compress

    py_files = sorted(
        (REPO_ROOT / "entroly").rglob("*.py"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )[:n_files]

    if not py_files:
        return {"error": "no source files found", "pass": False}

    results = []
    corpus_hash = hashlib.sha256()

    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        corpus_hash.update(text.encode())
        original_tokens = _tokens(text)

        if original_tokens < 400:
            continue

        # Target 25% of original — aggressive but preserves key structure.
        target_ratio = 0.25
        t0 = time.perf_counter()
        compressed, actual_ratio, _method = universal_compress(text, target_ratio, "code")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        compressed_tokens = _tokens(compressed)
        ratio = compressed_tokens / original_tokens
        results.append({
            "file": path.relative_to(REPO_ROOT).as_posix(),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": round(ratio, 3),
            "target_ratio": target_ratio,
            "ms": round(elapsed_ms, 1),
        })

    if not results:
        return {"error": "all files too small", "pass": False}

    avg_ratio = sum(r["ratio"] for r in results) / len(results)
    # Pass: compressed output is 10–80% of original (meaningful but not destructive).
    passed = 0.10 < avg_ratio < 0.80 and len(results) >= 3

    return {
        "pass": passed,
        "files_tested": len(results),
        "avg_compression_ratio": round(avg_ratio, 3),
        "avg_tokens_saved_pct": round((1 - avg_ratio) * 100, 1),
        "corpus_sha256": corpus_hash.hexdigest()[:16],
        "details": results,
    }


# ══════════════════════════════════════════════════════════════════════
# B — ARCHETYPE CLASSIFIER ACCURACY
# ══════════════════════════════════════════════════════════════════════

# Ground-truth labels derived from _ARCHETYPE_PATTERNS order.
# Each prompt has exactly one unambiguous winning pattern.
LABELED_PROMPTS: list[tuple[str, str]] = [
    # test/run — pattern 1 (tool name)
    ("run pytest",                                   "test/run"),
    ("run jest to check the components",             "test/run"),
    ("run vitest on the frontend",                   "test/run"),
    # test/run — pattern 2 (run/execute + test/suite)
    ("execute the test suite",                       "test/run"),
    ("run the test suite now",                       "test/run"),

    # test/write — pattern 3
    ("write a unit test for the router",             "test/write"),
    ("add a test for the payment processor",         "test/write"),
    ("create a spec file for the auth module",       "test/write"),

    # test/fix — pattern 4 (fix|debug|failing + test)
    ("fix the failing test",                         "test/fix"),
    ("debug the broken test case",                   "test/fix"),

    # build/run — pattern 5
    ("cargo build the rust crate",                   "build/run"),
    ("compile the typescript sources",               "build/run"),
    ("run webpack to bundle the assets",             "build/run"),

    # lint/run — pattern 7
    ("run eslint on the src directory",              "lint/run"),
    ("format the codebase with prettier",            "lint/run"),
    ("run clippy on the rust module",                "lint/run"),
    ("run ruff to find issues",                      "lint/run"),

    # typecheck/run — pattern 8
    ("run mypy on the entire codebase",              "typecheck/run"),
    ("typecheck the module with pyright",            "typecheck/run"),

    # code/refactor — pattern 9
    ("refactor the authentication module",           "code/refactor"),
    ("rename the PaymentProcessor class",            "code/refactor"),
    ("extract the common utility functions",         "code/refactor"),

    # code/implement — pattern 10
    ("implement a caching function for the API",     "code/implement"),
    ("add a method to validate user input",          "code/implement"),
    ("create a new component for the dashboard",     "code/implement"),

    # code/fix_bug — pattern 11
    ("fix the authentication bug in production",     "code/fix_bug"),
    ("debug the crash in the payment processor",     "code/fix_bug"),
    ("resolve the memory leak issue",                "code/fix_bug"),

    # code/edit — pattern 12
    ("update the configuration file",               "code/edit"),
    ("change the default timeout value",             "code/edit"),
    ("modify the proxy settings",                    "code/edit"),

    # explain — pattern 13
    ("explain how context compression works",        "explain"),
    ("what is the knapsack algorithm",               "explain"),
    ("how does the Bayesian router work",             "explain"),
    ("describe the vault data structure",            "explain"),

    # inspect — pattern 14
    ("read the proxy module carefully",              "inspect"),
    ("review the authentication code",               "inspect"),
    ("find the function for token validation",       "inspect"),
    ("grep for TODO comments in the codebase",       "inspect"),

    # git/op — pattern 15
    ("git push the changes to origin",               "git/op"),
    ("commit all staged changes",                    "git/op"),
    ("merge the feature branch into main",           "git/op"),

    # setup — pattern 16
    ("install all python dependencies",              "setup"),
    ("configure the development environment",        "setup"),
    ("pip install the requirements file",            "setup"),
]


def run_classifier() -> dict:
    from entroly.ravs.router import classify_archetype

    correct = 0
    wrong: list[dict] = []
    per_label: dict[str, dict[str, int]] = {}

    for prompt, expected in LABELED_PROMPTS:
        actual = classify_archetype(prompt)
        per_label.setdefault(expected, {"tp": 0, "fp": 0, "fn": 0})
        per_label.setdefault(actual,   {"tp": 0, "fp": 0, "fn": 0})

        if actual == expected:
            correct += 1
            per_label[expected]["tp"] += 1
        else:
            wrong.append({"prompt": prompt, "expected": expected, "actual": actual})
            per_label[expected]["fn"] += 1
            per_label[actual]["fp"] += 1

    total = len(LABELED_PROMPTS)
    accuracy = correct / total

    # macro precision / recall
    precisions, recalls = [], []
    for label, c in per_label.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    macro_p = sum(precisions) / len(precisions) if precisions else 0
    macro_r = sum(recalls) / len(recalls) if recalls else 0
    f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0

    return {
        "pass": accuracy >= 0.90,
        "accuracy": round(accuracy, 3),
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "macro_f1": round(f1, 3),
        "correct": correct,
        "total": total,
        "misclassified": wrong,
    }


# ══════════════════════════════════════════════════════════════════════
# C — HOOK CLASSIFIER COVERAGE
# ══════════════════════════════════════════════════════════════════════

# One representative command per tool in _PATTERNS (50 patterns → 50 commands)
HOOK_COMMANDS: list[tuple[str, int, str]] = [
    # (command, exit_code, expected_category)
    # Tests
    ("pytest tests/ -v",                    0,  "test"),
    ("python -m pytest",                    1,  "test"),
    ("python -m unittest discover",         0,  "test"),
    ("nose2",                               0,  "test"),
    ("jest --coverage",                     0,  "test"),
    ("npx jest",                            1,  "test"),
    ("vitest run",                          0,  "test"),
    ("npx vitest",                          0,  "test"),
    ("mocha test/",                         0,  "test"),
    ("npx mocha",                           1,  "test"),
    ("npx cypress run",                     0,  "test"),
    ("npx playwright test",                 0,  "test"),
    ("go test ./...",                       0,  "test"),
    ("cargo test --lib",                    0,  "test"),
    ("dotnet test",                         1,  "test"),
    ("rspec spec/",                         0,  "test"),
    ("phpunit tests/",                      1,  "test"),
    ("mvn test",                            0,  "test"),
    ("gradle test",                         1,  "test"),
    ("npm test",                            0,  "test"),
    ("yarn test",                           1,  "test"),
    ("pnpm test",                           0,  "test"),
    ("zig test src/main.zig",               0,  "test"),
    ("swift test",                          1,  "test"),
    ("mix test",                            0,  "test"),
    # Builds
    ("cargo build --release",               0,  "build"),
    ("go build ./...",                      0,  "build"),
    ("npm run build",                       1,  "build"),
    ("tsc --outDir dist",                   0,  "build"),
    ("webpack --config webpack.prod.js",    1,  "build"),
    ("vite build",                          0,  "build"),
    ("make all",                            0,  "build"),
    ("cmake --build .",                     1,  "build"),
    ("gcc -o app main.c",                   0,  "build"),
    ("javac Main.java",                     1,  "build"),
    ("dotnet build",                        0,  "build"),
    ("zig build",                           1,  "build"),
    # Lint
    ("ruff check entroly/",                 0,  "lint"),
    ("eslint src/",                         1,  "lint"),
    ("pylint entroly/",                     0,  "lint"),
    ("flake8 .",                            1,  "lint"),
    ("cargo clippy -- -D warnings",         0,  "lint"),
    ("golangci-lint run",                   1,  "lint"),
    ("rubocop app/",                        0,  "lint"),
    ("shellcheck scripts/*.sh",             1,  "lint"),
    ("hadolint Dockerfile",                 0,  "lint"),
    # Typecheck
    ("mypy entroly/",                       0,  "typecheck"),
    ("pyright src/",                        1,  "typecheck"),
    ("tsc --noEmit",                        0,  "typecheck"),
    ("cargo check",                         0,  "typecheck"),
    # Format
    ("black --check .",                     0,  "format"),
    ("npx prettier --check src/",           1,  "format"),
    ("isort --check-only .",                0,  "format"),
    ("cargo fmt -- --check",                1,  "format"),
    ("gofmt -l .",                          0,  "format"),
]


def run_hook_coverage() -> dict:
    from entroly.ravs.hook_classifier import classify

    covered = 0
    uncovered: list[str] = []
    category_hits: dict[str, int] = {}
    wrong_category: list[dict] = []

    for cmd, exit_code, expected_cat in HOOK_COMMANDS:
        result = classify(cmd, exit_code)
        if result is None:
            uncovered.append(cmd)
        else:
            covered += 1
            category_hits[result.category] = category_hits.get(result.category, 0) + 1
            if result.category != expected_cat:
                wrong_category.append({
                    "cmd": cmd,
                    "expected": expected_cat,
                    "actual": result.category,
                })

    total = len(HOOK_COMMANDS)
    coverage_pct = covered / total * 100

    return {
        "pass": coverage_pct == 100.0 and not wrong_category,
        "coverage_pct": round(coverage_pct, 1),
        "covered": covered,
        "total": total,
        "uncovered": uncovered,
        "wrong_category": wrong_category,
        "category_distribution": category_hits,
    }


# ══════════════════════════════════════════════════════════════════════
# D — ROUTER LOGIC
# ══════════════════════════════════════════════════════════════════════

def _make_events_jsonl(path: Path, archetype: str, n_pass: int, n_fail: int) -> None:
    """Write synthetic RAVS events so the router has cells to load."""
    events = []
    for i in range(n_pass):
        events.append(json.dumps({
            "kind": "outcome",
            "event_type": archetype.split("/")[0],
            "tool": archetype.split("/")[1] if "/" in archetype else archetype,
            "value": "pass",
            "request_id": f"synth-{i}",
        }))
    for i in range(n_fail):
        events.append(json.dumps({
            "kind": "outcome",
            "event_type": archetype.split("/")[0],
            "tool": archetype.split("/")[1] if "/" in archetype else archetype,
            "value": "fail",
            "request_id": f"synth-fail-{i}",
        }))
    path.write_text("\n".join(events) + "\n", encoding="utf-8")


def run_router_logic() -> dict:
    from entroly.ravs.router import BayesianRouter

    cases = []

    # Case 1: disabled router → always use original
    r_off = BayesianRouter(enabled=False)
    d = r_off.route("claude-3-opus-20240229", "run pytest")
    cases.append({
        "case": "disabled_router",
        "prompt": "run pytest",
        "expected_use_original": True,
        "actual_use_original": d.use_original,
        "pass": d.use_original is True,
    })

    # Case 2: high-risk prompt → always use original (even if enabled)
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "events.jsonl"
        _make_events_jsonl(log, "test/pytest", n_pass=30, n_fail=0)
        r = BayesianRouter(log_path=str(log), min_samples=10, ci_threshold=0.80, enabled=True)
        d = r.route("claude-3-opus-20240229", "fix the authentication security vulnerability")
        cases.append({
            "case": "high_risk_blocked",
            "prompt": "fix the authentication security vulnerability",
            "expected_use_original": True,
            "actual_use_original": d.use_original,
            "pass": d.use_original is True,
        })

    # Case 3: insufficient data → fail closed
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "events.jsonl"
        _make_events_jsonl(log, "test/pytest", n_pass=5, n_fail=0)  # only 5, need 10
        r = BayesianRouter(log_path=str(log), min_samples=10, ci_threshold=0.80, enabled=True)
        d = r.route("claude-3-opus-20240229", "run pytest")
        cases.append({
            "case": "insufficient_data",
            "prompt": "run pytest (only 5 samples, need 10)",
            "expected_use_original": True,
            "actual_use_original": d.use_original,
            "pass": d.use_original is True,
        })

    # Case 4: sufficient high-confidence data → route to cheaper
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "events.jsonl"
        _make_events_jsonl(log, "test/pytest", n_pass=30, n_fail=0)  # 30 passes, 0 fails
        r = BayesianRouter(log_path=str(log), min_samples=10, ci_threshold=0.80, enabled=True)
        # Force cache refresh
        r._cache.loaded_at = 0.0
        d = r.route("claude-3-opus-20240229", "run pytest")
        cases.append({
            "case": "high_confidence_routes",
            "prompt": "run pytest (30 passes, CI should exceed 0.80)",
            "expected_use_original": False,
            "actual_use_original": d.use_original,
            "recommended_model": d.recommended_model,
            "confidence": d.confidence,
            "pass": d.use_original is False,
        })

    # Case 5: low CI data → fail closed
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "events.jsonl"
        _make_events_jsonl(log, "test/pytest", n_pass=12, n_fail=8)  # 60% pass rate, CI < 0.80
        r = BayesianRouter(log_path=str(log), min_samples=10, ci_threshold=0.80, enabled=True)
        r._cache.loaded_at = 0.0
        d = r.route("claude-3-opus-20240229", "run pytest")
        cases.append({
            "case": "low_ci_blocked",
            "prompt": "run pytest (12 pass / 8 fail, CI < 0.80)",
            "expected_use_original": True,
            "actual_use_original": d.use_original,
            "confidence": d.confidence,
            "pass": d.use_original is True,
        })

    all_pass = all(c["pass"] for c in cases)
    return {
        "pass": all_pass,
        "cases_total": len(cases),
        "cases_passed": sum(1 for c in cases if c["pass"]),
        "cases": cases,
    }


# ══════════════════════════════════════════════════════════════════════
# E — DETERMINISM
# ══════════════════════════════════════════════════════════════════════

def run_determinism() -> dict:
    from entroly.sdk import compress

    # Use a fixed multi-type sample: code + prose + JSON mixed
    sample = (
        "def authenticate(token: str) -> bool:\n"
        "    if not token:\n"
        "        raise ValueError('empty token')\n"
        "    return hmac.compare_digest(token, SECRET)\n\n"
        "class PaymentProcessor:\n"
        "    def charge(self, amount: float) -> dict:\n"
        "        if amount <= 0:\n"
        "            raise ValueError('amount must be positive')\n"
        "        return self._gateway.process({'amount': amount})\n\n"
        "This module handles all financial transactions. "
        "It validates inputs, delegates to the gateway, and logs results. "
        "Do not modify the validation logic without reviewing the compliance docs.\n"
        '{"schema": "v2", "rules": [{"id": 1, "action": "block", "score": 0.9}]}\n'
    ) * 8  # repeat to get a meaningful size

    budget = _tokens(sample) // 3

    sha1 = _sha256(compress(sample, budget=budget, content_type="code"))
    sha2 = _sha256(compress(sample, budget=budget, content_type="code"))
    sha3 = _sha256(compress(sample, budget=budget, content_type="code"))

    deterministic = sha1 == sha2 == sha3

    return {
        "pass": deterministic,
        "deterministic": deterministic,
        "run_1_sha256": sha1[:16],
        "run_2_sha256": sha2[:16],
        "run_3_sha256": sha3[:16],
        "input_tokens": _tokens(sample),
        "target_tokens": budget,
    }


# ══════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def _badge(passed: bool) -> str:
    return _c(GREEN, "PASS") if passed else _c(RED, "FAIL")


def _bar(ratio: float, width: int = 20) -> str:
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio*100:.0f}%"


def print_report(results: dict, quiet: bool = False) -> None:
    all_pass = results.get("meta", {}).get("all_pass", False)

    print()
    print(_c(BOLD, "  Entroly Trust Benchmark"))
    print(_c(DIM,  "  ─────────────────────────────────────────────"))
    print()

    # A — Compression
    a = results["compression"]
    if "error" not in a:
        label = _badge(a["pass"])
        ratio = a["avg_compression_ratio"]
        saved = a["avg_tokens_saved_pct"]
        saved_str = _c(CYAN, "{:.0f}% tokens saved".format(saved))
        print(f"  A  Compression   {label}  "
              f"{_bar(ratio, 18)}  "
              f"{saved_str}  "
              f"across {a['files_tested']} real source files")
        if not quiet:
            for r in a["details"][:5]:
                print(f"     {_c(DIM, r['file'][:50]): <52} "
                      f"{r['original_tokens']:>5} → {r['compressed_tokens']:>4} tok  "
                      f"({r['ratio']*100:.0f}%)  {r['ms']:.0f}ms")
            if len(a["details"]) > 5:
                extra = len(a["details"]) - 5
                print(f"     {_c(DIM, f'... and {extra} more files')}")
    else:
        print(f"  A  Compression   {_badge(False)}  {a['error']}")

    print()

    # B — Classifier
    b = results["classifier"]
    label = _badge(b["pass"])
    acc_pct = _c(CYAN, "{:.0f}%".format(b["accuracy"] * 100))
    print(f"  B  Classifier    {label}  "
          f"accuracy={acc_pct}  "
          f"precision={b['macro_precision']:.2f}  "
          f"recall={b['macro_recall']:.2f}  "
          f"F1={b['macro_f1']:.2f}  "
          f"({b['correct']}/{b['total']} prompts)")
    if not quiet and b["misclassified"]:
        print(f"     {_c(YELLOW, 'Misclassified:')}")
        for m in b["misclassified"]:
            snippet = _c(DIM, m["prompt"][:48])
            actual  = _c(RED, m["actual"])
            print(f"     {snippet: <50} expected={m['expected']}  got={actual}")

    print()

    # C — Hook coverage
    c = results["hook_coverage"]
    label = _badge(c["pass"])
    cov_pct = _c(CYAN, "{:.0f}%".format(c["coverage_pct"]))
    print(f"  C  Hook Cover    {label}  "
          f"coverage={cov_pct}  "
          f"({c['covered']}/{c['total']} commands classified)")
    if not quiet:
        dist = c["category_distribution"]
        cats = " | ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"     {_c(DIM, cats)}")
    if not quiet and c["uncovered"]:
        for cmd in c["uncovered"]:
            print(f"     {_c(RED, 'UNCOVERED:')} {cmd}")
    if not quiet and c["wrong_category"]:
        for w in c["wrong_category"]:
            print(f"     {_c(YELLOW, 'WRONG CAT:')} {w['cmd'][:45]} "
                  f"expected={w['expected']} got={w['actual']}")

    print()

    # D — Router logic
    d = results["router_logic"]
    label = _badge(d["pass"])
    print(f"  D  Router Logic  {label}  "
          f"({d['cases_passed']}/{d['cases_total']} gate cases correct)")
    if not quiet:
        for case in d["cases"]:
            icon = _c(GREEN, "✓") if case["pass"] else _c(RED, "✗")
            conf = f"  ci={case['confidence']:.3f}" if case.get("confidence") else ""
            model = f"  → {case.get('recommended_model', '')}" if not case.get("actual_use_original", True) else ""
            print(f"     {icon} {case['case']:<28} {case['prompt'][:42]}{conf}{model}")

    print()

    # E — Determinism
    e = results["determinism"]
    label = _badge(e["pass"])
    print(f"  E  Determinism   {label}  "
          f"SHA-256: {e['run_1_sha256']} (3× identical)  "
          f"{e['input_tokens']} → {e['target_tokens']} tok")

    print()
    print(_c(DIM, "  ─────────────────────────────────────────────"))

    summary_color = GREEN if all_pass else RED
    summary_word = "ALL PASS" if all_pass else "SOME FAILURES"
    print(f"  {_c(BOLD + summary_color, summary_word)}  "
          f"compression={a.get('avg_tokens_saved_pct', 0):.0f}% savings  "
          f"classifier={b['accuracy']*100:.0f}% accuracy  "
          f"router={d['cases_passed']}/{d['cases_total']} gate cases")

    if not all_pass:
        print()
        print(_c(DIM, "  Re-run with --quiet to suppress detail, "
                      "or --json for machine-readable output."))
    print()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    # Ensure Unicode box-drawing / check-mark characters render on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Entroly trust benchmark")
    ap.add_argument("--json",   action="store_true", help="emit JSON to stdout")
    ap.add_argument("--quiet",  action="store_true", help="summary line only")
    ap.add_argument("--corpus", type=int, default=15, metavar="N",
                    help="number of real source files to compress (default 15)")
    args = ap.parse_args()

    t_total = time.perf_counter()

    if not args.json and not args.quiet:
        print(_c(DIM, "\n  running …"), end="", flush=True)

    results = {}

    t0 = time.perf_counter()
    results["compression"] = run_compression(args.corpus)
    results["compression"]["wall_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    results["classifier"] = run_classifier()
    results["classifier"]["wall_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    results["hook_coverage"] = run_hook_coverage()
    results["hook_coverage"]["wall_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    results["router_logic"] = run_router_logic()
    results["router_logic"]["wall_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    results["determinism"] = run_determinism()
    results["determinism"]["wall_s"] = round(time.perf_counter() - t0, 2)

    results["meta"] = {
        "total_wall_s": round(time.perf_counter() - t_total, 2),
        "python": sys.version.split()[0],
        "repo_root": str(REPO_ROOT),
        "all_pass": all(v.get("pass", False) for v in results.values()
                        if isinstance(v, dict) and "pass" in v),
    }

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.quiet:
        m = results["meta"]
        a = results["compression"]
        b = results["classifier"]
        d = results["router_logic"]
        status = "PASS" if m["all_pass"] else "FAIL"
        print(
            f"{status}  "
            f"compression={a.get('avg_tokens_saved_pct', 0):.0f}%  "
            f"classifier={b['accuracy']*100:.0f}%  "
            f"router={d['cases_passed']}/{d['cases_total']}  "
            f"time={m['total_wall_s']}s"
        )
    else:
        if not args.json and not args.quiet:
            print("\r" + " " * 20 + "\r", end="")
        print_report(results, quiet=False)

    all_pass = results["meta"]["all_pass"]
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
