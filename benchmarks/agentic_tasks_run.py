#!/usr/bin/env python3
"""Run coding tasks through a real model, with and without compression.

Every number this emits is measured:

  * the model is called over HTTP and must answer;
  * token counts come from the provider's own accounting
    (`prompt_eval_count` / `eval_count`), never from an estimator;
  * success is decided by executing the task's test, so the oracle is the
    interpreter's exit code and not a similarity score;
  * arms are paired -- same task, same model, same seed, same decoding
    parameters, same prompt template. Only the context differs.

There is no simulated path. If the model is unreachable the run fails.

Scope, stated up front so the artifact cannot be read as more than it is: this
is a harness-validation pilot on a handful of synthetic tasks with a small
local model. It is not the preregistered experiment in
AGENTIC_TASKS_PREREGISTRATION.md, which requires mined repository tasks at
n >= 400 and a dev/holdout split by repository. Nothing here licenses a
frontier claim.

Usage:
    python benchmarks/agentic_tasks_run.py --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from entroly.native_status import native_status


def _engine_provenance() -> dict[str, Any]:
    """Which selector actually ran this comparison.

    A stale entroly_core in the interpreter's site-packages downgrades
    silently to the pure-Python fallback -- a WARNING-level log line easy to
    lose among a long run's other output, or filtered out entirely by a
    caller piping through `grep -v` to cut noise, which is exactly how this
    ran unnoticed for most of a session: every artifact from that stretch
    compared the fallback selector, not the one users get by default. Fixed
    on the same real inputs, the native engine compressed harder (1,063 vs
    1,613 tokens) and kept a task the fallback dropped, so which one ran is
    not a footnote. Recorded here so it is asserted from the artifact
    itself, not reconstructed from logs after the fact.
    """
    status = native_status()
    return {
        "native_engine_active": status.ok,
        "entroly_core_version": status.version,
        "entroly_core_version_ok": status.version_ok,
    }

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agentic_arms import (  # noqa: E402
    Arm,
    BuiltContext,
    Fragment,
    build_for_arm,
)
from benchmarks.engine_isolation import (  # noqa: E402
    assert_engine_isolated,
    isolated_engine_dir,
)

DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass
class Task:
    """A bug plus the distractors an agent would also be handed."""

    task_id: str
    query: str
    target_file: str
    broken_source: str
    test_file: str
    test_source: str
    distractors: dict[str, str] = field(default_factory=dict)
    # Set only for dependency-bearing tasks. The fix cannot be written without
    # reading this file, so dropping it is the failure this benchmark exists to
    # detect. Empty for the self-contained set, where no such file exists.
    dependency_file: str = ""
    dependency_source: str = ""
    hidden_shape: str = ""

    def fragments(self) -> list[Fragment]:
        """Everything a naive agent would put in context, in a fixed order."""
        items = [Fragment(source=self.target_file, content=self.broken_source)]
        # The dependency is ordered among the distractors rather than placed
        # directly after the target. Privileging its position would let a
        # selector score well by accident of ordering instead of by relevance.
        pool = dict(self.distractors)
        if self.dependency_file:
            pool[self.dependency_file] = self.dependency_source
        items.extend(
            Fragment(source=name, content=body)
            for name, body in sorted(pool.items())
        )
        return items


def _distractors(count: int) -> dict[str, str]:
    """Plausible but irrelevant modules, the bulk of a real repository."""
    bodies = {
        "billing/invoice.py": (
            "import datetime\n\n"
            "TAX_RATE = 0.2\n\n"
            "def build_invoice(customer, lines):\n"
            "    subtotal = sum(line['amount'] for line in lines)\n"
            "    tax = round(subtotal * TAX_RATE, 2)\n"
            "    return {'customer': customer, 'subtotal': subtotal,\n"
            "            'tax': tax, 'total': subtotal + tax,\n"
            "            'issued': datetime.date.today().isoformat()}\n"
        ),
        "reporting/exporter.py": (
            "import csv\n\n"
            "def export_rows(path, rows, headers):\n"
            "    with open(path, 'w', newline='') as handle:\n"
            "        writer = csv.DictWriter(handle, fieldnames=headers)\n"
            "        writer.writeheader()\n"
            "        for row in rows:\n"
            "            writer.writerow(row)\n"
            "    return len(rows)\n"
        ),
        "notifications/mailer.py": (
            "TEMPLATES = {'welcome': 'Hello {name}', 'reset': 'Reset: {link}'}\n\n"
            "def render(template_name, **fields):\n"
            "    template = TEMPLATES[template_name]\n"
            "    return template.format(**fields)\n\n"
            "def queue_mail(address, body):\n"
            "    return {'to': address, 'body': body, 'status': 'queued'}\n"
        ),
        "inventory/warehouse.py": (
            "class Warehouse:\n"
            "    def __init__(self):\n"
            "        self._stock = {}\n\n"
            "    def receive(self, sku, quantity):\n"
            "        self._stock[sku] = self._stock.get(sku, 0) + quantity\n"
            "        return self._stock[sku]\n\n"
            "    def available(self, sku):\n"
            "        return self._stock.get(sku, 0)\n"
        ),
        "analytics/metrics.py": (
            "def percentile(values, fraction):\n"
            "    if not values:\n"
            "        return 0.0\n"
            "    ordered = sorted(values)\n"
            "    index = int(round(fraction * (len(ordered) - 1)))\n"
            "    return ordered[index]\n\n"
            "def mean(values):\n"
            "    return sum(values) / len(values) if values else 0.0\n"
        ),
        "geo/routing.py": (
            "import math\n\n"
            "def haversine(a, b):\n"
            "    lat1, lon1 = a\n"
            "    lat2, lon2 = b\n"
            "    radius = 6371.0\n"
            "    dlat = math.radians(lat2 - lat1)\n"
            "    dlon = math.radians(lon2 - lon1)\n"
            "    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \\\n"
            "        math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2\n"
            "    return 2 * radius * math.asin(math.sqrt(h))\n"
        ),
    }
    names = sorted(bodies)
    return {name: bodies[name] for name in names[:count]}


def build_dependency_tasks(distractor_count: int) -> list[Task]:
    """Tasks whose fix cannot be written without reading another file.

    The self-contained set below cannot measure context selection: its null
    control passes 4/4, because each bug is decidable from the query and the
    broken function alone. Parity between arms on those tasks is guaranteed no
    matter what the selector delivers, so it is not evidence about compression.

    These come from ``agentic_task_set``, which was written for exactly this
    and had no consumer. Each carries a ``hidden_shape`` -- a fact that appears
    only in the dependency file -- so an arm that drops the dependency should
    fail the oracle, which is what makes a difference between arms detectable.
    """
    from benchmarks.agentic_task_set import build_dependent_tasks

    return [
        Task(
            task_id=item.task_id,
            query=item.query,
            target_file=item.target_file,
            broken_source=item.broken_source,
            test_file=item.test_file,
            test_source=item.test_source,
            distractors=dict(item.distractors),
            dependency_file=item.dependency_file,
            dependency_source=item.dependency_source,
            hidden_shape=item.hidden_shape,
        )
        for item in build_dependent_tasks(distractor_count)
    ]


def build_tasks(distractor_count: int) -> list[Task]:
    """Small, deterministic bugs whose tests are unambiguous oracles.

    Retained as a harness-validation set only. Its null control passes 4/4, so
    it must not be used for context-quality claims -- see build_dependency_tasks.
    """
    shared = _distractors(distractor_count)
    return [
        Task(
            task_id="off_by_one_slice",
            query="fix the bug in last_n so it returns the last n items",
            target_file="core/windows.py",
            broken_source=(
                "def last_n(items, n):\n"
                "    \"\"\"Return the final n items, in order.\"\"\"\n"
                "    return items[-n - 1:]\n"
            ),
            test_file="test_windows.py",
            test_source=(
                "from core.windows import last_n\n\n"
                "def test_last_n():\n"
                "    assert last_n([1, 2, 3, 4, 5], 2) == [4, 5]\n"
                "    assert last_n([1, 2, 3], 3) == [1, 2, 3]\n"
            ),
            distractors=shared,
        ),
        Task(
            task_id="wrong_comparison",
            query="fix is_adult so it treats 18 as an adult",
            target_file="core/eligibility.py",
            broken_source=(
                "MIN_AGE = 18\n\n"
                "def is_adult(age):\n"
                "    \"\"\"True when age meets the minimum.\"\"\"\n"
                "    return age > MIN_AGE\n"
            ),
            test_file="test_eligibility.py",
            test_source=(
                "from core.eligibility import is_adult\n\n"
                "def test_is_adult():\n"
                "    assert is_adult(18) is True\n"
                "    assert is_adult(17) is False\n"
            ),
            distractors=shared,
        ),
        Task(
            task_id="missing_zero_guard",
            query="fix average so an empty list does not raise",
            target_file="core/stats.py",
            broken_source=(
                "def average(values):\n"
                "    \"\"\"Mean of values; 0.0 when there are none.\"\"\"\n"
                "    return sum(values) / len(values)\n"
            ),
            test_file="test_stats.py",
            test_source=(
                "from core.stats import average\n\n"
                "def test_average():\n"
                "    assert average([2, 4]) == 3\n"
                "    assert average([]) == 0.0\n"
            ),
            distractors=shared,
        ),
        Task(
            task_id="reversed_boolean",
            query="fix has_access so denied users are rejected",
            target_file="core/access.py",
            broken_source=(
                "DENIED = {'banned', 'suspended'}\n\n"
                "def has_access(status):\n"
                "    \"\"\"True unless the status is denied.\"\"\"\n"
                "    return status in DENIED\n"
            ),
            test_file="test_access.py",
            test_source=(
                "from core.access import has_access\n\n"
                "def test_has_access():\n"
                "    assert has_access('active') is True\n"
                "    assert has_access('banned') is False\n"
            ),
            distractors=shared,
        ),
    ]


PROMPT = """You are fixing a bug in a Python project.

Task: {query}

Here is the project context:

{context}

The file {target} is failing this test:

{test}

Rewrite {target} so the test passes.
Reply with ONLY the complete corrected contents of {target}.
No explanation, no markdown fences."""


def call_model(
    *, base_url: str, model: str, prompt: str, seed: int, timeout: float
) -> dict[str, Any]:
    """One generation. Raises rather than returning a placeholder."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Held identical across arms: Entroly changes context, never
        # generation parameters.
        "options": {"temperature": 0, "seed": seed, "num_predict": 512},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"model call failed: {exc}") from exc
    elapsed = time.perf_counter() - started

    if "response" not in body:
        raise RuntimeError(f"model returned no response field: {sorted(body)}")

    return {
        "text": body["response"],
        "latency_s": round(elapsed, 3),
        # Provider accounting. Absent means unknown, never zero.
        "input_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
    }


def extract_source(text: str) -> str:
    """Recover Python source from a chat reply.

    Extraction quality is part of the measurement: if a fenced block is not
    unwrapped, or surrounding prose is left in, the file fails to parse and the
    task scores as failed for a reason that has nothing to do with the context
    the arm was given. The first version of this harness lost a task that way,
    so this prefers the largest fenced block and falls back to trimming
    non-code prose rather than returning the reply verbatim.
    """
    stripped = text.strip()

    fenced = re.findall(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)(?:```|\Z)", stripped, re.S)
    if fenced:
        return max((block.strip() for block in fenced), key=len)

    # No fences: drop leading/trailing prose lines that cannot begin a
    # top-level Python statement, which is what chat models tend to add.
    lines = stripped.splitlines()
    start = 0
    while start < len(lines) and not _looks_like_code(lines[start]):
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end]).strip()


def _looks_like_code(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    keywords = (
        "def ", "class ", "import ", "from ", "@", "#", "if ", "return ",
        "async ", "try:", "with ", "for ", "while ",
    )
    return stripped.startswith(keywords) or bool(
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*[:=]", stripped)
    )


def classify_failure(passed: bool, output: str) -> str:
    """Separate a wrong fix from output this harness could not use.

    Scoring both as 'failed' conflates a claim about compression with a claim
    about answer formatting. Only `wrong_fix` speaks to whether the context was
    sufficient.
    """
    if passed:
        return "passed"
    lowered = output.lower()
    if "error during collection" in lowered or "syntaxerror" in lowered:
        return "unusable_output"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "unusable_output"
    if "assert" in lowered or "failed" in lowered:
        return "wrong_fix"
    return "unknown"


def run_oracle(task: Task, patched_source: str) -> tuple[bool, str]:
    """Write the model's answer into a scratch project and run its test."""
    with tempfile.TemporaryDirectory(prefix="entroly-task-") as tmp:
        root = Path(tmp)
        target = root / task.target_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patched_source, encoding="utf-8")

        # Packages the test imports through.
        for parent in [target.parent, *target.parent.parents]:
            if parent == root:
                break
            (parent / "__init__.py").write_text("", encoding="utf-8")

        written = dict(task.distractors)
        if task.dependency_file:
            # Always written, regardless of whether the arm put it in context.
            # The oracle measures whether the model's patch is correct, and the
            # patch can only be correct if the dependency was read -- so the
            # dependency must exist at test time even when context omitted it.
            written[task.dependency_file] = task.dependency_source
        for name, body in written.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            (path.parent / "__init__.py").write_text("", encoding="utf-8")

        (root / task.test_file).write_text(task.test_source, encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "-m", "pytest", task.test_file, "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return completed.returncode == 0, completed.stdout[-400:]


def run_arm(
    task: Task, arm: Arm, *, budget: int, model: str, base_url: str,
    seed: int, timeout: float, optimize_fn: Any,
) -> dict[str, Any]:
    built: BuiltContext = build_for_arm(
        arm, task.fragments(), query=task.query, budget=budget,
        optimize_fn=optimize_fn,
    )
    prompt = PROMPT.format(
        query=task.query, context=built.text, target=task.target_file,
        test=task.test_source,
    )
    generation = call_model(
        base_url=base_url, model=model, prompt=prompt, seed=seed, timeout=timeout
    )
    passed, detail = run_oracle(task, extract_source(generation["text"]))

    return {
        "task_id": task.task_id,
        "arm": arm.value,
        "passed": passed,
        "outcome": classify_failure(passed, detail),
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "latency_s": generation["latency_s"],
        "context": built.to_dict(),
        "oracle_tail": detail.strip()[-200:],
    }


def mcnemar_exact(left_only: int, right_only: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    import math

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** discordant))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--budget", type=int, default=220)
    parser.add_argument("--distractors", type=int, default=6)
    parser.add_argument(
        "--task-set",
        choices=("dependency", "self-contained"),
        default="dependency",
        help=(
            "dependency (default): the fix requires reading another file, so "
            "dropping it is detectable. self-contained: harness validation "
            "only -- its null control passes 4/4 and it cannot support a "
            "context-quality claim."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "benchmarks" / "results"
                        / "agentic_tasks_pilot.json")
    args = parser.parse_args()

    tasks = (
        build_dependency_tasks(args.distractors)
        if args.task_set == "dependency"
        else build_tasks(args.distractors)
    )
    arms = [Arm.RAW, Arm.COMPRESS]
    rows: list[dict[str, Any]] = []

    with isolated_engine_dir():
        assert_engine_isolated()
        from entroly import optimize  # imported inside the isolated context

        for task in tasks:
            for arm in arms:
                print(f"  {task.task_id:22} {arm.value:22} ...", end="", flush=True)
                row = run_arm(
                    task, arm, budget=args.budget, model=args.model,
                    base_url=args.base_url, seed=args.seed,
                    timeout=args.timeout, optimize_fn=optimize,
                )
                rows.append(row)
                print(f" {'PASS' if row['passed'] else 'FAIL'}"
                      f"  in={row['input_tokens']} out={row['output_tokens']}"
                      f" {row['latency_s']}s")

    summary: dict[str, Any] = {}
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm.value]
        inputs = [r["input_tokens"] for r in selected if r["input_tokens"]]
        summary[arm.value] = {
            "tasks": len(selected),
            "passed": sum(r["passed"] for r in selected),
            "total_input_tokens": sum(inputs),
            "mean_latency_s": round(
                sum(r["latency_s"] for r in selected) / max(1, len(selected)), 3
            ),
        }

    raw_by_task = {r["task_id"]: r["passed"] for r in rows if r["arm"] == "raw"}
    cmp_by_task = {
        r["task_id"]: r["passed"] for r in rows
        if r["arm"] == Arm.COMPRESS.value
    }
    raw_only = sum(
        1 for t in raw_by_task if raw_by_task[t] and not cmp_by_task.get(t)
    )
    cmp_only = sum(
        1 for t in cmp_by_task if cmp_by_task[t] and not raw_by_task.get(t)
    )

    raw_tokens = summary["raw"]["total_input_tokens"]
    cmp_tokens = summary[Arm.COMPRESS.value]["total_input_tokens"]
    reduction = (
        round(1 - cmp_tokens / raw_tokens, 4) if raw_tokens else None
    )

    artifact = {
        "metadata": {
            "model": args.model,
            "seed": args.seed,
            "budget": args.budget,
            "distractors": args.distractors,
            "task_set": args.task_set,
            "simulated": False,
            "token_source": "ollama prompt_eval_count / eval_count",
            "oracle": "pytest exit code",
            "engine": _engine_provenance(),
        },
        "summary": summary,
        "paired": {
            "raw_only_successes": raw_only,
            "compress_only_successes": cmp_only,
            "mcnemar_exact_p": round(mcnemar_exact(raw_only, cmp_only), 6),
            # Named, not just counted: a failure is only diagnosable if you can
            # see which task diverged and what its arm was shown.
            "regressions": sorted(
                t for t in raw_by_task
                if raw_by_task[t] and not cmp_by_task.get(t)
            ),
        },
        # Per-task rows carry each arm's included/omitted sources, so a lost
        # task can be attributed to a specific dropped fragment rather than to
        # "compression" in the abstract.
        "rows": rows,
        "input_token_reduction": reduction,
        "limitations": [
            "Harness-validation pilot, not the preregistered experiment in "
            "benchmarks/AGENTIC_TASKS_PREREGISTRATION.md, which requires "
            "mined repository tasks at n >= 400 with a dev/holdout split.",
            f"n = {len(tasks)} synthetic single-file bugs; far too few for a "
            "non-inferiority claim at any useful margin.",
            f"One small local model ({args.model}); results do not transfer to "
            "frontier models.",
            "Distractor context is synthetic and uniform, so the selection "
            "problem is easier than a real repository.",
            "Single seed, no repeats, so per-task variance is unmeasured.",
            "No CLOSED-LOOP arm: recovery-on-failure is not exercised here.",
            "CEILING EFFECT: every arm solves every task, so this run has no "
            "power to detect a difference in success. Parity here is the "
            "absence of evidence, not evidence of equivalence. The tasks must "
            "be made hard enough that the raw arm sometimes fails before any "
            "non-inferiority statement is meaningful.",
            "Answer extraction was a confound and materially changed the "
            "result. An earlier version of this harness unwrapped fenced code "
            "naively; the model's prose survived, the file failed to parse, "
            "and the run reported raw 3/4 versus compressed 2/4 -- an apparent "
            "regression caused by the harness, not by compression. With robust "
            "extraction both arms score 4/4. Outcomes are now classified so "
            "'wrong_fix' (context may have been insufficient) is never "
            "conflated with 'unusable_output' (the reply could not be used).",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print("\n=== measured ===")
    for arm_name, stats in summary.items():
        print(f"{arm_name:22} {stats['passed']}/{stats['tasks']} passed  "
              f"input_tokens={stats['total_input_tokens']}  "
              f"mean_latency={stats['mean_latency_s']}s")
    print(f"input token reduction : {reduction}")
    print(f"paired McNemar p      : {artifact['paired']['mcnemar_exact_p']}")
    print(f"artifact              : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
