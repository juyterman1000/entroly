"""Run an evidence-gated, same-input compression comparison.

The gauntlet compares public compressor entry points in isolated Python
environments. Every participant receives byte-identical fixtures. A participant
is eligible to win a scenario only when it preserves every preregistered answer
needle, produces deterministic output, and reduces tokens under one shared
tokenizer. This is a compression/evidence benchmark, not an LLM answer-quality
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from html import escape
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "entroly.compression-gauntlet.v1"
FIXTURE_VERSION = "agent-tool-evidence-v1"
TOKENIZER = "o200k_base"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_BUDGET = 1_200
DEFAULT_RUNS = 3
DEFAULT_WARMUPS = 1
_SUBPROCESS_ENV_ALLOWLIST = (
    "APPDATA",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    category: str
    query: str
    content: str
    evidence_needles: tuple[str, ...]

    def public_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["evidence_needles"] = list(self.evidence_needles)
        record["content_sha256"] = _sha256(self.content)
        return record


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _distribution_record_sha256(package: str) -> str | None:
    try:
        record = importlib.metadata.distribution(package).read_text("RECORD")
    except importlib.metadata.PackageNotFoundError:
        return None
    return _sha256(record) if record else None


def _versions(packages: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _implementation_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_rows(rows: list[dict[str, Any]], *, compact: bool) -> str:
    if compact:
        return json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)


def build_scenarios() -> list[Scenario]:
    build_lines = [f"   Compiling crate_{index:04d} v0.1.{index % 10}" for index in range(2_200)]
    build_lines.extend(
        [
            "error[E0382]: borrow of moved value: `session`",
            " --> src/auth/session.rs:184:9",
            "incident AUTH-REFRESH-184 exceeded retry window",
            "hint: increase token refresh slack before retry",
            "Build finished with exit code 1",
        ]
    )

    incident_rows = [
        {
            "id": index,
            "latency_ms": index % 117,
            "message": "normal request completed",
            "payload": "x" * 60,
            "service": "auth" if index % 3 == 0 else "billing",
            "status": "ok",
        }
        for index in range(900)
    ]
    incident_rows[667] = {
        "id": 667,
        "incident_id": "INC-JSON-667",
        "latency_ms": 9912,
        "message": "refresh timeout at retry boundary",
        "payload": "z" * 1_000,
        "service": "auth",
        "status": "failed",
    }

    sre_lines = [
        f"2026-07-13T08:{index % 60:02d}:00Z INFO worker heartbeat shard={index % 32}"
        for index in range(1_800)
    ]
    sre_lines.extend(
        [
            "2026-07-13T08:44:19Z WARN api latency p99 crossed threshold service=auth",
            "2026-07-13T08:44:22Z ERROR upstream refused connection service=auth region=us-west-2",
            "2026-07-13T08:44:25Z FATAL incident INC-9281 auth unavailable",
            "rollback candidate: deploy 2026.07.13.4",
        ]
    )

    search_rows = [
        {
            "path": f"src/components/component_{index:04d}.py",
            "score": round(1.0 - index / 2_000, 4),
            "snippet": "def render_component(state): return state",
            "symbol": f"render_component_{index:04d}",
        }
        for index in range(1_000)
    ]
    search_rows[731] = {
        "commit": "9f4c2ab",
        "incident": "AUTH-RACE-771",
        "path": "src/oauth/refresh_state.py:219",
        "retry_epoch": 184467,
        "score": 0.1337,
        "snippet": "compare_and_swap_refresh_token prevents the OAuth refresh race",
        "symbol": "compare_and_swap_refresh_token",
    }

    return [
        Scenario(
            scenario_id="cargo-build-failure",
            category="build_log",
            query="why did the auth refresh build fail and what should change",
            content="\n".join(build_lines),
            evidence_needles=(
                "AUTH-REFRESH-184",
                "src/auth/session.rs:184:9",
                "increase token refresh slack",
                "exit code 1",
            ),
        ),
        Scenario(
            scenario_id="json-incident-middle",
            category="json",
            query="find the failed auth refresh timeout incident and exact latency",
            content=_json_rows(incident_rows, compact=False),
            evidence_needles=(
                "INC-JSON-667",
                "9912",
                "refresh timeout at retry boundary",
                "failed",
            ),
        ),
        Scenario(
            scenario_id="sre-incident-tail",
            category="sre_log",
            query="which incident caused the auth outage and what should roll back",
            content="\n".join(sre_lines),
            evidence_needles=(
                "INC-9281",
                "upstream refused connection",
                "us-west-2",
                "deploy 2026.07.13.4",
            ),
        ),
        Scenario(
            scenario_id="code-search-middle",
            category="search_json",
            query="find the OAuth refresh race fix, source path, commit, and retry epoch",
            content=_json_rows(search_rows, compact=True),
            evidence_needles=(
                "AUTH-RACE-771",
                "src/oauth/refresh_state.py:219",
                "9f4c2ab",
                "184467",
            ),
        ),
    ]


def _messages(scenario: Scenario) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": scenario.query},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{scenario.scenario_id}",
                    "type": "function",
                    "function": {"name": "benchmark_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{scenario.scenario_id}",
            "content": scenario.content,
        },
    ]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_repeated(
    compress: Callable[[Scenario], tuple[str, dict[str, Any]]],
    scenario: Scenario,
    *,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        compress(scenario)

    latencies: list[float] = []
    outputs: list[str] = []
    metadata: dict[str, Any] = {}
    for _ in range(runs):
        started = time.perf_counter()
        output, metadata = compress(scenario)
        latencies.append((time.perf_counter() - started) * 1_000)
        outputs.append(output)

    hashes = {_sha256(output) for output in outputs}
    return {
        "scenario_id": scenario.scenario_id,
        "output_text": outputs[-1],
        "output_sha256": _sha256(outputs[-1]),
        "deterministic": len(hashes) == 1,
        "latency_ms": {
            "samples": [round(value, 3) for value in latencies],
            "p50": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "native_metrics": metadata,
    }


def _entroly_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    from entroly import __version__
    from entroly.compression_proxy import compress_proxy_payload

    budget = int(payload["budget_tokens"])

    def compress(scenario: Scenario) -> tuple[str, dict[str, Any]]:
        result = compress_proxy_payload(
            {"model": payload["model"], "messages": _messages(scenario)},
            query=scenario.query,
            budget_tokens=budget,
            include_receipt_header=False,
        )
        output = str(result.body["messages"][-1]["content"])
        return output, {
            "changed": result.changed,
            "compressed_blocks": result.receipt.compressed_blocks,
            "receipt_count": len(result.receipt.receipts),
        }

    root = Path(__file__).resolve().parents[1]
    return {
        "system": "entroly",
        "package": "entroly",
        "version": __version__,
        "algorithm": "query-conditioned evidence locks + entropy/outlier selection",
        "config": {"budget_tokens": budget, "receipt_header": False},
        "runtime": {
            "python": platform.python_version(),
            "import_origin": "repository source checkout",
            "native_acceleration_used": False,
            "dependencies": _versions(("tiktoken",)),
            "implementation_sha256": _implementation_sha256(
                (
                    Path(__file__).resolve(),
                    root / "entroly" / "compression_proxy.py",
                    root / "entroly" / "evidence_locked_compression.py",
                )
            ),
        },
        "results": [
            _run_repeated(
                compress,
                scenario,
                warmups=int(payload["warmups"]),
                runs=int(payload["runs"]),
            )
            for scenario in _scenarios_from_payload(payload)
        ],
    }


def _headroom_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    from headroom import compress as headroom_compress

    version = importlib.metadata.version("headroom-ai")

    def compress(scenario: Scenario) -> tuple[str, dict[str, Any]]:
        messages = json.loads(json.dumps(_messages(scenario)))
        result = headroom_compress(
            messages,
            model=str(payload["model"]),
            protect_recent=0,
            savings_profile="agent-90",
        )
        output = str(result.messages[-1]["content"])
        return output, {
            "tokens_before": int(result.tokens_before),
            "tokens_after": int(result.tokens_after),
            "transforms": list(result.transforms_applied),
        }

    return {
        "system": "headroom",
        "package": "headroom-ai",
        "version": version,
        "algorithm": "released public compress() pipeline with agent-90 profile",
        "config": {
            "model": payload["model"],
            "protect_recent": 0,
            "savings_profile": "agent-90",
        },
        "runtime": {
            "python": platform.python_version(),
            "dependencies": _versions(
                ("headroom-ai", "litellm", "onnxruntime", "tiktoken", "transformers")
            ),
            "distribution_record_sha256": _distribution_record_sha256("headroom-ai"),
        },
        "results": [
            _run_repeated(
                compress,
                scenario,
                warmups=int(payload["warmups"]),
                runs=int(payload["runs"]),
            )
            for scenario in _scenarios_from_payload(payload)
        ],
    }


def _scenarios_from_payload(payload: dict[str, Any]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for record in payload["scenarios"]:
        scenarios.append(
            Scenario(
                scenario_id=str(record["scenario_id"]),
                category=str(record["category"]),
                query=str(record["query"]),
                content=str(record["content"]),
                evidence_needles=tuple(str(value) for value in record["evidence_needles"]),
            )
        )
    return scenarios


def run_adapter(system: str, payload: dict[str, Any]) -> dict[str, Any]:
    if system == "entroly":
        return _entroly_adapter(payload)
    if system == "headroom":
        return _headroom_adapter(payload)
    raise ValueError(f"unsupported adapter: {system}")


def _adapter_command(python: str, system: str) -> list[str]:
    return [python, "-m", "benchmarks.compression_gauntlet", "adapter", "--system", system]


def _subprocess_env() -> dict[str, str]:
    """Pass only runtime essentials; benchmark adapters never need user secrets."""
    environment = {
        key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _invoke_adapter(
    command: list[str], payload: dict[str, Any], *, root: Path, timeout: float
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=_subprocess_env(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_000:]
        raise RuntimeError(f"adapter failed ({completed.returncode}): {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"adapter returned invalid JSON: {completed.stdout[-2_000:]}") from error
    if not isinstance(report, dict):
        raise RuntimeError("adapter report must be a JSON object")
    report["stderr_sha256"] = _sha256(completed.stderr)
    return report


def _token_counter() -> tuple[Callable[[str], int], str]:
    import tiktoken

    encoding = tiktoken.get_encoding(TOKENIZER)

    def count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    return count, importlib.metadata.version("tiktoken")


def analyze(
    *,
    scenarios: list[Scenario],
    adapters: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    count_tokens, tokenizer_version = _token_counter()
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    rows: list[dict[str, Any]] = []
    participant_meta: dict[str, Any] = {}

    for adapter in adapters:
        system = str(adapter["system"])
        participant_meta[system] = {
            key: adapter[key]
            for key in (
                "package",
                "version",
                "algorithm",
                "config",
                "runtime",
                "stderr_sha256",
            )
            if key in adapter
        }
        seen: set[str] = set()
        for result in adapter.get("results", []):
            scenario_id = str(result["scenario_id"])
            if scenario_id not in scenario_by_id or scenario_id in seen:
                raise ValueError(f"{system} returned an unknown or duplicate scenario {scenario_id!r}")
            seen.add(scenario_id)
            scenario = scenario_by_id[scenario_id]
            output = str(result["output_text"])
            found = [needle for needle in scenario.evidence_needles if needle in output]
            input_tokens = count_tokens(scenario.content)
            output_tokens = count_tokens(output)
            savings = 0.0 if input_tokens == 0 else 1.0 - output_tokens / input_tokens
            evidence_recall = len(found) / len(scenario.evidence_needles)
            valid = (
                bool(result.get("deterministic"))
                and evidence_recall == 1.0
                and 0 <= output_tokens <= input_tokens
            )
            rows.append(
                {
                    "system": system,
                    "scenario_id": scenario_id,
                    "category": scenario.category,
                    "input_sha256": _sha256(scenario.content),
                    "output_sha256": str(result["output_sha256"]),
                    "output_text": output,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_saved": max(0, input_tokens - output_tokens),
                    "savings_ratio": round(savings, 6),
                    "evidence_found": found,
                    "evidence_missing": [
                        needle for needle in scenario.evidence_needles if needle not in output
                    ],
                    "evidence_recall": round(evidence_recall, 6),
                    "deterministic": bool(result.get("deterministic")),
                    "valid": valid,
                    "compressed": valid and output_tokens < input_tokens,
                    "latency_ms": result["latency_ms"],
                    "native_metrics": result.get("native_metrics", {}),
                }
            )
        if seen != set(scenario_by_id):
            missing = sorted(set(scenario_by_id).difference(seen))
            raise ValueError(f"{system} returned an incomplete matrix: {missing}")

    aggregates: dict[str, Any] = {}
    for system in participant_meta:
        system_rows = [row for row in rows if row["system"] == system]
        input_total = sum(row["input_tokens"] for row in system_rows)
        output_total = sum(row["output_tokens"] for row in system_rows)
        aggregates[system] = {
            "scenarios": len(system_rows),
            "valid_scenarios": sum(row["valid"] for row in system_rows),
            "compressed_scenarios": sum(row["compressed"] for row in system_rows),
            "evidence_recall": round(
                statistics.mean(row["evidence_recall"] for row in system_rows), 6
            ),
            "weighted_savings_ratio": round(1.0 - output_total / input_total, 6),
            "macro_savings_ratio": round(
                statistics.mean(row["savings_ratio"] for row in system_rows), 6
            ),
            "p50_latency_ms": round(
                statistics.median(row["latency_ms"]["p50"] for row in system_rows), 3
            ),
            "passed": all(row["valid"] for row in system_rows),
        }

    eligible_systems = [
        (system, aggregate)
        for system, aggregate in aggregates.items()
        if aggregate["passed"]
    ]
    eligible_systems.sort(
        key=lambda item: (-item[1]["weighted_savings_ratio"], item[0])
    )
    has_strict_winner = (
        len(eligible_systems) >= 2
        and eligible_systems[0][1]["weighted_savings_ratio"]
        > eligible_systems[1][1]["weighted_savings_ratio"]
    )
    suite_winner = eligible_systems[0][0] if has_strict_winner else None
    runner_up = eligible_systems[1][0] if has_strict_winner else None
    margin = (
        aggregates[suite_winner]["weighted_savings_ratio"]
        - aggregates[runner_up]["weighted_savings_ratio"]
        if suite_winner and runner_up
        else None
    )

    fixture_manifest = [scenario.public_record() for scenario in scenarios]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            **protocol,
            "fixture_version": FIXTURE_VERSION,
            "fixture_sha256": _sha256(
                json.dumps(fixture_manifest, sort_keys=True, separators=(",", ":"))
            ),
            "tokenizer": TOKENIZER,
            "tokenizer_package": f"tiktoken=={tokenizer_version}",
            "winner_gate": (
                "complete matrix + deterministic output + 100% preregistered evidence "
                "retention + no token inflation"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "participants": participant_meta,
        "fixtures": fixture_manifest,
        "results": rows,
        "aggregates": aggregates,
        "suite_winner": suite_winner,
        "runner_up": runner_up,
        "winner_margin": round(margin, 6) if margin is not None else None,
        "headline_eligible": suite_winner is not None,
        "claim_scope": (
            "Deterministic compression and preregistered string-evidence retention on "
            f"{len(scenarios)} generated agent-tool fixtures; no LLM answer-quality claim."
        ),
        "caveats": [
            "This suite measures compressed prompt evidence, not downstream model answers.",
            "Synthetic fixtures are useful regression controls but do not replace public real-task datasets.",
            "A suite win applies only to the pinned versions, entry points, configuration, and tokenizer.",
            "Latency includes each public compression call but excludes process startup and model-provider latency.",
            "Entroly's gauntlet adapter is deterministic statistical selection; it is not evidence of neural-model superiority.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    verify_report(report)
    winner = report.get("suite_winner")
    if winner:
        result_line = (
            f"**SUITE WIN: {winner}** by "
            f"{report['winner_margin'] * 100:.1f} percentage points of weighted token savings "
            "after the evidence and determinism gates."
        )
    else:
        result_line = "**NO COMPETITIVE CLAIM.** Fewer than two participants passed every gate."

    lines = [
        "# Agent Context Compression Gauntlet",
        "",
        result_line,
        "",
        report["claim_scope"],
        "",
        "## Overall",
        "",
        "| System | Version | Evidence recall | Weighted savings | Median latency | Valid scenarios | Compressed | Result |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for system, aggregate in report["aggregates"].items():
        version = report["participants"][system]["version"]
        lines.append(
            f"| {system} | {version} | {aggregate['evidence_recall']:.1%} | "
            f"{aggregate['weighted_savings_ratio']:.1%} | {aggregate['p50_latency_ms']:.1f} ms | "
            f"{aggregate['valid_scenarios']}/{aggregate['scenarios']} | "
            f"{aggregate['compressed_scenarios']}/{aggregate['scenarios']} | "
            f"{'PASS' if aggregate['passed'] else 'NO CLAIM'} |"
        )

    lines.extend(
        [
            "",
            "## Per-fixture evidence gate",
            "",
            "| Fixture | System | Input tokens | Output tokens | Savings | Evidence | Deterministic | Valid |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in report["results"]:
        lines.append(
            f"| {row['scenario_id']} | {row['system']} | {row['input_tokens']:,} | "
            f"{row['output_tokens']:,} | {row['savings_ratio']:.1%} | "
            f"{row['evidence_recall']:.0%} | {'yes' if row['deterministic'] else 'no'} | "
            f"{'yes' if row['valid'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Fixtures: `{report['protocol']['fixture_version']}` (`{report['protocol']['fixture_sha256']}`)",
            f"- Tokenizer: `{report['protocol']['tokenizer_package']}` / `{report['protocol']['tokenizer']}`",
            f"- Runs: {report['protocol']['runs']} measured after {report['protocol']['warmups']} warm-up(s)",
            f"- Budget: {report['protocol']['budget_tokens']:,} Entroly tokens per tool block",
            f"- Gate: {report['protocol']['winner_gate']}",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m benchmarks.compression_gauntlet run \\",
            "  --headroom-python /path/to/headroom-0.31.0-venv/bin/python \\",
            "  --require-comparator \\",
            f"  --runs {report['protocol']['runs']} --warmups {report['protocol']['warmups']} \\",
            "  --output benchmarks/results/compression_gauntlet.json \\",
            "  --markdown benchmarks/results/compression_gauntlet.md",
            "```",
            "",
            "The JSON artifact is the source of truth. The Markdown file is generated from it.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_report(report: dict[str, Any]) -> None:
    """Fail closed if a published artifact no longer matches its raw evidence."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    fixtures = report.get("fixtures")
    results = report.get("results")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("report is missing fixtures")
    if not isinstance(results, list) or not results:
        raise ValueError("report is missing results")

    fixture_manifest = json.dumps(fixtures, sort_keys=True, separators=(",", ":"))
    if _sha256(fixture_manifest) != report["protocol"].get("fixture_sha256"):
        raise ValueError("fixture_sha256 does not match fixtures")
    by_id: dict[str, dict[str, Any]] = {}
    for fixture in fixtures:
        scenario_id = str(fixture.get("scenario_id", ""))
        content = fixture.get("content")
        if not scenario_id or scenario_id in by_id or not isinstance(content, str):
            raise ValueError("fixtures contain a missing or duplicate scenario_id")
        if _sha256(content) != fixture.get("content_sha256"):
            raise ValueError(f"content_sha256 does not match {scenario_id}")
        by_id[scenario_id] = fixture

    count_tokens, tokenizer_version = _token_counter()
    expected_tokenizer = f"tiktoken=={tokenizer_version}"
    if report["protocol"].get("tokenizer_package") != expected_tokenizer:
        raise ValueError(
            "verification tokenizer version differs from the artifact; "
            f"expected {report['protocol'].get('tokenizer_package')}, got {expected_tokenizer}"
        )

    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        system = str(row.get("system", ""))
        scenario_id = str(row.get("scenario_id", ""))
        key = (system, scenario_id)
        if not system or scenario_id not in by_id or key in seen:
            raise ValueError("results contain an unknown or duplicate system/scenario row")
        seen.add(key)
        fixture = by_id[scenario_id]
        output = row.get("output_text")
        if not isinstance(output, str) or _sha256(output) != row.get("output_sha256"):
            raise ValueError(f"output_sha256 does not match {system}/{scenario_id}")
        input_tokens = count_tokens(str(fixture["content"]))
        output_tokens = count_tokens(output)
        if row.get("input_sha256") != fixture["content_sha256"]:
            raise ValueError(f"input_sha256 does not match {system}/{scenario_id}")
        if row.get("input_tokens") != input_tokens or row.get("output_tokens") != output_tokens:
            raise ValueError(f"token counts do not match {system}/{scenario_id}")
        needles = [str(value) for value in fixture["evidence_needles"]]
        found = [needle for needle in needles if needle in output]
        missing = [needle for needle in needles if needle not in output]
        if row.get("evidence_found") != found or row.get("evidence_missing") != missing:
            raise ValueError(f"evidence lists do not match {system}/{scenario_id}")
        evidence_recall = len(found) / len(needles)
        savings = 0.0 if input_tokens == 0 else 1.0 - output_tokens / input_tokens
        valid = (
            bool(row.get("deterministic"))
            and evidence_recall == 1.0
            and output_tokens <= input_tokens
        )
        if row.get("savings_ratio") != round(savings, 6):
            raise ValueError(f"savings_ratio does not match {system}/{scenario_id}")
        if row.get("evidence_recall") != round(evidence_recall, 6):
            raise ValueError(f"evidence_recall does not match {system}/{scenario_id}")
        if row.get("valid") is not valid:
            raise ValueError(f"valid gate does not match {system}/{scenario_id}")
        if row.get("compressed") is not (valid and output_tokens < input_tokens):
            raise ValueError(f"compressed gate does not match {system}/{scenario_id}")
        grouped.setdefault(system, []).append(row)

    expected_matrix = {
        (system, scenario_id) for system in grouped for scenario_id in by_id
    }
    if seen != expected_matrix:
        raise ValueError("results contain an incomplete condition matrix")
    if set(grouped) != set(report.get("participants", {})):
        raise ValueError("participants do not match result systems")
    if set(grouped) != set(report.get("aggregates", {})):
        raise ValueError("aggregates do not match result systems")

    entroly_runtime = report.get("participants", {}).get("entroly", {}).get("runtime", {})
    if entroly_runtime.get("import_origin") == "repository source checkout":
        from entroly import __version__ as current_entroly_version

        if report["participants"]["entroly"].get("version") != current_entroly_version:
            raise ValueError("Entroly version changed after the artifact was generated")
        root = Path(__file__).resolve().parents[1]
        implementation_sha256 = _implementation_sha256(
            (
                Path(__file__).resolve(),
                root / "entroly" / "compression_proxy.py",
                root / "entroly" / "evidence_locked_compression.py",
            )
        )
        if entroly_runtime.get("implementation_sha256") != implementation_sha256:
            raise ValueError("Entroly implementation changed after the artifact was generated")

    recomputed: dict[str, dict[str, Any]] = {}
    for system, system_rows in grouped.items():
        input_total = sum(int(row["input_tokens"]) for row in system_rows)
        output_total = sum(int(row["output_tokens"]) for row in system_rows)
        recomputed[system] = {
            "scenarios": len(system_rows),
            "valid_scenarios": sum(bool(row["valid"]) for row in system_rows),
            "compressed_scenarios": sum(bool(row["compressed"]) for row in system_rows),
            "evidence_recall": round(
                statistics.mean(float(row["evidence_recall"]) for row in system_rows), 6
            ),
            "weighted_savings_ratio": round(1.0 - output_total / input_total, 6),
            "macro_savings_ratio": round(
                statistics.mean(float(row["savings_ratio"]) for row in system_rows), 6
            ),
            "p50_latency_ms": round(
                statistics.median(float(row["latency_ms"]["p50"]) for row in system_rows), 3
            ),
            "passed": all(bool(row["valid"]) for row in system_rows),
        }
    if report["aggregates"] != recomputed:
        raise ValueError("aggregates do not match raw result rows")

    ranked = sorted(
        (
            (system, aggregate)
            for system, aggregate in recomputed.items()
            if aggregate["passed"]
        ),
        key=lambda item: (-item[1]["weighted_savings_ratio"], item[0]),
    )
    has_strict_winner = (
        len(ranked) >= 2
        and ranked[0][1]["weighted_savings_ratio"]
        > ranked[1][1]["weighted_savings_ratio"]
    )
    winner = ranked[0][0] if has_strict_winner else None
    runner_up = ranked[1][0] if has_strict_winner else None
    margin = (
        recomputed[winner]["weighted_savings_ratio"]
        - recomputed[runner_up]["weighted_savings_ratio"]
        if winner and runner_up
        else None
    )
    if report.get("suite_winner") != winner or report.get("runner_up") != runner_up:
        raise ValueError("winner fields do not match aggregates")
    if report.get("winner_margin") != (round(margin, 6) if margin is not None else None):
        raise ValueError("winner_margin does not match aggregates")
    if report.get("headline_eligible") is not (winner is not None):
        raise ValueError("headline_eligible does not match the winner gate")


def render_svg(report: dict[str, Any]) -> str:
    """Render a shareable card without separating the numbers from caveats."""
    verify_report(report)
    systems = list(report["aggregates"])
    if len(systems) < 2:
        raise ValueError("the social card requires at least two participants")
    winner = str(report["suite_winner"] or systems[0])
    ordered = [winner, *[system for system in systems if system != winner]][:2]
    colors = ("#35D0BA", "#8A9BB0")
    rows: list[str] = []
    for index, (system, color) in enumerate(zip(ordered, colors)):
        aggregate = report["aggregates"][system]
        participant = report["participants"][system]
        savings = float(aggregate["weighted_savings_ratio"])
        bar_width = round(700 * max(0.0, min(1.0, savings)), 1)
        y = 275 + index * 120
        rows.extend(
            [
                f'<text x="90" y="{y}" class="system">{escape(system.title())}</text>',
                f'<text x="90" y="{y + 34}" class="version">v{escape(str(participant["version"]))}</text>',
                f'<rect x="330" y="{y - 33}" width="700" height="48" rx="12" fill="#17253A"/>',
                f'<rect x="330" y="{y - 33}" width="{bar_width}" height="48" rx="12" fill="{color}"/>',
                f'<text x="1065" y="{y + 3}" class="value" text-anchor="end">{savings:.1%}</text>',
                f'<text x="330" y="{y + 43}" class="detail">'
                f'{aggregate["evidence_recall"]:.0%} evidence retained · '
                f'{aggregate["compressed_scenarios"]}/{aggregate["scenarios"]} fixtures compressed</text>',
            ]
        )

    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="title desc">',
            '<title id="title">Entroly Agent Context Compression Gauntlet</title>',
            '<desc id="desc">Same-input synthetic compression comparison with evidence retention gates.</desc>',
            '<rect width="1200" height="630" fill="#09111F"/>',
            '<rect x="38" y="38" width="1124" height="554" rx="28" fill="#0D192B" stroke="#213552" stroke-width="2"/>',
            "<style>",
            "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#F4F8FC}",
            ".eyebrow{font-size:20px;font-weight:700;letter-spacing:2px;fill:#35D0BA}",
            ".headline{font-size:46px;font-weight:800}",
            ".subhead{font-size:22px;fill:#B7C4D6}",
            ".system{font-size:27px;font-weight:750}",
            ".version{font-size:17px;fill:#8FA2B9}",
            ".value{font-size:32px;font-weight:800}",
            ".detail{font-size:17px;fill:#AFC0D3}",
            ".footer{font-size:16px;fill:#8FA2B9}",
            "</style>",
            '<text x="90" y="105" class="eyebrow">AGENT CONTEXT COMPRESSION GAUNTLET</text>',
            '<text x="90" y="165" class="headline">Same inputs. Same evidence. Fewer tokens.</text>',
            '<text x="90" y="207" class="subhead">Weighted token reduction · shared o200k_base tokenizer</text>',
            *rows,
            '<line x1="90" y1="520" x2="1110" y2="520" stroke="#213552" stroke-width="2"/>',
            f'<text x="90" y="557" class="footer">{len(report["fixtures"])} deterministic agent-tool fixtures · '
            f'{report["protocol"]["runs"]} measured runs after {report["protocol"]["warmups"]} warm-ups</text>',
            '<text x="90" y="582" class="footer">Synthetic, no-model result — not downstream answer quality or ML superiority · raw artifacts committed</text>',
            "</svg>",
            "",
        ]
    )


def _adapter_payload(args: argparse.Namespace, scenarios: list[Scenario]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "budget_tokens": args.budget,
        "runs": args.runs,
        "warmups": args.warmups,
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }


def _run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    scenarios = build_scenarios()
    payload = _adapter_payload(args, scenarios)
    commands = [_adapter_command(sys.executable, "entroly")]
    if args.headroom_python:
        commands.append(_adapter_command(str(Path(args.headroom_python).resolve()), "headroom"))

    adapters = [
        _invoke_adapter(command, payload, root=root, timeout=args.timeout)
        for command in commands
    ]
    report = analyze(
        scenarios=scenarios,
        adapters=adapters,
        protocol={
            "model": args.model,
            "budget_tokens": args.budget,
            "runs": args.runs,
            "warmups": args.warmups,
        },
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    markdown = render_markdown(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(render_svg(report), encoding="utf-8")
    print(markdown, end="")
    return 0 if report["headline_eligible"] or not args.require_comparator else 1


def _adapter(args: argparse.Namespace) -> int:
    payload = json.load(sys.stdin)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    print(json.dumps(run_adapter(args.system, payload), sort_keys=True))
    return 0


def _render(args: argparse.Namespace) -> int:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rendered = render_markdown(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(render_svg(payload), encoding="utf-8")
    print(rendered, end="")
    return 0


def _verify(args: argparse.Namespace) -> int:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(payload)
    print(
        f"VERIFIED {args.input}: {len(payload['fixtures'])} fixtures, "
        f"{len(payload['participants'])} participants, winner={payload['suite_winner']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the same fixtures through isolated adapters")
    run.add_argument("--headroom-python")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    run.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    run.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--output", type=Path)
    run.add_argument("--markdown", type=Path)
    run.add_argument("--svg", type=Path)
    run.add_argument("--require-comparator", action="store_true")
    run.set_defaults(func=_run)

    adapter = subparsers.add_parser("adapter", help=argparse.SUPPRESS)
    adapter.add_argument("--system", choices=("entroly", "headroom"), required=True)
    adapter.set_defaults(func=_adapter)

    render = subparsers.add_parser("render", help="Render Markdown from a JSON result")
    render.add_argument("input", type=Path)
    render.add_argument("--output", type=Path)
    render.add_argument("--svg", type=Path)
    render.set_defaults(func=_render)

    verify = subparsers.add_parser("verify", help="Verify hashes, evidence, tokens, and aggregates")
    verify.add_argument("input", type=Path)
    verify.set_defaults(func=_verify)

    args = parser.parse_args()
    if getattr(args, "budget", 1) < 1:
        parser.error("--budget must be positive")
    if getattr(args, "runs", 1) < 1:
        parser.error("--runs must be positive")
    if getattr(args, "warmups", 0) < 0:
        parser.error("--warmups must be non-negative")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
