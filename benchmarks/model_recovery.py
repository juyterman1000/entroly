"""Auditable end-to-end model-triggered recovery benchmark.

The compressor sees an initial question. A different future question is only
revealed after compression. The model can answer from active context or return
``RETRIEVE``. Only that exact response invokes the participant's persistent
recovery store and permits one retry. Errors and wrong answers remain in the
matrix as zero-score outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmarks.compression_frontier import _ollama_identity, _ollama_post

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmarks" / "model_recovery_protocol_v7.json"
SCHEMA_VERSION = "entroly.model-recovery.v1"
_SUBPROCESS_ENV_ALLOWLIST = (
    "ALL_PROXY",
    "APPDATA",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
)
_ANSWER_RE = re.compile(r"[^a-z0-9-]+")
_PROMPT_TEMPLATE = """You are answering an audit question from the supplied context.
Return only the exact recovery code, with no prose or punctuation.
If the exact code is not present in the context, return exactly RETRIEVE.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _canonical_answer(value: str) -> str:
    return _ANSWER_RE.sub("", value.casefold().strip())


def _count_tokens(value: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(value))
    except ImportError:
        return max(1, (len(value) + 3) // 4)


def _limitations(model: str) -> list[str]:
    return [
        (
            f"Local {model} short-answer behavior is not evidence about "
            "hosted frontier models."
        ),
        "Synthetic JSON audit logs isolate query shift and recovery; they do not represent every agent workload.",
        "Headroom recovery composes public compress() with its persistent CompressionStore contract; MCP transport latency is excluded.",
        "A single exact RETRIEVE token is required; wrong or verbose first responses do not receive oracle recovery.",
        "No aggregate product score or universal superiority claim is allowed.",
    ]


def _messages(query: str, content: str, fixture_id: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{fixture_id}",
                    "type": "function",
                    "function": {"name": "audit_log_query", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{fixture_id}",
            "content": content,
        },
    ]


def build_fixtures(protocol: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    config = protocol["phases"][phase]
    seed = int(config["seed"])
    trials = int(config["trials"])
    rows = int(protocol["fixture_rows"])
    ordinary_chars = int(protocol["ordinary_payload_chars"])
    future_chars = int(protocol["future_payload_chars"])
    fixtures: list[dict[str, Any]] = []
    for trial in range(trials):
        rng = random.Random(seed + trial * 1009)
        fixture_id = f"{phase}-{trial:03d}"
        primary_index, future_index = rng.sample(range(12, rows - 12), 2)
        primary_id = f"INC-{seed % 10000:04d}-{trial:03d}"
        audit_case = f"CASE-{seed % 10000:04d}-{trial:03d}"
        expected = f"RCV-{rng.randrange(10**8, 10**9):09d}"
        records: list[dict[str, Any]] = []
        for index in range(rows):
            record: dict[str, Any] = {
                "audit_case": f"CASE-ORD-{index:04d}",
                "latency_ms": 20 + (index * 17 + trial) % 83,
                "message": "routine audit event completed",
                "payload": (f"trace-{index:04d}-" + "x" * ordinary_chars)[
                    :ordinary_chars
                ],
                "service": ("auth", "billing", "search")[index % 3],
                "status": "ok",
            }
            if index == primary_index:
                record.update(
                    {
                        "incident_id": primary_id,
                        "latency_ms": 9911 + trial,
                        "message": "initial refresh boundary incident",
                        "service": "auth",
                        "status": "failed",
                    }
                )
            if index == future_index:
                padding = "future-audit-payload-" + "q" * future_chars
                record.update(
                    {
                        "audit_case": audit_case,
                        "message": "archived compliance recovery record",
                        "payload": f"{padding[: future_chars // 2]} {expected} {padding[future_chars // 2 :]}",
                        "recovery_code": expected,
                    }
                )
            records.append(record)
        content = json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True)
        fixture = {
            "fixture_id": fixture_id,
            "compression_query": (
                f"Find initial auth incident {primary_id}, its latency, and failure status."
            ),
            "future_question": f"What recovery code belongs to audit case {audit_case}?",
            "expected_answer": expected,
            "content": content,
            "content_sha256": _sha256(content),
            "input_tokens": _count_tokens(content),
            "primary_index": primary_index,
            "future_index": future_index,
        }
        fixtures.append(fixture)
    return fixtures


def _participant_version(system: str) -> str:
    if system == "entroly":
        from entroly import __version__

        return str(__version__)
    return importlib.metadata.version("headroom-ai")


def _compress_entroly(payload: dict[str, Any]) -> dict[str, Any]:
    from entroly.compression_proxy import compress_proxy_payload
    from entroly.compression_retrieval_store import CompressionRetrievalStore

    fixture = payload["fixture"]
    store_path = Path(payload["store_path"])
    store = CompressionRetrievalStore(store_path)
    result = compress_proxy_payload(
        {
            "model": payload["headroom_model"],
            "messages": _messages(
                fixture["compression_query"], fixture["content"], fixture["fixture_id"]
            ),
        },
        query=fixture["compression_query"],
        budget_tokens=int(payload["entroly_budget_tokens"]),
        include_receipt_header=False,
        retrieval_store=store,
    )
    active = str(result.body["messages"][-1]["content"])
    receipts = [
        value.get("retrieval", {})
        for value in result.receipt.receipts
        if isinstance(value, dict) and isinstance(value.get("retrieval"), dict)
    ]
    return {
        "active_context": active,
        "active_sha256": _sha256(active),
        "active_tokens": _count_tokens(active),
        "changed": bool(result.changed),
        "retrieval_handle": {
            "receipt_ids": [str(value.get("receipt_id", "")) for value in receipts]
        },
        "native_metrics": {
            "compressed_blocks": int(result.receipt.compressed_blocks),
            "receipt_count": len(result.receipt.receipts),
            "stored_span_count": sum(
                int(value.get("span_count", 0)) for value in receipts
            ),
        },
    }


def _compress_headroom(payload: dict[str, Any]) -> dict[str, Any]:
    from headroom import compress
    from headroom.cache.backends.sqlite import SQLiteBackend
    from headroom.cache.compression_store import (
        CompressionStore,
        clear_request_compression_store,
        set_request_compression_store,
    )

    fixture = payload["fixture"]
    store = CompressionStore(backend=SQLiteBackend(payload["store_path"]))
    set_request_compression_store(store)
    try:
        result = compress(
            json.loads(
                json.dumps(
                    _messages(
                        fixture["compression_query"],
                        fixture["content"],
                        fixture["fixture_id"],
                    )
                )
            ),
            model=payload["headroom_model"],
            protect_recent=0,
            savings_profile=payload["headroom_savings_profile"],
            target_ratio=float(payload["headroom_target_ratio"]),
            min_tokens_to_compress=0,
        )
        active = str(result.messages[-1]["content"])
        handle = store.store(
            original=fixture["content"],
            compressed=active,
            original_tokens=int(result.tokens_before),
            compressed_tokens=int(result.tokens_after),
            tool_name="model_recovery_benchmark",
            query_context=fixture["compression_query"],
            compression_strategy="public_compress_plus_mcp_store_contract",
            ttl=3600,
        )
    finally:
        clear_request_compression_store()
    return {
        "active_context": active,
        "active_sha256": _sha256(active),
        "active_tokens": _count_tokens(active),
        "changed": int(result.tokens_after) < int(result.tokens_before),
        "retrieval_handle": {"hash": handle},
        "native_metrics": {
            "tokens_before": int(result.tokens_before),
            "tokens_after": int(result.tokens_after),
            "transforms": list(result.transforms_applied),
        },
    }


def _retrieve_entroly(payload: dict[str, Any]) -> dict[str, Any]:
    from entroly.compression_retrieval_store import CompressionRetrievalStore

    store = CompressionRetrievalStore(payload["store_path"])
    spans = store.search_exact_excerpts(
        payload["query"],
        limit=int(payload["retrieval_limit"]),
        max_tokens_per_span=int(payload["recovery_excerpt_tokens"]),
        record_retrieval=True,
        retrieval_id=payload["retrieval_id"],
    )
    content = "\n\n".join(
        f"[Recovered Entroly span {span.span_id}]\n{span.content}" for span in spans
    )
    return {
        "status": "ok" if spans else "not_found",
        "content": content,
        "content_sha256": _sha256(content),
        "tokens": _count_tokens(content) if content else 0,
        "items": [
            {
                "receipt_id": span.receipt_id,
                "span_id": span.span_id,
                "content_sha256": _sha256(span.content),
            }
            for span in spans
        ],
    }


def _retrieve_headroom(payload: dict[str, Any]) -> dict[str, Any]:
    from headroom.cache.backends.sqlite import SQLiteBackend
    from headroom.cache.compression_store import CompressionStore

    store = CompressionStore(backend=SQLiteBackend(payload["store_path"]))
    entry = store.retrieve(payload["handle"]["hash"], query=payload["query"])
    if entry is None:
        return {
            "status": "not_found",
            "content": "",
            "content_sha256": _sha256(""),
            "tokens": 0,
            "items": [],
        }
    content = entry.original_content
    return {
        "status": "ok",
        "content": content,
        "content_sha256": _sha256(content),
        "tokens": _count_tokens(content),
        "items": [{"hash": entry.hash, "content_sha256": _sha256(content)}],
    }


def run_adapter(system: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    expected = payload["expected_version"]
    observed = _participant_version(system)
    if observed != expected:
        raise RuntimeError(
            f"{system} version mismatch: expected {expected}, observed {observed}"
        )
    if operation == "compress":
        result = (
            _compress_entroly(payload)
            if system == "entroly"
            else _compress_headroom(payload)
        )
    elif operation == "retrieve":
        result = (
            _retrieve_entroly(payload)
            if system == "entroly"
            else _retrieve_headroom(payload)
        )
    else:
        raise ValueError(f"unsupported adapter operation: {operation}")
    return {
        "system": system,
        "operation": operation,
        "version": observed,
        "python": platform.python_version(),
        **result,
    }


def _subprocess_env() -> dict[str, str]:
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


def _call_adapter(
    python: str,
    system: str,
    operation: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    process = subprocess.run(
        [
            python,
            "-m",
            "benchmarks.model_recovery",
            "adapter",
            "--system",
            system,
            "--operation",
            operation,
        ],
        cwd=ROOT,
        env=_subprocess_env(),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "no adapter output"
        raise RuntimeError(f"{system} {operation} adapter failed: {detail}")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{system} {operation} adapter returned invalid JSON: {process.stdout[:500]}"
        ) from error
    if process.stderr.strip():
        value["stderr"] = process.stderr.strip()
    return value


def _model_call(
    *,
    base_url: str,
    model: str,
    context: str,
    question: str,
    generation: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    started = time.perf_counter()
    error: str | None = None
    try:
        response = _ollama_post(
            base_url,
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": generation,
            },
            timeout,
        )
    except RuntimeError as caught:
        response = {}
        error = str(caught)
    if error is None and not bool(response.get("done")):
        error = "Ollama generation did not complete"
    prediction = str(response.get("response", "")).strip()
    return {
        "context_sha256": _sha256(context),
        "context_tokens": _count_tokens(context),
        "prompt_sha256": _sha256(prompt),
        "prediction": prediction,
        "prediction_sha256": _sha256(prediction),
        "canonical_prediction": _canonical_answer(prediction),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "eval_count": response.get("eval_count"),
        "done": bool(response.get("done")),
        "error": error,
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON checkpoint without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    protocol_sha256: str,
    phase: str,
    model_digest: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    contract = {
        "schema_version": "entroly.model-recovery-checkpoint.v1",
        "protocol_sha256": protocol_sha256,
        "phase": phase,
        "model_digest": model_digest,
    }
    if any(value.get(key) != expected for key, expected in contract.items()):
        raise ValueError(f"checkpoint contract mismatch: {path}")
    responses = value.get("responses")
    if not isinstance(responses, dict):
        raise ValueError(f"checkpoint responses are invalid: {path}")
    return responses


def _write_checkpoint(
    path: Path,
    *,
    protocol_sha256: str,
    phase: str,
    model_digest: str,
    responses: dict[str, dict[str, Any]],
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": "entroly.model-recovery-checkpoint.v1",
            "protocol_sha256": protocol_sha256,
            "phase": phase,
            "model_digest": model_digest,
            "responses": responses,
        },
    )


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    import math

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    systems: dict[str, Any] = {}
    for system in ("entroly", "headroom"):
        selected = [row for row in rows if row["system"] == system]
        systems[system] = {
            "trials": len(selected),
            "active_exact": sum(bool(row["active_exact"]) for row in selected),
            "retrieval_triggers": sum(
                bool(row["retrieval_triggered"]) for row in selected
            ),
            "retrieval_successes": sum(
                row.get("retrieval_status") == "ok" for row in selected
            ),
            "final_exact": sum(bool(row["final_exact"]) for row in selected),
            "active_exact_accuracy": round(
                sum(bool(row["active_exact"]) for row in selected) / len(selected), 6
            )
            if selected
            else 0.0,
            "final_exact_accuracy": round(
                sum(bool(row["final_exact"]) for row in selected) / len(selected), 6
            )
            if selected
            else 0.0,
            "mean_active_token_ratio": round(
                sum(row["active_tokens"] / row["input_tokens"] for row in selected)
                / len(selected),
                6,
            )
            if selected
            else 0.0,
            "mean_effective_token_ratio": round(
                sum(row["effective_tokens"] / row["input_tokens"] for row in selected)
                / len(selected),
                6,
            )
            if selected
            else 0.0,
            "errors": sum(bool(row["errors"]) for row in selected),
        }
    entroly_only = sum(
        bool(left["final_exact"]) and not bool(right["final_exact"])
        for left, right in zip(
            [row for row in rows if row["system"] == "entroly"],
            [row for row in rows if row["system"] == "headroom"],
        )
    )
    headroom_only = sum(
        bool(right["final_exact"]) and not bool(left["final_exact"])
        for left, right in zip(
            [row for row in rows if row["system"] == "entroly"],
            [row for row in rows if row["system"] == "headroom"],
        )
    )
    return {
        "trials": total // 2,
        "systems": systems,
        "paired": {
            "entroly_only_final_exact": entroly_only,
            "headroom_only_final_exact": headroom_only,
            "mcnemar_exact_p": _mcnemar_exact(entroly_only, headroom_only),
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    phase = args.phase
    fixtures = build_fixtures(protocol, phase)
    ollama = protocol["ollama"]
    base_url = args.ollama_base_url or ollama["base_url"]
    model = args.ollama_model or ollama["model"]
    timeout = float(args.timeout or ollama["timeout_seconds"])
    identity = _ollama_identity(base_url, model, timeout)
    generation = {
        key: int(ollama[key])
        for key in ("temperature", "seed", "num_predict", "num_ctx")
        if key in ollama
    }
    protocol_sha256 = _canonical_sha256(protocol)
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else Path(str(args.output) + ".checkpoint.json").resolve()
    )
    checkpoint_responses = _load_checkpoint(
        checkpoint_path,
        protocol_sha256=protocol_sha256,
        phase=phase,
        model_digest=str(identity["digest"]),
    )
    python_by_system = {"entroly": sys.executable, "headroom": args.headroom_python}
    participant_versions = {
        system: str(protocol["participants"][system]["version"])
        for system in ("entroly", "headroom")
    }
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="entroly-model-recovery-") as temp:
        temp_root = Path(temp)
        compression: dict[tuple[str, str], dict[str, Any]] = {}
        state_paths: dict[tuple[str, str], Path] = {}
        for fixture in fixtures:
            for system in ("entroly", "headroom"):
                suffix = ".json" if system == "entroly" else ".sqlite3"
                store_path = temp_root / f"{system}-{fixture['fixture_id']}{suffix}"
                state_paths[(system, fixture["fixture_id"])] = store_path
                payload = {
                    "fixture": fixture,
                    "store_path": str(store_path),
                    "expected_version": participant_versions[system],
                    "entroly_budget_tokens": protocol["entroly_budget_tokens"],
                    "headroom_model": protocol["headroom_model"],
                    "headroom_savings_profile": protocol["headroom_savings_profile"],
                    "headroom_target_ratio": protocol["headroom_target_ratio"],
                }
                try:
                    compression[(system, fixture["fixture_id"])] = _call_adapter(
                        python_by_system[system],
                        system,
                        "compress",
                        payload,
                        timeout=timeout,
                    )
                except Exception as error:  # failures remain explicit and in sample
                    compression[(system, fixture["fixture_id"])] = {"error": str(error)}

        model_order = [("raw", fixture["fixture_id"]) for fixture in fixtures]
        model_order += [
            (system, fixture["fixture_id"])
            for fixture in fixtures
            for system in ("entroly", "headroom")
        ]
        random.Random(int(protocol["phases"][phase]["seed"]) + 17).shuffle(model_order)
        fixture_by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}
        first_pass: dict[tuple[str, str], dict[str, Any]] = {}
        for index, (system, fixture_id) in enumerate(model_order, 1):
            fixture = fixture_by_id[fixture_id]
            context = (
                fixture["content"]
                if system == "raw"
                else str(compression[(system, fixture_id)].get("active_context", ""))
            )
            checkpoint_key = f"first:{system}:{fixture_id}"
            result = checkpoint_responses.get(checkpoint_key)
            if result is not None and result.get("context_sha256") != _sha256(context):
                raise ValueError(f"checkpoint context mismatch for {checkpoint_key}")
            if result is None:
                result = _model_call(
                    base_url=base_url,
                    model=model,
                    context=context,
                    question=fixture["future_question"],
                    generation=generation,
                    timeout=timeout,
                )
                checkpoint_responses[checkpoint_key] = result
                _write_checkpoint(
                    checkpoint_path,
                    protocol_sha256=protocol_sha256,
                    phase=phase,
                    model_digest=str(identity["digest"]),
                    responses=checkpoint_responses,
                )
            print(
                f"model {index}/{len(model_order)} {system}/{fixture_id}: "
                f"{result['canonical_prediction'] or result['error'] or 'empty'}",
                flush=True,
            )
            first_pass[(system, fixture_id)] = result

        for fixture in fixtures:
            raw = first_pass[("raw", fixture["fixture_id"])]
            raw["exact"] = raw["error"] is None and raw[
                "canonical_prediction"
            ] == _canonical_answer(fixture["expected_answer"])
            raw_rows.append({"fixture_id": fixture["fixture_id"], **raw})
            for system in ("entroly", "headroom"):
                compressed = compression[(system, fixture["fixture_id"])]
                active = first_pass[(system, fixture["fixture_id"])]
                errors = [
                    value
                    for value in (compressed.get("error"), active.get("error"))
                    if value
                ]
                expected = _canonical_answer(fixture["expected_answer"])
                active_exact = (
                    active["error"] is None
                    and active["canonical_prediction"] == expected
                )
                triggered = (
                    active["error"] is None
                    and active["canonical_prediction"] == "retrieve"
                )
                retrieval: dict[str, Any] | None = None
                retry: dict[str, Any] | None = None
                if triggered and not compressed.get("error"):
                    payload = {
                        "store_path": str(state_paths[(system, fixture["fixture_id"])]),
                        "expected_version": participant_versions[system],
                        "query": fixture["future_question"],
                        "handle": compressed.get("retrieval_handle", {}),
                        "retrieval_limit": protocol["entroly_retrieval_limit"],
                        "recovery_excerpt_tokens": protocol[
                            "entroly_recovery_excerpt_tokens"
                        ],
                        "retrieval_id": f"{phase}:{fixture['fixture_id']}:{system}:retry-1",
                    }
                    try:
                        retrieval = _call_adapter(
                            python_by_system[system],
                            system,
                            "retrieve",
                            payload,
                            timeout=timeout,
                        )
                    except Exception as error:
                        retrieval = {
                            "status": "error",
                            "content": "",
                            "tokens": 0,
                            "error": str(error),
                        }
                    if retrieval.get("status") == "ok":
                        recovered_context = (
                            str(compressed["active_context"])
                            + "\n\n[RECOVERED EXACT CONTEXT]\n"
                            + str(retrieval["content"])
                        )
                        retry_key = f"retry:{system}:{fixture['fixture_id']}"
                        retry = checkpoint_responses.get(retry_key)
                        if retry is not None and retry.get("context_sha256") != _sha256(
                            recovered_context
                        ):
                            raise ValueError(
                                f"checkpoint context mismatch for {retry_key}"
                            )
                        if retry is None:
                            retry = _model_call(
                                base_url=base_url,
                                model=model,
                                context=recovered_context,
                                question=fixture["future_question"],
                                generation=generation,
                                timeout=timeout,
                            )
                            checkpoint_responses[retry_key] = retry
                            _write_checkpoint(
                                checkpoint_path,
                                protocol_sha256=protocol_sha256,
                                phase=phase,
                                model_digest=str(identity["digest"]),
                                responses=checkpoint_responses,
                            )
                        print(
                            f"retry {system}/{fixture['fixture_id']}: "
                            f"{retry['canonical_prediction'] or retry['error'] or 'empty'}",
                            flush=True,
                        )
                        if retry.get("error"):
                            errors.append(str(retry["error"]))
                    elif retrieval.get("error"):
                        errors.append(str(retrieval["error"]))
                final = retry or active
                final_exact = (
                    final["error"] is None and final["canonical_prediction"] == expected
                )
                recovery_tokens = int(retrieval.get("tokens", 0)) if retrieval else 0
                active_tokens = int(compressed.get("active_tokens", 0))
                rows.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "system": system,
                        "input_tokens": fixture["input_tokens"],
                        "active_tokens": active_tokens,
                        "recovery_tokens": recovery_tokens,
                        "effective_tokens": active_tokens + recovery_tokens,
                        "active_contains_expected": fixture["expected_answer"]
                        in str(compressed.get("active_context", "")),
                        "changed": bool(compressed.get("changed", False)),
                        "compression": compressed,
                        "active_response": active,
                        "active_exact": active_exact,
                        "retrieval_triggered": triggered,
                        "retrieval_status": retrieval.get("status")
                        if retrieval
                        else None,
                        "retrieval": retrieval,
                        "retry_response": retry,
                        "final_exact": final_exact,
                        "errors": errors,
                    }
                )

    aggregates = _aggregate(rows)
    gates = {
        "complete_matrix": len(rows) == 2 * len(fixtures),
        "raw_exact_accuracy": round(
            sum(bool(row["exact"]) for row in raw_rows) / len(raw_rows), 6
        ),
        "no_errors": not any(row["errors"] for row in rows)
        and not any(row["error"] for row in raw_rows),
        "persistent_recovery_exact_bytes": all(
            row["retrieval"] is None
            or row["retrieval_status"] != "ok"
            or fixture_by_id[row["fixture_id"]]["expected_answer"]
            in str(row["retrieval"]["content"])
            for row in rows
        ),
    }
    gates["passed"] = (
        gates["complete_matrix"]
        and gates["raw_exact_accuracy"]
        == float(protocol["quality_gates"]["raw_exact_accuracy"])
        and gates["no_errors"]
        and gates["persistent_recovery_exact_bytes"]
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "model": identity,
            "base_url": base_url,
        },
        "generation": generation,
        "prompt_template_sha256": _sha256(_PROMPT_TEMPLATE),
        "fixtures": fixtures,
        "raw_rows": raw_rows,
        "rows": rows,
        "aggregates": aggregates,
        "quality_gates": gates,
        "limitations": _limitations(model),
    }
    report["payload_sha256"] = _canonical_sha256(report)
    return report


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    payload_hash = report.get("payload_sha256")
    unsigned = dict(report)
    unsigned.pop("payload_sha256", None)
    if payload_hash != _canonical_sha256(unsigned):
        raise ValueError("payload_sha256 mismatch")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or report.get(
        "protocol_sha256"
    ) != _canonical_sha256(protocol):
        raise ValueError("protocol hash mismatch")
    fixtures = report.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("missing fixtures")
    expected_fixtures = build_fixtures(protocol, str(report["phase"]))
    if fixtures != expected_fixtures:
        raise ValueError("fixture matrix does not match frozen generator")
    by_id = {fixture["fixture_id"]: fixture for fixture in fixtures}
    if len(by_id) != len(fixtures):
        raise ValueError("duplicate fixture ids")
    raw_rows = report.get("raw_rows")
    rows = report.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(fixtures):
        raise ValueError("raw matrix is incomplete")
    if not isinstance(rows, list) or len(rows) != len(fixtures) * 2:
        raise ValueError("participant matrix is incomplete")
    expected_pairs = {
        (system, fixture_id)
        for system in ("entroly", "headroom")
        for fixture_id in by_id
    }
    observed_pairs = {(str(row["system"]), str(row["fixture_id"])) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError("participant matrix pairs mismatch")
    environment = report.get("environment")
    if not isinstance(environment, dict) or not isinstance(
        environment.get("model"), dict
    ):
        raise ValueError("model environment is missing")
    protocol_model = str(protocol["ollama"]["model"])
    if environment["model"].get("name") != protocol_model:
        raise ValueError("model identity does not match frozen protocol")
    expected_generation = {
        key: protocol["ollama"][key]
        for key in ("temperature", "seed", "num_predict", "num_ctx")
    }
    if report.get("generation") != expected_generation:
        raise ValueError("generation settings do not match frozen protocol")
    if report.get("limitations") != _limitations(protocol_model):
        raise ValueError("model-scope limitations mismatch")
    if report.get("prompt_template_sha256") != _sha256(_PROMPT_TEMPLATE):
        raise ValueError("prompt template hash mismatch")

    def verify_response(
        response: dict[str, Any],
        *,
        context: str,
        question: str,
        label: str,
    ) -> None:
        prediction = str(response["prediction"])
        if response["context_sha256"] != _sha256(context):
            raise ValueError(f"{label} context hash mismatch")
        if int(response["context_tokens"]) != _count_tokens(context):
            raise ValueError(f"{label} context token mismatch")
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        if response["prompt_sha256"] != _sha256(prompt):
            raise ValueError(f"{label} prompt hash mismatch")
        if response["prediction_sha256"] != _sha256(prediction):
            raise ValueError(f"{label} prediction hash mismatch")
        if response["canonical_prediction"] != _canonical_answer(prediction):
            raise ValueError(f"{label} canonical prediction mismatch")

    for raw in raw_rows:
        fixture = by_id[str(raw["fixture_id"])]
        verify_response(
            raw,
            context=fixture["content"],
            question=fixture["future_question"],
            label="raw",
        )
        exact = raw["error"] is None and raw[
            "canonical_prediction"
        ] == _canonical_answer(fixture["expected_answer"])
        if bool(raw["exact"]) != exact:
            raise ValueError("raw score mismatch")
    for row in rows:
        fixture = by_id[str(row["fixture_id"])]
        active = row["active_response"]
        compression = row["compression"]
        active_context = str(compression.get("active_context", ""))
        if compression.get("system") != row["system"]:
            raise ValueError("compression system mismatch")
        expected_version = protocol["participants"][row["system"]]["version"]
        if compression.get("version") != expected_version:
            raise ValueError("participant version mismatch")
        if compression.get("active_sha256") != _sha256(active_context):
            raise ValueError("active context hash mismatch")
        observed_active_tokens = _count_tokens(active_context)
        if int(compression.get("active_tokens", -1)) != observed_active_tokens:
            raise ValueError("compression active token mismatch")
        if int(row["active_tokens"]) != observed_active_tokens:
            raise ValueError("row active token mismatch")
        if bool(row["active_contains_expected"]) != (
            fixture["expected_answer"] in active_context
        ):
            raise ValueError("active evidence-presence mismatch")
        verify_response(
            active,
            context=active_context,
            question=fixture["future_question"],
            label="active",
        )
        triggered = (
            active["error"] is None and active["canonical_prediction"] == "retrieve"
        )
        if bool(row["retrieval_triggered"]) != triggered:
            raise ValueError("retrieval trigger mismatch")
        if row["retrieval"] is not None and not triggered:
            raise ValueError("oracle retrieval detected")
        retrieval = row["retrieval"]
        retry = row["retry_response"]
        if retrieval is None:
            if retry is not None or int(row["recovery_tokens"]) != 0:
                raise ValueError("recovery data exists without retrieval")
            expected_retrieval_status = None
        else:
            content = str(retrieval.get("content", ""))
            if retrieval.get("system") != row["system"]:
                raise ValueError("retrieval system mismatch")
            if retrieval.get("version") != expected_version:
                raise ValueError("retrieval version mismatch")
            if retrieval.get("content_sha256") != _sha256(content):
                raise ValueError("retrieval content hash mismatch")
            observed_recovery_tokens = _count_tokens(content) if content else 0
            if int(retrieval.get("tokens", -1)) != observed_recovery_tokens:
                raise ValueError("retrieval token mismatch")
            if int(row["recovery_tokens"]) != observed_recovery_tokens:
                raise ValueError("row recovery token mismatch")
            expected_retrieval_status = retrieval.get("status")
            if expected_retrieval_status == "ok":
                if fixture["expected_answer"] not in content:
                    raise ValueError("recovery omitted expected source evidence")
                items = retrieval.get("items")
                if not isinstance(items, list) or not items:
                    raise ValueError("successful recovery is missing provenance items")
                if row["system"] == "entroly":
                    exact_chunks = re.findall(
                        r"\[Recovered Entroly span [^\]]+\]\n"
                        r"(.*?)(?=\n\n\[Recovered Entroly span |\Z)",
                        content,
                        flags=re.DOTALL,
                    )
                    if len(exact_chunks) != len(items):
                        raise ValueError("Entroly recovery provenance count mismatch")
                    for chunk, item in zip(exact_chunks, items, strict=True):
                        if chunk not in fixture["content"]:
                            raise ValueError("Entroly recovery bytes are not in source")
                        if item.get("content_sha256") != _sha256(chunk):
                            raise ValueError("Entroly recovery item hash mismatch")
                else:
                    if content != fixture["content"]:
                        raise ValueError("Headroom recovery is not the stored source")
                    if len(items) != 1 or items[0].get("content_sha256") != _sha256(
                        content
                    ):
                        raise ValueError("Headroom recovery item hash mismatch")
                if retry is None:
                    raise ValueError("successful retrieval is missing its retry")
                recovered_context = (
                    active_context + "\n\n[RECOVERED EXACT CONTEXT]\n" + content
                )
                verify_response(
                    retry,
                    context=recovered_context,
                    question=fixture["future_question"],
                    label="retry",
                )
            elif retry is not None:
                raise ValueError("retry exists after unsuccessful retrieval")
        if row["retrieval_status"] != expected_retrieval_status:
            raise ValueError("retrieval status mismatch")
        active_exact = active["error"] is None and active[
            "canonical_prediction"
        ] == _canonical_answer(fixture["expected_answer"])
        if bool(row["active_exact"]) != active_exact:
            raise ValueError("active score mismatch")
        final = row["retry_response"] or active
        expected_exact = final["error"] is None and final[
            "canonical_prediction"
        ] == _canonical_answer(fixture["expected_answer"])
        if bool(row["final_exact"]) != expected_exact:
            raise ValueError("final score mismatch")
        expected_effective = int(row["active_tokens"]) + int(row["recovery_tokens"])
        if int(row["effective_tokens"]) != expected_effective:
            raise ValueError("effective token accounting mismatch")
    if report.get("aggregates") != _aggregate(rows):
        raise ValueError("aggregate mismatch")
    raw_accuracy = sum(bool(row["exact"]) for row in raw_rows) / len(raw_rows)
    expected_gates = {
        "complete_matrix": True,
        "raw_exact_accuracy": round(raw_accuracy, 6),
        "no_errors": all(raw["error"] is None for raw in raw_rows)
        and all(not row["errors"] for row in rows),
        "persistent_recovery_exact_bytes": all(
            not row["retrieval_triggered"]
            or row["retrieval_status"] != "ok"
            or by_id[row["fixture_id"]]["expected_answer"]
            in str(row["retrieval"]["content"])
            for row in rows
        ),
    }
    expected_gates["passed"] = (
        expected_gates["complete_matrix"]
        and expected_gates["raw_exact_accuracy"]
        == float(protocol["quality_gates"]["raw_exact_accuracy"])
        and expected_gates["no_errors"]
        and expected_gates["persistent_recovery_exact_bytes"]
    )
    if report.get("quality_gates") != expected_gates:
        raise ValueError("quality gate mismatch")


def _adapter_main(args: argparse.Namespace) -> int:
    payload = json.load(sys.stdin)
    result = run_adapter(args.system, args.operation, payload)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapter = subparsers.add_parser("adapter")
    adapter.add_argument("--system", choices=("entroly", "headroom"), required=True)
    adapter.add_argument("--operation", choices=("compress", "retrieve"), required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    run.add_argument("--phase", choices=("development", "holdout"), required=True)
    run.add_argument("--headroom-python", required=True)
    run.add_argument("--ollama-model")
    run.add_argument("--ollama-base-url")
    run.add_argument("--timeout", type=float)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "adapter":
        return _adapter_main(args)
    if args.command == "run":
        report = run_benchmark(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(args.output, report)
        verify_report(report)
        print(
            f"VERIFIED {args.output}: phase={report['phase']} "
            f"raw={report['quality_gates']['raw_exact_accuracy']:.1%} "
            f"entroly={report['aggregates']['systems']['entroly']['final_exact_accuracy']:.1%} "
            f"headroom={report['aggregates']['systems']['headroom']['final_exact_accuracy']:.1%}"
        )
        if report["quality_gates"]["passed"]:
            checkpoint_path = (
                args.checkpoint
                if args.checkpoint
                else Path(str(args.output) + ".checkpoint.json")
            )
            checkpoint_path.unlink(missing_ok=True)
            return 0
        return 2
    report = json.loads(args.input.read_text(encoding="utf-8"))
    verify_report(report)
    print(f"VERIFIED {args.input}: payload_sha256={report['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
