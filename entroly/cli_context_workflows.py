"""Production CLI workflows for evidence operations.

All workflows are local-first, preserve explicit claim boundaries, and write
machine-readable receipts atomically.  They do not infer answer quality from a
process exit code or infer provider billing from Entroly token estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_STATE_DIR_OVERRIDE: Path | None = None
_WRAP_AGENTS: Mapping[str, dict[str, Any]] = {}


def _unconfigured_agent_names() -> str:
    return "none configured"


def _unconfigured_start_proxy(_port: int) -> bool:
    return False


def _unconfigured_wrap_env(_spec: dict[str, Any], _port: int) -> dict[str, str]:
    return {}


_WRAP_AGENT_NAMES: Callable[[], str] = _unconfigured_agent_names
_START_PROXY: Callable[[int], bool] = _unconfigured_start_proxy
_RESOLVED_WRAP_ENV: Callable[[dict[str, Any], int], dict[str, str]] = _unconfigured_wrap_env


class _Colors:
    _enabled = "NO_COLOR" not in os.environ
    BOLD = "\033[1m" if _enabled else ""
    CYAN = "\033[38;5;45m" if _enabled else ""
    YELLOW = "\033[38;5;220m" if _enabled else ""
    RESET = "\033[0m" if _enabled else ""


def configure_cli_runtime(
    *,
    state_dir: Path,
    wrap_agents: Mapping[str, dict[str, Any]],
    wrap_agent_names: Callable[[], str],
    start_proxy: Callable[[int], bool],
    resolved_wrap_env: Callable[[dict[str, Any], int], dict[str, str]],
) -> None:
    """Bind workflows to the CLI runtime without introducing an import cycle."""
    global _STATE_DIR_OVERRIDE, _WRAP_AGENTS, _WRAP_AGENT_NAMES, _START_PROXY, _RESOLVED_WRAP_ENV
    _STATE_DIR_OVERRIDE = Path(state_dir)
    _WRAP_AGENTS = wrap_agents
    _WRAP_AGENT_NAMES = wrap_agent_names
    _START_PROXY = start_proxy
    _RESOLVED_WRAP_ENV = resolved_wrap_env


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _state_dir() -> Path:
    if _STATE_DIR_OVERRIDE is not None:
        return _STATE_DIR_OVERRIDE
    explicit = os.environ.get("ENTROLY_DIR")
    return Path(explicit).expanduser() if explicit else Path.home() / ".entroly"


def _default_recovery_store_path() -> str:
    explicit = os.environ.get("ENTROLY_DIR")
    if explicit:
        return str(Path(explicit).expanduser() / "recovery.json")
    from .config import _project_checkpoint_dir

    return str(_project_checkpoint_dir() / "recovery.json")


def _stats(port: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=3) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise RuntimeError("proxy stats did not return an object")
    return payload


def _delta(after: dict[str, Any], before: dict[str, Any], *keys: str) -> int:
    def read(value: dict[str, Any]) -> int:
        current: Any = value
        for key in keys:
            current = current.get(key, {}) if isinstance(current, dict) else {}
        try:
            return int(current)
        except (TypeError, ValueError):
            return 0

    return max(0, read(after) - read(before))


def _set_bypass(port: int, enabled: bool) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/bypass",
        data=json.dumps({"enabled": enabled}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.loads(response.read())
    if bool(payload.get("bypass")) != enabled:
        raise RuntimeError("proxy did not enter the requested bypass state")


def cmd_history(args: Any) -> int:
    from .history_audit import audit_histories, custom_roots

    explicit = list(getattr(args, "history_root", None) or ())
    report = audit_histories(
        custom_roots(explicit) if explicit else None,
        max_files=max(1, int(getattr(args, "max_files", 200))),
        max_bytes=max(1, int(getattr(args, "max_bytes", 64 * 1024 * 1024))),
        max_file_bytes=max(1, int(getattr(args, "max_file_bytes", 8 * 1024 * 1024))),
    )
    if getattr(args, "json_output", False):
        print(json.dumps(report, indent=2))
        return 0

    C = _Colors

    scope = report["scope"]
    known = report["provider_reported"]["known_semantics"]
    estimate = report["structural_estimate"]
    print(f"\n{C.CYAN}{C.BOLD}  Entroly Evidence Audit — Local Agent History{C.RESET}\n")
    print(
        f"  Read {scope['files_read']:,} files / {scope['records_read']:,} records "
        f"({scope['bytes_read']:,} bytes)."
    )
    print(f"  {C.BOLD}Adapter-interpreted provider/session usage:{C.RESET}")
    print(f"    Input tokens:       {known['input_tokens']:,}")
    print(f"    Cache-read tokens:  {known['cache_read_tokens']:,}")
    print(f"    Output tokens:      {known['output_tokens']:,}")
    print(f"  {C.BOLD}Largest structural pressure (estimates):{C.RESET}")
    for sink in estimate["sinks"][:5]:
        print(
            f"    {sink['category']:<22} {sink['estimated_tokens']:>10,} tokens "
            f"({sink['share_pct']:>5.1f}%)"
        )
    if report["recommendations"]:
        print(f"  {C.BOLD}Reversible experiments to consider:{C.RESET}")
        for recommendation in report["recommendations"]:
            print(f"    {recommendation['id']}: {recommendation['proposed_action']}")
    print(
        f"\n  {C.YELLOW}Boundary:{C.RESET} structural counts are estimates; unknown usage "
        "semantics are excluded from comparable totals. Nothing was changed.\n"
    )
    return 0


def _trial_command(args: Any) -> list[str]:
    command = list(getattr(args, "agent_command", None) or ())
    if command and command[0] == "--":
        command = command[1:]
    return command


def _command_digest(command: Iterable[str]) -> str:
    payload = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def _load_evaluation(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation must be a JSON object")
    required = {"task_success", "evidence_retained", "evaluator"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"evaluation is missing: {', '.join(missing)}")
    if not isinstance(value["task_success"], bool) or not isinstance(value["evidence_retained"], bool):
        raise ValueError("evaluation task_success and evidence_retained must be booleans")
    evaluator = str(value["evaluator"]).strip()
    if not evaluator or len(evaluator) > 160:
        raise ValueError("evaluation evaluator must be 1-160 characters")
    return {
        "task_success": value["task_success"],
        "evidence_retained": value["evidence_retained"],
        "evaluator": evaluator,
        "artifact_sha256": str(value.get("artifact_sha256") or "") or None,
    }


def _experiment_dir(experiment: str) -> Path:
    if not _EXPERIMENT_ID.fullmatch(experiment):
        raise ValueError("experiment id must be 1-64 letters, numbers, dot, underscore, or hyphen")
    return _state_dir() / "experiments" / experiment


def _trial_report(experiment: str) -> dict[str, Any]:
    directory = _experiment_dir(experiment)
    receipts: list[dict[str, Any]] = []
    ignored_receipts = 0
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        valid = (
            isinstance(value, dict)
            and value.get("schema_version") == "entroly.trial-run.v2"
            and value.get("arm") in {"baseline", "optimized"}
            and isinstance(value.get("traffic"), dict)
            and isinstance(value.get("usage"), dict)
            and isinstance(value.get("quality"), dict)
            and isinstance(value.get("economics"), dict)
        )
        if valid:
            try:
                int(value["usage"]["provider_reported_active_input_tokens"])
                cost_value = value["economics"].get("cost_usd")
                if cost_value is not None and (
                    isinstance(cost_value, bool) or not math.isfinite(float(cost_value))
                ):
                    valid = False
                if value["traffic"].get("evidence_gate") not in {"passed", "failed"}:
                    valid = False
                task_success = value["quality"].get("task_success")
                if task_success is not None and not isinstance(task_success, bool):
                    valid = False
                evidence_retained = value["quality"].get("evidence_retained")
                if evidence_retained is not None and not isinstance(evidence_retained, bool):
                    valid = False
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
        if valid:
            receipts.append(value)
        else:
            ignored_receipts += 1
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "optimized"):
        rows = [row for row in receipts if row.get("arm") == arm]
        provider_input = sum(int(row["usage"]["provider_reported_active_input_tokens"]) for row in rows)
        cost = sum(float(row["economics"]["cost_usd"] or 0.0) for row in rows)
        task_successes = sum(bool(row["quality"]["task_success"]) for row in rows)
        evidence_successes = sum(
            bool(row["quality"]["task_success"] is True and row["quality"]["evidence_retained"] is True)
            for row in rows
        )
        arms[arm] = {
            "runs": len(rows),
            "traffic_gates_passed": sum(row["traffic"]["evidence_gate"] == "passed" for row in rows),
            "task_successes": task_successes,
            "evidence_supported_successes": evidence_successes,
            "provider_reported_active_input_tokens": provider_input,
            "cost_usd": round(cost, 6) if cost else None,
            "cost_per_evidence_supported_success_usd": (
                round(cost / evidence_successes, 6) if cost and evidence_successes else None
            ),
        }
    command_digests = {row.get("command_sha256") for row in receipts}
    matched_command = (
        len(command_digests) == 1
        and all(
            isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71
            for digest in command_digests
        )
    )
    comparable = (
        matched_command
        and arms["baseline"]["runs"] >= 1
        and arms["baseline"]["runs"] == arms["optimized"]["runs"]
        and all(row["traffic"]["evidence_gate"] == "passed" for row in receipts)
    )
    enough_for_directional = comparable and arms["baseline"]["runs"] >= 3
    return {
        "schema_version": "entroly.trial-report.v2",
        "experiment": experiment,
        "receipts": {"accepted": len(receipts), "ignored_invalid": ignored_receipts},
        "arms": arms,
        "comparison": {
            "matched_command": matched_command,
            "balanced_arms": arms["baseline"]["runs"] == arms["optimized"]["runs"],
            "status": "directional" if enough_for_directional else "insufficient-evidence",
            "provider_input_token_difference": (
                arms["baseline"]["provider_reported_active_input_tokens"]
                - arms["optimized"]["provider_reported_active_input_tokens"]
                if comparable else None
            ),
            "claim_boundary": (
                "Three matched runs permit a directional operational comparison only. "
                "No statistical significance, causality, or general savings claim is inferred."
            ),
        },
    }


def cmd_trial(args: Any) -> int:
    """Record one explicit baseline/optimized arm or report an experiment."""
    report_id = getattr(args, "report", None)
    if report_id:
        try:
            report = _trial_report(report_id)
        except ValueError as exc:
            print(f"  {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2))
        return 0

    from .response_contract import environment_contract

    command = _trial_command(args)
    if not command:
        print("  Usage: entroly trial --experiment ID --arm baseline|optimized -- <agent> [args...]", file=sys.stderr)
        return 2
    experiment = str(getattr(args, "experiment", None) or "").strip()
    arm = str(getattr(args, "arm", None) or "").strip()
    if arm not in {"baseline", "optimized"}:
        print("  --arm must be baseline or optimized", file=sys.stderr)
        return 2
    try:
        directory = _experiment_dir(experiment)
        evaluation = _load_evaluation(getattr(args, "evaluation", None))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"  Invalid trial input: {exc}", file=sys.stderr)
        return 2

    executable_name = Path(command[0]).stem.lower()
    spec = _WRAP_AGENTS.get(executable_name)
    if not spec or spec.get("kind") != "cli":
        print(f"  Trial requires a supported CLI agent ({_WRAP_AGENT_NAMES()}); got {command[0]!r}.", file=sys.stderr)
        return 2
    executable = shutil.which(command[0])
    if executable is None:
        print(f"  Agent executable not found: {command[0]}", file=sys.stderr)
        return 127

    port = int(getattr(args, "port", None) or 9377)
    if not _START_PROXY(port):
        return 1
    before = _stats(port)
    previous_bypass = bool(before.get("bypass_mode", False))
    env = os.environ.copy()
    env.update(_RESOLVED_WRAP_ENV(spec, port))
    env.update(environment_contract())
    started = time.monotonic()
    try:
        _set_bypass(port, arm == "baseline")
        completed = subprocess.run(
            [executable, *command[1:]],
            env=env,
            check=False,
            stdout=sys.stderr if getattr(args, "json_output", False) else None,
        )
    finally:
        try:
            _set_bypass(port, previous_bypass)
        except Exception:
            pass
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    after = _stats(port)

    requests = _delta(after, before, "requests_total")
    provider_requests = _delta(after, before, "usage_accounting", "live", "requests")
    uncached = _delta(after, before, "usage_accounting", "live", "uncached_input_tokens")
    cache_read = _delta(after, before, "usage_accounting", "live", "cache_read_tokens")
    cache_write = _delta(after, before, "usage_accounting", "live", "cache_write_tokens")
    output = _delta(after, before, "usage_accounting", "live", "output_tokens")
    raw_estimate = _delta(after, before, "tokens", "original_total")
    active_estimate = _delta(after, before, "tokens", "optimized_total")
    cost_micro = _delta(after, before, "usage_accounting", "ledger", "cost_micro_usd")
    traffic_gate = requests > 0 and provider_requests > 0
    task_success: bool | None = evaluation["task_success"] if evaluation else None
    evidence_retained: bool | None = evaluation["evidence_retained"] if evaluation else None
    timestamp = time.time_ns()
    receipt = {
        "schema_version": "entroly.trial-run.v2",
        "experiment": experiment,
        "arm": arm,
        "agent": executable_name,
        "command_sha256": _command_digest([Path(command[0]).name, *command[1:]]),
        "task": {"process_exit_code": completed.returncode, "latency_ms": latency_ms},
        "traffic": {
            "proxy_requests": requests,
            "provider_usage_records": provider_requests,
            "evidence_gate": "passed" if traffic_gate else "failed",
        },
        "usage": {
            "provider_reported_active_input_tokens": uncached + cache_read + cache_write,
            "provider_reported": {
                "uncached_input_tokens": uncached,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "output_tokens": output,
            },
            "local_original_counter_estimate": raw_estimate,
            "local_selected_counter_estimate": active_estimate,
        },
        "quality": {
            "task_success": task_success,
            "evidence_retained": evidence_retained,
            "evaluation": evaluation,
            "process_success": completed.returncode == 0,
        },
        "economics": {
            "cost_usd": round(cost_micro / 1_000_000, 6) if cost_micro else None,
            "provenance": "provider usage plus configured pricing ledger" if cost_micro else "unavailable",
        },
        "claim_boundary": (
            "Process exit is not task quality. A comparison requires matched commands, "
            "balanced arms, traffic gates, and an external evaluation artifact."
        ),
    }
    receipt_path = Path(getattr(args, "receipt", None) or (directory / f"{timestamp}-{arm}.json"))
    _atomic_json(receipt_path, receipt)
    if getattr(args, "json_output", False):
        print(json.dumps(receipt, indent=2))
    else:
        print(f"\n  Trial arm:       {arm}")
        print(f"  Process exit:    {completed.returncode}")
        print(f"  Traffic gate:    {'passed' if traffic_gate else 'FAILED'}")
        print(f"  Provider input:  {uncached + cache_read + cache_write:,} tokens")
        print(f"  Evaluation:      {'attached' if evaluation else 'not attached'}")
        print(f"  Receipt:         {receipt_path}\n")
    return completed.returncode if traffic_gate else 3


def _compress_stream(text: str, *, source_id: str, budget: int, store_path: str) -> tuple[str, dict[str, Any]]:
    from .codec import RecoveryStore, estimate_tokens
    from .codecs_builtin import ShellCodec

    before = estimate_tokens(text) if text else 0
    if not text:
        return "", {"tokens_before": 0, "tokens_after": 0, "recovery_digest": None, "mode": "empty"}
    store = RecoveryStore(store_path)
    reps = ShellCodec(store).representations(text, source_id=source_id, budget=budget, tool_name=source_id)
    usable = [rep for rep in reps if rep.recovery is not None or rep.text == text]
    chosen = min(usable or reps, key=lambda rep: rep.token_cost)
    recovery = chosen.recovery
    if recovery is not None:
        try:
            if store.recover(recovery) != text:
                raise ValueError("recovery mismatch")
        except (KeyError, ValueError):
            chosen = reps[0]
            recovery = None
    return chosen.text, {
        "tokens_before": before,
        "tokens_after": chosen.token_cost,
        "recovery_digest": recovery.digest if recovery else None,
        "mode": "compressed" if recovery else "passthrough",
        "protected_evidence": list(chosen.protected_evidence),
        "source_sha256": chosen.source_sha256,
    }


def _write_bytes(stream: Any, path: Path) -> None:
    with path.open("rb") as handle:
        shutil.copyfileobj(handle, stream)
    stream.flush()


def cmd_shrink(args: Any) -> int:
    """Run a command through a bounded, recoverable output envelope."""
    command = list(getattr(args, "command_args", None) or ())
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("  Usage: entroly shrink [--budget 1200] -- <command> [args...]", file=sys.stderr)
        return 2
    executable = shutil.which(command[0])
    if executable is None:
        print(f"  Command not found: {command[0]}", file=sys.stderr)
        return 127
    store_path = getattr(args, "store_path", None) or _default_recovery_store_path()
    budget = max(64, int(getattr(args, "budget", 1200)))
    max_bytes = max(1024, int(getattr(args, "max_bytes", 64 * 1024 * 1024)))
    with tempfile.TemporaryDirectory(prefix="entroly-command-") as temporary:
        stdout_path = Path(temporary) / "stdout.bin"
        stderr_path = Path(temporary) / "stderr.bin"
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            completed = subprocess.run(
                [executable, *command[1:]],
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        receipts: dict[str, dict[str, Any]] = {}
        for name, path, stream in (
            ("stdout", stdout_path, sys.stdout.buffer),
            ("stderr", stderr_path, sys.stderr.buffer),
        ):
            size = path.stat().st_size
            if size > max_bytes:
                _write_bytes(stream, path)
                receipts[name] = {
                    "mode": "passthrough-oversize",
                    "bytes": size,
                    "max_bytes": max_bytes,
                    "recovery_digest": None,
                }
                continue
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                stream.write(raw)
                stream.flush()
                receipts[name] = {
                    "mode": "passthrough-non-utf8",
                    "bytes": size,
                    "recovery_digest": None,
                }
                continue
            compact, stream_receipt = _compress_stream(
                text,
                source_id=f"{Path(command[0]).name}:{name}",
                budget=budget,
                store_path=store_path,
            )
            stream.write(compact.encode("utf-8", "surrogateescape"))
            stream.flush()
            receipts[name] = stream_receipt

    receipt = {
        "schema_version": "entroly.command-envelope.v2",
        "command": {"executable": Path(command[0]).name, "argv_sha256": _command_digest(command)},
        "exit_code": completed.returncode,
        "streams": receipts,
        "recovery_store": store_path,
        "claim_boundary": "Token counts are local estimates; the command exit code is preserved.",
    }
    receipt_path = Path(
        getattr(args, "receipt", None)
        or (_state_dir() / "command-receipts" / f"{time.time_ns()}.json")
    )
    _atomic_json(receipt_path, receipt)
    print("\n[Entroly command envelope]", file=sys.stderr)
    for name in ("stdout", "stderr"):
        row = receipts[name]
        if "tokens_before" in row:
            print(f"  {name}: {row['tokens_before']} -> {row['tokens_after']} tokens; mode={row['mode']}", file=sys.stderr)
        else:
            print(f"  {name}: {row['bytes']} bytes; mode={row['mode']}", file=sys.stderr)
        if row.get("recovery_digest"):
            print(f"    exact recovery: entroly recover {row['recovery_digest']}", file=sys.stderr)
    print(f"  receipt: {receipt_path}", file=sys.stderr)
    return completed.returncode


def cmd_browser(args: Any) -> int:
    from .browser_context import capture_accessibility_snapshot, compress_accessibility_snapshot, query_fingerprint
    from .codec import RecoveryStore

    snapshot_path = getattr(args, "snapshot", None)
    if snapshot_path:
        path = Path(snapshot_path)
        max_bytes = max(1024, int(getattr(args, "max_bytes", 16 * 1024 * 1024)))
        if path.stat().st_size > max_bytes:
            print(f"  Snapshot exceeds --max-bytes ({max_bytes}).", file=sys.stderr)
            return 2
        snapshot = path.read_text(encoding="utf-8", errors="replace")
        source_id = "local-browser-snapshot"
    else:
        url = getattr(args, "url", None)
        if not url:
            print("  Provide a URL or --snapshot PATH.", file=sys.stderr)
            return 2
        try:
            snapshot = capture_accessibility_snapshot(
                url,
                timeout_ms=int(args.timeout * 1000),
                allow_private_network=bool(getattr(args, "allow_private_network", False)),
                max_snapshot_bytes=max(
                    1024, int(getattr(args, "max_bytes", 16 * 1024 * 1024))
                ),
            )
        except Exception as exc:
            print(f"  Browser capture failed; no context was altered: {exc}", file=sys.stderr)
            return 1
        source_id = "rendered-page"
    store_path = getattr(args, "store_path", None) or _default_recovery_store_path()
    query = getattr(args, "query", "") or ""
    result = compress_accessibility_snapshot(
        snapshot,
        query=query,
        budget=max(64, int(getattr(args, "budget", 2000))),
        store=RecoveryStore(store_path),
        source_id=source_id,
    )
    receipt = result.receipt()
    receipt["query_fingerprint"] = query_fingerprint(query)
    receipt["recovery_store"] = store_path
    if getattr(args, "receipt", None):
        _atomic_json(Path(args.receipt), receipt)
    if getattr(args, "json_output", False):
        print(json.dumps({"context": result.text, "receipt": receipt}, indent=2))
    else:
        sys.stdout.write(result.text)
        if result.text and not result.text.endswith("\n"):
            sys.stdout.write("\n")
        print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
    return 0


def cmd_response(args: Any) -> int:
    from .response_contract import CONTRACTS, load_contract, set_contract

    action = getattr(args, "response_action", None)
    scope = getattr(args, "scope", "project")
    if action == "list":
        payload = {name: value["description"] for name, value in CONTRACTS.items()}
    elif action == "show":
        try:
            payload = load_contract(scope)
        except ValueError as exc:
            print(f"  {exc}", file=sys.stderr)
            return 1
    elif action == "set":
        try:
            payload = set_contract(args.name, scope=scope)
        except (OSError, ValueError) as exc:
            print(f"  Could not set response contract: {exc}", file=sys.stderr)
            return 1
    elif action == "disable":
        try:
            payload = set_contract("off", scope=scope)
        except (OSError, ValueError) as exc:
            print(f"  Could not disable response contract: {exc}", file=sys.stderr)
            return 1
    else:
        print("  Usage: entroly response {list|show|set|disable}", file=sys.stderr)
        return 2
    if getattr(args, "json_output", False) or action in {"list", "show"}:
        print(json.dumps(payload, indent=2))
    else:
        print(f"  Response contract {payload['action']}: {payload['name']} ({payload['scope']})")
        print(f"  Reversible: yes; receipt digest: {payload['new_digest']}")
        if payload.get("backup"):
            print(f"  Backup: {payload['backup']}")
        print("  Boundary: this is an instruction contract, not a measured savings claim.")
    return 0


__all__ = ["cmd_browser", "cmd_history", "cmd_response", "cmd_shrink", "cmd_trial"]
