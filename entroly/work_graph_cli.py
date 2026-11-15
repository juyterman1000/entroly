"""CLI orchestration for the shared Rust AI Work Graph."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from .work_graph import WorkGraphUnavailableError
from .work_graph_content_digest import enrich_worktree_content_digests
from .work_graph_recovery_ack import (
    RecoveryAcknowledgementRequired,
    acknowledge as recovery_acknowledge,
    arm as recovery_arm,
    recovery_token,
    require_acknowledged as recovery_require_acknowledged,
)
from .work_graph_repo import discover_repository_identity, discover_repository_observation
from .work_graph_resume import (
    attach_current_worktree_changes,
    recover_workstream,
    recovery_unknowns,
    workspace_commitment,
)
from .work_graph_store import (
    continuation_outstanding_refs,
    WorkGraphLockTimeout,
    WorkGraphStateError,
    WorkGraphStore,
    WorkGraphStoreError,
)

_MAX_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_EVIDENCE = 4096
_MAX_ID_CHARS = 512
_MAX_LABEL_CHARS = 8192


def _project_path(project: str) -> Path:
    return Path(project or ".").expanduser().resolve()


def _store_for_path(project: Path) -> WorkGraphStore:
    identity = discover_repository_identity(project)
    return WorkGraphStore(identity["repo_id"])


def _passive_observation(project: Path) -> dict[str, Any]:
    observation = discover_repository_observation(project)
    return enrich_worktree_content_digests(project, observation)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _add_field(lines: list[str], label: str, value: object) -> None:
    if value in (None, "", [], {}):
        return
    if isinstance(value, (list, tuple, set)):
        value = len(value)
    elif isinstance(value, dict):
        value = len(value)
    text = str(value)
    if len(text) > 160:
        text = text[:157] + "..."
    lines.append(f"  {label:<12} {text}")


def _coordination_conflicts(coordination: object) -> int | None:
    data = _mapping(coordination)
    for key in ("conflicts", "active_conflicts", "overlaps"):
        count = _count(data.get(key))
        if count is not None:
            return count
    return None


def _summary_lines(summary: object) -> list[str]:
    data = _mapping(summary)
    lines: list[str] = []
    metrics = (
        ("Events", ("event_count", "events")),
        ("Nodes", ("node_count", "nodes")),
        ("Edges", ("edge_count", "edges")),
        ("Tasks", ("task_count", "tasks")),
        ("Workstreams", ("workstream_count", "workstreams")),
        ("Evidence", ("evidence_count", "evidence")),
    )
    for label, keys in metrics:
        value = _first(data, *keys)
        count = _count(value)
        _add_field(lines, label, count if count is not None else value)
    return lines


def _human_lines(payload: dict[str, Any]) -> list[str]:
    lines = ["Entroly Work Graph"]

    if payload.get("kind") == "work_pending_handoffs":
        _add_field(lines, "Repository", payload.get("repo_id"))
        _add_field(lines, "Project", payload.get("project_root"))
        _add_field(lines, "Recipient", payload.get("to_agent"))
        handoffs = payload.get("handoffs", [])
        lines.append(f"  Handoffs     {len(handoffs)}")
        for item in handoffs:
            label = "current" if item.get("current") else "stale"
            lines.append(f"  {label:<12} {item.get('ref', '')}")

    elif payload.get("kind") == "work_record_review":
        review = _mapping(payload.get("review"))
        lines.append("  Status       Review recorded as an agent statement.")
        _add_field(lines, "Review", review.get("review_id"))
        _add_field(lines, "Verdict", review.get("verdict"))
        _add_field(lines, "Findings", review.get("findings"))

    elif payload.get("kind") == "work_reviews":
        reviews = payload.get("reviews", [])
        lines.append(f"  Reviews      {len(reviews)}")
        for item in reviews:
            label = "current" if item.get("current") else "stale"
            lines.append(f"  {label:<12} {item.get('review_id', '')}")

    elif "repo_id" in payload:
        _add_field(lines, "Repository", payload.get("repo_id"))
        _add_field(lines, "Project", payload.get("project_root"))
        lines.extend(_summary_lines(payload.get("summary")))
        unfinished = _count(payload.get("unfinished"))
        if unfinished is not None:
            _add_field(lines, "Unfinished", unfinished)
        conflicts = _coordination_conflicts(payload.get("coordination"))
        if conflicts is not None:
            _add_field(lines, "Conflicts", conflicts)
        if unfinished == 0:
            lines.append("  State        No unfinished work is currently recorded.")

    elif "lease_id" in payload:
        lines.append("  Status       Work claim recorded.")
        _add_field(lines, "Lease", payload.get("lease_id"))
        lines.extend(_summary_lines(payload.get("summary")))
        conflicts = _coordination_conflicts(payload.get("coordination"))
        if conflicts is not None:
            _add_field(lines, "Conflicts", conflicts)

    elif "resume" in payload:
        view = _mapping(payload.get("resume"))
        lines.append("  Status       Resume package ready.")
        _add_field(lines, "Workstream", _first(view, "workstream_id", "workstream", "id"))
        _add_field(lines, "Title", _first(view, "title", "task_title", "label"))
        _add_field(lines, "Work status", _first(view, "status", "work_status"))
        _add_field(lines, "Tasks", _first(view, "task_ids", "tasks"))
        _add_field(lines, "Agents", _first(view, "agent_ids", "agents"))
        _add_field(lines, "Live changes", view.get("current_worktree_changed_paths"))
        _add_field(lines, "Past changes", _first(view, "changed_paths", "changes"))
        _add_field(lines, "Evidence", _first(view, "evidence_ids", "evidence"))
        _add_field(lines, "Remaining", _first(view, "remaining_work", "remaining"))
        acknowledgement = _mapping(payload.get("acknowledgement"))
        _add_field(lines, "Ack token", acknowledgement.get("token"))
        if acknowledgement.get("required"):
            lines.append("  Next         Review the recovered state, then run acknowledge --token <token>.")
        if len(lines) == 2:
            lines.append("  State        Recovery succeeded; use --json for the complete view.")

    elif "acknowledged" in payload:
        lines.append("  Status       Recovered work state acknowledged.")

    elif payload.get("kind") == "work_evidence":
        _add_field(lines, "Evidence", payload.get("record_id"))
        _add_field(lines, "Verdict", f"{payload.get('verdict')} (exit {payload.get('exit_code')})")
        trace = payload.get("trace") or {}
        _add_field(lines, "Trace", "complete" if trace.get("complete")
                   else "incomplete: " + ", ".join(trace.get("reasons") or []))
        _add_field(lines, "Inputs", sum((payload.get("inputs") or {}).values()))

    elif payload.get("kind") == "work_evidence_check":
        _add_field(lines, "Evidence", payload.get("record_id"))
        _add_field(lines, "Decision", payload.get("decision"))
        _add_field(lines, "Reasons", ", ".join(payload.get("reasons") or []))
        for item in (payload.get("changed") or [])[:10]:
            lines.append(f"  Changed      {item.get('kind')} {item.get('key')}")

    elif payload.get("kind") == "work_accept":
        epoch = payload.get("epoch") or {}
        _add_field(lines, "Outcome", payload.get("outcome"))
        _add_field(lines, "Holder", f"{epoch.get('holder')} (epoch {epoch.get('epoch')}, {epoch.get('authority')})")
        for action in payload.get("next_actions") or []:
            lines.append(f"  Next         {action}")
        for warning in payload.get("warnings") or []:
            lines.append(f"  Warning      {warning}")

    elif payload.get("kind") == "work_authority":
        current = payload.get("current") or {}
        _add_field(lines, "Workstream", payload.get("workstream"))
        _add_field(lines, "Epoch", payload.get("epoch"))
        if current:
            _add_field(lines, "Holder", f"{current.get('holder')} ({current.get('authority')})")
            _add_field(lines, "Expired", "yes" if payload.get("expired") else "no")

    elif payload.get("kind") == "work_host_key":
        _add_field(lines, "Public key", payload.get("public_key"))
        _add_field(lines, "Key id", payload.get("key_id"))
        _add_field(lines, "Private key", payload.get("private_key_file"))
        lines.append("  Note         Keep the private key on the host; pin the public key in the trust file.")

    elif payload.get("kind") == "work_host_assertion":
        claims = (payload.get("assertion") or {}).get("claims") or {}
        _add_field(lines, "Subject", claims.get("sub"))
        _add_field(lines, "Expires", claims.get("exp"))
        lines.append("  State        Assertion signed; use --json or --out for the file.")

    elif "handoff" in payload:
        receipt = _mapping(payload.get("handoff"))
        lines.append("  Status       Handoff sealed to the current Work Graph.")
        _add_field(lines, "Workstream", _first(receipt, "workstream_id", "workstream"))
        _add_field(lines, "From", _first(receipt, "source_agent", "from_agent"))
        _add_field(lines, "To", _first(receipt, "target_agent", "to_agent"))
        _add_field(lines, "Receipt", _first(receipt, "receipt_id", "handoff_id", "id"))
        _add_field(lines, "Inbox ref", payload.get("handoff_ref"))
        _add_field(lines, "Commitment", _first(receipt, "graph_commitment", "commitment"))
        if payload.get("offer_digest"):
            _add_field(lines, "Offer digest", payload.get("offer_digest"))
            _add_field(lines, "Repository", _mapping(payload.get("offer")).get("repo_id"))
        if len(lines) == 2:
            lines.append("  State        Handoff created; use --json for the complete receipt.")

    else:
        lines.append("  Status       Operation completed.")

    lines.append("  Detail       Use --json for the complete machine-readable result.")
    return lines


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    status = payload.get("status", "ok")
    if status == "error":
        code = payload.get("error", "work_graph_error")
        detail = payload.get("detail", "unknown")
        print(f"Entroly Work Graph error [{code}]: {detail}", file=sys.stderr)
        return
    print("\n".join(_human_lines(payload)))


def _error(exc: Exception) -> dict[str, Any]:
    from .evidence_applicability import EvidenceError
    from .handoff_authority import AuthorityRefused

    if isinstance(exc, AuthorityRefused):
        return {
            "status": "error",
            "error": "handoff_authority_refused",
            "code": exc.code,
            "detail": exc.detail[:2000],
        }
    if isinstance(exc, EvidenceError):
        return {"status": "error", "error": "evidence_error", "detail": str(exc)[:2000]}
    if isinstance(exc, WorkGraphUnavailableError):
        code = "native_work_graph_unavailable"
    elif isinstance(exc, WorkGraphLockTimeout):
        code = "work_graph_lock_timeout"
    elif isinstance(exc, WorkGraphStateError):
        code = "work_graph_state_error"
    elif isinstance(exc, WorkGraphStoreError):
        code = "work_graph_store_error"
    elif isinstance(exc, RecoveryAcknowledgementRequired):
        code = "recovery_acknowledgement_required"
    elif isinstance(exc, ValueError):
        code = "invalid_work_graph_request"
    else:
        code = "work_graph_error"
    return {"status": "error", "error": code, "detail": str(exc)[:2000]}


def _required_id(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > _MAX_ID_CHARS or "\x00" in text:
        raise ValueError(f"{name} may not exceed {_MAX_ID_CHARS} characters or contain NUL")
    return text


def _claim(project: Path, args: Any) -> tuple[Any, str]:
    ttl = float(args.ttl)
    if not math.isfinite(ttl) or not 0 < ttl <= _MAX_TTL_SECONDS:
        raise ValueError(f"ttl must be > 0 and <= {_MAX_TTL_SECONDS}")
    agent = _required_id(args.agent, "--agent")
    title = str(args.task).strip()
    if not title:
        raise ValueError("--task must not be empty")
    if len(title) > _MAX_LABEL_CHARS:
        raise ValueError(f"--task may not exceed {_MAX_LABEL_CHARS} characters")
    task_id = str(args.task_id or "").strip()
    session_id = str(args.session or "").strip()
    if len(task_id) > _MAX_ID_CHARS or "\x00" in task_id:
        raise ValueError(f"--task-id may not exceed {_MAX_ID_CHARS} characters or contain NUL")
    if len(session_id) > _MAX_ID_CHARS or "\x00" in session_id:
        raise ValueError(f"--session may not exceed {_MAX_ID_CHARS} characters or contain NUL")
    store = _store_for_path(project)
    recovery_require_acknowledged(store.repo_dir)
    now_ms = int(time.time() * 1000)
    lease_id = uuid.uuid4().hex
    observation = discover_repository_observation(
        project,
        agent_id=agent,
        session_id=session_id,
        task_hint={
            "task_id": task_id,
            "title": title,
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": [],
            "source_kind": "user_statement",
            "source_ref": f"work-claim:{lease_id}",
        },
        observed_at_ms=now_ms,
    )
    observation["leases"] = [{
        "lease_id": lease_id,
        "agent_id": agent,
        "task_id": task_id,
        "scope_paths": sorted(set(args.path or [])),
        "scope_symbols": sorted(set(args.symbol or [])),
        "expires_at_ms": now_ms + max(1, int(ttl * 1000)),
        "source_ref": f"work-lease:{lease_id}",
    }]
    enrich_worktree_content_digests(project, observation)
    return store.submit_repository_observation(
        observation, repository_path=project
    ), lease_id


def _authority_inputs(args: Any) -> tuple[dict[str, Any] | None, Any]:
    """The caller's host assertion and the pinned trust anchors, if supplied."""
    import os

    from .handoff_authority import ASSERTION_FILE_ENV, load_assertion, load_trust

    path = getattr(args, "assertion", "") or os.environ.get(ASSERTION_FILE_ENV, "")
    if not path:
        return None, None
    from .handoff_authority import AuthorityRefused

    try:
        trust = load_trust(getattr(args, "trust_file", "") or None)
    except AuthorityRefused:
        # Needed only if the workstream is held; require_write_authority asks
        # for it then, so a global assertion does not break unheld work.
        trust = None
    return load_assertion(path), trust


def _evidence_ref(store: WorkGraphStore, record_id: str) -> dict[str, Any]:
    from .evidence_applicability import EvidenceError, load_record, record_digest, verify_record

    record = load_record(store.repo_dir, _required_id(record_id, "--evidence"))
    problems = verify_record(record)
    if problems:
        raise EvidenceError(f"evidence {record_id} fails its integrity check: {', '.join(problems)}")
    return {
        "record_id": record["record_id"],
        "record_digest": record_digest(record),
        "name": record.get("name", ""),
        "argv": list(record["command"]["argv"]),
        "verdict": record["result"]["verdict"],
    }


def _evidence_action(project: Path, args: Any) -> dict[str, Any]:
    from .evidence_applicability import (
        EvidenceError,
        assess,
        load_record,
        provenance,
        record_digest,
        run_and_record,
        save_record,
        state_digest_secret,
        summarize_inputs,
    )

    store = _store_for_path(project)
    if args.evidence_action == "run":
        command = list(args.command or [])
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise ValueError("give the verification command after --")
        workstream = str(args.workstream or "").strip()
        authority = None
        if workstream:
            from .handoff_authority import require_write_authority

            assertion, trust = _authority_inputs(args)
            authority = require_write_authority(
                store.repo_dir, repo_id=store.repo_id, workstream=workstream,
                assertion=assertion, trust=trust,
            )
        timeout = args.timeout
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("--timeout must be a positive number of seconds")
        record, output = run_and_record(
            project,
            command,
            name=str(args.name or ""),
            timeout_seconds=timeout,
            work_dir=store.repo_dir / "evidence-runs",
            excluded_roots=[store.root],
            recorded_by={
                "agent": str(args.agent or ""),
                "workstream": workstream or None,
                "authority_epoch": authority["epoch"] if authority else None,
            },
            repository_id=store.repo_id,
            digest_secret=state_digest_secret(store.repo_dir),
        )
        save_record(store.repo_dir, record)
        return {
            "status": "ok",
            "kind": "work_evidence",
            "record_id": record["record_id"],
            "record_digest": record_digest(record),
            "verdict": record["result"]["verdict"],
            "exit_code": record["result"]["exit_code"],
            "trace": record["trace"],
            "inputs": summarize_inputs(record),
            "provenance": provenance(record),
            "output": output,
        }
    if args.evidence_action == "check":
        record_id = _required_id(args.record, "--record")
        try:
            record = load_record(store.repo_dir, record_id)
        except EvidenceError as exc:
            return {
                "status": "ok",
                "kind": "work_evidence_check",
                "record_id": record_id,
                "decision": "unknown",
                "reasons": ["record-unavailable"],
                "detail": str(exc),
            }
        decision = assess(record, project, digest_secret=state_digest_secret(store.repo_dir))
        return {
            "status": "ok",
            "kind": "work_evidence_check",
            **decision,
            "provenance": provenance(record),
        }
    raise ValueError("choose evidence run or evidence check")


def _next_action(item: dict[str, Any], decision: dict[str, Any]) -> str:
    argv = " ".join(item.get("argv") or ()) or "the verification"
    verdict = item.get("verdict") or "unknown"
    choice = decision.get("decision", "unknown")
    if choice == "reuse":
        return f"reuse: '{argv}' {verdict}; every input it read is unchanged"
    if choice == "revalidate":
        return f"rerun '{argv}': {decision.get('changed_count', 0)} input(s) it read have changed"
    return f"rerun '{argv}': applicability unknown ({', '.join(decision.get('reasons', []))})"


def _accept_action(project: Path, args: Any) -> dict[str, Any]:
    from .evidence_applicability import EvidenceError, assess, load_record, provenance, state_digest_secret
    from .handoff_authority import accept_handoff, load_assertion, load_trust

    store = _store_for_path(project)
    trust = load_trust(getattr(args, "trust_file", "") or None)
    assertion = load_assertion(_required_id(args.assertion, "--assertion"))
    observation = _passive_observation(project)
    details: dict[str, dict[str, Any]] = {}

    def assess_one(item: dict[str, Any]) -> dict[str, Any]:
        record_id = str(item.get("record_id") or "")
        try:
            record = load_record(store.repo_dir, record_id)
        except EvidenceError:
            result = {"decision": "unknown", "reasons": ["record-unavailable"], "changed_count": 0}
        else:
            try:
                result = assess(
                    record, project, expected_digest=str(item.get("record_digest") or ""),
                    digest_secret=state_digest_secret(store.repo_dir),
                )
            except Exception:  # a record that cannot be assessed is never reused
                result = {"decision": "unknown", "reasons": ["record-malformed"], "changed_count": 0}
                details[record_id] = result
                return result
            result["provenance"] = provenance(record, bound_by="record digest in the host-signed offer (host did not inspect it)")
        details[record_id] = result
        return result

    accepted = accept_handoff(
        store.repo_dir,
        repo_id=store.repo_id,
        handoff_ref=_required_id(args.handoff_ref, "--handoff-ref"),
        assertion=assertion,
        trust=trust,
        workspace_commitment=workspace_commitment(observation),
        assess_evidence=assess_one,
    )
    from .handoff_authority import load_offer

    _, offer = load_offer(store.repo_dir, args.handoff_ref)
    handed = {str(item.get("record_id")): item for item in offer.get("evidence") or ()}
    for record_id, item in handed.items():
        if record_id not in details:  # an idempotent retry still reports current applicability
            assess_one(item)
    next_actions = [
        _next_action(handed.get(record_id, {}), decision)
        for record_id, decision in details.items()
    ] or ["no verification evidence was handed over: run the verification before relying on it"]
    warnings = []
    if accepted.get("epoch", {}).get("authority") != "host-protected-trust":
        warnings.append(
            "advisory authority: the trust file is writable by this user, so a local process could "
            "have pinned its own key; the host should verify this epoch's key_id against its own keys"
        )
    return {
        "status": "ok",
        "kind": "work_accept",
        **accepted,
        "evidence_details": details,
        "next_actions": next_actions,
        "warnings": warnings,
    }


def _host_keygen(args: Any) -> dict[str, Any]:
    import os

    from .handoff_authority import generate_host_key, key_id

    target = Path(_required_id(args.private_key_out, "--private-key-out")).expanduser()
    private_hex, public_hex = generate_host_key()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(private_hex + "\n")
    return {
        "status": "ok",
        "kind": "work_host_key",
        "private_key_file": str(target),
        "public_key": public_hex,
        "key_id": key_id(public_hex),
        "note": "keep the private key on the host; pin the public key in the trust file",
    }


def _issue_assertion(args: Any) -> dict[str, Any]:
    from .handoff_authority import issue_assertion

    key_file = Path(_required_id(args.private_key_file, "--private-key-file")).expanduser()
    private_hex = key_file.read_text(encoding="utf-8").strip()
    ttl = float(args.ttl)
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("--ttl must be a positive number of seconds")
    assertion = issue_assertion(
        private_hex,
        issuer=_required_id(args.issuer, "--issuer"),
        audience=_required_id(args.audience, "--audience"),
        subject=_required_id(args.subject, "--subject"),
        workspace=_required_id(args.workspace, "--workspace"),
        task=_required_id(args.task, "--task"),
        handoff=_required_id(args.handoff, "--handoff"),
        scope=args.scope or ("work:accept", "work:write"),
        ttl_seconds=ttl,
    )
    if args.out:
        import os

        fd = os.open(Path(args.out).expanduser(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(assertion, handle, sort_keys=True)
    return {"status": "ok", "kind": "work_host_assertion", "assertion": assertion}


def run(args: Any) -> int:
    action = getattr(args, "work_action", None)
    json_output = bool(getattr(args, "json_output", False))
    try:
        project = _project_path(getattr(args, "project", "."))
        if action == "state":
            store = _store_for_path(project)
            graph = store.load()
            payload = {
                "status": "ok",
                "project_root": str(project),
                "repo_id": graph.repo_id,
                "summary": graph.summary(),
                "unfinished": graph.unfinished(),
                "coordination": graph.coordination(int(time.time() * 1000)),
            }
        elif action == "claim":
            graph, lease_id = _claim(project, args)
            payload = {
                "status": "ok",
                "lease_id": lease_id,
                "summary": graph.summary(),
                "coordination": graph.coordination(int(time.time() * 1000)),
            }
        elif action == "resume":
            max_evidence = args.max_evidence
            if (
                not isinstance(max_evidence, int)
                or isinstance(max_evidence, bool)
                or not 0 <= max_evidence <= _MAX_EVIDENCE
            ):
                raise ValueError(
                    f"max_evidence must be an integer between 0 and {_MAX_EVIDENCE}"
                )
            workstream = str(args.workstream or "").strip()
            if len(workstream) > _MAX_ID_CHARS or "\x00" in workstream:
                raise ValueError(
                    f"--workstream may not exceed {_MAX_ID_CHARS} characters or contain NUL"
                )
            store = _store_for_path(project)
            # Resume is an explicit recovery action: refresh bounded durable
            # repository/checkpoint facts plus exact unstaged worktree identity
            # before Rust reconstructs the workstream.
            resume_view = recover_workstream(
                store, project, _passive_observation(project),
                workstream_id=workstream, max_evidence=max_evidence,
            )
            payload = {
                "status": "ok",
                "resume": resume_view,
                "acknowledgement": recovery_arm(
                    store.repo_dir,
                    recovery_token(resume_view),
                    recovery_unknowns(resume_view),
                    workspace_commitment=resume_view["current_workspace_commitment"],
                ),
            }
            target_agent = str(getattr(args, "to_agent", "") or "").strip()
            if target_agent:
                target_agent = _required_id(target_agent, "--to-agent")
                payload["continuation_proof"] = store.reconstructed_continuation_proof(
                    str(resume_view["selected_workstream"]["node_id"]),
                    target_agent,
                    outstanding_work_refs=continuation_outstanding_refs(resume_view),
                    created_at_ms=int(time.time() * 1000),
                )
        elif action == "acknowledge":
            token = _required_id(args.token, "--token")
            store = _store_for_path(project)
            payload = {
                "status": "ok",
                **recovery_acknowledge(
                    store.repo_dir, token,
                    workspace_commitment=workspace_commitment(_passive_observation(project)),
                ),
            }
        elif action == "handoff":
            workstream = _required_id(args.workstream, "--workstream")
            source_agent = _required_id(args.from_agent, "--from-agent")
            target_agent = _required_id(args.to_agent, "--to-agent")
            store = _store_for_path(project)
            recovery_require_acknowledged(store.repo_dir)
            from .handoff_authority import build_offer, offer_digest, require_write_authority

            # A held workstream may only be handed on by its current holder.
            assertion, trust = _authority_inputs(args)
            authority = require_write_authority(
                store.repo_dir, repo_id=store.repo_id, workstream=workstream,
                assertion=assertion, trust=trust,
            )
            evidence = [
                _evidence_ref(store, record_id)
                for record_id in getattr(args, "evidence", None) or ()
            ]
            # A handoff receipt must commit to the latest durable worktree and
            # checkpoint facts, including exact unstaged worktree content identity.
            observation = _passive_observation(project)
            store.submit_repository_observation(observation, repository_path=project)
            handoff = store.handoff(workstream, source_agent, target_agent)
            resume_view = attach_current_worktree_changes(
                store.resume(workstream, max_evidence=128), observation
            )
            offer = build_offer(
                handoff_receipt=handoff,
                workstream=workstream,
                from_agent=source_agent,
                to_agent=target_agent,
                repo_id=store.repo_id,
                base_epoch=int(authority["epoch"]) if authority else 0,
                workspace_commitment=resume_view["current_workspace_commitment"],
                evidence=evidence,
                from_authority=authority,
            )
            payload = {
                "status": "ok",
                "handoff": handoff,
                "continuation_proof": store.continuation_proof(
                    handoff,
                    outstanding_work_refs=continuation_outstanding_refs(resume_view),
                    created_at_ms=int(time.time() * 1000),
                ),
                "offer": offer,
                "offer_digest": offer_digest(offer),
            }
            from .work_graph_inbox import save_handoff

            payload["handoff_ref"] = save_handoff(
                store, payload["handoff"], payload["continuation_proof"], offer=offer
            )
        elif action == "evidence":
            payload = _evidence_action(project, args)
        elif action == "accept":
            payload = _accept_action(project, args)
        elif action == "authority":
            from .handoff_authority import authority_view

            store = _store_for_path(project)
            payload = {
                "status": "ok",
                "kind": "work_authority",
                **authority_view(store.repo_dir, _required_id(args.workstream, "--workstream")),
            }
        elif action == "host-keygen":
            payload = _host_keygen(args)
        elif action == "issue-assertion":
            payload = _issue_assertion(args)
        elif action == "handoffs":
            target_agent = _required_id(args.to_agent, "--to-agent")
            store = _store_for_path(project)
            from .work_graph_inbox import pending_handoffs

            payload = pending_handoffs(root=project, store=store, to_agent=target_agent)
            payload["project_root"] = str(project)
            payload["repo_id"] = store.repo_id
        elif action == "review":
            reviewer = _required_id(args.reviewer, "--reviewer")
            handoff_ref = _required_id(args.handoff_ref, "--handoff-ref")
            if re.fullmatch(r"handoff:[0-9a-f]{64}", handoff_ref) is None:
                raise ValueError("--handoff-ref must be a handoff: SHA-256 reference")
            findings = [
                {"path": path, "line": int(line), "severity": severity, "message": message}
                for path, line, severity, message in args.finding
            ]
            checks = [
                {"name": name, "exit_code": int(exit_code)}
                for name, exit_code in args.check
            ]
            from .work_graph_inbox import pending_handoffs
            from .work_graph_review import record_review

            store = _store_for_path(project)
            entries = pending_handoffs(root=project, store=store, to_agent=reviewer)["handoffs"]
            selected = next((item for item in entries if item["ref"] == handoff_ref), None)
            if selected is None:
                raise ValueError("handoff reference is not in this repository's bounded inbox")
            if not selected["current"]:
                raise ValueError("handoff is stale against current repository source")
            payload = record_review(
                handoff=selected["handoff"], reviewer_id=reviewer,
                verdict=args.verdict, findings=findings, summary=args.summary,
                checks=checks, _root=project,
            )
        elif action == "reviews":
            workstream = str(args.workstream or "").strip()
            if len(workstream) > _MAX_ID_CHARS or "\x00" in workstream:
                raise ValueError(f"--workstream may not exceed {_MAX_ID_CHARS} characters or contain NUL")
            from .work_graph_review import reviews

            payload = reviews(workstream_id=workstream, _root=project)
        else:
            payload = {
                "status": "error",
                "error": "missing_work_action",
                "detail": (
                    "choose state, claim, resume, acknowledge, handoff, handoffs, review, "
                    "reviews, evidence, accept, authority, host-keygen, or issue-assertion"
                ),
            }
    except Exception as exc:
        payload = _error(exc)
    _emit(payload, json_output=json_output)
    if payload.get("status") != "ok":
        return 1
    if payload.get("kind") == "work_evidence":
        # The verification's own result: wrapping a failing test run must not
        # turn a red build green.
        code = payload.get("exit_code")
        return code if isinstance(code, int) and 0 <= code < 256 else 1
    if payload.get("kind") == "work_evidence_check":
        return 0 if payload.get("decision") == "reuse" else 3
    return 0


def _add_authority_arguments(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--assertion", default="",
        help=f"{help_text} (or ENTROLY_HANDOFF_ASSERTION)",
    )
    parser.add_argument("--trust-file", default="", help="Pinned host keys (or ENTROLY_HANDOFF_TRUST_FILE)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entroly-work",
        description="Inspect, resume, coordinate, and hand off evidence-backed repository work.",
    )
    parser.add_argument("--project", default=".", help="Git worktree to inspect")
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit the complete machine-readable result",
    )
    actions = parser.add_subparsers(dest="work_action", required=True)
    actions.add_parser("state", help="Show shared repository work state")
    claim = actions.add_parser("claim", help="Claim work with an advisory lease")
    claim.add_argument("--agent", required=True)
    claim.add_argument("--task", required=True)
    claim.add_argument("--task-id", default="")
    claim.add_argument("--session", default="")
    claim.add_argument("--path", action="append", default=[])
    claim.add_argument("--symbol", action="append", default=[])
    claim.add_argument("--ttl", type=float, default=900.0)
    resume = actions.add_parser("resume", help="Recover an unfinished workstream")
    resume.add_argument("--workstream", default="")
    resume.add_argument("--max-evidence", type=int, default=128)
    resume.add_argument(
        "--to-agent",
        default="",
        help="Also seal an evidence-bounded no-handoff continuation proof",
    )
    acknowledge = actions.add_parser(
        "acknowledge", help="Accept responsibility for the recovered state"
    )
    acknowledge.add_argument("--token", required=True)
    handoff = actions.add_parser("handoff", help="Seal a graph-bound cross-agent handoff")
    handoff.add_argument("--workstream", required=True)
    handoff.add_argument("--from-agent", required=True)
    handoff.add_argument("--to-agent", required=True)
    handoff.add_argument(
        "--evidence", action="append", default=[], metavar="EVIDENCE_ID",
        help="Bind a recorded verification (from 'evidence run') into the offer; repeatable",
    )
    _add_authority_arguments(handoff, "Current holder's host assertion, required once the workstream is held")
    evidence = actions.add_parser(
        "evidence", help="Record a verification Entroly runs itself, or check whether one still applies",
    )
    evidence_actions = evidence.add_subparsers(dest="evidence_action", required=True)
    evidence_run = evidence_actions.add_parser(
        "run", help="Run a verification command with input tracing and record the result",
    )
    evidence_run.add_argument("--name", default="", help="Label for the verification")
    evidence_run.add_argument("--agent", default="", help="Label of the recording agent (not authenticated)")
    evidence_run.add_argument("--workstream", default="", help="Workstream the evidence belongs to")
    evidence_run.add_argument("--timeout", type=float, default=None, help="Seconds before the run is abandoned")
    _add_authority_arguments(evidence_run, "Holder's host assertion, required when the workstream is held")
    evidence_run.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command and its arguments")
    evidence_check = evidence_actions.add_parser(
        "check", help="Decide reuse, revalidate or unknown for a recorded verification",
    )
    evidence_check.add_argument("--record", required=True, help="evidence:<id> from 'evidence run'")
    accept = actions.add_parser("accept", help="Accept a handoff with a host assertion and take authority")
    accept.add_argument("--handoff-ref", required=True)
    accept.add_argument("--assertion", required=True, help="Host-signed assertion file for this successor")
    accept.add_argument("--trust-file", default="", help="Pinned host keys (or ENTROLY_HANDOFF_TRUST_FILE)")
    authority = actions.add_parser("authority", help="Show who holds a workstream and since which epoch")
    authority.add_argument("--workstream", required=True)
    keygen = actions.add_parser("host-keygen", help="Host side: create an Ed25519 key for signing assertions")
    keygen.add_argument("--private-key-out", required=True, help="New file for the private key (mode 0600)")
    issue = actions.add_parser("issue-assertion", help="Host side: sign an assertion for one successor")
    issue.add_argument("--private-key-file", required=True)
    issue.add_argument("--issuer", required=True)
    issue.add_argument("--audience", required=True)
    issue.add_argument("--subject", required=True, help="Successor agent id; must equal the offer's to_agent")
    issue.add_argument("--workspace", required=True, help="Work Graph repo_id")
    issue.add_argument("--task", required=True, help="Workstream id")
    issue.add_argument("--handoff", required=True, help="offer_digest printed by 'handoff'")
    issue.add_argument("--scope", action="append", default=[], choices=("work:accept", "work:write"))
    issue.add_argument("--ttl", type=float, default=900.0, help="Validity in seconds")
    issue.add_argument("--out", default="", help="Also write the assertion to this new file")
    handoffs = actions.add_parser("handoffs", help="List current and stale handoffs for one agent")
    handoffs.add_argument("--to-agent", required=True)
    review = actions.add_parser("review", help="Record an independent review of an inbox handoff")
    review.add_argument("--handoff-ref", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--verdict", required=True, choices=("approve", "changes_requested", "comment"))
    review.add_argument("--summary", required=True)
    review.add_argument(
        "--finding", action="append", nargs=4, default=[],
        metavar=("PATH", "LINE", "SEVERITY", "MESSAGE"),
        help="Repeat for each source-bound finding",
    )
    review.add_argument(
        "--check", action="append", nargs=2, default=[],
        metavar=("NAME", "EXIT_CODE"),
        help="Repeat for each check actually run; the value remains a caller statement",
    )
    reviews = actions.add_parser("reviews", help="List recorded reviews and source freshness")
    reviews.add_argument("--workstream", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
