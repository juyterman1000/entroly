"""Local JSONL bridge for the Entroly OpenClaw context-engine plugin."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from .context_receipts.retrieval import tokenize
from .sdk import compress

BRIDGE_SCHEMA = "entroly.openclaw.bridge.v2"
RECEIPT_SCHEMA = "entroly.openclaw.receipt.v2"
DEFAULT_PRESERVE_LAST_N = 4
PROVIDER_MODE = "openclaw_managed"
_BUDGET_SOURCES = {
    "openclaw_token_budget",
    "openclaw_runtime_settings",
    "operator_fallback",
}
DEFAULT_RECEIPT_MAX_FILES = 512
DEFAULT_RECEIPT_MAX_BYTES = 64 * 1024 * 1024
_EMPTY_ACCEPTANCE_SIGNATURE = "0" * 64


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signing_key_id(key: bytes) -> str:
    """Return a non-secret identifier that detects receipt-key rotation."""
    return hashlib.sha256(b"entroly.openclaw.receipt-key.v1\0" + key).hexdigest()


def _hmac_sha256(key: bytes, label: str, value: str) -> str:
    payload = f"entroly.openclaw.{label}.v1:{value}".encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _estimate_value_tokens(value: Any) -> int:
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4) if value else 0
    if value is None or isinstance(value, (bool, int, float)):
        return 1
    if isinstance(value, list):
        return sum(_estimate_value_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _estimate_value_tokens(key) + _estimate_value_tokens(item)
            for key, item in value.items()
        )
    return _estimate_value_tokens(str(value))


def _compressible_text_locations(message: dict[str, Any]) -> list[tuple[int | None, str]]:
    """Return unsigned normalized text fields that Entroly may safely rewrite."""
    content = message.get("content")
    if isinstance(content, str):
        return [(None, content)]
    if not isinstance(content, list):
        return []
    locations: list[tuple[int | None, str]] = []
    for index, block in enumerate(content):
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and set(block) <= {"type", "text"}
        ):
            locations.append((index, block["text"]))
    return locations


def _message_text(message: dict[str, Any], *, compressible_only: bool) -> str:
    if compressible_only:
        return "\n".join(text for _, text in _compressible_text_locations(message))
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block["text"])
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _provider_visible_message_tokens(message: dict[str, Any]) -> int:
    """Estimate normalized prompt content without routing/accounting metadata."""
    role = message.get("role")
    if role not in {"user", "assistant", "toolResult", "system", "developer", "human"}:
        return _estimate_value_tokens(message) + 3

    content = message.get("content")
    content_tokens = _estimate_value_tokens(content)
    envelope_tokens = 4
    if role == "toolResult":
        envelope_tokens += _estimate_value_tokens(message.get("toolCallId"))
        envelope_tokens += _estimate_value_tokens(message.get("toolName"))
        envelope_tokens += 1  # isError
    return content_tokens + envelope_tokens


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate provider-visible normalized context, independent of routing metadata."""
    return sum(_provider_visible_message_tokens(message) for message in messages)


def _compressible_text_tokens(message: dict[str, Any]) -> int:
    return sum(_estimate_value_tokens(text) for _, text in _compressible_text_locations(message))


def _fixed_message_tokens(message: dict[str, Any]) -> int:
    return max(0, _provider_visible_message_tokens(message) - _compressible_text_tokens(message))


def _protected_message(message: dict[str, Any]) -> bool:
    """Return whether a message must remain byte-for-byte equivalent."""
    if message.get("role") in {"system", "developer"}:
        return True
    return not _compressible_text_locations(message)


def _message_relevance(
    messages: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    query_terms = sorted(set(tokenize(query)))
    if not query_terms:
        return [
            {"score": 0.0, "matched_terms": [], "token_count": 0}
            for _ in messages
        ]

    documents = [
        tokenize(_message_text(message, compressible_only=True))
        for message in messages
    ]
    document_frequency: dict[str, int] = defaultdict(int)
    for terms in documents:
        for term in set(terms):
            document_frequency[term] += 1
    document_count = max(1, len(documents))
    average_length = max(
        1.0, sum(len(terms) for terms in documents) / document_count
    )

    ranked: list[dict[str, Any]] = []
    for index, terms in enumerate(documents):
        frequencies = Counter(terms)
        document_length = max(1, len(terms))
        matched = sorted(term for term in query_terms if frequencies.get(term, 0))
        score = 0.0
        for term in matched:
            frequency = frequencies[term]
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1.0
                + (document_count - frequency_docs + 0.5)
                / (frequency_docs + 0.5)
            )
            normalization = 1.0 - 0.75 + 0.75 * document_length / average_length
            score += inverse_frequency * (frequency * 2.2) / (
                frequency + 1.2 * normalization
            )
        coverage = len(matched) / max(1, len(query_terms))
        pin_eligible = True
        security_flags: list[str] = []
        pin_blocked_reason: str | None = None
        if matched:
            try:
                from .context_firewall import scan

                scan_result = scan(
                    _message_text(messages[index], compressible_only=False),
                    source=f"openclaw_message_{index}",
                    check_repetition=False,
                )
                security_flags = sorted(
                    {
                        f"{threat.severity}:{threat.threat_type}"
                        for threat in scan_result.threats
                    }
                )
                pin_eligible = scan_result.is_safe
                if not pin_eligible:
                    pin_blocked_reason = "context_firewall"
            except Exception:
                pin_eligible = False
                security_flags = ["critical:scanner_error"]
                pin_blocked_reason = "context_firewall_error"
        ranked.append(
            {
                "score": round(score * (1.0 + coverage), 6),
                "matched_terms": matched,
                "token_count": len(terms),
                "pin_eligible": pin_eligible,
                "security_flags": security_flags,
                "pin_blocked_reason": pin_blocked_reason,
            }
        )
    return ranked


def _query_for_request(
    request: dict[str, Any], messages: list[dict[str, Any]]
) -> str:
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    for message in reversed(messages):
        if message.get("role") not in {"user", "human"}:
            continue
        content = _message_text(message, compressible_only=False)
        if content.strip():
            return content.strip()
    return ""


def _compress_message_bodies(
    messages: list[dict[str, Any]],
    *,
    content_budget: int,
    distill: bool,
    query: str,
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]], int]:
    text_tokens = [
        max(1, _compressible_text_tokens(message)) for message in messages
    ]
    relevance = _message_relevance(messages, query)
    reserve_for_compression = min(int(content_budget * 0.35), len(messages) * 8)
    pin_budget = min(
        int(content_budget * 0.65), max(0, content_budget - reserve_for_compression)
    )
    pinned: set[int] = set()
    for index in sorted(
        range(len(messages)),
        key=lambda item: (-relevance[item]["score"], text_tokens[item], item),
    ):
        if (
            relevance[index]["score"] <= 0
            or not relevance[index]["matched_terms"]
            or not relevance[index]["pin_eligible"]
        ):
            continue
        if text_tokens[index] <= pin_budget:
            pinned.add(index)
            pin_budget -= text_tokens[index]

    unpinned = [index for index in range(len(messages)) if index not in pinned]
    remaining_budget = max(
        0, content_budget - sum(text_tokens[index] for index in pinned)
    )
    allocations = {index: text_tokens[index] for index in pinned}
    if unpinned:
        distributable = max(0, remaining_budget - len(unpinned))
        max_score = max((relevance[index]["score"] for index in unpinned), default=0.0)
        weights = {
            index: math.sqrt(text_tokens[index])
            * (1.0 + 2.0 * relevance[index]["score"] / max(1.0, max_score))
            for index in unpinned
        }
        total_weight = max(1.0, sum(weights.values()))
        for index in unpinned:
            allocations[index] = 1 + int(
                distributable * weights[index] / total_weight
            )

    result: list[dict[str, Any]] = []
    distillation_failures = 0
    for index, message in enumerate(messages):
        if index in pinned:
            result.append(copy.deepcopy(message))
            continue
        compressed_message = copy.deepcopy(message)
        locations = _compressible_text_locations(message)
        location_tokens = [max(1, _estimate_value_tokens(text)) for _, text in locations]
        total_tokens = max(1, sum(location_tokens))
        remaining = max(1, allocations[index])
        location_budgets: list[int] = []
        for location_index, token_count in enumerate(location_tokens):
            locations_left = len(location_tokens) - location_index - 1
            if locations_left == 0:
                allocated = max(1, remaining)
            else:
                proportional = max(1, int(allocations[index] * token_count / total_tokens))
                allocated = min(proportional, max(1, remaining - locations_left))
            location_budgets.append(allocated)
            remaining -= allocated

        for (block_index, raw_text), block_budget in zip(
            locations, location_budgets, strict=True
        ):
            content = raw_text
            if distill and message.get("role") == "assistant":
                try:
                    from .proxy_transform import distill_response

                    content, _, _ = distill_response(content, mode="full")
                except Exception:
                    distillation_failures += 1
            compressed_text = compress(content, budget=max(1, block_budget))
            if block_index is None:
                compressed_message["content"] = compressed_text
            else:
                compressed_message["content"][block_index]["text"] = compressed_text
        result.append(compressed_message)
    for index, item in enumerate(relevance):
        item["allocated_tokens"] = allocations[index]
    return result, pinned, relevance, distillation_failures


def _safe_session_name(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id).strip("-.")
    return (safe or "session")[:80]


def _bounded_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:limit]


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _runtime_audit_metadata(request: dict[str, Any]) -> dict[str, Any]:
    """Copy only the bounded, non-secret OpenClaw runtime fields we audit."""
    raw = request.get("openclaw_runtime")
    raw = raw if isinstance(raw, dict) else {}
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    limits = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    resolved = _bounded_text(model.get("resolved"), 256)
    if resolved is None:
        resolved = _bounded_text(request.get("model"), 256)
    return {
        "schema_version": 1 if raw.get("schema_version") == 1 else None,
        "runtime": {
            "host": _bounded_text(runtime.get("host"), 64),
            "mode": _bounded_text(runtime.get("mode"), 64),
            "harness_id": _bounded_text(runtime.get("harness_id"), 128),
            "runtime_id": _bounded_text(runtime.get("runtime_id"), 128),
        },
        "model": {
            "requested": _bounded_text(model.get("requested"), 256),
            "resolved": resolved,
            "provider": _bounded_text(model.get("provider"), 128),
            "family": _bounded_text(model.get("family"), 128),
        },
        "limits": {
            "prompt_token_budget": _positive_int_or_none(
                limits.get("prompt_token_budget")
            ),
            "max_output_tokens": _positive_int_or_none(
                limits.get("max_output_tokens")
            ),
        },
    }


def _validate_assembly_invariants(
    source_messages: list[dict[str, Any]],
    assembled_messages: list[dict[str, Any]],
    protected_indexes: set[int],
) -> None:
    """Prove that only eligible unsigned text fields changed."""
    if len(source_messages) != len(assembled_messages):
        raise ValueError("message count changed")
    for index, (source, assembled) in enumerate(
        zip(source_messages, assembled_messages, strict=True)
    ):
        if not isinstance(assembled, dict) or source.get("role") != assembled.get("role"):
            raise ValueError(f"message role changed at index {index}")
        source_metadata = {key: value for key, value in source.items() if key != "content"}
        assembled_metadata = {
            key: value for key, value in assembled.items() if key != "content"
        }
        if _canonical_json(source_metadata) != _canonical_json(assembled_metadata):
            raise ValueError(f"message metadata changed at index {index}")

        source_content = source.get("content")
        assembled_content = assembled.get("content")
        if index in protected_indexes:
            if _canonical_json(source_content) != _canonical_json(assembled_content):
                raise ValueError(f"protected message changed at index {index}")
            continue
        if isinstance(source_content, str):
            if not isinstance(assembled_content, str):
                raise ValueError(f"text shape changed at index {index}")
            continue
        if not isinstance(source_content, list) or not isinstance(
            assembled_content, list
        ):
            if _canonical_json(source_content) != _canonical_json(assembled_content):
                raise ValueError(f"opaque content changed at index {index}")
            continue
        if len(source_content) != len(assembled_content):
            raise ValueError(f"content block count changed at index {index}")
        eligible = {
            block_index
            for block_index, _ in _compressible_text_locations(source)
            if block_index is not None
        }
        for block_index, (source_block, assembled_block) in enumerate(
            zip(source_content, assembled_content, strict=True)
        ):
            if block_index in eligible:
                if not (
                    isinstance(assembled_block, dict)
                    and assembled_block.get("type") == "text"
                    and isinstance(assembled_block.get("text"), str)
                    and set(assembled_block) <= {"type", "text"}
                ):
                    raise ValueError(
                        f"text block shape changed at index {index}:{block_index}"
                    )
            elif _canonical_json(source_block) != _canonical_json(assembled_block):
                raise ValueError(
                    f"opaque block changed at index {index}:{block_index}"
                )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _receipt_signing_key_path() -> Path:
    configured = os.environ.get("ENTROLY_OPENCLAW_RECEIPT_KEY_FILE")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "ENTROLY_OPENCLAW_RECEIPT_KEY_FILE must be an absolute path outside the receipt store"
            )
        return path.absolute()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        state_root = Path(os.environ["LOCALAPPDATA"])
    else:
        state_root = Path(
            os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        )
    return (state_root.expanduser().absolute() / "entroly" / "openclaw-receipt.key")


@contextmanager
def _receipt_key_init_lock(parent: Path):
    lock_path = parent / ".openclaw-receipt-key.lock"
    deadline = time.monotonic() + 2.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 30:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"OpenClaw receipt signing key initialization is busy at {lock_path}"
                )
            time.sleep(0.025)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _read_receipt_signing_key(path: Path) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"OpenClaw receipt signing key must not be a symlink: {path}")
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError(
            f"OpenClaw receipt signing key is corrupted at {path}; preserve the file and restore its 32-byte backup before retrying"
        )
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise PermissionError(
            f"OpenClaw receipt signing key must use 0600 permissions: {path}"
        )
    return key


def _load_receipt_signing_key(
    *forbidden_roots: Path, create: bool = True
) -> bytes:
    path = _receipt_signing_key_path()
    resolved_path = path.resolve(strict=False)
    for forbidden_root in forbidden_roots:
        resolved_root = forbidden_root.expanduser().resolve()
        if resolved_path == resolved_root or resolved_path.is_relative_to(
            resolved_root
        ):
            raise PermissionError(
                "OpenClaw receipt signing key must be stored outside the "
                f"workspace and receipt store: {path}"
            )
    parent = path.parent
    parent_existed = parent.exists()
    if not parent_existed and not create:
        raise FileNotFoundError(
            f"OpenClaw receipt signing key is missing at {path}; restore the original key before committing existing receipts"
        )
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.resolve() != parent:
        raise PermissionError("OpenClaw receipt signing key directory contains a symlink")
    if os.name != "nt":
        if not parent_existed or not os.environ.get(
            "ENTROLY_OPENCLAW_RECEIPT_KEY_FILE"
        ):
            os.chmod(parent, 0o700)
        elif parent.stat().st_mode & 0o022:
            raise PermissionError(
                f"OpenClaw receipt signing key directory is writable by other users: {parent}"
            )
    with _receipt_key_init_lock(parent):
        if path.exists():
            if path.stat().st_size != 0:
                return _read_receipt_signing_key(path)
            if not create:
                raise ValueError(
                    f"OpenClaw receipt signing key is interrupted at {path}; restore the original 32-byte key before committing existing receipts"
                )
            quarantine = path.with_name(
                f"{path.name}.incomplete-{secrets.token_hex(8)}"
            )
            os.replace(path, quarantine)
            _fsync_directory(parent)
            print(
                f"entroly: quarantined interrupted empty receipt signing key at {quarantine}; creating a new key",
                file=sys.stderr,
                flush=True,
            )

        if not create:
            raise FileNotFoundError(
                f"OpenClaw receipt signing key is missing at {path}; restore the original key before committing existing receipts"
            )

        key = secrets.token_bytes(32)
        descriptor = -1
        temporary_name = ""
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, path)
            temporary_name = ""
            if os.name != "nt":
                os.chmod(path, 0o600)
            _fsync_directory(parent)
            return key
        finally:
            if descriptor != -1:
                os.close(descriptor)
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)


def _receipt_directory(request: dict[str, Any]) -> Path:
    configured = request.get("receipt_dir")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser().resolve()
    workspace = request.get("workspace_dir")
    root = (
        Path(workspace).expanduser()
        if isinstance(workspace, str) and workspace
        else Path.cwd()
    ).resolve()
    directory = root / ".entroly" / "receipts" / "openclaw"
    current = root
    for component in (".entroly", "receipts", "openclaw"):
        current /= component
        if current.is_symlink():
            raise PermissionError(
                "default OpenClaw receipt store must not contain symlink components; "
                "configure receiptDir explicitly if an external store is intentional"
            )
    resolved = directory.resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError("default OpenClaw receipt store escaped the workspace")
    return directory


def _receipt_integrity_sha256(receipt: dict[str, Any]) -> str:
    immutable = {key: value for key, value in receipt.items() if key != "proposal_sha256"}
    immutable["acceptance_status"] = "proposed"
    immutable["acceptance_signature"] = _EMPTY_ACCEPTANCE_SIGNATURE
    immutable["acceptance_commit_sha256"] = _EMPTY_ACCEPTANCE_SIGNATURE
    return _sha256_text(_canonical_json(immutable))


def _acceptance_commit_sha256(proposal_sha256: str, proof: str) -> str:
    return _sha256_text(
        f"entroly.openclaw.accept.v1:{proposal_sha256}:{proof}"
    )


def _acceptance_signature(
    signing_key: bytes,
    receipt: dict[str, Any],
    proposal_sha256: str,
    acceptance_commit_sha256: str,
) -> str:
    signed = _canonical_json(
        {
            "schema_version": receipt.get("schema_version"),
            "receipt_id": receipt.get("receipt_id"),
            "proposal_id": receipt.get("proposal_id"),
            "proposal_sha256": proposal_sha256,
            "acceptance_actor": receipt.get("acceptance_actor"),
            "acceptance_challenge_sha256": receipt.get(
                "acceptance_challenge_sha256"
            ),
            "acceptance_commit_sha256": acceptance_commit_sha256,
            "acceptance_status": "accepted",
        }
    )
    return _hmac_sha256(signing_key, "receipt_acceptance", signed)


def _validate_acceptance_state(
    receipt: dict[str, Any], proposal_sha256: str, signing_key: bytes
) -> None:
    status = receipt.get("acceptance_status")
    signature = receipt.get("acceptance_signature")
    commit_sha256 = receipt.get("acceptance_commit_sha256")
    challenge_sha256 = receipt.get("acceptance_challenge_sha256")
    if not isinstance(challenge_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", challenge_sha256
    ):
        raise ValueError("receipt proposal has an invalid acceptance challenge")
    if status == "proposed":
        if (
            signature != _EMPTY_ACCEPTANCE_SIGNATURE
            or commit_sha256 != _EMPTY_ACCEPTANCE_SIGNATURE
        ):
            raise ValueError("proposed receipt contains a forged acceptance signature")
        return
    if status != "accepted" or not isinstance(signature, str) or not re.fullmatch(
        r"[0-9a-f]{64}", signature
    ):
        raise ValueError("receipt proposal has an invalid acceptance state")
    if not isinstance(commit_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", commit_sha256
    ):
        raise ValueError("receipt acceptance commit failed integrity validation")
    expected = _acceptance_signature(
        signing_key, receipt, proposal_sha256, commit_sha256
    )
    if not hmac.compare_digest(signature, expected):
        raise ValueError("receipt acceptance signature failed integrity validation")


def _positive_receipt_limit(
    request: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int
) -> int:
    value = request.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
    return value


def _ensure_private_receipt_directory(
    directory: Path, *, repair_default_permissions: bool
) -> None:
    created = False
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
    except FileExistsError:
        pass
    if directory.resolve() != directory:
        raise PermissionError("OpenClaw receipt directory resolved through a symlink")
    if os.name != "nt":
        if created or repair_default_permissions:
            os.chmod(directory, 0o700)
        elif directory.stat().st_mode & 0o077:
            raise PermissionError(
                f"configured receiptDir is not private at {directory}; use a dedicated 0700 directory"
            )


@contextmanager
def _receipt_store_lock(directory: Path):
    lock_path = directory / ".receipt-store.lock"
    deadline = time.monotonic() + 1.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 30:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "OpenClaw receipt store is busy; retry after the active assembly finishes"
                )
            time.sleep(0.025)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _atomic_write_private_json(destination: Path, payload: dict[str, Any]) -> None:
    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, destination)
        temporary_name = ""
        if os.name != "nt":
            os.chmod(destination, 0o600)
            _fsync_directory(destination.parent)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _enforce_receipt_quota(
    directory: Path, *, new_bytes: int, max_files: int, max_bytes: int
) -> None:
    receipts = [path for path in directory.glob("*.json") if path.is_file()]
    existing_bytes = sum(path.stat().st_size for path in receipts)
    if len(receipts) + 1 > max_files or existing_bytes + new_bytes > max_bytes:
        raise ValueError(
            "OpenClaw receipt quota reached; archive old receipts or increase "
            "receiptMaxFiles/receiptMaxBytes before retrying"
        )


def _write_receipt(
    request: dict[str, Any],
    source_messages: list[dict[str, Any]],
    assembled_messages: list[dict[str, Any]],
    *,
    source_tokens: int,
    assembled_tokens: int,
    warnings: list[str],
    evidence: dict[int, dict[str, Any]],
    strategy: str,
    query: str,
    budget_source: str,
    runtime_metadata: dict[str, Any],
) -> tuple[str, str | None, bool, str | None, str | None]:
    source_json = _canonical_json(source_messages)
    assembled_json = _canonical_json(assembled_messages)
    source_hash = _sha256_text(source_json)
    assembled_hash = _sha256_text(assembled_json)
    write_receipt = request.get("write_receipt", True) is not False
    directory: Path | None = None
    signing_key: bytes | None = None
    workspace_root: Path | None = None
    if write_receipt:
        directory = _receipt_directory(request)
        workspace = request.get("workspace_dir")
        workspace_root = (
            Path(workspace).expanduser()
            if isinstance(workspace, str) and workspace
            else Path.cwd()
        ).resolve()
        signing_key = _load_receipt_signing_key(directory, workspace_root)
    source_digest = (
        _hmac_sha256(signing_key, "source_context", source_json)
        if signing_key is not None
        else source_hash
    )
    assembled_digest = (
        _hmac_sha256(signing_key, "assembled_context", assembled_json)
        if signing_key is not None
        else assembled_hash
    )
    identity = _canonical_json(
        {
            "source_digest": source_digest,
            "assembled_digest": assembled_digest,
            "session_id": str(request.get("session_id") or ""),
            "budget": request.get("token_budget"),
            "budget_source": budget_source,
            "openclaw_runtime": runtime_metadata,
        }
    )
    receipt_id = "ocr_" + _sha256_text(identity)[:20]
    if not write_receipt:
        return receipt_id, None, False, None, None
    assert signing_key is not None
    assert directory is not None
    assert workspace_root is not None

    message_decisions = []
    for index, (source, assembled) in enumerate(
        zip(source_messages, assembled_messages)
    ):
        source_content = source.get("content")
        assembled_content = assembled.get("content")
        source_content_json = _canonical_json(source_content)
        assembled_content_json = _canonical_json(assembled_content)
        evidence_item = evidence.get(index, {})
        is_pinned = bool(evidence_item.get("evidence_pinned"))
        message_decisions.append(
            {
                "message_index": index,
                "role": source.get("role"),
                "action": (
                    "evidence_pinned"
                    if is_pinned
                    else "preserved"
                    if source_content_json == assembled_content_json
                    else "compressed"
                ),
                "source_content_hmac_sha256": _hmac_sha256(
                    signing_key, "source_message", source_content_json
                ),
                "assembled_content_hmac_sha256": _hmac_sha256(
                    signing_key, "assembled_message", assembled_content_json
                ),
                "source_chars": len(source_content) if isinstance(source_content, str) else None,
                "assembled_chars": (
                    len(assembled_content) if isinstance(assembled_content, str) else None
                ),
                "relevance_score": evidence_item.get("score"),
                "matched_query_term_count": len(
                    evidence_item.get("matched_terms", [])
                ),
                "allocated_tokens": evidence_item.get("allocated_tokens"),
                "pin_eligible": evidence_item.get("pin_eligible"),
                "security_flags": evidence_item.get("security_flags", []),
                "pin_blocked_reason": evidence_item.get("pin_blocked_reason"),
            }
        )
    acceptance_challenge_sha256 = request.get("receipt_commit_challenge_sha256")
    if not isinstance(acceptance_challenge_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", acceptance_challenge_sha256
    ):
        raise ValueError(
            "receipt_commit_challenge_sha256 must be a host-generated SHA-256 commitment"
        )
    proposal_id = "ocp_" + secrets.token_hex(16)
    tokens_saved = max(0, source_tokens - assembled_tokens)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "proposal_id": proposal_id,
        "session_id": str(request.get("session_id") or ""),
        "query_chars": len(query),
        "matched_query_term_storage": "count_only",
        "model": runtime_metadata["model"]["resolved"],
        "openclaw_runtime": runtime_metadata,
        "provider_mode": PROVIDER_MODE,
        "provider_independent": True,
        "budget_source": budget_source,
        "budget_authority": (
            "openclaw"
            if budget_source
            in {"openclaw_token_budget", "openclaw_runtime_settings"}
            else "operator"
        ),
        "token_budget": request.get("token_budget"),
        "source_tokens_estimated": source_tokens,
        "assembled_tokens_estimated": assembled_tokens,
        "tokens_saved_estimated": tokens_saved,
        "reduction_pct_estimated": round(
            (tokens_saved / source_tokens * 100.0) if source_tokens else 0.0, 2
        ),
        "source_message_count": len(source_messages),
        "assembled_message_count": len(assembled_messages),
        "source_hmac_sha256": source_digest,
        "assembled_hmac_sha256": assembled_digest,
        "changed": source_hash != assembled_hash,
        "message_decisions": message_decisions,
        "assembly_strategy": strategy,
        "evidence_pinned_count": sum(
            1 for item in evidence.values() if item.get("evidence_pinned")
        ),
        "evidence_pin_blocked_count": sum(
            1 for item in evidence.values() if item.get("pin_blocked_reason")
        ),
        "recovery_source": "openclaw_transcript_unmodified",
        "warnings": warnings,
        "local_only": True,
        "acceptance_actor": "openclaw_plugin",
        "signing_key_id": _signing_key_id(signing_key),
        "workspace_root_hmac_sha256": _hmac_sha256(
            signing_key,
            "workspace_root",
            str(workspace_root),
        ),
        "acceptance_challenge_sha256": acceptance_challenge_sha256,
        "acceptance_status": "proposed",
        "acceptance_signature": _EMPTY_ACCEPTANCE_SIGNATURE,
        "acceptance_commit_sha256": _EMPTY_ACCEPTANCE_SIGNATURE,
    }
    proposal_sha256 = _receipt_integrity_sha256(receipt)
    receipt["proposal_sha256"] = proposal_sha256
    configured_receipt_dir = isinstance(request.get("receipt_dir"), str) and bool(
        request["receipt_dir"].strip()
    )
    _ensure_private_receipt_directory(
        directory,
        repair_default_permissions=not configured_receipt_dir,
    )
    session_name = _safe_session_name(str(request.get("session_id") or "session"))
    destination = directory / f"{session_name}-{receipt_id}-{proposal_id}.json"
    serialized_bytes = len(
        (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    max_files = _positive_receipt_limit(
        request,
        "receipt_max_files",
        DEFAULT_RECEIPT_MAX_FILES,
        minimum=8,
        maximum=10000,
    )
    max_bytes = _positive_receipt_limit(
        request,
        "receipt_max_bytes",
        DEFAULT_RECEIPT_MAX_BYTES,
        minimum=1024 * 1024,
        maximum=1024 * 1024 * 1024,
    )
    with _receipt_store_lock(directory):
        if destination.exists():
            raise FileExistsError("receipt proposal identity already exists")
        _enforce_receipt_quota(
            directory,
            new_bytes=serialized_bytes,
            max_files=max_files,
            max_bytes=max_bytes,
        )
        _atomic_write_private_json(destination, receipt)
    return receipt_id, str(destination), True, proposal_id, proposal_sha256


def commit_receipt(request: dict[str, Any]) -> dict[str, Any]:
    """Idempotently validate one append-only receipt proposal."""
    receipt_id = request.get("receipt_id")
    proposal_id = request.get("proposal_id")
    proposal_sha256 = request.get("proposal_sha256")
    receipt_path = request.get("receipt_path")
    receipt_commit_token = request.get("receipt_commit_token")
    if not isinstance(receipt_id, str) or not re.fullmatch(r"ocr_[0-9a-f]{20}", receipt_id):
        raise ValueError("receipt_id must identify an OpenClaw receipt")
    if not isinstance(proposal_id, str) or not re.fullmatch(r"ocp_[0-9a-f]{32}", proposal_id):
        raise ValueError("proposal_id must identify one receipt proposal")
    if not isinstance(proposal_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", proposal_sha256
    ):
        raise ValueError("proposal_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        raise ValueError("receipt_path is required to recover the proposal")
    if not isinstance(receipt_commit_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", receipt_commit_token
    ):
        raise ValueError("receipt_commit_token must be a 256-bit lowercase hex secret")
    raw_destination = Path(receipt_path).expanduser().absolute()
    current = Path(raw_destination.anchor)
    for component in raw_destination.parts[1:]:
        current /= component
        if current.is_symlink():
            raise PermissionError("receipt proposal path must not contain symlinks")
    destination = raw_destination.resolve(strict=True)
    expected_suffix = f"-{receipt_id}-{proposal_id}.json"
    if not destination.name.endswith(expected_suffix):
        raise ValueError("receipt proposal path does not match its identity")
    if os.name != "nt" and (
        destination.parent.stat().st_mode & 0o077
        or destination.stat().st_mode & 0o077
    ):
        raise PermissionError(
            "receipt proposal or directory is not private; require 0700/0600 permissions"
        )
    workspace = request.get("workspace_dir")
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError(
            "workspace_dir is required to bind receipt acceptance to the OpenClaw workspace"
        )
    workspace_root = Path(workspace).expanduser().resolve()
    signing_key = _load_receipt_signing_key(
        destination.parent,
        workspace_root,
        create=False,
    )
    with _receipt_store_lock(destination.parent):
        receipt = json.loads(destination.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("receipt_id") != receipt_id
            or receipt.get("proposal_id") != proposal_id
            or receipt.get("proposal_sha256") != proposal_sha256
            or _receipt_integrity_sha256(receipt) != proposal_sha256
        ):
            raise ValueError("receipt proposal failed integrity validation")
        if receipt.get("signing_key_id") != _signing_key_id(signing_key):
            raise ValueError(
                "receipt proposal signing key changed; restore the original key before committing"
            )
        if receipt.get("workspace_root_hmac_sha256") != _hmac_sha256(
            signing_key,
            "workspace_root",
            str(workspace_root),
        ):
            raise PermissionError(
                "receipt proposal belongs to a different OpenClaw workspace"
            )
        _validate_acceptance_state(receipt, proposal_sha256, signing_key)
        expected_name = (
            f"{_safe_session_name(str(receipt.get('session_id') or 'session'))}"
            f"-{receipt_id}-{proposal_id}.json"
        )
        if destination.name != expected_name:
            raise ValueError("receipt proposal filename failed integrity validation")
        if _sha256_text(receipt_commit_token) != receipt.get(
            "acceptance_challenge_sha256"
        ):
            raise ValueError("receipt commit token does not match the host commitment")
        acceptance_commit_sha256 = _acceptance_commit_sha256(
            proposal_sha256, receipt_commit_token
        )
        if receipt.get("acceptance_status") != "accepted":
            receipt["acceptance_status"] = "accepted"
            receipt["acceptance_commit_sha256"] = acceptance_commit_sha256
            receipt["acceptance_signature"] = _acceptance_signature(
                signing_key,
                receipt,
                proposal_sha256,
                acceptance_commit_sha256,
            )
            _atomic_write_private_json(destination, receipt)
        elif receipt.get("acceptance_commit_sha256") != acceptance_commit_sha256:
            raise ValueError("receipt commit is not idempotent for this host proof")
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "receipt_id": receipt_id,
        "proposal_id": proposal_id,
        "proposal_sha256": proposal_sha256,
        "receipt_path": str(destination),
        "acceptance_commit_sha256": acceptance_commit_sha256,
        "committed": True,
    }


def assemble(request: dict[str, Any]) -> dict[str, Any]:
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, dict) for message in raw_messages
    ):
        raise ValueError("messages must be a list of objects")

    messages: list[dict[str, Any]] = copy.deepcopy(raw_messages)
    budget = request.get("token_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("token_budget must be a positive integer")
    preserve_last_n = request.get("preserve_last_n", DEFAULT_PRESERVE_LAST_N)
    if (
        isinstance(preserve_last_n, bool)
        or not isinstance(preserve_last_n, int)
        or preserve_last_n < 0
    ):
        raise ValueError("preserve_last_n must be a non-negative integer")
    budget_source = request.get("budget_source")
    if budget_source is None:
        budget_source = "openclaw_token_budget"
    if budget_source not in _BUDGET_SOURCES:
        raise ValueError("budget_source is not recognized")
    runtime_metadata = _runtime_audit_metadata(request)

    source_tokens = estimate_messages_tokens(messages)
    evidence_pinning = request.get("evidence_pinning", True) is not False
    query = _query_for_request(request, messages) if evidence_pinning else ""
    strategy = (
        "query_aware_evidence_pinning"
        if evidence_pinning
        else "uniform_budget_compression"
    )
    warnings = [
        "Token counts are deterministic estimates, not provider-billed usage."
    ]
    if budget_source == "operator_fallback":
        warnings.append(
            "OpenClaw did not provide a finite prompt budget; Entroly used the "
            "operator-configured fallbackTokenBudget."
        )
    evidence: dict[int, dict[str, Any]] = {}
    protected_indexes = {
        index for index, message in enumerate(messages) if _protected_message(message)
    }
    if preserve_last_n:
        protected_indexes.update(range(max(0, len(messages) - preserve_last_n), len(messages)))
    protected_tokens = estimate_messages_tokens(
        [messages[index] for index in sorted(protected_indexes)]
    )

    if source_tokens <= budget:
        assembled = messages
    elif protected_tokens >= budget:
        assembled = messages
        warnings.append(
            "Protected system, structured, and recent messages exceed the token budget; "
            "Entroly returned the exact original context."
        )
    else:
        compressible_indexes = [
            index for index in range(len(messages)) if index not in protected_indexes
        ]
        compressible = [messages[index] for index in compressible_indexes]
        fixed_tokens = sum(_fixed_message_tokens(message) for message in compressible)
        content_budget = budget - protected_tokens - fixed_tokens
        if content_budget <= 0:
            assembled = messages
            warnings.append(
                "Protected messages and message metadata exceed the token budget; "
                "Entroly returned the exact original context."
            )
        else:
            compressed, pinned, relevance, distillation_failures = _compress_message_bodies(
                compressible,
                content_budget=content_budget,
                distill=bool(request.get("distill", True)),
                query=query,
            )
            assembled = copy.deepcopy(messages)
            for local_index, (index, compressed_message) in enumerate(
                zip(compressible_indexes, compressed, strict=True)
            ):
                assembled[index] = compressed_message
                evidence[index] = {
                    **relevance[local_index],
                    "evidence_pinned": local_index in pinned,
                }
            if distillation_failures:
                warnings.append(
                    "Assistant distillation was unavailable for "
                    f"{distillation_failures} message(s); Entroly compressed their "
                    "original text instead."
                )

    try:
        _validate_assembly_invariants(messages, assembled, protected_indexes)
    except ValueError:
        assembled = messages
        evidence = {}
        warnings.append(
            "An internal preservation invariant rejected the assembled context; "
            "Entroly returned the exact original context."
        )

    assembled_tokens = estimate_messages_tokens(assembled)
    if assembled_tokens > budget and assembled != messages:
        assembled = messages
        assembled_tokens = source_tokens
        evidence = {}
        warnings.append(
            "The minimum safe structured context could not fit the token budget; "
            "Entroly returned the exact original context for OpenClaw recovery."
        )
    blocked_count = sum(
        1 for item in evidence.values() if item.get("pin_blocked_reason")
    )
    if blocked_count:
        warnings.append(
            f"Context firewall blocked verbatim evidence pinning for {blocked_count} "
            "message(s); those messages remained subject to normal compression."
        )
    (
        receipt_id,
        receipt_path,
        receipt_commit_required,
        proposal_id,
        proposal_sha256,
    ) = _write_receipt(
        request,
        messages,
        assembled,
        source_tokens=source_tokens,
        assembled_tokens=assembled_tokens,
        warnings=warnings,
        evidence=evidence,
        strategy=strategy,
        query=query,
        budget_source=budget_source,
        runtime_metadata=runtime_metadata,
    )
    pinned_indexes = sorted(
        index for index, item in evidence.items() if item.get("evidence_pinned")
    )
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "messages": assembled,
        "estimated_tokens": assembled_tokens,
        "source_tokens": source_tokens,
        "tokens_saved": max(0, source_tokens - assembled_tokens),
        "changed": assembled != messages,
        "receipt_id": receipt_id,
        "receipt_path": receipt_path,
        "receipt_commit_required": receipt_commit_required,
        "proposal_id": proposal_id,
        "proposal_sha256": proposal_sha256,
        "provider_mode": PROVIDER_MODE,
        "provider_independent": True,
        "budget_source": budget_source,
        "model": runtime_metadata["model"]["resolved"],
        "provider_hint": runtime_metadata["model"]["provider"],
        "assembly_strategy": strategy,
        "evidence_pinned": len(pinned_indexes),
        "evidence_pin_blocked": blocked_count,
        "pinned_message_indexes": pinned_indexes,
        "warnings": warnings,
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    if operation == "health":
        workspace = request.get("workspace_dir")
        workspace_root = (
            Path(workspace).expanduser().resolve()
            if isinstance(workspace, str) and workspace.strip()
            else None
        )
        key_path = _receipt_signing_key_path()
        initialize_key = request.get("write_receipt", True) is not False
        if workspace_root is not None and initialize_key:
            receipt_directory = _receipt_directory(request)
            _load_receipt_signing_key(receipt_directory, workspace_root)
            receipt_key_status = "ready"
        elif key_path.exists():
            forbidden_roots = (
                (_receipt_directory(request), workspace_root)
                if workspace_root is not None
                else ()
            )
            _load_receipt_signing_key(*forbidden_roots, create=False)
            receipt_key_status = "ready"
        else:
            receipt_key_status = "uninitialized"
        return {
            "schema_version": BRIDGE_SCHEMA,
            "ok": True,
            "status": "ready",
            "receipt_key_status": receipt_key_status,
            "provider_mode": PROVIDER_MODE,
            "provider_independent": True,
            "requires_host_token_budget": True,
            "receipt_commit_protocol": "two_phase",
        }
    if operation == "assemble":
        return assemble(request)
    if operation == "commit_receipt":
        return commit_receipt(request)
    raise ValueError(f"unsupported operation: {operation!r}")


def serve(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> int:
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            request_id = payload.get("request_id")
            response = handle_request(payload)
        except Exception as error:
            response = {
                "schema_version": BRIDGE_SCHEMA,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        response["request_id"] = request_id
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="store_true", help="serve newline-delimited JSON")
    args = parser.parse_args(argv)
    if not args.jsonl:
        parser.error("--jsonl is required")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
