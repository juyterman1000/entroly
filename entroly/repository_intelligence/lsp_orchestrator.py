"""Bounded local Language Server Protocol orchestration.

The language server is an explicitly configured external process. Entroly
controls framing, bounds, environment exposure, and workspace URI acceptance;
it cannot prove or enforce what network activity that executable performs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import unquote, urlparse

from .models import RepositoryIndex, Symbol

LSP_ORCHESTRATION_SCHEMA_VERSION = "entroly.lsp-orchestration.v1"
_ENV_ALLOWLIST = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
    "LOCALAPPDATA", "APPDATA", "USERPROFILE", "HOME", "LANG", "LC_ALL",
})


def _resolve_symbol(index: RepositoryIndex, query: str) -> Symbol:
    lowered = query.strip().lower()
    matches = sorted(
        (
            symbol
            for symbol in index.symbols.values()
            if symbol.symbol_id.lower() == lowered
            or symbol.qualified_name.lower() == lowered
            or symbol.name.lower() == lowered
        ),
        key=lambda symbol: symbol.symbol_id,
    )
    if len(matches) != 1:
        status = "ambiguous" if matches else "not-found"
        raise ValueError(f"LSP symbol query is {status}")
    return matches[0]


def _utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _definition_position(root: Path, index: RepositoryIndex, symbol: Symbol) -> tuple[dict[str, int], dict[str, object]]:
    record = index.files[symbol.path]
    candidate = (root / symbol.path).resolve(strict=True)
    candidate.relative_to(root)
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        raise ValueError("LSP source is stale")
    signature = symbol.signature.encode("utf-8", errors="surrogateescape")
    signature_start = raw.find(signature, symbol.start_byte, symbol.end_byte)
    if signature_start < 0:
        raise ValueError("LSP definition signature is unverifiable")
    name = symbol.name.encode("utf-8")
    start = signature_start
    while True:
        start = raw.find(name, start, signature_start + len(signature))
        if start < 0:
            raise ValueError("LSP definition identifier is unverifiable")
        end = start + len(name)
        left = raw[start - 1:start] if start else b""
        right = raw[end:end + 1]
        if not left.isalnum() and left not in {b"_", b"$"} and not right.isalnum() and right not in {b"_", b"$"}:
            break
        start += 1
    line = raw[:start].count(b"\n")
    line_start = raw.rfind(b"\n", 0, start) + 1
    prefix = raw[line_start:start].decode("utf-8", errors="surrogateescape")
    start_character = _utf16_units(prefix)
    end_character = start_character + _utf16_units(symbol.name)
    position = {"line": line, "character": start_character}
    location = {
        "path": symbol.path,
        "line": line,
        "start_character": start_character,
        "end_character": end_character,
    }
    return position, location


def _frame(message: Mapping[str, object]) -> bytes:
    body = json.dumps(
        dict(message), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


class _FramedReader(threading.Thread):
    def __init__(self, stream, *, max_messages: int, max_output_bytes: int) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.max_messages = max_messages
        self.max_output_bytes = max_output_bytes
        self.messages: queue.Queue[dict[str, object] | BaseException | None] = queue.Queue()

    def run(self) -> None:
        total = 0
        count = 0
        try:
            while count < self.max_messages:
                headers: dict[str, str] = {}
                while True:
                    line = self.stream.readline()
                    if not line:
                        self.messages.put(None)
                        return
                    total += len(line)
                    if total > self.max_output_bytes:
                        raise ValueError("LSP output limit exceeded")
                    if line in {b"\r\n", b"\n"}:
                        break
                    name, separator, value = line.decode("ascii", errors="strict").partition(":")
                    if not separator:
                        raise ValueError("invalid LSP header")
                    headers[name.strip().lower()] = value.strip()
                length = int(headers.get("content-length", "-1"))
                if length < 0:
                    raise ValueError("invalid LSP content length")
                if length > self.max_output_bytes - total:
                    raise ValueError("LSP output limit exceeded")
                body = self.stream.read(length)
                if len(body) != length:
                    raise ValueError("truncated LSP message")
                total += length
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("LSP message must be an object")
                self.messages.put(payload)
                count += 1
            raise ValueError("LSP message limit exceeded")
        except BaseException as exc:
            self.messages.put(exc)


class _FramedWriter(threading.Thread):
    """Keep pipe backpressure from escaping the orchestration deadline."""

    def __init__(self, stream) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.messages: queue.Queue[
            tuple[bytes, threading.Event, list[BaseException]] | None
        ] = queue.Queue()

    def run(self) -> None:
        while True:
            item = self.messages.get()
            if item is None:
                return
            frame, completed, errors = item
            try:
                self.stream.write(frame)
                self.stream.flush()
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

    def stop(self) -> None:
        self.messages.put(None)


class _StderrReader(threading.Thread):
    def __init__(self, stream, max_bytes: int = 64 * 1024) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.max_bytes = max_bytes
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def run(self) -> None:
        while self.bytes_read < self.max_bytes:
            chunk = self.stream.read(min(4096, self.max_bytes - self.bytes_read))
            if not chunk:
                return
            self.digest.update(chunk)
            self.bytes_read += len(chunk)


def _workspace_path(root: Path, uri: object) -> str | None:
    if not isinstance(uri, str):
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    raw_path = unquote(parsed.path)
    if os.name == "nt" and len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]
    try:
        candidate = Path(raw_path).resolve(strict=True)
        relative = candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return relative.as_posix()


def _location(value: object, root: Path) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    uri = value.get("uri", value.get("targetUri"))
    raw_range = value.get("range", value.get("targetRange"))
    path = _workspace_path(root, uri)
    if path is None or not isinstance(raw_range, Mapping):
        return None
    start = raw_range.get("start")
    end = raw_range.get("end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        return None
    try:
        line = int(start["line"])
        start_character = int(start["character"])
        end_line = int(end["line"])
        end_character = int(end["character"])
    except (KeyError, TypeError, ValueError):
        return None
    if line < 0 or end_line != line or end_character <= start_character:
        return None
    return {
        "path": path,
        "line": line,
        "start_character": start_character,
        "end_character": end_character,
    }


def collect_lsp_references(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    *,
    command: Sequence[str],
    language_id: str,
    timeout_seconds: float = 15.0,
    max_relationships: int = 10_000,
    max_messages: int = 10_000,
    max_output_bytes: int = 32 * 1024 * 1024,
) -> dict[str, object]:
    """Launch one configured server and collect bounded workspace references."""
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError("LSP command must be an argument array")
    arguments = tuple(str(item) for item in command)
    if not arguments or len(arguments) > 32 or any(not item or len(item) > 4096 for item in arguments):
        raise ValueError("LSP command must contain 1 to 32 bounded arguments")
    executable = shutil.which(arguments[0])
    if executable is None:
        raise ValueError("configured LSP executable was not found")
    timeout = max(1.0, min(float(timeout_seconds), 30.0))
    relationship_limit = max(1, min(int(max_relationships), 100_000))
    message_limit = max(10, min(int(max_messages), 100_000))
    output_limit = max(1024, min(int(max_output_bytes), 128 * 1024 * 1024))
    symbol = _resolve_symbol(index, symbol_query)
    position, target_location = _definition_position(root, index, symbol)
    source_path = (root / symbol.path).resolve(strict=True)
    source_text = source_path.read_text(encoding="utf-8", errors="surrogateescape")
    env = {key: value for key, value in os.environ.items() if key.upper() in _ENV_ALLOWLIST}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        (executable, *arguments[1:]),
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=env,
        creationflags=creationflags,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    reader = _FramedReader(
        process.stdout, max_messages=message_limit, max_output_bytes=output_limit
    )
    writer = _FramedWriter(process.stdin)
    stderr_reader = _StderrReader(process.stderr)
    reader.start()
    writer.start()
    stderr_reader.start()
    request_id = 0
    received = 0
    pending: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout

    def send(message: Mapping[str, object]) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("LSP operation timed out while writing")
        completed = threading.Event()
        errors: list[BaseException] = []
        writer.messages.put((_frame(message), completed, errors))
        if not completed.wait(timeout=remaining):
            raise ValueError("LSP operation timed out while writing")
        if errors:
            raise ValueError("LSP process rejected protocol input") from None

    def request(method: str, params: Mapping[str, object]) -> object:
        nonlocal request_id, received
        request_id += 1
        current_id = request_id
        send({"jsonrpc": "2.0", "id": current_id, "method": method, "params": dict(params)})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(f"LSP request timed out: {method}")
            try:
                message = reader.messages.get(timeout=remaining)
            except queue.Empty:
                raise ValueError(f"LSP request timed out: {method}") from None
            if message is None:
                raise ValueError("LSP process closed its output")
            if isinstance(message, BaseException):
                raise ValueError(str(message)) from None
            received += 1
            if message.get("id") == current_id:
                if "error" in message:
                    raise ValueError(f"LSP request failed: {method}")
                return message.get("result")
            if "id" in message and isinstance(message.get("method"), str):
                server_method = str(message["method"])
                if server_method == "workspace/configuration":
                    raw_items = message.get("params", {})
                    items = raw_items.get("items", []) if isinstance(raw_items, Mapping) else []
                    result: object = [None] * len(items) if isinstance(items, list) else []
                    send({"jsonrpc": "2.0", "id": message["id"], "result": result})
                elif server_method in {
                    "client/registerCapability", "client/unregisterCapability",
                    "window/workDoneProgress/create",
                }:
                    send({"jsonrpc": "2.0", "id": message["id"], "result": None})
                else:
                    send({
                        "jsonrpc": "2.0", "id": message["id"],
                        "error": {"code": -32601, "message": "method not supported by bounded client"},
                    })
                continue
            pending.append(message)

    relationships: list[dict[str, object]] = []
    omissions: dict[str, int] = {}
    initialized = False
    operation_succeeded = False
    try:
        initialize_result = request("initialize", {
            "processId": None,
            "rootUri": root.as_uri(),
            "capabilities": {"general": {"positionEncodings": ["utf-16"]}},
            "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
        })
        capabilities = (
            initialize_result.get("capabilities", {})
            if isinstance(initialize_result, Mapping)
            else {}
        )
        selected_encoding = (
            capabilities.get("positionEncoding")
            if isinstance(capabilities, Mapping)
            else None
        )
        if selected_encoding not in {None, "utf-16"}:
            raise ValueError("LSP server did not select UTF-16 positions")
        initialized = True
        send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        send({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {"textDocument": {
                "uri": source_path.as_uri(),
                "languageId": str(language_id).strip()[:64],
                "version": 1,
                "text": source_text,
            }},
        })
        result = request("textDocument/references", {
            "textDocument": {"uri": source_path.as_uri()},
            "position": position,
            "context": {"includeDeclaration": True},
        })
        locations = result if isinstance(result, list) else []
        for value in locations:
            if len(relationships) >= relationship_limit:
                omissions["relationship-limit"] = omissions.get("relationship-limit", 0) + 1
                break
            source_location = _location(value, root)
            if source_location is None:
                omissions["invalid-or-outside-workspace-location"] = (
                    omissions.get("invalid-or-outside-workspace-location", 0) + 1
                )
                continue
            relationships.append({
                "kind": "reference",
                "source": source_location,
                "target": target_location,
            })
        operation_succeeded = True
    finally:
        if process.poll() is None:
            if operation_succeeded and initialized:
                try:
                    request("shutdown", {})
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    send({"jsonrpc": "2.0", "method": "exit", "params": {}})
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=min(5.0, timeout))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            else:
                process.kill()
                process.wait(timeout=5)
        writer.stop()
        try:
            process.stdin.close()
        except OSError:
            pass
        writer.join(timeout=1)
        reader.join(timeout=1)
        stderr_reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass

    return {
        "schema_version": LSP_ORCHESTRATION_SCHEMA_VERSION,
        "provider": f"lsp:{Path(executable).name}",
        "symbol_id": symbol.symbol_id,
        "language_id": str(language_id).strip()[:64],
        "position_encoding": "utf-16",
        "relationships": relationships,
        "omissions_by_reason": dict(sorted(omissions.items())),
        "process": {
            "exit_code": process.returncode,
            "messages_received": received,
            "ignored_messages": len(pending),
            "stderr_bytes_hashed": stderr_reader.bytes_read,
            "stderr_sha256": stderr_reader.digest.hexdigest(),
            "environment": "allowlisted-non-secret-operational-variables",
            "shell": False,
            "network_control": "not-enforced-external-process",
        },
        "bounds": {
            "total_deadline_seconds": timeout,
            "max_relationships": relationship_limit,
            "max_messages": message_limit,
            "max_output_bytes": output_limit,
        },
        "remote_calls_by_entroly": 0,
    }


def build_committed_lsp_rename_preview(
    orchestration: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "entroly.lsp-rename-preview.v1",
        "orchestration": dict(orchestration),
        "plan": dict(plan),
        "receipt": {
            "remote_calls_by_entroly": 0,
            "external_process_network_control": "not-enforced",
            "writes_performed": 0,
            "commitment_scope": "payload-excluding-generation-command-and-lsp-preview-sha256",
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["receipt"]["lsp_preview_sha256"] = hashlib.sha256(canonical).hexdigest()  # type: ignore[index]
    return payload


def verify_lsp_rename_preview_commitment(payload: Mapping[str, object]) -> bool:
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("lsp_preview_sha256"))
        canonical = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "LSP_ORCHESTRATION_SCHEMA_VERSION",
    "build_committed_lsp_rename_preview",
    "collect_lsp_references",
    "verify_lsp_rename_preview_commitment",
]
