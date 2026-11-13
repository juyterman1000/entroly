"""High-efficiency Evidence-Locked Compression proxy surface.

This module is the product-facing bridge between Entroly's compression
algorithms and LLM API payloads. It is intentionally provider-light: callers can
use it from the existing proxy, SDKs, MCP tools, or benchmarks without starting
an HTTP server.

Design principle:

    Compress aggressively around evidence, never through evidence.

The proxy surface preserves normal user/assistant text by default and compresses
heavy tool/function payloads using Evidence-Locked Compression receipts. When a
retrieval store is supplied, omitted spans are saved locally and can be fetched
back by receipt/span id.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .compression_retrieval_store_secure import CompressionRetrievalStore
from .evidence_locked_compression import compress_evidence_locked, estimate_tokens

_QUERY_WORD_RE = re.compile(r"\b[\w.-]{3,}\b")
_QUERY_STOPWORDS = {
    "about",
    "after",
    "again",
    "agent",
    "agents",
    "build",
    "built",
    "cause",
    "caused",
    "check",
    "code",
    "could",
    "debug",
    "did",
    "does",
    "error",
    "errors",
    "fail",
    "failed",
    "fails",
    "failure",
    "find",
    "fix",
    "from",
    "happen",
    "happened",
    "issue",
    "log",
    "logs",
    "output",
    "please",
    "proxy",
    "result",
    "results",
    "show",
    "test",
    "tests",
    "tool",
    "tools",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


@dataclass(slots=True)
class ProxyCompressionReceipt:
    provider: str
    mode: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    savings_ratio: float
    compressed_blocks: int
    receipts: list[dict[str, object]] = field(default_factory=list)
    protected_blocks: int = 0
    deduped_blocks: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProxyCompressionResult:
    body: dict[str, Any]
    receipt: ProxyCompressionReceipt
    changed: bool

    def headers(self) -> dict[str, str]:
        """Headers suitable for an HTTP proxy response/debug surface."""
        return {
            "x-entroly-compression-mode": self.receipt.mode,
            "x-entroly-original-tokens": str(self.receipt.original_tokens),
            "x-entroly-compressed-tokens": str(self.receipt.compressed_tokens),
            "x-entroly-tokens-saved": str(self.receipt.tokens_saved),
            "x-entroly-savings-ratio": f"{self.receipt.savings_ratio:.4f}",
            "x-entroly-compressed-blocks": str(self.receipt.compressed_blocks),
            "x-entroly-protected-blocks": str(self.receipt.protected_blocks),
            "x-entroly-deduped-blocks": str(self.receipt.deduped_blocks),
        }


# ── which tool results the lossy codec may touch ─────────────────────────

# Tools whose result *is* the file the model asked to read. Matched on the
# last segment of the name, so ``mcp__filesystem__read_file`` and
# ``functions.read_file`` are both ``read_file``.
FILE_READ_TOOLS = frozenset(
    {
        "read",
        "read_file",
        "readfile",
        "read_text_file",
        "read_files",
        "read_many_files",
        "read_multiple_files",
        "file_read",
        "fs_read",
        "view",
        "view_file",
        "open_file",
        "cat",
        "get_file_contents",
        "notebookread",
        "read_notebook",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "text_editor",
    }
)

_NAME_SEPARATORS_RE = re.compile(r"__|[./:]")
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)(?:\t|\u2192|\s*[|\u2502]\s?)(.*)$")
_JSON_KEY_LINE_RE = re.compile(r'^\s*"[^"]*"\s*:')
_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:async\s+)?def\s+\w|class\s+\w|import\s+\w|from\s+[\w.]+\s+import\b|@\w[\w.]*"
    r"|return\b|(?:el)?if\b.*:\s*$|else:\s*$|for\b.*\bin\b.*:\s*$|while\b.*:\s*$"
    r"|try:\s*$|except\b.*:\s*$|finally:\s*$|with\b.*:\s*$"
    r"|(?:pub(?:\([\w:]+\))?\s+)?(?:fn|struct|enum|impl|trait|mod|use)\b"
    r"|(?:export\s+)?(?:default\s+)?(?:const|let|var|function|interface|type)\s+\w"
    r"|(?:public|private|protected|static|final)\s+\w|func\s+\w|package\s+\w"
    r"|#include\b|namespace\s+\w"
    r")"
)
_STATEMENT_END_RE = re.compile(r"[;{}]\s*$")
_SOURCE_SAMPLE_LINES = 400


def tool_basename(name: str) -> str:
    """Last segment of a tool name, lowercased."""
    cleaned = str(name or "").strip().lower()
    if not cleaned:
        return ""
    return _NAME_SEPARATORS_RE.split(cleaned)[-1]


def looks_like_source(text: str) -> bool:
    """True when ``text`` reads as a file view or as program source.

    The line codec keeps lines that look like evidence -- errors, paths,
    identifiers, outliers -- and drops the rest. That is the right trade for a
    log, where most lines repeat. It is the wrong one for source, where the
    dropped lines are the body of the function the model asked to see:
    measured on ten file reads from one agent session, 36% of the non-blank
    lines reached the model.

    Two shapes count. A numbered listing (``cat -n``, most Read tools) is a
    file view whatever the file holds, unless the file is JSON data, which the
    structured codec summarises without losing its outliers. Otherwise the
    text is source when enough of its lines parse as statements. A wrong
    "yes" forwards a block uncompressed; a wrong "no" is the old behaviour.
    """
    lines = [ln for ln in text.splitlines()[:_SOURCE_SAMPLE_LINES] if ln.strip()]
    if len(lines) < 5:
        return False

    numbered = [m for m in (_NUMBERED_LINE_RE.match(ln) for ln in lines) if m]
    if len(numbered) >= 5 and len(numbered) >= 0.8 * len(lines):
        numbers = [int(m.group(1)) for m in numbered]
        ordered = sum(1 for a, b in zip(numbers, numbers[1:]) if b >= a)
        if ordered >= 0.9 * (len(numbers) - 1):
            if numbered[0].group(2).lstrip()[:1] in ("[", "{"):
                matches = (_NUMBERED_LINE_RE.match(ln) for ln in text.splitlines())
                body = "\n".join(m.group(2) for m in matches if m)
                try:
                    json.loads(body)
                    return False
                except Exception:  # noqa: BLE001 - not JSON, so a file view
                    pass
            return True

    stripped = text.strip()
    if stripped[:1] in "[{":
        try:
            json.loads(stripped)
            return False
        except Exception:  # noqa: BLE001
            pass
    code = sum(
        1 for ln in lines if _CODE_LINE_RE.match(ln) or _STATEMENT_END_RE.search(ln)
    )
    data = sum(1 for ln in lines if _JSON_KEY_LINE_RE.match(ln))
    if data >= code:
        return False
    return code >= 10 and code >= 0.25 * len(lines)


@dataclass(frozen=True, slots=True)
class ToolResultPolicy:
    """What the proxy may do to a tool result before forwarding it.

    ``protected_tools`` are never lossy-compressed. ``protect_file_reads``
    adds :data:`FILE_READ_TOOLS` and anything :func:`looks_like_source`
    recognises. ``dedupe_repeats`` replaces a tool result that is
    byte-identical to an earlier one in the same request with one line that
    points at it; nothing is lost, because the earlier copy is still there.
    """

    protected_tools: frozenset[str] = frozenset()
    protect_file_reads: bool = True
    dedupe_repeats: bool = True
    dedupe_min_chars: int = 1024
    # A rewrite must remove at least this share of a block. A block the
    # provider has not seen is billed as input if left alone and as a cache
    # write once rewritten; a 5-minute write costs 1.25x input, so a rewrite
    # that keeps more than 80% of a block is a loss on the turn it is made
    # (r_in * N < r_write * 0.8 N), and it drops lines to get there.
    min_savings: float = 0.20

    @classmethod
    def from_names(
        cls,
        protected_tools: str | Iterable[str] | None = None,
        *,
        protect_file_reads: bool = True,
        dedupe_repeats: bool = True,
    ) -> "ToolResultPolicy":
        if isinstance(protected_tools, str):
            names: Iterable[str] = protected_tools.split(",")
        else:
            names = protected_tools or ()
        return cls(
            protected_tools=frozenset(
                tool_basename(name) for name in names if tool_basename(name)
            ),
            protect_file_reads=protect_file_reads,
            dedupe_repeats=dedupe_repeats,
        )

    def protection(self, tool_name: str, text: str) -> str:
        """Why this result must be forwarded as sent, or ``""``."""
        base = tool_basename(tool_name)
        if base and base in self.protected_tools:
            return "excluded-tool"
        if not self.protect_file_reads:
            return ""
        if base and base in FILE_READ_TOOLS:
            return "file-read"
        if looks_like_source(text):
            return "source"
        return ""


@dataclass(slots=True)
class _ToolResultState:
    """Per-request bookkeeping for the policy; holds digests, never content."""

    policy: ToolResultPolicy | None
    call_names: dict[str, str] = field(default_factory=dict)
    seen: dict[str, tuple[str, str]] = field(default_factory=dict)
    protected_blocks: int = 0
    deduped_blocks: int = 0

    @property
    def min_savings(self) -> float | None:
        return None if self.policy is None else self.policy.min_savings


def _call_names(*sequences: Any) -> dict[str, str]:
    """Map a tool call id to the name of the tool that was called.

    A result names only the call it answers; the tool's name is on the
    assistant turn that made the call (Anthropic ``tool_use``, OpenAI
    ``tool_calls``, Responses ``function_call``).
    """
    names: dict[str, str] = {}
    for sequence in sequences:
        if not isinstance(sequence, list):
            continue
        for item in sequence:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"function_call", "custom_tool_call"}:
                call_id, name = item.get("call_id"), item.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    names[call_id] = name
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and isinstance(block.get("id"), str)
                        and isinstance(block.get("name"), str)
                    ):
                        names[block["id"]] = block["name"]
            calls = item.get("tool_calls")
            for call in calls if isinstance(calls, list) else []:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                fn = call.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    names[call["id"]] = fn["name"]
    return names


def _repeat_pointer(first_ref: str, tool_name: str, text: str) -> str:
    """One deterministic line standing in for a repeated tool result.

    Deterministic in its inputs, so the same request always forwards the same
    bytes and the provider's prefix cache keeps hitting.
    """
    lines = len(text.splitlines()) or 1
    who = f"the earlier {tool_name} result" if tool_name else "an earlier tool result"
    ref = f" (call {first_ref})" if first_ref else ""
    return (
        f"[entroly: byte-identical to {who}{ref} in this conversation; "
        f"{lines:,} lines not repeated]"
    )


def _apply_tool_result_policy(
    text: str,
    *,
    tool_name: str,
    call_ref: str,
    state: _ToolResultState,
    budget_tokens: int,
) -> tuple[str, str, dict[str, object] | None]:
    """Decide a tool result's fate before the lossy codec sees it.

    Returns ``(text, verdict, receipt)`` where verdict is ``"deduped"`` (text
    is the pointer), ``"protected"`` (forward as sent) or ``""`` (eligible).
    """
    policy = state.policy
    if policy is None:
        return text, "", None
    if policy.dedupe_repeats and len(text) >= policy.dedupe_min_chars:
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
        first = state.seen.get(digest)
        if first is None:
            state.seen[digest] = (call_ref, tool_name)
        else:
            pointer = _repeat_pointer(first[0], first[1] or tool_name, text)
            if len(pointer) < len(text):
                original = estimate_tokens(text)
                compressed = estimate_tokens(pointer)
                state.deduped_blocks += 1
                return (
                    pointer,
                    "deduped",
                    {
                        "content_type": "duplicate",
                        "original_tokens": original,
                        "compressed_tokens": compressed,
                        "tokens_saved": max(0, original - compressed),
                        "duplicate_of": first[0],
                    },
                )
    # Below the budget the codec leaves a block alone anyway; only a block it
    # would have cut is worth classifying, or worth counting as protected.
    if estimate_tokens(text) <= max(128, budget_tokens):
        return text, "", None
    if policy.protection(tool_name, text):
        state.protected_blocks += 1
        return text, "protected", None
    return text, "", None


def compress_proxy_payload_from_env(
    body: dict[str, Any],
    *,
    provider: str = "openai",
    query: str = "",
) -> ProxyCompressionResult:
    """Compress using environment-driven live-proxy settings.

    Intended for the HTTP proxy path. Defaults are safe: mode is off unless
    ``ENTROLY_COMPRESSION_PROXY_MODE=elc`` is set.
    """
    mode = os.environ.get("ENTROLY_COMPRESSION_PROXY_MODE", "off").strip().lower()
    budget = _env_budget_tokens()
    compress_user = os.environ.get("ENTROLY_ELC_COMPRESS_USER", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    store_path = os.environ.get("ENTROLY_COMPRESSION_STORE")
    store = CompressionRetrievalStore(Path(store_path)) if store_path else None
    return compress_proxy_payload(
        body,
        provider=provider,
        query=query,
        budget_tokens=budget,
        mode="elc" if mode == "elc" else "off",
        compress_user_messages=compress_user,
        retrieval_store=store,
    )


def _env_budget_tokens(default: int = 1200) -> int:
    raw = os.environ.get("ENTROLY_ELC_BUDGET_TOKENS")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def compress_proxy_payload(
    body: dict[str, Any],
    *,
    provider: str = "openai",
    query: str = "",
    budget_tokens: int = 1200,
    mode: str = "elc",
    include_receipt_header: bool = True,
    compress_user_messages: bool = False,
    retrieval_store: CompressionRetrievalStore | None = None,
    policy: ToolResultPolicy | None = None,
) -> ProxyCompressionResult:
    """Compress heavy payload blocks before they reach an LLM provider.

    Supported shapes:
    - OpenAI chat/completions: ``messages`` with string content or text blocks.
    - Anthropic messages: ``messages`` with list content and ``tool_result`` blocks.
    - OpenAI Responses-style ``input`` arrays with message/content blocks.

    By default, only tool/function/tool_result payloads are compressed. User
    messages are preserved unless ``compress_user_messages=True`` because the
    user's latest request is usually the semantic target, not background noise.

    ``policy`` decides which tool results the lossy codec may not touch and
    whether byte-identical repeats collapse to a pointer. ``None`` keeps the
    behaviour of earlier releases: every tool result is eligible and repeats
    are compressed independently.
    """
    if mode not in {"elc", "off"}:
        raise ValueError("mode must be 'elc' or 'off'")
    if mode == "off":
        tokens = _estimate_body_tokens(body)
        receipt = ProxyCompressionReceipt(provider, mode, tokens, tokens, 0, 0.0, 0, [])
        return ProxyCompressionResult(dict(body), receipt, changed=False)

    original_tokens = _estimate_body_tokens(body)
    new_body = dict(body)
    receipts: list[dict[str, object]] = []
    compressed_blocks = 0
    state = _ToolResultState(
        policy=policy,
        call_names=_call_names(body.get("messages"), body.get("input"))
        if policy is not None
        else {},
    )

    if isinstance(body.get("messages"), list):
        messages, changed_count, block_receipts = _compress_messages(
            body["messages"],
            query=query,
            budget_tokens=budget_tokens,
            include_receipt_header=include_receipt_header,
            compress_user_messages=compress_user_messages,
            retrieval_store=retrieval_store,
            state=state,
        )
        if changed_count:
            new_body["messages"] = messages
            compressed_blocks += changed_count
            receipts.extend(block_receipts)

    if "input" in body and isinstance(body.get("input"), list):
        new_input, changed_count, block_receipts = _compress_input_items(
            body["input"],
            query=query,
            budget_tokens=budget_tokens,
            include_receipt_header=include_receipt_header,
            compress_user_messages=compress_user_messages,
            retrieval_store=retrieval_store,
            state=state,
        )
        if changed_count:
            new_body["input"] = new_input
            compressed_blocks += changed_count
            receipts.extend(block_receipts)

    if "contents" in body and isinstance(body.get("contents"), list):
        new_contents, changed_count, block_receipts = _compress_gemini_contents(
            body["contents"],
            query=query,
            budget_tokens=budget_tokens,
            include_receipt_header=include_receipt_header,
            retrieval_store=retrieval_store,
            state=state,
        )
        if changed_count:
            new_body["contents"] = new_contents
            compressed_blocks += changed_count
            receipts.extend(block_receipts)

    compressed_tokens = _estimate_body_tokens(new_body)
    saved = max(0, original_tokens - compressed_tokens)
    receipt = ProxyCompressionReceipt(
        provider=provider,
        mode="elc",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        tokens_saved=saved,
        savings_ratio=0.0 if original_tokens == 0 else saved / original_tokens,
        compressed_blocks=compressed_blocks,
        receipts=receipts,
        protected_blocks=state.protected_blocks,
        deduped_blocks=state.deduped_blocks,
    )
    return ProxyCompressionResult(new_body, receipt, changed=compressed_blocks > 0)


def _compress_messages(
    messages: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    compress_user_messages: bool,
    retrieval_store: CompressionRetrievalStore | None,
    state: _ToolResultState | None = None,
) -> tuple[list[Any], int, list[dict[str, object]]]:
    state = state or _ToolResultState(policy=None)
    out: list[Any] = []
    changed_count = 0
    receipts: list[dict[str, object]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = str(msg.get("role", ""))
        content = msg.get("content")
        should_compress_role = role in {"tool", "function"} or (
            compress_user_messages and role == "user"
        )

        if isinstance(content, str) and role in {"tool", "function"}:
            call_ref = str(msg.get("tool_call_id") or "")
            pointer, verdict, policy_receipt = _apply_tool_result_policy(
                content,
                tool_name=str(msg.get("name") or state.call_names.get(call_ref, "")),
                call_ref=call_ref,
                state=state,
                budget_tokens=budget_tokens,
            )
            if verdict == "deduped" and policy_receipt is not None:
                new_msg = dict(msg)
                new_msg["content"] = pointer
                out.append(new_msg)
                changed_count += 1
                receipts.append(policy_receipt)
                continue
            if verdict == "protected":
                out.append(msg)
                continue

        if isinstance(content, str) and should_compress_role:
            new_text, changed, receipt = _compress_text_block(
                content,
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
                retrieval_store=retrieval_store,
                min_savings=state.min_savings,
            )
            if changed:
                new_msg = dict(msg)
                new_msg["content"] = new_text
                out.append(new_msg)
                changed_count += 1
                receipts.append(receipt)
                continue

        if isinstance(content, list):
            new_content, changed, block_receipts = _compress_content_blocks(
                content,
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
                compress_user_text=compress_user_messages and role == "user",
                retrieval_store=retrieval_store,
                state=state,
            )
            if changed:
                new_msg = dict(msg)
                new_msg["content"] = new_content
                out.append(new_msg)
                changed_count += len(block_receipts)
                receipts.extend(block_receipts)
                continue

        out.append(msg)
    return out, changed_count, receipts


def _compress_input_items(
    items: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    compress_user_messages: bool,
    retrieval_store: CompressionRetrievalStore | None,
    state: _ToolResultState | None = None,
) -> tuple[list[Any], int, list[dict[str, object]]]:
    state = state or _ToolResultState(policy=None)
    out: list[Any] = []
    changed_count = 0
    receipts: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        role = str(item.get("role", ""))
        if isinstance(item.get("content"), list):
            new_content, changed, block_receipts = _compress_content_blocks(
                item["content"],
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
                compress_user_text=compress_user_messages and role == "user",
                retrieval_store=retrieval_store,
                state=state,
            )
            if changed:
                new_item = dict(item)
                new_item["content"] = new_content
                out.append(new_item)
                changed_count += len(block_receipts)
                receipts.extend(block_receipts)
                continue
        out.append(item)
    return out, changed_count, receipts


def _compress_gemini_contents(
    contents: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    retrieval_store: CompressionRetrievalStore | None,
    state: _ToolResultState | None = None,
) -> tuple[list[Any], int, list[dict[str, object]]]:
    """Compress only textual tool-result payloads inside Gemini Parts.

    Function-call IDs, thought signatures, part ordering, and every unsupported
    field are copied byte-for-byte. Ambiguous structured results are left
    untouched instead of risking an invalid Gemini request.
    """
    state = state or _ToolResultState(policy=None)
    output = copy.deepcopy(contents)
    changed_count = 0
    receipts: list[dict[str, object]] = []
    for content in output:
        if not isinstance(content, dict) or not isinstance(
            content.get("parts"),
            list,
        ):
            continue
        for part in content["parts"]:
            if not isinstance(part, dict):
                continue
            targets: list[tuple[dict[str, Any], str]] = []
            tool_name = ""
            call_ref = ""
            function_response = part.get("functionResponse")
            if isinstance(function_response, dict):
                tool_name = str(function_response.get("name") or "")
                call_ref = str(function_response.get("id") or tool_name)
                response = function_response.get("response")
                if isinstance(response, dict):
                    targets.extend(
                        (response, key)
                        for key in ("output", "result", "content", "text")
                        if isinstance(response.get(key), str)
                    )
            code_result = part.get("codeExecutionResult")
            if isinstance(code_result, dict) and isinstance(
                code_result.get("output"),
                str,
            ):
                targets.append((code_result, "output"))
            for container, key in targets:
                pointer, verdict, policy_receipt = _apply_tool_result_policy(
                    container[key],
                    tool_name=tool_name,
                    call_ref=call_ref,
                    state=state,
                    budget_tokens=budget_tokens,
                )
                if verdict == "deduped" and policy_receipt is not None:
                    container[key] = pointer
                    changed_count += 1
                    receipts.append(policy_receipt)
                    continue
                if verdict == "protected":
                    continue
                compressed, changed, receipt = _compress_text_block(
                    container[key],
                    query=query,
                    budget_tokens=budget_tokens,
                    include_receipt_header=include_receipt_header,
                    retrieval_store=retrieval_store,
                    min_savings=state.min_savings,
                )
                if not changed:
                    continue
                container[key] = compressed
                changed_count += 1
                receipts.append(receipt)
    return output, changed_count, receipts


def _compress_content_blocks(
    blocks: list[Any],
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    compress_user_text: bool,
    retrieval_store: CompressionRetrievalStore | None,
    state: _ToolResultState | None = None,
) -> tuple[list[Any], bool, list[dict[str, object]]]:
    state = state or _ToolResultState(policy=None)
    out: list[Any] = []
    changed = False
    receipts: list[dict[str, object]] = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        block_type = str(block.get("type", ""))
        is_tool_result = block_type == "tool_result"
        is_text = block_type in {"text", "input_text"}
        if is_tool_result:
            value = block.get("content", "")
            if isinstance(value, str):
                call_ref = str(block.get("tool_use_id") or "")
                pointer, verdict, policy_receipt = _apply_tool_result_policy(
                    value,
                    tool_name=state.call_names.get(call_ref, ""),
                    call_ref=call_ref,
                    state=state,
                    budget_tokens=budget_tokens,
                )
                if verdict == "deduped" and policy_receipt is not None:
                    new_block = dict(block)
                    new_block["content"] = pointer
                    out.append(new_block)
                    changed = True
                    receipts.append(policy_receipt)
                    continue
                if verdict == "protected":
                    out.append(block)
                    continue
                new_text, did_change, receipt = _compress_text_block(
                    value,
                    query=query,
                    budget_tokens=budget_tokens,
                    include_receipt_header=include_receipt_header,
                    retrieval_store=retrieval_store,
                    min_savings=state.min_savings,
                )
                if did_change:
                    new_block = dict(block)
                    new_block["content"] = new_text
                    out.append(new_block)
                    changed = True
                    receipts.append(receipt)
                    continue
        elif is_text and compress_user_text and isinstance(block.get("text"), str):
            new_text, did_change, receipt = _compress_text_block(
                block["text"],
                query=query,
                budget_tokens=budget_tokens,
                include_receipt_header=include_receipt_header,
                retrieval_store=retrieval_store,
            )
            if did_change:
                new_block = dict(block)
                new_block["text"] = new_text
                out.append(new_block)
                changed = True
                receipts.append(receipt)
                continue
        out.append(block)
    return out, changed, receipts


def _compress_text_block(
    text: str,
    *,
    query: str,
    budget_tokens: int,
    include_receipt_header: bool,
    retrieval_store: CompressionRetrievalStore | None,
    min_savings: float | None = None,
) -> tuple[str, bool, dict[str, object]]:
    cleaned_query = _clean_query_for_compression(query)
    if min_savings is None:
        result = compress_evidence_locked(
            text, query=cleaned_query, budget_tokens=budget_tokens
        )
    else:
        result = compress_evidence_locked(
            text,
            query=cleaned_query,
            budget_tokens=budget_tokens,
            min_savings=min_savings,
        )
    receipt = result.receipt.as_dict()
    if not result.changed:
        return text, False, receipt
    rendered = (
        result.with_receipt_header()
        if include_receipt_header
        else result.compressed
    )
    if retrieval_store is not None:
        # Reserve enough space for the deterministic recovery marker before
        # writing anything. Borderline compression must not create an unused
        # recovery record and then forward the original block.
        if estimate_tokens(rendered) + 32 >= estimate_tokens(text):
            return text, False, receipt
        # The ELC receipt intentionally caps its diagnostic omission list.
        # A recovery handle must have a stronger contract: one exact span for
        # the complete original block, regardless of that diagnostic cap.
        recovery_receipt = dict(receipt)
        # Count lines the way the store slices them (str.splitlines), which
        # also breaks on "\r", U+2028, NEL and form feed. Counting only "\n"
        # made the stored "complete original" stop short of the real end.
        line_count = max(1, len(text.splitlines(keepends=True)))
        recovery_receipt["omitted_spans"] = [
            {
                "start_line": 1,
                "end_line": line_count,
                "line_count": line_count,
                "reason": "compression_proxy_full_recovery",
            }
        ]
        stored = retrieval_store.put(
            original_text=text,
            compressed_text=result.compressed,
            receipt=recovery_receipt,
            metadata={"query": query, "cleaned_query": cleaned_query, "component": "compression_proxy"},
        )
        if len(stored.spans) != 1 or stored.spans[0].content != text:
            # Fail closed: never forward a compacted block whose recovery
            # handle would hand back anything but the exact original.
            return text, False, receipt
        receipt["retrieval"] = {
            "receipt_id": stored.receipt_id,
            "span_count": len(stored.spans),
            "span_ids": [span.span_id for span in stored.spans],
        }
        span_id = stored.spans[0].span_id if stored.spans else "none"
        rendered = (
            f"{rendered}\n"
            f"[entroly-recovery:{stored.receipt_id}:{span_id}]"
        )
    if estimate_tokens(rendered) >= estimate_tokens(text):
        return text, False, receipt
    return rendered, True, receipt


def _clean_query_for_compression(query: str) -> str:
    """Keep only specific intent terms for ELC query locks.

    Generic words such as "build", "log", or "fail" appear on thousands of
    lines in CI output. Treating them as evidence locks prevents compression.
    This filter preserves specific entities like service names, file names,
    error codes, and incident ids while removing broad task words.
    """
    terms = []
    for word in _QUERY_WORD_RE.findall(query or ""):
        lowered = word.lower()
        if lowered in _QUERY_STOPWORDS:
            continue
        if len(lowered) < 4 and not any(ch.isdigit() for ch in lowered):
            continue
        terms.append(word)
    return " ".join(terms)


def _estimate_body_tokens(body: dict[str, Any]) -> int:
    return estimate_tokens(_stable_body_text(body))


def _stable_body_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


__all__ = [
    "ProxyCompressionReceipt",
    "ProxyCompressionResult",
    "compress_proxy_payload",
    "FILE_READ_TOOLS",
    "ToolResultPolicy",
    "looks_like_source",
    "compress_proxy_payload_from_env",
]
