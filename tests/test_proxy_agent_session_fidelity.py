"""An agent session through the proxy: what it costs and what the model sees.

Measured by replaying one 25-turn tool session (file reads, greps, test runs,
a 34k-token JSON response) through ``entroly proxy`` against a provider stub
that counts tokens and simulates prefix-cache billing:

* every OpenAI tool turn waited ~5.8 s, because the selection pipeline ran
  again for the question asked many turns earlier;
* every new question gained ~740 tokens that were an alphabetical file
  listing, identical for a haiku and for a question about ``handle_proxy``;
* 36% of the lines of the files the agent read reached the model, with no
  tool or instruction for getting the rest;
* a test log the agent produced twice was sent, and compressed, twice.

Each test below pins one of those.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from entroly.compression_proxy import (
    ToolResultPolicy,
    compress_proxy_payload,
    looks_like_source,
    tool_basename,
)
from entroly.proxy import (
    PromptCompilerProxy,
    _hierarchy_is_overview_only,
    _hierarchy_seed_ids,
)
from entroly.proxy_config import ProxyConfig, resolve_context_injection
from entroly.proxy_transform import latest_turn_is_user_text, request_declares_tools

# ── fixtures ─────────────────────────────────────────────────────────────

SOURCE = "\n".join(
    f"def handler_{n}(request, ledger):\n"
    f"    total = ledger.balance({n})\n"
    f"    if total < request.amount:\n"
    f"        raise InsufficientFunds(request.account, total)\n"
    f"    ledger.debit(request.account, request.amount)\n"
    f"    return Receipt(request.id, total - request.amount)\n"
    for n in range(120)
)
NUMBERED_SOURCE = "\n".join(
    f"{i:>6}\t{line}" for i, line in enumerate(SOURCE.splitlines(), 1)
)
LOG = "\n".join(
    [f"2026-07-24 12:00:{i % 60:02d} INFO worker={i % 5} batch {i} complete" for i in range(900)]
    + ["2026-07-24 12:15:00 FATAL payment-service E_CONNRESET after 3 retries"]
)
RECORDS = json.dumps(
    {
        "items": [
            {"id": i, "status": "ok" if i != 217 else "error",
             "error": None if i != 217 else "upstream 529 overloaded_error",
             "latency_ms": 100 + (i * 37) % 900}
            for i in range(400)
        ]
    },
    indent=1,
)


def _anthropic_session(*results: tuple[str, str]) -> dict:
    """One user question followed by ``(tool name, result)`` tool turns."""
    messages: list[dict] = [
        {"role": "user", "content": [{"type": "text", "text": "Fix the debit handler."}]}
    ]
    for index, (tool, result) in enumerate(results):
        call_id = f"toolu_{index:02d}"
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": tool, "input": {}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": result}]})
    return {"model": "claude-sonnet-4-5", "max_tokens": 64, "messages": messages,
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}]}


def _openai_session(*results: tuple[str, str]) -> dict:
    messages: list[dict] = [{"role": "user", "content": "Fix the debit handler."}]
    for index, (tool, result) in enumerate(results):
        call_id = f"call_{index:02d}"
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": tool, "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    return {"model": "gpt-4o", "messages": messages,
            "tools": [{"type": "function", "function": {"name": "Read"}}]}


def _tool_results(body: dict) -> list[str]:
    out: list[str] = []
    for message in body["messages"]:
        content = message.get("content")
        if message.get("role") == "tool":
            out.append(content)
        elif isinstance(content, list):
            out.extend(b["content"] for b in content if b.get("type") == "tool_result")
    return out


PROTECTIVE = ToolResultPolicy.from_names("")


# ── the pipeline runs for a new question, not for a tool turn ────────────

def test_an_openai_tool_turn_is_not_a_new_question():
    body = _openai_session(("Read", "contents"))
    assert latest_turn_is_user_text(body, "openai") is False
    body["messages"].append({"role": "user", "content": "Now run the tests."})
    assert latest_turn_is_user_text(body, "openai") is True


def test_reminder_text_beside_a_tool_result_is_still_a_tool_turn():
    body = _anthropic_session(("Read", "contents"))
    body["messages"][-1]["content"].append(
        {"type": "text", "text": "<system-reminder>todo list is empty</system-reminder>"}
    )
    assert latest_turn_is_user_text(body, "anthropic") is False
    assert latest_turn_is_user_text(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]},
        "anthropic",
    ) is True


@pytest.mark.parametrize(
    ("body", "provider", "expected"),
    [
        ({"input": "explain the ledger"}, "openai", True),
        ({"input": [{"role": "user", "content": [{"type": "input_text", "text": "q"}]}]},
         "openai", True),
        ({"input": [{"role": "user", "content": "q"},
                    {"type": "function_call", "call_id": "c1", "name": "shell"},
                    {"type": "function_call_output", "call_id": "c1", "output": "ok"}]},
         "openai", False),
        ({"contents": [{"role": "user", "parts": [{"text": "q"}]}]}, "gemini", True),
        ({"contents": [{"role": "user", "parts": [
            {"functionResponse": {"name": "read_file", "response": {"output": "x"}}}]}]},
         "gemini", False),
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "partial"}]}, "openai", False),
        ({"messages": []}, "openai", False),
    ],
)
def test_latest_turn_detection_across_request_shapes(body, provider, expected):
    assert latest_turn_is_user_text(body, provider) is expected


def test_tools_are_detected_in_either_declaration():
    assert request_declares_tools({"tools": [{"name": "Read"}]})
    assert request_declares_tools({"functions": [{"name": "read"}]})
    assert not request_declares_tools({"tools": []})
    assert not request_declares_tools({"messages": []})


# ── injection is the operator's decision, and off adds nothing ───────────

@pytest.mark.parametrize(
    ("raw", "mode"),
    [("off", "off"), ("chat", "chat"), ("always", "always"), ("1", "always"),
     ("", "off"), (None, "off"), ("ALWAYS", "always"), ("alwys", "off")],
)
def test_an_unrecognised_injection_mode_adds_nothing(raw, mode):
    assert resolve_context_injection(raw) == mode


def test_injection_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv("ENTROLY_CONTEXT_INJECTION", raising=False)
    assert ProxyConfig().context_injection == "off"
    assert ProxyConfig.from_env().context_injection == "off"
    monkeypatch.setenv("ENTROLY_CONTEXT_INJECTION", "chat")
    assert ProxyConfig.from_env().context_injection == "chat"


class _CountingEngine:
    """Counts selection calls: the object under test is when the proxy asks."""

    def __init__(self) -> None:
        self._turn_counter = 0
        self.calls = 0

    def advance_turn(self) -> None:
        return None

    def optimize_context(self, token_budget: int, query: str) -> dict:
        self.calls += 1
        return {
            "selected_fragments": [{
                "id": "ledger-1", "source": "src/ledger.py",
                "content": "def debit(account, amount):\n    return account - amount\n",
                "token_count": 14, "entropy_score": 1.0, "variant": "full",
            }],
            "query_analysis": {},
        }


def _config(mode: str) -> ProxyConfig:
    config = ProxyConfig(witness_mode="off")
    config.enable_adaptive_budget = False
    config.enable_dynamic_budget = False
    config.enable_hierarchical_compression = False
    config.enable_passive_feedback = False
    config.enable_context_scaffold = False
    config.context_injection = mode
    return config


def _forward(body: dict, mode: str, path: str, tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    engine = _CountingEngine()
    captured: dict = {}

    async def run():
        proxy = PromptCompilerProxy(engine, _config(mode))
        proxy._confidence_threshold = 0.0

        async def capture(_url, _headers, forwarded, *_args, **kwargs):
            captured["body"] = json.loads(json.dumps(forwarded))
            return JSONResponse({"ok": True}, headers=kwargs.get("extra_headers") or {})

        proxy._forward_response = capture
        app = Starlette(routes=[Route(path, proxy.handle_proxy, methods=["POST"])])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            headers = (
                {"x-api-key": "sk-ant-api03-test", "anthropic-version": "2023-06-01"}
                if path == "/v1/messages" else {"authorization": "Bearer sk-test"}
            )
            response = await client.post(path, headers=headers, json=body)
        return proxy, response

    proxy, response = asyncio.run(run())
    return engine, proxy, response, captured.get("body") or {}


def test_with_injection_off_a_chat_request_is_forwarded_as_sent(tmp_path, monkeypatch):
    sent = {"model": "gpt-4o",
            "messages": [{"role": "user", "content": "How does debit handle overdrafts?"}]}
    engine, _proxy, response, forwarded = _forward(
        sent, "off", "/v1/chat/completions", tmp_path, monkeypatch)

    assert response.status_code == 200
    assert engine.calls == 0, "selection ran for a request that takes no context"
    assert forwarded["messages"] == sent["messages"]
    assert response.headers["x-entroly-context-injection"] == "off"
    assert response.headers["x-entroly-context-tokens-added"] == "0"


def test_chat_mode_serves_a_chat_and_leaves_an_agent_alone(tmp_path, monkeypatch):
    question = "How does debit handle overdrafts?"
    chat = {"model": "gpt-4o", "messages": [{"role": "user", "content": question}]}
    engine, _proxy, response, forwarded = _forward(
        chat, "chat", "/v1/chat/completions", tmp_path, monkeypatch)
    assert engine.calls == 1
    assert "def debit(account, amount)" in json.dumps(forwarded)
    assert int(response.headers["x-entroly-context-tokens-added"]) > 0

    agent = dict(chat, tools=[{"type": "function", "function": {"name": "read_file"}}])
    engine, _proxy, response, forwarded = _forward(
        agent, "chat", "/v1/chat/completions", tmp_path, monkeypatch)
    assert engine.calls == 0
    assert forwarded["messages"] == agent["messages"]
    assert response.headers["x-entroly-context-injection"] == "agent-has-tools"


def test_always_mode_selects_once_per_question_not_once_per_tool_call(tmp_path, monkeypatch):
    body = _openai_session(("Bash", "ok"), ("Bash", "ok"), ("Bash", "ok"))
    engine, _proxy, response, _forwarded = _forward(
        body, "always", "/v1/chat/completions", tmp_path, monkeypatch)
    assert engine.calls == 0
    assert response.headers["x-entroly-context-injection"] == "not-a-new-question"

    body["messages"].append({"role": "user", "content": "Now explain the debit path."})
    engine, _proxy, _response, _forwarded = _forward(
        body, "always", "/v1/chat/completions", tmp_path, monkeypatch)
    assert engine.calls == 1


# ── the hierarchy is rendered from the selection, or not at all ──────────

def test_synthetic_selection_ids_are_never_used_as_hierarchy_seeds():
    synthetic = [{"id": "qccr::file:pkg/a.py", "source": "file:pkg/a.py",
                  "source_fragment_ids": ["f1_000001", "f1_000002"]}]
    assert _hierarchy_seed_ids(synthetic) is None
    native = [{"id": "f1_000001"}, {"fragment_id": "f1_000002"}, {"id": "f1_000001"}]
    assert _hierarchy_seed_ids(native) == ["f1_000001", "f1_000002"]
    assert _hierarchy_seed_ids([]) == []


def test_a_file_listing_alone_is_not_a_hierarchy():
    listing = {"level1_map": "file:a.py\nfile:b.py", "level1_tokens": 500,
               "level2_tokens": 0, "level3_tokens": 0, "level3_fragments": []}
    assert _hierarchy_is_overview_only(listing)
    assert not _hierarchy_is_overview_only(dict(listing, level2_cluster="a -> b", level2_tokens=40))
    assert not _hierarchy_is_overview_only(
        dict(listing, level3_fragments=[{"source": "a.py"}], level3_tokens=900))


# ── what the model asked to read is what it gets ─────────────────────────

def test_a_numbered_file_view_and_raw_source_both_read_as_source():
    assert looks_like_source(NUMBERED_SOURCE)
    assert looks_like_source(SOURCE)
    assert not looks_like_source(LOG)
    assert not looks_like_source(RECORDS)
    numbered_json = "\n".join(
        f"{i:>6}\t{line}" for i, line in enumerate(RECORDS.splitlines(), 1))
    assert not looks_like_source(numbered_json), "a JSON data file is not source"


def test_tool_names_match_on_their_last_segment():
    assert tool_basename("mcp__filesystem__read_file") == "read_file"
    assert tool_basename("functions.Read") == "read"
    assert tool_basename("") == ""


@pytest.mark.parametrize("session", [_anthropic_session, _openai_session])
def test_a_file_read_reaches_the_model_whole(session):
    body = session(("Read", NUMBERED_SOURCE), ("Bash", LOG))
    result = compress_proxy_payload(body, provider="anthropic", policy=PROTECTIVE)
    read, log = _tool_results(result.body)

    assert read == NUMBERED_SOURCE
    assert len(log) < len(LOG) // 2 and "E_CONNRESET" in log
    assert result.receipt.protected_blocks == 1
    assert result.headers()["x-entroly-protected-blocks"] == "1"


def test_source_printed_by_a_shell_is_still_source():
    body = _anthropic_session(("Bash", SOURCE))
    result = compress_proxy_payload(body, provider="anthropic", policy=PROTECTIVE)
    assert _tool_results(result.body) == [SOURCE]


def test_structured_data_is_still_summarised_with_its_outlier():
    body = _anthropic_session(("Bash", RECORDS))
    result = compress_proxy_payload(body, provider="anthropic", policy=PROTECTIVE)
    (forwarded,) = _tool_results(result.body)
    assert len(forwarded) < len(RECORDS) // 5
    assert "overloaded_error" in forwarded


def test_an_operator_exclusion_is_honoured_on_the_default_path():
    body = _anthropic_session(("fetch_orders", LOG))
    policy = ToolResultPolicy.from_names("mcp__shop__fetch_orders, other")
    result = compress_proxy_payload(body, provider="anthropic", policy=policy)
    assert _tool_results(result.body) == [LOG]


def test_protection_can_be_switched_off_and_the_sdk_default_is_unchanged():
    body = _anthropic_session(("Read", NUMBERED_SOURCE))
    aggressive = ToolResultPolicy.from_names("", protect_file_reads=False, dedupe_repeats=False)
    for policy in (aggressive, None):
        result = compress_proxy_payload(body, provider="anthropic", policy=policy)
        (forwarded,) = _tool_results(result.body)
        assert len(forwarded) < len(NUMBERED_SOURCE)


# ── a repeat costs one line ───────────────────────────────────────────────

@pytest.mark.parametrize("session", [_anthropic_session, _openai_session])
def test_an_identical_repeat_becomes_a_pointer_to_the_first_copy(session):
    body = session(("Read", NUMBERED_SOURCE), ("Bash", "ok"), ("Read", NUMBERED_SOURCE))
    result = compress_proxy_payload(body, provider="anthropic", policy=PROTECTIVE)
    first, _ack, repeat = _tool_results(result.body)

    assert first == NUMBERED_SOURCE, "the copy the pointer refers to must be intact"
    assert len(repeat) < 200 and "byte-identical" in repeat and "Read" in repeat
    assert ("toolu_00" in repeat) or ("call_00" in repeat)
    assert result.receipt.deduped_blocks == 1
    assert result.changed


def test_a_near_repeat_and_a_short_repeat_are_left_alone():
    changed = NUMBERED_SOURCE.replace("handler_7(", "handler_seven(", 1)
    body = _anthropic_session(("Read", NUMBERED_SOURCE), ("Read", changed),
                              ("Bash", "ok"), ("Bash", "ok"))
    result = compress_proxy_payload(body, provider="anthropic", policy=PROTECTIVE)
    assert _tool_results(result.body) == [NUMBERED_SOURCE, changed, "ok", "ok"]
    assert result.receipt.deduped_blocks == 0


def test_the_forwarded_prefix_is_stable_as_the_session_grows():
    """Turn N's forwarded messages must be a prefix of turn N+1's."""
    steps = [("Read", NUMBERED_SOURCE), ("Bash", LOG), ("Read", NUMBERED_SOURCE),
             ("Bash", RECORDS), ("Bash", LOG)]
    previous: list | None = None
    for count in range(1, len(steps) + 1):
        body = _anthropic_session(*steps[:count])
        forwarded = compress_proxy_payload(
            body, provider="anthropic", policy=PROTECTIVE).body["messages"]
        again = compress_proxy_payload(
            body, provider="anthropic", policy=PROTECTIVE).body["messages"]
        assert forwarded == again
        if previous is not None:
            assert forwarded[: len(previous)] == previous
        previous = forwarded


# ── what the proxy did is what it reports ────────────────────────────────

def test_a_compression_only_request_is_booked_as_reduced(tmp_path, monkeypatch):
    import entroly.value_tracker as value_tracker

    tracker = value_tracker.ValueTracker(data_dir=tmp_path / "value")
    monkeypatch.setattr(value_tracker, "_tracker", tracker)

    body = _anthropic_session(("Read", NUMBERED_SOURCE), ("Bash", LOG), ("Bash", LOG))
    engine, proxy, response, forwarded = _forward(
        body, "off", "/v1/messages", tmp_path, monkeypatch)

    assert response.status_code == 200 and engine.calls == 0
    read, log, repeat = _tool_results(forwarded)
    assert read == NUMBERED_SOURCE
    assert "E_CONNRESET" in log and len(log) < len(LOG) // 2
    assert "byte-identical" in repeat

    saved = proxy._total_original_tokens - proxy._total_optimized_tokens
    assert saved > 0 and proxy._requests_optimized == 1
    assert saved == (PromptCompilerProxy._estimate_body_tokens(body)
                     - PromptCompilerProxy._estimate_body_tokens(forwarded))
    provider = tracker.get_value_receipt()["provider_path"]
    assert provider["requests_observed"] == 1
    assert provider["requests_optimized"] == 1
    assert provider["requests_enlarged"] == 0
    assert provider["input_tokens_reduced"] == saved
    assert provider["input_tokens_added"] == 0
    assert response.headers["x-entroly-protected-blocks"] == "1"
    assert response.headers["x-entroly-deduped-blocks"] == "1"


# ── a rewrite has to be worth making ──────────────────────────────────────

def _marginal_block() -> str:
    """A block the codec trims by under a fifth: one just over its budget.

    Sized by measurement, because the token count of a line depends on which
    tokenizer this installation has.
    """
    for lines in range(60, 240):
        text = "\n".join(
            f"src/pkg/module_{i}.py:{i + 10}:    value = compute_total(order_{i}, ledger)"
            for i in range(lines)
        )
        legacy = compress_proxy_payload(
            _anthropic_session(("Grep", text)), provider="anthropic", policy=None)
        (forwarded,) = _tool_results(legacy.body)
        if forwarded != text and len(forwarded) > 0.85 * len(text):
            return text
    pytest.skip("no block size is trimmed marginally with this tokenizer")


def test_a_rewrite_that_keeps_most_of_a_block_is_not_made():
    """Just over the block budget, the codec trims ~10%: lines lost, nothing gained."""
    marginal = _marginal_block()
    body = _anthropic_session(("Grep", marginal))

    kept = compress_proxy_payload(body, provider="anthropic", policy=PROTECTIVE)
    assert _tool_results(kept.body) == [marginal]
    assert not kept.changed


# ── the conservative ledger charges a cache read as a cache read ─────────

def _inventories():
    from entroly import net_savings as ns

    old_raw = "x" * 6000          # a block the client sent many turns ago
    old_rewritten = "y" * 2000    # what Entroly forwards in its place
    newest = "z" * 9000           # this turn's own content, forwarded as sent
    raw = {"messages": [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": old_raw},
        {"role": "tool", "content": newest},
    ]}
    out = {"messages": [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": old_rewritten},
        {"role": "tool", "content": newest},
    ]}
    return ns, ns.prompt_inventory(raw), ns.prompt_inventory(out)


def test_a_rewrite_forwarded_before_and_inside_the_cached_prefix_was_read():
    ns, raw, out = _inventories()
    rewritten_digest, rewritten_bytes = out.units[1]
    total = 3000
    question_end = -(-out.units[0][1] * total // out.unit_bytes)
    prefix = dict(total_input_tokens=total, forwarded_before={rewritten_digest})

    # The provider read past the start of the rewritten block.
    assert ns.added_bytes_in_cached_prefix(
        raw, out, cache_read_tokens=question_end + 1, **prefix) == rewritten_bytes
    # The cached prefix ends where the block starts: it was written, not read.
    assert ns.added_bytes_in_cached_prefix(
        raw, out, cache_read_tokens=question_end, **prefix) == 0
    assert ns.added_bytes_in_cached_prefix(
        raw, out, cache_read_tokens=0, **prefix) == 0
    # Never forwarded before: the provider cannot have had it, wherever it sits.
    assert ns.added_bytes_in_cached_prefix(
        raw, out, cache_read_tokens=total, total_input_tokens=total,
        forwarded_before=set()) == 0
    assert ns.added_bytes_in_cached_prefix(
        raw, raw, cache_read_tokens=total, **prefix) == 0


def test_of_identical_blocks_the_later_one_is_the_added_one():
    from entroly import net_savings as ns

    raw = ns.prompt_inventory({"messages": [
        {"role": "tool", "content": "same"}, {"role": "tool", "content": "other"}]})
    out = ns.prompt_inventory({"messages": [
        {"role": "tool", "content": "same"}, {"role": "tool", "content": "same"}]})
    ((digest, length, start),) = ns.added_units(raw, out)
    assert (digest, length) == out.units[1] and start == out.units[0][1]


def test_a_moved_cache_marker_does_not_make_a_block_new():
    """Clients move ``cache_control`` to the newest block every turn."""
    from entroly import net_savings as ns

    block = {"type": "tool_result", "tool_use_id": "t1", "content": "payload " * 200}
    marked = dict(block, cache_control={"type": "ephemeral"})
    plain = ns.prompt_inventory({"messages": [{"role": "user", "content": [block]}]})
    moved = ns.prompt_inventory({"messages": [{"role": "user", "content": [marked]}]})
    assert plain.units == moved.units

    seen = ns.SeenUnits()
    empty = ns.prompt_inventory({"messages": []})
    assert seen.classify_and_remember(moved.units, moved, empty) == (0, moved.units[0][1])
    assert seen.classify_and_remember(plain.units, plain, empty) == (plain.units[0][1], 0)
    assert seen.known([plain.units[0][0], "never-sent"]) == {plain.units[0][0]}


def test_new_client_content_is_not_billed_to_an_old_rewrite():
    """A turn that only appends the client's own content must not book a loss."""
    ns, raw, out = _inventories()
    removed, added = ns.inventory_delta(raw, out)
    seen_bytes = sum(length for _, length in removed)   # sent on an earlier turn
    rates = ns.resolve_rates("anthropic", "claude-sonnet-4-5-20250929")[0]
    rewritten_digest, rewritten_bytes = out.units[1]
    total = 3000
    read = -(-(out.units[0][1] + rewritten_bytes) * total // out.unit_bytes)
    usage = dict(uncached_input_tokens=0, cache_read_tokens=read,
                 cache_write_tokens=total - read)

    def estimate(cached_bytes: int):
        return ns.estimate_net_saving(
            raw=raw, outbound=out, removed_seen_bytes=seen_bytes, removed_new_bytes=0,
            added_bytes=added, interference_bytes=0, rates=rates,
            pricing_source="test", added_cached_bytes=cached_bytes, **usage)

    pooled = estimate(0)
    positioned = estimate(ns.added_bytes_in_cached_prefix(
        raw, out, cache_read_tokens=read, total_input_tokens=total,
        forwarded_before={rewritten_digest}))

    assert pooled.micro_usd < 0, "the old estimate charged the rewrite as a fresh write"
    assert positioned.micro_usd > 0, "6,000 bytes read from cache became 2,000"
    assert positioned.added_cached_tokens > 0
    assert positioned.details()["added_cached_tokens"] == positioned.added_cached_tokens
    # First sighting of the rewrite: nothing is assumed, the pool is charged.
    first_turn = estimate(ns.added_bytes_in_cached_prefix(
        raw, out, cache_read_tokens=read, total_input_tokens=total,
        forwarded_before=set()))
    assert first_turn.micro_usd == pooled.micro_usd
