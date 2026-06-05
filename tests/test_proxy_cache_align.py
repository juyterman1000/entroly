"""Proxy cache-alignment: stabilize Entroly's injected context block across
turns so provider prefix caches keep hitting — without ever touching the
model, generation params, tools, or the user's messages.

Guards the cost fix that wires CacheAligner into the HTTP proxy (previously
SDK-only), and the compliance invariant that alignment is context-only.
"""
from __future__ import annotations

from entroly.cache_aligner import CacheAligner
from entroly.proxy import PromptCompilerProxy

_key = PromptCompilerProxy._conversation_key


def test_conversation_key_stable_as_conversation_grows():
    """Same model + anchor (first system/user msg) => same key across turns,
    even as later turns append messages. That's what keeps the prefix stable."""
    turn1 = {"model": "claude-x", "messages": [
        {"role": "system", "content": "You are a coding agent for repo X."},
        {"role": "user", "content": "fix the auth bug"},
    ]}
    turn3 = {"model": "claude-x", "messages": [
        {"role": "system", "content": "You are a coding agent for repo X."},
        {"role": "user", "content": "fix the auth bug"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "now add a test"},
    ]}
    assert _key(turn1) and _key(turn1) == _key(turn3)


def test_conversation_key_differs_by_model_and_anchor():
    base = {"model": "m1", "messages": [{"role": "user", "content": "hello world"}]}
    diff_model = {"model": "m2", "messages": [{"role": "user", "content": "hello world"}]}
    diff_anchor = {"model": "m1", "messages": [{"role": "user", "content": "different"}]}
    assert _key(base) != _key(diff_model)
    assert _key(base) != _key(diff_anchor)


def test_conversation_key_is_read_only_compliance():
    """Deriving the key must NOT mutate the request body in any way
    (no model/param/message tampering)."""
    import copy
    body = {"model": "m", "temperature": 0.7, "max_tokens": 512,
            "tools": [{"name": "t"}],
            "messages": [{"role": "user", "content": "q"}]}
    snapshot = copy.deepcopy(body)
    _key(body)
    assert body == snapshot  # untouched


def test_conversation_key_empty_without_anchor():
    assert _key({}) == ""
    assert _key({"messages": []}) == ""
    assert _key("not-a-dict") == ""
    assert _key({"messages": [{"role": "assistant", "content": "x"}]}) == ""


def test_align_reuses_similar_context_verbatim():
    """An unchanged/near-identical injected block on the next turn is reused
    byte-for-byte => provider cached prefix hits."""
    a = CacheAligner()
    ctx = "\n".join(f"file{i}.py: def f{i}(): return {i}" for i in range(50))
    out1, hit1 = a.align("conv1", ctx)
    assert out1 == ctx and hit1 is False  # first turn: stored
    # next turn: one line changed (>=90% Jaccard similar)
    ctx2 = ctx.replace("file49.py: def f49(): return 49", "file49.py: def f49(): return 99")
    out2, hit2 = a.align("conv1", ctx2)
    assert hit2 is True
    assert out2 == ctx  # reused verbatim -> stable prefix -> cache hit


def test_align_does_not_reuse_materially_changed_context():
    """If the repo context materially changed, use the fresh block (no stale
    reuse) — correctness over cache."""
    a = CacheAligner()
    ctx = "\n".join(f"alpha{i}" for i in range(40))
    a.align("conv2", ctx)
    different = "\n".join(f"beta{i}" for i in range(40))
    out, hit = a.align("conv2", different)
    assert hit is False
    assert out == different
