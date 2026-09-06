"""Python and Rust must agree on every governance value, byte for byte.

Rust is the core; Python and Node are wrappers over it. `governance_bindings.rs`
says so outright -- "All governance computation lives in `entroly-engine`" --
and the audit-chain binding promises hashes "used by all three distributions to
guarantee identical chain hashes". But only `identity.py` calls into the core;
`audit.py` and `policy.py` reimplement it in Python. Nothing checked the two
agreed, and for one primitive they did not.

`_identity_payload_json` canonicalized with `ensure_ascii=True`, emitting
`\\u00e4` where serde_json emits raw UTF-8. Rust re-serializes the payload it
parsed rather than hashing the string it was handed, so the two hashed
different bytes for any identity with a non-ASCII field. Pure ASCII agreed,
which is why it looked correct everywhere anyone tried it. The failure was
cross-distribution: an identity minted where the native engine is absent failed
verification where it is present, and `resolve_identity` fails closed by
downgrading to anonymous -- so an agent belonging to a user with an accent in
their name silently lost every granted scope, and the log blamed the token.

These tests run both implementations in one process and compare. That is the
only arrangement that can see the divergence: each side is self-consistent, so
neither can detect it alone.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from entroly.governance import identity as identity_module
from entroly.native_status import usable_core

REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE = usable_core()

pytestmark = pytest.mark.skipif(
    _CORE is None or not hasattr(_CORE, "governance_compute_identity_token"),
    reason="native governance bindings unavailable; build with `maturin develop --release`",
)

# Deliberately spans the encoding boundary. An ASCII-only matrix passes against
# the exact bug this file exists to prevent.
IDENTITY_CASES = [
    ("ascii", {"agent_id": "a1", "agent_type": "claude-code",
               "scopes": ["read"], "created_at": 1.0}),
    ("accented_user", {"agent_id": "agent", "agent_type": "codex",
                       "user": "josé", "scopes": ["read", "write"],
                       "created_at": 1.5}),
    ("latin1_agent_id", {"agent_id": "ägent", "agent_type": "codex",
                         "scopes": ["read"], "created_at": 1.5}),
    ("cjk", {"agent_id": "代理", "agent_type": "human",
             "scopes": ["read"], "created_at": 1.0}),
    ("astral_emoji", {"agent_id": "\U0001f916", "agent_type": "service",
                      "scopes": ["read"], "created_at": 2.0}),
    ("rtl", {"agent_id": "agent", "agent_type": "human",
             "organization": "مؤسسة",
             "scopes": ["read"], "created_at": 3.0}),
    ("all_fields", {"agent_id": "a", "agent_type": "ci-bot", "organization": "o",
                    "user": "u", "team": "t", "session_id": "s", "model": "m",
                    "scopes": ["admin", "read"], "created_at": 1699999999.123}),
]


@pytest.mark.parametrize("name,data", IDENTITY_CASES, ids=[c[0] for c in IDENTITY_CASES])
@pytest.mark.parametrize("key", ["k", "operator-key", "a" * 200, "ключ"])
def test_identity_token_is_identical_in_python_and_rust(name, data, key):
    """The Python fallback must reproduce the native token exactly.

    `compute_identity_token` uses Rust when present and this fallback when not,
    and the two must be interchangeable -- that is the whole premise of having
    a fallback at all.
    """
    payload_json = identity_module._identity_payload_json(data)
    python_token = identity_module._python_compute_token(payload_json, key)
    rust_token = _CORE.governance_compute_identity_token(payload_json, key)
    assert python_token == rust_token, (
        f"{name}: Python and Rust disagree on the identity token.\n"
        f"  python: {python_token}\n  rust  : {rust_token}\n"
        f"  payload: {payload_json}"
    )


@pytest.mark.parametrize("name,data", IDENTITY_CASES, ids=[c[0] for c in IDENTITY_CASES])
def test_a_token_minted_without_the_native_engine_verifies_with_it(name, data, monkeypatch):
    """The cross-distribution path: mint on pure Python, verify on native.

    This is the scenario users actually hit, and the one that failed. Each
    engine agreed with itself, so only crossing the boundary reveals it.
    """
    from entroly.governance.domain import AgentIdentity

    key = "operator-key"
    monkeypatch.setattr(identity_module, "_RUST_AVAILABLE", False)
    payload_json = identity_module._identity_payload_json(data)
    token = identity_module._python_compute_token(payload_json, key)

    identity = AgentIdentity(
        agent_id=str(data.get("agent_id", "")),
        agent_type=str(data.get("agent_type", "")),
        organization=str(data.get("organization", "")),
        user=str(data.get("user", "")),
        team=str(data.get("team", "")),
        session_id=str(data.get("session_id", "")),
        model=str(data.get("model", "")),
        created_at=data.get("created_at", 0),
        identity_token=token,
        scopes=frozenset(data.get("scopes", ())),
    )

    monkeypatch.setattr(identity_module, "_RUST_AVAILABLE", True)
    assert identity_module.verify_token(identity, key), (
        f"{name}: an identity minted without the native engine was rejected by "
        f"it. resolve_identity treats that as a forged token and downgrades the "
        f"agent to anonymous/read-only."
    )


def test_payload_canonicalization_matches_serde_not_ascii_escapes():
    """Pin the encoding, not just the outcome.

    Rust canonicalizes with `serde_json::to_string`, which never escapes
    non-ASCII. A payload that still carried `\\uXXXX` escapes would hash
    differently no matter how the tokens happened to compare.
    """
    payload_json = identity_module._identity_payload_json(
        {"agent_id": "ägent", "agent_type": "codex",
         "user": "josé", "scopes": ["read"], "created_at": 1.0}
    )
    assert "\\u00e4" not in payload_json, (
        "canonical payload is ASCII-escaped; serde_json emits raw UTF-8, so "
        "Rust would hash different bytes for the same identity"
    )
    assert "ägent" in payload_json


def test_rust_struct_declaration_order_matches_the_python_payload():
    """Read the Rust struct itself; asserting on Python's output proves nothing.

    serde emits fields in *declaration* order, so the contract lives in
    governance.rs, not here. An earlier version of this test checked that
    Python's own keys came out sorted -- which they do whether or not
    `sort_keys` is set, because the dict literal is already alphabetical, and
    which says nothing at all about the Rust side.

    A reorder in governance.rs would invalidate every issued token and still
    pass `cargo test`. The token differentials above would go red, but with a
    hash mismatch that points nowhere. This one names the cause.
    """
    source = REPO_ROOT / "entroly-engine" / "src" / "governance.rs"
    if not source.exists():
        pytest.skip("entroly-engine sources not present in this checkout")

    body = re.search(
        r"pub struct AgentIdentityPayload\s*\{(.*?)\n\}", source.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert body, "could not locate AgentIdentityPayload in governance.rs"
    rust_fields = re.findall(r"^\s*pub\s+(\w+)\s*:", body.group(1), re.MULTILINE)
    assert rust_fields, "parsed no fields from AgentIdentityPayload"

    assert rust_fields == sorted(rust_fields), (
        f"AgentIdentityPayload fields are no longer alphabetical: {rust_fields}. "
        f"serde emits declaration order, so this silently changes every identity "
        f"token and invalidates all issued credentials."
    )

    python_keys = list(json.loads(
        identity_module._identity_payload_json({"agent_id": "a", "created_at": 1.0})
    ))
    assert rust_fields == python_keys, (
        f"the two canonical payloads have drifted apart.\n"
        f"  rust  : {rust_fields}\n  python: {python_keys}"
    )


@pytest.mark.parametrize(
    "prev,event_id,payload",
    [
        ("", "evt-1", {"scope": "write"}),
        ("a" * 64, "evt-2", {"scope": "deploy", "n": 3}),
        ("deadbeef", "evt-3", {}),
        ("", "évt-4", {"path": "src/ünïcode.py"}),
        ("", "evt-5", {"note": "\U0001f916 agent"}),
    ],
)
def test_audit_chain_hash_is_identical_in_python_and_rust(prev, event_id, payload):
    """`audit.py` computes this in Python and never calls the core.

    The Rust binding documents itself as "used by all three distributions to
    guarantee identical chain hashes", which is only true if the Python
    implementation happens to match. It does -- so pin it, because an audit
    chain that forks between distributions cannot be verified where it was not
    written, and tamper-evidence is the entire point of the chain.
    """
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    python_hash = hashlib.sha256(
        f"{prev}|{event_id}|{payload_json}".encode("utf-8")
    ).hexdigest()
    rust_hash = _CORE.governance_compute_audit_chain_hash(prev, event_id, payload_json)
    assert python_hash == rust_hash, (
        f"chain hash diverged for {event_id!r}\n"
        f"  python: {python_hash}\n  rust  : {rust_hash}"
    )
