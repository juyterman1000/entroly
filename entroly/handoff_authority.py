"""Who may continue a workstream after a handoff, and on whose authority.

Before this module, a Work Graph handoff named its successor with a string.
``--to-agent codex`` was a label anyone could write, and any process that
could reach the state directory could act as that agent. The label describes
intent. It does not establish authority.

Here authority comes from a **host assertion**. The trusted host runs the
agents and holds their credentials, outside the agent's sandbox. It signs a
short-lived statement with an Ed25519 key, and Entroly verifies it with a
public key the operator pinned. The statement names:

* ``iss`` and ``aud``: who issued it and for which verifier;
* ``sub``: the successor agent, which must equal the offer's ``to_agent``;
* ``scope``: ``work:accept`` to take over, ``work:write`` to act afterwards;
* ``workspace`` and ``task``: the Work Graph repository and workstream;
* ``handoff``: the digest of the exact offer, which commits to the handoff
  receipt, the workspace state and every evidence record handed over;
* ``jti``: one attempt; ``iat``, ``nbf`` and ``exp``: its validity window.

Acceptance is a single state transition per workstream. Each accepted handoff
appends ``epoch-<n>.json`` to a hash-chained log. The record is written in
full to a private temporary file and published with ``os.link``. ``link``
fails if the name exists, so concurrent acceptances of the same epoch have
exactly one winner. A crash before the link changes nothing. A crash after
it leaves a complete record, and repeating the same assertion returns it.

Fencing: once a workstream has an epoch, its writes (a new handoff, recorded
evidence) require the current holder's assertion. That means the same attempt,
the same subject and ``work:write``, inside the validity window. The
predecessor's assertion names an older attempt and is refused, as is the
holder's own after it expires.

Ambiguity stops the transition. Missing trust anchors, an unpinned key, an
offer that does not match its receipt, a log that does not chain or an
outstanding recovery acknowledgement all refuse. Nothing is guessed.

What this does not do: the verifier runs inside the workspace. A process that
can rewrite the pinned trust file or the state directory can subvert it. The
trust file belongs on a read-only path provided by the host, and the host can
re-verify every epoch record against its own key log afterwards. No network
call is made.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASSERTION_SCHEMA = "entroly.handoff-assertion.v1"
TRUST_SCHEMA = "entroly.handoff-trust.v1"
OFFER_SCHEMA = "entroly.handoff-offer.v1"
EPOCH_SCHEMA = "entroly.handoff-epoch.v1"
TRUST_FILE_ENV = "ENTROLY_HANDOFF_TRUST_FILE"
ASSERTION_FILE_ENV = "ENTROLY_HANDOFF_ASSERTION"
SCOPE_ACCEPT = "work:accept"
SCOPE_WRITE = "work:write"
KNOWN_SCOPES = frozenset({SCOPE_ACCEPT, SCOPE_WRITE})
CLAIM_NAMES = ("aud", "exp", "handoff", "iat", "iss", "jti", "nbf", "scope", "sub", "task", "workspace")
_TEXT_CLAIMS = ("aud", "handoff", "iss", "jti", "sub", "task", "workspace")
_TIME_CLAIMS = ("exp", "iat", "nbf")
_SIGNATURE_DOMAIN = b"entroly.handoff-assertion.v1\x00"
_DEFAULT_MAX_LIFETIME_MS = 8 * 60 * 60 * 1000
_DEFAULT_SKEW_MS = 60 * 1000
_MAX_TEXT = 512
_MAX_FILE_BYTES = 256 * 1024
_EPOCH_NAME = re.compile(r"^epoch-(\d{6})\.json$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AuthorityRefused(RuntimeError):
    """The transition or write was refused; ``code`` says why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ── encoding ─────────────────────────────────────────────────────────


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT or "\x00" in value:
        raise AuthorityRefused("malformed-assertion", f"{name} must be text of 1-{_MAX_TEXT} characters")
    return value


def _read_json(path: str | os.PathLike[str], what: str, code: str) -> Any:
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise AuthorityRefused(code, f"{what} must not be a symbolic link")
        with open(candidate, "rb") as handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
    except FileNotFoundError as exc:
        raise AuthorityRefused(code, f"{what} not found: {path}") from exc
    except OSError as exc:
        raise AuthorityRefused(code, f"{what} cannot be read: {exc}") from exc
    if len(raw) > _MAX_FILE_BYTES:
        raise AuthorityRefused(code, f"{what} is larger than {_MAX_FILE_BYTES} bytes")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AuthorityRefused(code, f"{what} is not valid JSON") from exc


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            PublicFormat,
        )
    except ImportError as exc:
        raise AuthorityRefused(
            "crypto-unavailable",
            "verifying host assertions requires the cryptography package: "
            "pip install 'entroly[receipt-proof]'",
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature, Encoding, NoEncryption, PrivateFormat, PublicFormat


def key_id(public_key_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()[:16]


# ── trust anchors ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrustAnchors:
    audience: str
    issuers: Mapping[str, tuple[str, ...]]
    max_lifetime_ms: int
    clock_skew_ms: int
    source: str


def load_trust(path: str | None = None, environ: Mapping[str, str] | None = None) -> TrustAnchors:
    """Load the operator's pinned issuer keys. No file means no authority."""
    env = os.environ if environ is None else environ
    location = path or env.get(TRUST_FILE_ENV, "")
    if not location:
        raise AuthorityRefused(
            "no-trust-anchors",
            f"no pinned host keys: pass --trust-file or set {TRUST_FILE_ENV}",
        )
    data = _read_json(location, "trust file", "no-trust-anchors")
    if not isinstance(data, dict) or data.get("schema") != TRUST_SCHEMA:
        raise AuthorityRefused("no-trust-anchors", f"trust file is not {TRUST_SCHEMA}")
    unknown = set(data) - {"schema", "audience", "issuers", "max_lifetime_seconds", "clock_skew_seconds"}
    if unknown:
        raise AuthorityRefused("no-trust-anchors", f"trust file has unknown fields: {sorted(unknown)}")
    audience = data.get("audience")
    if not isinstance(audience, str) or not audience:
        raise AuthorityRefused("no-trust-anchors", "trust file must name the expected audience")
    issuers_raw = data.get("issuers")
    if not isinstance(issuers_raw, dict) or not issuers_raw:
        raise AuthorityRefused("no-trust-anchors", "trust file pins no issuers")
    issuers: dict[str, tuple[str, ...]] = {}
    for issuer, keys in issuers_raw.items():
        if not isinstance(issuer, str) or not issuer or not isinstance(keys, list) or not keys:
            raise AuthorityRefused("no-trust-anchors", "each issuer needs a non-empty key list")
        for key in keys:
            if not isinstance(key, str) or not _HEX64.match(key):
                raise AuthorityRefused("no-trust-anchors", "keys are 32-byte Ed25519 public keys in hex")
        issuers[issuer] = tuple(keys)
    def seconds(name: str, default_ms: int) -> int:
        value = data.get(name)
        if value is None:
            return default_ms
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AuthorityRefused("no-trust-anchors", f"{name} must be a positive integer")
        return value * 1000
    return TrustAnchors(
        audience=audience,
        issuers=issuers,
        max_lifetime_ms=seconds("max_lifetime_seconds", _DEFAULT_MAX_LIFETIME_MS),
        clock_skew_ms=seconds("clock_skew_seconds", _DEFAULT_SKEW_MS),
        source=str(location),
    )


# ── host side: keys and assertions ───────────────────────────────────


def generate_host_key() -> tuple[str, str]:
    """A new Ed25519 key pair as ``(private_hex, public_hex)``. For the host, never the agent."""
    private_cls, _, _, encoding, no_encryption, private_format, public_format = _crypto()
    key = private_cls.generate()
    private_hex = key.private_bytes(encoding.Raw, private_format.Raw, no_encryption()).hex()
    public_hex = key.public_key().public_bytes(encoding.Raw, public_format.Raw).hex()
    return private_hex, public_hex


def issue_assertion(
    private_key_hex: str,
    *,
    issuer: str,
    audience: str,
    subject: str,
    workspace: str,
    task: str,
    handoff: str,
    scope: Sequence[str] = (SCOPE_ACCEPT, SCOPE_WRITE),
    ttl_seconds: float = 900.0,
    now_ms: int | None = None,
    not_before_ms: int | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Sign one assertion. This runs on the trusted host."""
    private_cls, _, _, encoding, _, _, public_format = _crypto()
    key = private_cls.from_private_bytes(bytes.fromhex(private_key_hex))
    issued = int(now_ms if now_ms is not None else time.time() * 1000)
    claims = {
        "aud": audience,
        "exp": issued + int(ttl_seconds * 1000),
        "handoff": handoff,
        "iat": issued,
        "iss": issuer,
        "jti": attempt_id or uuid.uuid4().hex,
        "nbf": issued if not_before_ms is None else int(not_before_ms),
        "scope": sorted(set(scope)),
        "sub": subject,
        "task": task,
        "workspace": workspace,
    }
    public_hex = key.public_key().public_bytes(encoding.Raw, public_format.Raw).hex()
    return {
        "schema": ASSERTION_SCHEMA,
        "claims": claims,
        "kid": key_id(public_hex),
        "signature": key.sign(_SIGNATURE_DOMAIN + _canonical(claims)).hex(),
    }


# ── verifier side ────────────────────────────────────────────────────


def load_assertion(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = _read_json(path, "assertion", "malformed-assertion")
    if not isinstance(data, dict):
        raise AuthorityRefused("malformed-assertion", "assertion must be a JSON object")
    return data


def verify_assertion(
    assertion: Mapping[str, Any],
    trust: TrustAnchors,
    *,
    now_ms: int,
    allow_expired: bool = False,
) -> dict[str, Any]:
    """Check shape, pinned signature and validity window. Return the claims."""
    if set(assertion) != {"schema", "claims", "kid", "signature"}:
        raise AuthorityRefused("malformed-assertion", "assertion fields are schema, claims, kid, signature")
    if assertion.get("schema") != ASSERTION_SCHEMA:
        raise AuthorityRefused("malformed-assertion", f"assertion is not {ASSERTION_SCHEMA}")
    claims = assertion.get("claims")
    if not isinstance(claims, dict):
        raise AuthorityRefused("malformed-assertion", "claims must be an object")
    missing = sorted(set(CLAIM_NAMES) - set(claims))
    if missing:
        raise AuthorityRefused("missing-claim", f"missing claims: {missing}")
    extra = sorted(set(claims) - set(CLAIM_NAMES))
    if extra:
        raise AuthorityRefused("unknown-claim", f"unknown claims: {extra}")
    for name in _TEXT_CLAIMS:
        _text(claims[name], name)
    for name in _TIME_CLAIMS:
        value = claims[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuthorityRefused("malformed-assertion", f"{name} must be a non-negative integer of milliseconds")
    scope = claims["scope"]
    if (
        not isinstance(scope, list) or not scope
        or len(set(scope)) != len(scope)
        or not all(isinstance(item, str) for item in scope)
    ):
        raise AuthorityRefused("malformed-assertion", "scope must be a non-empty list of distinct strings")
    if set(scope) - KNOWN_SCOPES:
        raise AuthorityRefused("malformed-assertion", f"unknown scope: {sorted(set(scope) - KNOWN_SCOPES)}")
    kid = assertion.get("kid")
    signature = assertion.get("signature")
    if not isinstance(kid, str) or not isinstance(signature, str):
        raise AuthorityRefused("malformed-assertion", "kid and signature must be text")

    keys = trust.issuers.get(claims["iss"])
    if keys is None:
        raise AuthorityRefused("unknown-issuer", f"issuer {claims['iss']!r} is not pinned in {trust.source}")
    pinned = [key for key in keys if key_id(key) == kid]
    if not pinned:
        raise AuthorityRefused("unpinned-key", f"key {kid} is not pinned for issuer {claims['iss']!r}")
    _, public_cls, invalid_signature, *_ = _crypto()
    try:
        signature_bytes = bytes.fromhex(signature)
        public_cls.from_public_bytes(bytes.fromhex(pinned[0])).verify(
            signature_bytes, _SIGNATURE_DOMAIN + _canonical(claims)
        )
    except (ValueError, invalid_signature) as exc:
        raise AuthorityRefused("bad-signature", "signature does not verify under the pinned key") from exc

    if claims["aud"] != trust.audience:
        raise AuthorityRefused("wrong-audience", f"assertion is for {claims['aud']!r}, not {trust.audience!r}")
    if not claims["exp"] > claims["nbf"] or not claims["exp"] > claims["iat"]:
        raise AuthorityRefused("malformed-assertion", "exp must follow nbf and iat")
    if claims["exp"] - claims["iat"] > trust.max_lifetime_ms:
        raise AuthorityRefused("lifetime-too-long", "assertion lifetime exceeds the configured maximum")
    if now_ms + trust.clock_skew_ms < claims["nbf"]:
        raise AuthorityRefused("not-yet-valid", "assertion is not valid yet")
    if not allow_expired and now_ms - trust.clock_skew_ms > claims["exp"]:
        raise AuthorityRefused("expired", "assertion has expired")
    return dict(claims)


def assertion_digest(assertion: Mapping[str, Any]) -> str:
    return _sha256(_canonical(dict(assertion)))


def attempt_digest(attempt_id: str) -> str:
    return _sha256(attempt_id.encode("utf-8"))


# ── offers ───────────────────────────────────────────────────────────


def build_offer(
    *,
    handoff_receipt: Mapping[str, Any],
    workstream: str,
    from_agent: str,
    to_agent: str,
    repo_id: str,
    base_epoch: int,
    workspace_commitment: str,
    evidence: Sequence[Mapping[str, Any]] = (),
    from_authority: Mapping[str, Any] | None = None,
    created_at_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": OFFER_SCHEMA,
        "repo_id": repo_id,
        "workstream": workstream,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "base_epoch": int(base_epoch),
        "handoff_commitment": str(handoff_receipt.get("payload_commitment", "")),
        "workspace_commitment": workspace_commitment,
        "evidence": [dict(item) for item in evidence],
        "from_authority": dict(from_authority) if from_authority else None,
        "created_at_ms": int(created_at_ms if created_at_ms is not None else time.time() * 1000),
    }


def offer_digest(offer: Mapping[str, Any]) -> str:
    return _sha256(_canonical(dict(offer)))


def load_offer(state_dir: str | os.PathLike[str], handoff_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(inbox_payload, offer)`` for ``handoff:<sha256>``; fail closed."""
    match = re.fullmatch(r"handoff:([0-9a-f]{64})", str(handoff_ref))
    if match is None:
        raise AuthorityRefused("offer-missing", "handoff refs look like handoff:<64 hex>")
    path = Path(state_dir) / "handoffs" / f"{match.group(1)}.json"
    payload = _read_json(path, "handoff offer", "offer-missing")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AuthorityRefused("offer-invalid", "inbox entry is not a schema 1 handoff")
    offer = payload.get("offer")
    if not isinstance(offer, dict) or offer.get("schema") != OFFER_SCHEMA:
        raise AuthorityRefused(
            "offer-missing",
            "this handoff carries no authority offer; recreate it with entroly-work handoff",
        )
    handoff = payload.get("handoff")
    if not isinstance(handoff, dict) or handoff.get("payload_commitment") != offer.get("handoff_commitment"):
        raise AuthorityRefused("offer-invalid", "offer does not commit to its handoff receipt")
    expected_name = hashlib.sha256(str(handoff["payload_commitment"]).encode("utf-8")).hexdigest()
    if expected_name != match.group(1):
        raise AuthorityRefused("offer-invalid", "inbox entry name does not match its receipt")
    from .work_graph import WorkGraph

    try:
        intact = WorkGraph.verify_handoff_integrity(handoff)
    except Exception:
        intact = False
    if not intact:
        raise AuthorityRefused("offer-invalid", "handoff receipt fails its integrity check")
    if not isinstance(offer.get("base_epoch"), int) or isinstance(offer.get("base_epoch"), bool):
        raise AuthorityRefused("offer-invalid", "offer has no base epoch")
    return payload, offer


# ── the epoch log ────────────────────────────────────────────────────


class AuthorityLog:
    """Append-only, hash-chained authority epochs for one workstream."""

    def __init__(self, state_dir: str | os.PathLike[str], workstream: str) -> None:
        self.workstream = workstream
        key = hashlib.sha256(workstream.encode("utf-8")).hexdigest()[:32]
        self.directory = Path(state_dir) / "authority" / key

    def _ensure(self) -> Path:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise AuthorityRefused("authority-log-corrupt", "authority directory is unsafe")
        return self.directory

    def history(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise AuthorityRefused("authority-log-corrupt", "authority directory is unsafe")
        numbered = []
        for entry in self.directory.iterdir():
            match = _EPOCH_NAME.match(entry.name)
            if match:
                numbered.append((int(match.group(1)), entry))
        numbered.sort()
        records: list[dict[str, Any]] = []
        previous = ""
        for expected, (number, entry) in enumerate(numbered, start=1):
            if number != expected or entry.is_symlink():
                raise AuthorityRefused("authority-log-corrupt", f"epoch {number} breaks the sequence")
            record = _read_json(entry, f"epoch {number}", "authority-log-corrupt")
            if (
                not isinstance(record, dict)
                or record.get("schema") != EPOCH_SCHEMA
                or record.get("epoch") != number
                or record.get("base_epoch") != number - 1
                or record.get("workstream") != self.workstream
                or record.get("prev_digest") != previous
            ):
                raise AuthorityRefused("authority-log-corrupt", f"epoch {number} does not chain")
            records.append(record)
            previous = _sha256(_canonical(record))
        return records

    def publish(self, record: Mapping[str, Any]) -> bool:
        """Publish epoch ``record['epoch']``. ``False`` means another acceptance won."""
        folder = self._ensure()
        target = folder / f"epoch-{int(record['epoch']):06d}.json"
        temporary = folder / f".epoch-{uuid.uuid4().hex}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical(dict(record)))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                return False
            _fsync_dir(folder)
            return True
        finally:
            temporary.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def chain_digest(record: Mapping[str, Any]) -> str:
    return _sha256(_canonical(dict(record)))


# ── transitions ──────────────────────────────────────────────────────


def _epoch_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": record.get("epoch"),
        "holder": record.get("holder"),
        "issuer": record.get("issuer"),
        "key_id": record.get("key_id"),
        "authority_expires_at_ms": record.get("authority_expires_at_ms"),
        "accepted_at_ms": record.get("accepted_at_ms"),
        "offer_digest": record.get("offer_digest"),
        "record_digest": chain_digest(record),
    }


def require_write_authority(
    state_dir: str | os.PathLike[str],
    *,
    repo_id: str,
    workstream: str,
    assertion: Mapping[str, Any] | None,
    trust: TrustAnchors | None,
    now_ms: int | None = None,
) -> dict[str, Any] | None:
    """Fencing. ``None`` when no authority was ever accepted for the workstream."""
    history = AuthorityLog(state_dir, workstream).history()
    if not history:
        return None
    current = history[-1]
    if assertion is None:
        raise AuthorityRefused(
            "fenced",
            f"workstream is held at epoch {current['epoch']} by {current['holder']!r}; "
            "writes need that holder's host assertion",
        )
    if trust is None:
        raise AuthorityRefused("no-trust-anchors", "writes to a held workstream need pinned host keys")
    timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
    claims = verify_assertion(assertion, trust, now_ms=timestamp, allow_expired=True)
    if SCOPE_WRITE not in claims["scope"]:
        raise AuthorityRefused("scope-missing", "assertion does not grant work:write")
    if claims["workspace"] != repo_id or claims["task"] != workstream:
        raise AuthorityRefused("fenced", "assertion is for another workspace or workstream")
    if attempt_digest(claims["jti"]) != current.get("attempt_digest") or claims["sub"] != current.get("holder"):
        raise AuthorityRefused(
            "fenced",
            f"authority moved to epoch {current['epoch']} ({current['holder']!r}); this assertion is not current",
        )
    if timestamp - trust.clock_skew_ms > claims["exp"]:
        raise AuthorityRefused("holder-expired", f"authority at epoch {current['epoch']} has expired")
    return _epoch_summary(current)


def accept_handoff(
    state_dir: str | os.PathLike[str],
    *,
    repo_id: str,
    handoff_ref: str,
    assertion: Mapping[str, Any],
    trust: TrustAnchors,
    workspace_commitment: str,
    assess_evidence: Any = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Verify the host assertion and move authority to its subject, once."""
    from .work_graph_recovery_ack import pending as recovery_pending

    if recovery_pending(state_dir) is not None:
        raise AuthorityRefused(
            "recovery-unacknowledged",
            "a recovered state awaits acknowledgement; acknowledge it before accepting a handoff",
        )
    timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
    _, offer = load_offer(state_dir, handoff_ref)
    digest = offer_digest(offer)
    claims = verify_assertion(assertion, trust, now_ms=timestamp)
    if SCOPE_ACCEPT not in claims["scope"]:
        raise AuthorityRefused("scope-missing", "assertion does not grant work:accept")
    if claims["workspace"] != repo_id or offer.get("repo_id") != repo_id:
        raise AuthorityRefused("wrong-workspace", "assertion or offer is for another workspace")
    if claims["task"] != offer.get("workstream"):
        raise AuthorityRefused("wrong-task", "assertion is for another workstream")
    if claims["handoff"] != digest:
        raise AuthorityRefused("wrong-handoff", "assertion is for a different offer than this inbox entry")
    if claims["sub"] != offer.get("to_agent"):
        raise AuthorityRefused("wrong-subject", "assertion names a different successor than the offer")

    log = AuthorityLog(state_dir, str(offer["workstream"]))
    history = log.history()
    attempt = attempt_digest(claims["jti"])
    for record in history:
        if record.get("attempt_digest") == attempt:
            if record is history[-1] and record.get("offer_digest") == digest:
                return {"outcome": "already-accepted", "epoch": _epoch_summary(record),
                        "evidence": record.get("evidence", [])}
            raise AuthorityRefused("replayed-attempt", "this attempt was already used")
    if offer["base_epoch"] != len(history):
        raise AuthorityRefused(
            "superseded",
            f"offer was made at epoch {offer['base_epoch']}; the workstream is now at epoch {len(history)}",
        )

    evidence = []
    for item in offer.get("evidence") or ():
        decision = assess_evidence(item) if assess_evidence is not None else None
        evidence.append({
            "record_id": item.get("record_id"),
            "record_digest": item.get("record_digest"),
            "decision": (decision or {}).get("decision", "unknown"),
            "reasons": (decision or {}).get("reasons", ["not-assessed"]),
            "changed_count": (decision or {}).get("changed_count", 0),
        })
    record = {
        "schema": EPOCH_SCHEMA,
        "workstream": offer["workstream"],
        "epoch": len(history) + 1,
        "base_epoch": len(history),
        "holder": claims["sub"],
        "issuer": claims["iss"],
        "key_id": assertion["kid"],
        "scope": claims["scope"],
        "attempt_digest": attempt,
        "assertion_digest": assertion_digest(assertion),
        "offer_digest": digest,
        "handoff_ref": handoff_ref,
        "offer_workspace_commitment": offer.get("workspace_commitment"),
        "workspace_commitment": workspace_commitment,
        "authority_expires_at_ms": claims["exp"],
        "accepted_at_ms": timestamp,
        "evidence": evidence,
        "prev_digest": chain_digest(history[-1]) if history else "",
    }
    if not log.publish(record):
        raise AuthorityRefused("superseded", "another acceptance for this workstream won the race")
    return {"outcome": "accepted", "epoch": _epoch_summary(record), "evidence": evidence}


def authority_view(state_dir: str | os.PathLike[str], workstream: str, *, now_ms: int | None = None) -> dict[str, Any]:
    timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
    history = AuthorityLog(state_dir, workstream).history()
    if not history:
        return {"workstream": workstream, "epoch": 0, "held": False}
    current = history[-1]
    return {
        "workstream": workstream,
        "epoch": current["epoch"],
        "held": True,
        "current": _epoch_summary(current),
        "expired": timestamp > int(current.get("authority_expires_at_ms", 0)),
        "history": [_epoch_summary(record) for record in history],
    }


__all__ = [
    "ASSERTION_FILE_ENV",
    "AuthorityLog",
    "AuthorityRefused",
    "SCOPE_ACCEPT",
    "SCOPE_WRITE",
    "TRUST_FILE_ENV",
    "TrustAnchors",
    "accept_handoff",
    "attempt_digest",
    "authority_view",
    "build_offer",
    "chain_digest",
    "generate_host_key",
    "issue_assertion",
    "key_id",
    "load_assertion",
    "load_offer",
    "load_trust",
    "offer_digest",
    "require_write_authority",
    "verify_assertion",
]
