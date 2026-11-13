"""Signed, offline-verifiable proof bundles for context evidence.

A proof bundle packages evidence Entroly already produces -- Context Commits,
a session receipt chain, WITNESS taint records -- into one file an outside
reviewer can check without Entroly's stores, a network, or the exporter's
cooperation:

* every Context Commit is re-verified from its own bytes (commit id, receipt
  hash, selected-context digest, recovery of every omitted chunk);
* the session receipt chain's parent links and chain hash are recomputed;
* one Merkle root covers every evidence item, in a fixed order;
* the whole bundle is content-addressed, and optionally signed with Ed25519.

What a bundle does not establish is stated in ``LIMITS`` and travels inside
every bundle, so the limits are covered by the digest and the signature too.

Verification fails closed: a field that cannot be recomputed, a recorded
result that differs from the recomputed one, or a signature that cannot be
checked makes the bundle invalid. An unsigned bundle is never reported as
authenticated.
"""

from __future__ import annotations

import copy
import hashlib
import html
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .context_receipts.models import stable_hash, stable_json

PROOF_BUNDLE_SCHEMA = "entroly.proof-bundle.v1"
_SIGNATURE_DOMAIN = "entroly.proof-bundle-signature.v1"
_KEY_DERIVATION_DOMAIN = b"entroly.proof-bundle-key.v1\x00"
_UNSIGNED_FIELDS = ("bundle_id", "bundle_digest", "signature")

SIGNING_KEY_ENV = "ENTROLY_PROOF_SIGNING_KEY"
IDENTITY_KEY_ENV = "ENTROLY_IDENTITY_KEY"

#: Travels inside every bundle. Each line is something a reader might assume
#: and should not.
LIMITS: tuple[str, ...] = (
    "Replay is of the context: the bundle shows which bytes were selected for "
    "the model and that every omitted chunk is recoverable. It does not "
    "reproduce or vouch for the model's output.",
    "WITNESS records are Entroly's own local checks of model output against "
    "the supplied context. They have measured false-positive and "
    "false-negative rates and are not a guarantee of factual accuracy.",
    "Timestamps are the exporter's clock. The bundle carries no trusted "
    "third-party timestamp.",
    "A signature shows the bundle was produced by the holder of the signing "
    "key and has not changed since. It identifies a person or organisation "
    "only if you obtained the public key from them through a channel you trust.",
    "The bundle covers the evidence it was given. It cannot show that no "
    "other commits or turns existed.",
    "This is technical evidence. It is not a certification and not a finding "
    "of compliance with any law, regulation or standard.",
)

STATUS_VERIFIED = "verified"                # signed, key pinned by the verifier
STATUS_SIGNED_UNPINNED = "signed_unpinned"  # signed, signer not authenticated
STATUS_INTEGRITY_ONLY = "integrity_only"    # unsigned, self-consistent
STATUS_INVALID = "invalid"


class ProofBundleError(RuntimeError):
    """The bundle could not be built: missing evidence or an unusable key."""


@dataclass(frozen=True)
class ProofVerification:
    status: str
    valid: bool
    bundle_id: str
    checks: dict[str, bool]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    signed: bool = False
    public_key: str = ""
    key_pinned: bool = False
    context_commits: int = 0
    session_turns: int = 0
    limits: tuple[str, ...] = field(default=LIMITS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── keys ──────────────────────────────────────────────────────────────────


def load_signing_key(environ: Mapping[str, str] | None = None) -> tuple[Any, str]:
    """Return ``(AttestationKey, key_source)`` or raise ``ProofBundleError``.

    ``ENTROLY_PROOF_SIGNING_KEY`` is a 32-byte Ed25519 private key in hex.
    Without it, the key is derived from ``ENTROLY_IDENTITY_KEY`` under a
    fixed domain label, so an operator who already holds an identity key
    needs no second secret. A derived key is exactly as strong as that
    secret; the bundle records which source was used.
    """
    from .receipt_attestation import AttestationKey

    env = os.environ if environ is None else environ
    explicit = (env.get(SIGNING_KEY_ENV) or "").strip()
    try:
        if explicit:
            raw = bytes.fromhex(explicit)
            if len(raw) != 32:
                raise ValueError("expected 32 bytes")
            return AttestationKey.from_private_hex(explicit), SIGNING_KEY_ENV
        identity = env.get(IDENTITY_KEY_ENV) or ""
        if identity.strip():
            seed = hashlib.sha256(_KEY_DERIVATION_DOMAIN + identity.encode("utf-8"))
            return (
                AttestationKey.from_private_hex(seed.hexdigest()),
                f"derived:{IDENTITY_KEY_ENV}",
            )
    except ValueError as exc:
        raise ProofBundleError(
            f"{SIGNING_KEY_ENV} must be a 32-byte Ed25519 private key in hex ({exc})"
        ) from exc
    except RuntimeError as exc:  # cryptography is not installed
        raise ProofBundleError(str(exc)) from exc
    raise ProofBundleError(
        f"signing needs {SIGNING_KEY_ENV} (Ed25519 private key, hex) or "
        f"{IDENTITY_KEY_ENV}; neither is set"
    )


def generate_signing_key() -> tuple[str, str]:
    """Return a fresh ``(private_hex, public_hex)`` Ed25519 pair."""
    from .receipt_attestation import AttestationKey

    try:
        key = AttestationKey.generate()
        return key.private_hex(), key.public_hex()
    except RuntimeError as exc:
        raise ProofBundleError(str(exc)) from exc


# ── building ──────────────────────────────────────────────────────────────


def _order_commits(commits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Parents before children; input order breaks ties and keeps strays."""
    remaining = [copy.deepcopy(dict(commit)) for commit in commits]
    ids = {str(commit.get("commit_id")) for commit in remaining}
    placed: set[str] = set()
    ordered: list[dict[str, Any]] = []
    while remaining:
        progressed = False
        for commit in list(remaining):
            parent = commit.get("parent_commit_id")
            if parent is None or str(parent) not in ids or str(parent) in placed:
                ordered.append(commit)
                placed.add(str(commit.get("commit_id")))
                remaining.remove(commit)
                progressed = True
        if not progressed:  # a parent cycle: keep input order, verification reports it
            ordered.extend(remaining)
            break
    return ordered


def _token_total(items: Any) -> int:
    total = 0
    if isinstance(items, list):
        for item in items:
            if isinstance(item, Mapping):
                value = item.get("tokens", item.get("token_count", 0))
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total += int(value)
    return total


def _commit_section(
    commits: Sequence[Mapping[str, Any]], *, embed_content: bool
) -> dict[str, Any]:
    from .context_commit import context_commit_anchor, verify_context_commit

    entries: list[dict[str, Any]] = []
    known: set[str] = set()
    for commit in _order_commits(commits):
        result = verify_context_commit(commit)
        receipt = commit.get("receipt") if isinstance(commit.get("receipt"), Mapping) else {}
        commit_id = str(commit.get("commit_id") or "")
        parent = commit.get("parent_commit_id")
        entry: dict[str, Any] = {
            "commit_id": commit_id,
            "anchor": context_commit_anchor(commit),
            "parent_commit_id": parent,
            "parent_in_bundle": parent is None or str(parent) in known,
            "query": str(receipt.get("query") or ""),
            "token_budget": receipt.get("token_budget"),
            "selected_tokens": _token_total(receipt.get("selected_context")),
            "omitted_tokens": _token_total(receipt.get("omitted_context")),
            "risk_summary": copy.deepcopy(receipt.get("risk_summary") or {}),
            "verification": result.to_dict(),
        }
        if embed_content:
            entry["commit"] = commit
        entries.append(entry)
        known.add(commit_id)
    return {
        "count": len(entries),
        "content_embedded": bool(embed_content),
        "head_commit_id": entries[-1]["commit_id"] if entries else None,
        "entries": entries,
    }


def _session_section(
    session_chain: Mapping[str, Any] | None, taint: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if session_chain is None:
        return None
    from .session_intelligence import SessionReceiptChain

    chain = SessionReceiptChain.from_dict(session_chain)
    section: dict[str, Any] = {
        "session_id": chain.session_id,
        "chain": copy.deepcopy(dict(session_chain)),
        "integrity": chain.verify_integrity(),
        "turns": [
            {
                "turn_index": link.turn_index,
                "receipt_id": link.receipt_id,
                "receipt_hash": link.receipt_hash,
                "token_budget": link.token_budget,
                "risk_summary": dict(link.risk_summary),
            }
            for link in chain.links
        ],
    }
    if taint is not None:
        suspects = taint.get("suspects")
        section["witness_taint"] = copy.deepcopy(dict(taint))
        section["witness_flagged_entities"] = (
            len(suspects) if isinstance(suspects, Mapping) else 0
        )
    return section


def _evidence_leaves(bundle: Mapping[str, Any]) -> list[bytes]:
    """One Merkle leaf per evidence item, in the order the bundle lists them."""
    from .receipt_merkle import leaf_hash

    leaves: list[bytes] = []
    commits = bundle.get("context_commits")
    if isinstance(commits, Mapping):
        for entry in commits.get("entries") or []:
            leaves.append(leaf_hash(f"commit:{entry.get('anchor')}".encode("utf-8")))
    session = bundle.get("session")
    if isinstance(session, Mapping):
        for turn in session.get("turns") or []:
            leaves.append(leaf_hash(f"turn:{turn.get('receipt_hash')}".encode("utf-8")))
        if "witness_taint" in session:
            leaves.append(
                leaf_hash(f"taint:{stable_hash(session['witness_taint'])}".encode("utf-8"))
            )
    return leaves


def _merkle_section(bundle: Mapping[str, Any]) -> dict[str, Any]:
    from .receipt_merkle import merkle_root

    leaves = _evidence_leaves(bundle)
    return {
        "hash": "sha256",
        "tree_size": len(leaves),
        "root_hash": merkle_root(leaves).hex() if leaves else "",
        "leaves": [leaf.hex() for leaf in leaves],
    }


def _digest(bundle: Mapping[str, Any]) -> str:
    return stable_hash({k: v for k, v in bundle.items() if k not in _UNSIGNED_FIELDS})


def _signature_message(digest: str, signed_at: float) -> bytes:
    return stable_json(
        {"domain": _SIGNATURE_DOMAIN, "bundle_digest": digest, "signed_at": signed_at}
    ).encode("utf-8")


def build_proof_bundle(
    *,
    commits: Sequence[Mapping[str, Any]] = (),
    session_chain: Mapping[str, Any] | None = None,
    taint: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    embed_content: bool = True,
    signing_key: Any | None = None,
    key_source: str = "",
    created_at: float | None = None,
) -> dict[str, Any]:
    """Assemble a proof bundle from evidence the caller already holds.

    ``embed_content=False`` leaves the Context Commits out and keeps only
    their anchors and the verification result recorded at export. Such a
    bundle is smaller and carries no source text, and a verifier can no
    longer re-run the replay and recovery checks; it will say so.
    """
    if not commits and session_chain is None:
        raise ProofBundleError("a proof bundle needs a Context Commit or a session chain")
    if taint is not None and session_chain is None:
        raise ProofBundleError("WITNESS taint records need the session chain they refer to")

    session = _session_section(session_chain, taint)
    if session_id is not None and session is not None and session["session_id"] != session_id:
        raise ProofBundleError(
            f"session chain is for {session['session_id']!r}, not {session_id!r}"
        )
    try:
        from . import __version__ as version
    except Exception:  # pragma: no cover - source tree without metadata
        version = "source"

    bundle: dict[str, Any] = {
        "schema_version": PROOF_BUNDLE_SCHEMA,
        "created_at": round(time.time() if created_at is None else float(created_at), 6),
        "generator": {"name": "entroly", "version": str(version)},
        "subject": {"session_id": session["session_id"] if session else session_id},
        "context_commits": _commit_section(commits, embed_content=embed_content),
        "session": session,
        "limits": list(LIMITS),
    }
    bundle["merkle"] = _merkle_section(bundle)
    digest = _digest(bundle)
    bundle["bundle_digest"] = "sha256:" + digest
    bundle["bundle_id"] = "proof_" + digest[:24]
    bundle["signature"] = None
    if signing_key is not None:
        signed_at = round(time.time(), 6)
        bundle["signature"] = {
            "algorithm": "ed25519",
            "public_key": signing_key.public_hex(),
            "key_source": key_source,
            "signed_at": signed_at,
            "signature": signing_key.sign(_signature_message(digest, signed_at)),
        }
    return bundle


# ── verifying ─────────────────────────────────────────────────────────────


def _signature_ok(signature: Mapping[str, Any], digest: str) -> tuple[bool, str]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return False, "signature present but the 'cryptography' package is not installed"
    if signature.get("algorithm") != "ed25519":
        return False, f"unsupported signature algorithm {signature.get('algorithm')!r}"
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(str(signature.get("public_key", "")))
        ).verify(
            bytes.fromhex(str(signature.get("signature", ""))),
            _signature_message(digest, signature.get("signed_at")),
        )
    except (InvalidSignature, TypeError, ValueError):
        return False, "signature does not match the bundle"
    return True, ""


def verify_proof_bundle(
    bundle: Mapping[str, Any] | object,
    *,
    public_key: str | None = None,
    require_signature: bool = False,
) -> ProofVerification:
    """Recompute every claim in ``bundle``; fail closed on any difference."""
    if not isinstance(bundle, Mapping):
        return ProofVerification(
            STATUS_INVALID, False, "", {}, ("bundle is not a JSON object",)
        )
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []

    def check(name: str, passed: bool, message: str) -> bool:
        checks[name] = bool(passed)
        if not passed:
            errors.append(message)
        return bool(passed)

    check(
        "schema",
        bundle.get("schema_version") == PROOF_BUNDLE_SCHEMA,
        f"unknown schema {bundle.get('schema_version')!r}",
    )

    try:
        digest = _digest(bundle)
    except (TypeError, ValueError) as exc:
        return ProofVerification(
            STATUS_INVALID, False, str(bundle.get("bundle_id") or ""),
            {**checks, "bundle_digest": False}, (f"bundle is not canonical JSON: {exc}",),
        )
    check(
        "bundle_digest",
        bundle.get("bundle_digest") == "sha256:" + digest
        and bundle.get("bundle_id") == "proof_" + digest[:24],
        "bundle content does not match its digest (modified after export)",
    )
    check(
        "limits",
        list(bundle.get("limits") or []) == list(LIMITS),
        "the bundle's statement of limits is missing or altered",
    )

    # Context Commits: re-verify from the embedded bytes where they exist.
    commit_count = 0
    section = bundle.get("context_commits")
    commits_ok = isinstance(section, Mapping) and isinstance(section.get("entries"), list)
    if commits_ok:
        from .context_commit import context_commit_anchor, verify_context_commit

        known: set[str] = set()
        embedded = bool(section.get("content_embedded"))
        for index, entry in enumerate(section["entries"]):
            if not isinstance(entry, Mapping):
                commits_ok = False
                errors.append(f"context commit {index} is malformed")
                continue
            commit_count += 1
            commit_id = str(entry.get("commit_id") or "")
            recorded = entry.get("verification")
            commit = entry.get("commit")
            if embedded:
                if not isinstance(commit, Mapping):
                    commits_ok = False
                    errors.append(f"{commit_id}: commit content is missing")
                    continue
                result = verify_context_commit(commit, trusted_anchor=str(entry.get("anchor")))
                if not result.valid:
                    commits_ok = False
                    errors.append(f"{commit_id}: " + ", ".join(result.errors))
                expected = result.to_dict()
                expected["checks"] = {
                    k: v for k, v in expected["checks"].items() if k != "trusted_anchor"
                }
                expected["errors"] = [e for e in expected["errors"] if e != "trusted_anchor"]
                expected["valid"] = not expected["errors"]
                same = (
                    isinstance(recorded, Mapping)
                    and dict(recorded.get("checks") or {}) == expected["checks"]
                    and bool(recorded.get("valid")) == expected["valid"]
                    and recorded.get("selected_chunks") == expected["selected_chunks"]
                    and recorded.get("omitted_chunks") == expected["omitted_chunks"]
                    and recorded.get("recovered_omitted_chunks")
                    == expected["recovered_omitted_chunks"]
                )
                if not same:
                    commits_ok = False
                    errors.append(f"{commit_id}: recorded verification differs from recomputed")
                if (
                    commit.get("commit_id") != commit_id
                    or context_commit_anchor(commit) != entry.get("anchor")
                    or commit.get("parent_commit_id") != entry.get("parent_commit_id")
                ):
                    commits_ok = False
                    errors.append(f"{commit_id}: summary does not match the embedded commit")
            elif not (isinstance(recorded, Mapping) and recorded.get("valid") is True):
                commits_ok = False
                errors.append(f"{commit_id}: exporter recorded a failed verification")
            parent = entry.get("parent_commit_id")
            in_bundle = parent is None or str(parent) in known
            if bool(entry.get("parent_in_bundle")) != in_bundle:
                commits_ok = False
                errors.append(f"{commit_id}: parent linkage is misreported")
            elif not in_bundle:
                warnings.append(f"{commit_id}: parent {parent} is not in this bundle")
            known.add(commit_id)
        if commit_count != section.get("count"):
            commits_ok = False
            errors.append("context commit count does not match the entries")
        if commit_count and not embedded:
            warnings.append(
                "commit content is not embedded: replay and recovery were checked by "
                "the exporter and cannot be re-run from this bundle"
            )
    elif section is not None:
        errors.append("context commit section is malformed")
    # A section with nothing in it is not reported as a passed check.
    if commit_count or not commits_ok:
        checks["context_commits"] = bool(commits_ok)

    # Session receipt chain.
    turns = 0
    session = bundle.get("session")
    session_ok = session is None
    if isinstance(session, Mapping):
        from .session_intelligence import SessionReceiptChain

        try:
            chain = SessionReceiptChain.from_dict(session.get("chain") or {})
            integrity = chain.verify_integrity()
            turns = len(chain.links)
            listed = [
                (t.get("turn_index"), t.get("receipt_id"), t.get("receipt_hash"))
                for t in session.get("turns") or []
            ]
            actual = [
                (link.turn_index, link.receipt_id, link.receipt_hash) for link in chain.links
            ]
            session_ok = (
                bool(integrity["valid"])
                and bool((session.get("chain") or {}).get("chain_hash"))
                and listed == actual
                and session.get("session_id") == chain.session_id
                and (bundle.get("subject") or {}).get("session_id") == chain.session_id
            )
            if not session_ok:
                errors.append(
                    "session receipt chain failed: "
                    + (", ".join(integrity["issues"]) or "summary does not match the chain")
                )
        except (KeyError, TypeError, ValueError) as exc:
            session_ok = False
            errors.append(f"session receipt chain is malformed: {exc}")
    elif session is not None:
        errors.append("session section is malformed")
    if session is not None:
        checks["session_chain"] = bool(session_ok)

    try:
        expected_merkle = _merkle_section(bundle)
    except (AttributeError, KeyError, TypeError, ValueError):
        expected_merkle = None
    check(
        "merkle_root",
        expected_merkle is not None
        and bundle.get("merkle") == expected_merkle
        and expected_merkle["tree_size"] > 0,
        "Merkle root does not cover the evidence in the bundle",
    )

    signature = bundle.get("signature")
    signed = isinstance(signature, Mapping)
    signer = str(signature.get("public_key") or "") if signed else ""
    pinned = False
    if signed:
        ok, reason = _signature_ok(signature, digest)
        check("signature", ok, reason or "signature invalid")
        if public_key is not None:
            pinned = check(
                "signer",
                ok and signer.lower() == public_key.strip().lower(),
                "bundle is not signed by the expected public key",
            )
    else:
        if signature is not None:
            check("signature", False, "signature field is malformed")
        if require_signature or public_key is not None:
            check("signature", False, "bundle is not signed")

    if errors:
        status = STATUS_INVALID
    elif signed and pinned:
        status = STATUS_VERIFIED
    elif signed:
        status = STATUS_SIGNED_UNPINNED
        warnings.append(
            "signed, but the signer was not checked: pass the public key you "
            "obtained from the issuer to authenticate who produced this bundle"
        )
    else:
        status = STATUS_INTEGRITY_ONLY
        warnings.append(
            "not signed: the bundle is self-consistent, but anyone could have produced it"
        )
    return ProofVerification(
        status=status,
        valid=not errors,
        bundle_id=str(bundle.get("bundle_id") or ""),
        checks=checks,
        errors=tuple(errors),
        warnings=tuple(warnings),
        signed=signed,
        public_key=signer,
        key_pinned=pinned,
        context_commits=commit_count,
        session_turns=turns,
    )


# ── report ────────────────────────────────────────────────────────────────

_STATUS_TEXT = {
    STATUS_VERIFIED: ("Verified", "Evidence is intact and signed by the expected key."),
    STATUS_SIGNED_UNPINNED: (
        "Intact, signer not authenticated",
        "Evidence is intact and signed. The signing key was not checked against "
        "one obtained from the issuer.",
    ),
    STATUS_INTEGRITY_ONLY: (
        "Intact, not signed",
        "Evidence is self-consistent. Nothing shows who produced it.",
    ),
    STATUS_INVALID: ("Failed verification", "Do not rely on this bundle."),
}


def render_html_report(bundle: Mapping[str, Any], verification: ProofVerification) -> str:
    """A static, dependency-free report of the bundle and its verification.

    Every value is escaped; the page loads nothing and runs no script.
    """
    e = lambda value: html.escape("" if value is None else str(value), quote=True)  # noqa: E731
    title, blurb = _STATUS_TEXT[verification.status]
    tone = {
        STATUS_VERIFIED: "#0a7b53", STATUS_SIGNED_UNPINNED: "#8a6a00",
        STATUS_INTEGRITY_ONLY: "#8a6a00", STATUS_INVALID: "#b3261e",
    }[verification.status]
    signature = bundle.get("signature") if isinstance(bundle.get("signature"), Mapping) else {}
    commits = bundle.get("context_commits") if isinstance(bundle.get("context_commits"), Mapping) else {}
    session = bundle.get("session") if isinstance(bundle.get("session"), Mapping) else {}
    merkle = bundle.get("merkle") if isinstance(bundle.get("merkle"), Mapping) else {}

    def rows(pairs: Sequence[tuple[str, Any]]) -> str:
        return "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in pairs)

    commit_rows = "".join(
        "<tr>"
        f"<td><code>{e(entry.get('commit_id'))}</code></td>"
        f"<td>{e(entry.get('query'))}</td>"
        f"<td>{e((entry.get('verification') or {}).get('selected_chunks'))}</td>"
        f"<td>{e((entry.get('verification') or {}).get('omitted_chunks'))}</td>"
        f"<td>{e((entry.get('verification') or {}).get('recovered_omitted_chunks'))}</td>"
        f"<td>{'pass' if (entry.get('verification') or {}).get('valid') else 'FAIL'}</td>"
        "</tr>"
        for entry in (commits.get("entries") or [])
        if isinstance(entry, Mapping)
    )
    turn_rows = "".join(
        "<tr>"
        f"<td>{e(turn.get('turn_index'))}</td>"
        f"<td><code>{e(turn.get('receipt_id'))}</code></td>"
        f"<td>{e(turn.get('token_budget'))}</td>"
        f"<td>{e(stable_json(turn.get('risk_summary') or {}))}</td>"
        "</tr>"
        for turn in (session.get("turns") or [])
        if isinstance(turn, Mapping)
    )
    check_rows = "".join(
        f"<tr><th>{e(name)}</th><td>{'pass' if ok else 'FAIL'}</td></tr>"
        for name, ok in verification.checks.items()
    )
    notes = "".join(f"<li>{e(item)}</li>" for item in (*verification.errors, *verification.warnings))
    limits = "".join(f"<li>{e(item)}</li>" for item in LIMITS)
    created = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(bundle.get("created_at") or 0)))
    replay_note = (
        "Re-verified from the embedded commits."
        if commits.get("content_embedded")
        else "Recorded by the exporter; commit content is not embedded, so it was not re-run."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly context proof report {e(verification.bundle_id)}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1b1f24;max-width:860px;margin:32px auto;padding:0 20px}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid #d9dee4;padding-bottom:4px}}
.status{{border-left:6px solid {tone};background:#f6f8fa;padding:12px 16px;margin:16px 0}}
.status b{{color:{tone};font-size:18px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #e6e9ed;vertical-align:top;word-break:break-word}}
th{{width:32%;font-weight:600;color:#3d444d}}thead th{{width:auto;background:#f6f8fa}}
code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}small{{color:#59636e}}
@media print{{body{{margin:0}}}}
</style></head><body>
<h1>Context proof report</h1>
<small>Entroly {e((bundle.get("generator") or {}).get("version"))} · exported {e(created)} (exporter's clock)</small>
<div class="status"><b>{e(title)}</b><br>{e(blurb)}</div>
<h2>Bundle</h2>
<table>{rows([
    ("Bundle id", verification.bundle_id),
    ("Bundle digest", bundle.get("bundle_digest")),
    ("Session", (bundle.get("subject") or {}).get("session_id") or "not given"),
    ("Context Commits", verification.context_commits),
    ("Session turns", verification.session_turns),
    ("Merkle root (" + str(merkle.get("tree_size", 0)) + " evidence items)", merkle.get("root_hash")),
    ("Signature", (str(signature.get("algorithm")) + ", key source " + str(signature.get("key_source") or "unspecified")) if signature else "none"),
    ("Signer public key", signature.get("public_key") if signature else "none"),
    ("Signer checked against a key you supplied", "yes" if verification.key_pinned else "no"),
])}</table>
<h2>Checks recomputed by the verifier</h2>
<table>{check_rows}</table>
{f"<ul>{notes}</ul>" if notes else ""}
<h2>Context Commits</h2>
<small>{e(replay_note)}</small>
<table><thead><tr><th>Commit</th><th>Query</th><th>Selected chunks</th><th>Omitted chunks</th><th>Omitted recovered</th><th>Result</th></tr></thead>
<tbody>{commit_rows or '<tr><td colspan="6">None in this bundle.</td></tr>'}</tbody></table>
<h2>Session turns</h2>
<table><thead><tr><th>Turn</th><th>Receipt</th><th>Token budget</th><th>Risk summary</th></tr></thead>
<tbody>{turn_rows or '<tr><td colspan="4">None in this bundle.</td></tr>'}</tbody></table>
<small>WITNESS flagged entities recorded: {e(session.get("witness_flagged_entities", "not included"))}</small>
<h2>What this report does not establish</h2>
<ul>{limits}</ul>
<h2>Verify it yourself</h2>
<p><code>entroly verify-proof BUNDLE.json --public-key {e(signature.get("public_key") if signature else "PUBLIC_KEY_HEX")}</code></p>
</body></html>
"""


__all__ = [
    "LIMITS",
    "PROOF_BUNDLE_SCHEMA",
    "ProofBundleError",
    "ProofVerification",
    "build_proof_bundle",
    "generate_signing_key",
    "load_signing_key",
    "render_html_report",
    "verify_proof_bundle",
]
