"""Proof bundles: signed, offline-verifiable packages of context evidence.

The properties that matter to someone handed a bundle:

* an intact bundle verifies, and says exactly how far that goes (signed and
  pinned, signed but signer unchecked, or merely self-consistent);
* any change to the evidence, the recorded results, the statement of limits
  or the metadata makes verification fail;
* a signature cannot be silently removed or replaced when the verifier asked
  for one;
* the exporter refuses to write a bundle whose evidence does not verify.
"""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from entroly.context_commit import create_context_commit  # noqa: E402
from entroly.proof_bundle import (  # noqa: E402
    LIMITS,
    PROOF_BUNDLE_SCHEMA,
    ProofBundleError,
    build_proof_bundle,
    generate_signing_key,
    load_signing_key,
    render_html_report,
    verify_proof_bundle,
)
from entroly.receipt_attestation import AttestationKey  # noqa: E402
from entroly.session_intelligence import (  # noqa: E402
    HallucinationTaintTracker,
    SessionReceiptChain,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ("refunds.md", "# Refund policy\n\n" + "\n".join(
        f"Rule {i}: refunds within {30 + i} days need approval tier {i % 4}." for i in range(120))),
    ("billing.py", "\n\n".join(
        f"def charge_{i}(amount):\n    '''charge handler {i}'''\n    return round(amount * 1.0{i % 9}, 2)"
        for i in range(80))),
]


def _commits() -> list[dict]:
    first = create_context_commit(DOCS, query="refund approval tier", token_budget=1500,
                                  prefer_rust=False)
    second = create_context_commit(DOCS, query="charge handler rounding", token_budget=300,
                                   prefer_rust=False, parent_commit_id=first["commit_id"])
    return [first, second]


def _session() -> tuple[dict, dict]:
    chain = SessionReceiptChain(session_id="sess-audit-1")
    for turn in range(3):
        chain.append({"receipt_id": f"r{turn}", "query": f"step {turn}", "token_budget": 900 + turn,
                      "risk_summary": {"high": turn % 2}, "selected_context": [f"chunk-{turn}"]})
    return chain.as_dict(), HallucinationTaintTracker().as_dict()


@pytest.fixture(scope="module")
def key() -> AttestationKey:
    return AttestationKey.generate()


@pytest.fixture(scope="module")
def signed(key) -> dict:
    chain, taint = _session()
    return build_proof_bundle(commits=_commits(), session_chain=chain, taint=taint,
                              signing_key=key, key_source="test")


# ── an intact bundle, and how far "verified" goes ────────────────────────


def test_signed_bundle_verifies_when_the_verifier_pins_the_key(signed, key) -> None:
    result = verify_proof_bundle(signed, public_key=key.public_hex())
    assert result.valid and result.status == "verified"
    assert result.key_pinned and result.signed
    assert all(result.checks.values())
    assert set(result.checks) == {"schema", "bundle_digest", "limits", "context_commits",
                                  "session_chain", "merkle_root", "signature", "signer"}
    assert (result.context_commits, result.session_turns) == (2, 3)
    assert signed["schema_version"] == PROOF_BUNDLE_SCHEMA
    assert signed["merkle"]["tree_size"] == 2 + 3 + 1  # commits + turns + taint record


def test_signed_bundle_without_a_pinned_key_is_not_called_verified(signed) -> None:
    result = verify_proof_bundle(signed)
    assert result.valid and result.status == "signed_unpinned"
    assert not result.key_pinned
    assert any("signer was not checked" in note for note in result.warnings)


def test_unsigned_bundle_is_integrity_only_and_fails_when_a_signature_is_required() -> None:
    bundle = build_proof_bundle(commits=_commits())
    assert bundle["signature"] is None
    loose = verify_proof_bundle(bundle)
    assert loose.valid and loose.status == "integrity_only" and not loose.signed
    assert any("anyone could have produced it" in note for note in loose.warnings)
    # A section that is absent is not reported as a passed check.
    assert "session_chain" not in loose.checks and "signature" not in loose.checks
    strict = verify_proof_bundle(bundle, require_signature=True)
    assert not strict.valid and strict.status == "invalid"
    assert "bundle is not signed" in strict.errors


def test_bundle_survives_a_json_round_trip(signed, key) -> None:
    again = json.loads(json.dumps(signed, ensure_ascii=False))
    assert verify_proof_bundle(again, public_key=key.public_hex()).status == "verified"


def test_limits_travel_inside_the_bundle(signed) -> None:
    assert signed["limits"] == list(LIMITS)
    text = " ".join(LIMITS)
    assert "not a certification" in text
    assert "does not reproduce or vouch for the model's output" in text


# ── tampering ────────────────────────────────────────────────────────────


def _edit_selected_text(bundle):
    item = bundle["context_commits"]["entries"][0]["commit"]["receipt"]["selected_context"][0]
    item["text"] = item["text"] + " "


def _edit_omitted_recovery(bundle):
    chunks = bundle["context_commits"]["entries"][0]["commit"]["recovery_bundle"]["chunks"]
    first = next(iter(chunks.values()))
    first["text"] = first["text"][:-1] + "X"


def _drop_commit(bundle):
    bundle["context_commits"]["entries"].pop()
    bundle["context_commits"]["count"] -= 1


def _drop_turn(bundle):
    bundle["session"]["turns"].pop()


def _edit_turn_budget(bundle):
    bundle["session"]["chain"]["links"][1]["token_budget"] = 1


TAMPERS = {
    "selected context text": _edit_selected_text,
    "recoverable omitted text": _edit_omitted_recovery,
    "recorded chunk count": lambda b: b["context_commits"]["entries"][0]["verification"]
        .__setitem__("omitted_chunks", 0),
    "recorded pass flag": lambda b: b["context_commits"]["entries"][1]["verification"]["checks"]
        .__setitem__("commit_id", False),
    "a commit removed": _drop_commit,
    "commit anchor": lambda b: b["context_commits"]["entries"][0].__setitem__("anchor", "sha256:" + "0" * 64),
    "parent linkage flag": lambda b: b["context_commits"]["entries"][1].__setitem__("parent_in_bundle", False),
    "a session turn removed": _drop_turn,
    "a turn's budget": _edit_turn_budget,
    "session id": lambda b: b["subject"].__setitem__("session_id", "someone-else"),
    "witness taint record": lambda b: b["session"]["witness_taint"].__setitem__("risk_half_life_turns", 99.0),
    "merkle root": lambda b: b["merkle"].__setitem__("root_hash", "0" * 64),
    "limits softened": lambda b: b.__setitem__("limits", b["limits"][:-1]),
    "export time": lambda b: b.__setitem__("created_at", 1.0),
    "generator version": lambda b: b["generator"].__setitem__("version", "9.9.9"),
    "signature bytes": lambda b: b["signature"].__setitem__("signature", "00" * 64),
    "signed_at": lambda b: b["signature"].__setitem__("signed_at", 1.0),
    "schema": lambda b: b.__setitem__("schema_version", "entroly.proof-bundle.v0"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_any_change_fails_verification(signed, key, name) -> None:
    bundle = copy.deepcopy(signed)
    TAMPERS[name](bundle)
    result = verify_proof_bundle(bundle, public_key=key.public_hex())
    assert not result.valid and result.status == "invalid", name
    assert result.errors


def test_tampered_bundle_fails_even_after_its_digest_is_recomputed(signed, key) -> None:
    """Fixing up the digest and id hides the edit from the digest check; the
    signature and the re-verified commit still catch it."""
    from entroly.proof_bundle import _digest

    bundle = copy.deepcopy(signed)
    _edit_selected_text(bundle)
    digest = _digest(bundle)
    bundle["bundle_digest"], bundle["bundle_id"] = "sha256:" + digest, "proof_" + digest[:24]
    result = verify_proof_bundle(bundle, public_key=key.public_hex())
    assert result.checks["bundle_digest"] is True
    assert result.checks["signature"] is False and result.checks["context_commits"] is False
    assert not result.valid


def test_stripping_the_signature_is_visible_and_fails_a_pinned_check(signed, key) -> None:
    bundle = copy.deepcopy(signed)
    bundle["signature"] = None
    assert verify_proof_bundle(bundle).status == "integrity_only"
    assert not verify_proof_bundle(bundle, public_key=key.public_hex()).valid
    assert not verify_proof_bundle(bundle, require_signature=True).valid


def test_resigning_with_another_key_fails_a_pinned_check(signed, key) -> None:
    other = AttestationKey.generate()
    chain, taint = _session()
    forged = build_proof_bundle(commits=_commits(), session_chain=chain, taint=taint,
                                signing_key=other, key_source="test")
    assert verify_proof_bundle(forged).status == "signed_unpinned"
    result = verify_proof_bundle(forged, public_key=key.public_hex())
    assert not result.valid
    assert "not signed by the expected public key" in " ".join(result.errors)


@pytest.mark.parametrize("junk", [None, [], "bundle", 7, {}])
def test_non_bundles_fail_closed(junk) -> None:
    result = verify_proof_bundle(junk)
    assert not result.valid and result.status == "invalid"


# ── content-free bundles, incomplete chains, bad inputs ──────────────────


def test_content_free_bundle_carries_no_source_text_and_says_replay_was_not_rerun(key) -> None:
    bundle = build_proof_bundle(commits=_commits(), embed_content=False, signing_key=key)
    blob = json.dumps(bundle)
    assert "approval tier 1" not in blob and "def charge_" not in blob
    assert all("commit" not in entry for entry in bundle["context_commits"]["entries"])
    result = verify_proof_bundle(bundle, public_key=key.public_hex())
    assert result.status == "verified"
    assert any("cannot be re-run" in note for note in result.warnings)


def test_a_missing_parent_is_reported_not_hidden() -> None:
    child = _commits()[1]
    result = verify_proof_bundle(build_proof_bundle(commits=[child]))
    assert result.valid
    assert any("is not in this bundle" in note for note in result.warnings)


def test_commits_are_ordered_parent_first() -> None:
    first, second = _commits()
    bundle = build_proof_bundle(commits=[second, first])
    ids = [entry["commit_id"] for entry in bundle["context_commits"]["entries"]]
    assert ids == [first["commit_id"], second["commit_id"]]
    assert all(entry["parent_in_bundle"] for entry in bundle["context_commits"]["entries"])


def test_builder_rejects_missing_or_mismatched_evidence() -> None:
    chain, taint = _session()
    with pytest.raises(ProofBundleError, match="needs a Context Commit or a session chain"):
        build_proof_bundle()
    with pytest.raises(ProofBundleError, match="need the session chain"):
        build_proof_bundle(commits=_commits(), taint=taint)
    with pytest.raises(ProofBundleError, match="not 'other'"):
        build_proof_bundle(session_chain=chain, session_id="other")


def test_a_corrupted_commit_exports_as_a_bundle_that_does_not_verify() -> None:
    broken = _commits()
    broken[0]["receipt"]["selected_context"][0]["text"] += "!"
    result = verify_proof_bundle(build_proof_bundle(commits=broken))
    assert not result.valid and result.checks["context_commits"] is False


# ── keys ─────────────────────────────────────────────────────────────────


def test_signing_key_sources() -> None:
    private_hex, public_hex = generate_signing_key()
    key, source = load_signing_key({"ENTROLY_PROOF_SIGNING_KEY": private_hex})
    assert (key.public_hex(), source) == (public_hex, "ENTROLY_PROOF_SIGNING_KEY")

    derived_a, source = load_signing_key({"ENTROLY_IDENTITY_KEY": "operator secret"})
    derived_b, _ = load_signing_key({"ENTROLY_IDENTITY_KEY": "operator secret"})
    other, _ = load_signing_key({"ENTROLY_IDENTITY_KEY": "another secret"})
    assert source == "derived:ENTROLY_IDENTITY_KEY"
    assert derived_a.public_hex() == derived_b.public_hex() != other.public_hex()

    # The explicit key wins over the derived one.
    key, source = load_signing_key(
        {"ENTROLY_PROOF_SIGNING_KEY": private_hex, "ENTROLY_IDENTITY_KEY": "operator secret"})
    assert key.public_hex() == public_hex


@pytest.mark.parametrize("env", [{}, {"ENTROLY_IDENTITY_KEY": "  "},
                                 {"ENTROLY_PROOF_SIGNING_KEY": "zz"},
                                 {"ENTROLY_PROOF_SIGNING_KEY": "ab" * 16}])
def test_signing_without_a_usable_key_is_an_error_not_an_unsigned_bundle(env) -> None:
    with pytest.raises(ProofBundleError):
        load_signing_key(env)


# ── report ───────────────────────────────────────────────────────────────


def test_html_report_is_static_escaped_and_states_the_limits(key) -> None:
    hostile = create_context_commit(
        DOCS, query='<script>alert("x")</script> refund', token_budget=300, prefer_rust=False)
    bundle = build_proof_bundle(commits=[hostile], signing_key=key)
    page = render_html_report(bundle, verify_proof_bundle(bundle, public_key=key.public_hex()))
    assert "<script" not in page and "&lt;script&gt;" in page
    assert "http://" not in page and "https://" not in page and "src=" not in page
    assert "Verified" in page and bundle["bundle_id"] in page
    assert "not a certification" in page
    assert key.public_hex() in page


def test_html_report_for_a_failed_bundle_says_so(signed, key) -> None:
    bundle = copy.deepcopy(signed)
    _edit_selected_text(bundle)
    page = render_html_report(bundle, verify_proof_bundle(bundle, public_key=key.public_hex()))
    assert "Failed verification" in page and "Do not rely on this bundle" in page


# ── command line ─────────────────────────────────────────────────────────


def _cli(args: list[str], cwd: Path, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items()
            if k not in {"ENTROLY_PROOF_SIGNING_KEY", "ENTROLY_IDENTITY_KEY"}}
    base.update({"PYTHONPATH": str(ROOT), "ENTROLY_DISABLE_TELEMETRY": "1", "NO_COLOR": "1"}, **env)
    return subprocess.run([sys.executable, "-m", "entroly", *args], cwd=cwd, env=base,
                          capture_output=True, text=True, timeout=120)


@pytest.fixture()
def workdir(tmp_path) -> Path:
    commits = tmp_path / ".entroly" / "context-commits"
    commits.mkdir(parents=True)
    for commit in _commits():
        (commits / f"{commit['commit_id']}.json").write_text(json.dumps(commit), encoding="utf-8")
    chain, taint = _session()
    (tmp_path / "session_chain.json").write_text(json.dumps(chain), encoding="utf-8")
    (tmp_path / "session_taint.json").write_text(json.dumps(taint), encoding="utf-8")
    return tmp_path


def test_cli_export_sign_verify_and_tamper(workdir) -> None:
    made = _cli(["export-proof", "--generate-key", "proof.key"], workdir)
    assert made.returncode == 0, made.stderr
    key_file = workdir / "proof.key"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert _cli(["export-proof", "--generate-key", "proof.key"], workdir).returncode == 1
    private_hex = key_file.read_text().strip()
    public_hex = AttestationKey.from_private_hex(private_hex).public_hex()

    out = _cli(["export-proof", "--sign", "--format", "html", "--session", "sess-audit-1",
                "--session-chain", "session_chain.json", "--commits-dir", ".entroly/context-commits",
                "--out", "bundle.json", "--json"], workdir, ENTROLY_PROOF_SIGNING_KEY=private_hex)
    assert out.returncode == 0, out.stderr
    summary = json.loads(out.stdout)
    assert summary["status"] == "verified" and summary["public_key"] == public_hex
    assert (summary["context_commits"], summary["session_turns"]) == (2, 3)
    assert (workdir / "bundle.html").read_text().startswith("<!doctype html>")

    ok = _cli(["verify-proof", "bundle.json", "--public-key", public_hex], workdir)
    assert ok.returncode == 0 and "VERIFIED" in ok.stdout

    bundle = json.loads((workdir / "bundle.json").read_text())
    bundle["session"]["chain"]["links"][0]["query"] = "something else"
    (workdir / "tampered.json").write_text(json.dumps(bundle))
    bad = _cli(["verify-proof", "tampered.json", "--public-key", public_hex], workdir)
    assert bad.returncode == 1 and "FAILED" in bad.stdout

    wrong = _cli(["verify-proof", "bundle.json", "--public-key", "ab" * 32], workdir)
    assert wrong.returncode == 1


def test_cli_sign_without_a_key_writes_nothing(workdir) -> None:
    out = _cli(["export-proof", "--sign", "--out", "bundle.json"], workdir)
    assert out.returncode == 1 and "neither is set" in out.stderr
    assert not (workdir / "bundle.json").exists()


@pytest.mark.parametrize("which", [0, 1])
def test_cli_refuses_to_export_evidence_that_does_not_verify(workdir, which) -> None:
    # Every commit has recovery chunks; not every commit selected any (the
    # second one here selects none), and directory order is not stable, so the
    # file is chosen by sorted name and corrupted where all commits have bytes.
    path = sorted((workdir / ".entroly" / "context-commits").glob("*.json"))[which]
    commit = json.loads(path.read_text())
    chunk = next(iter(commit["recovery_bundle"]["chunks"].values()))
    chunk["text"] += "!"
    path.write_text(json.dumps(commit))
    out = _cli(["export-proof", "--out", "bundle.json"], workdir)
    assert out.returncode == 1 and "Refusing to export" in out.stderr
    assert not (workdir / "bundle.json").exists()


def test_cli_wrong_session_id_and_missing_inputs(workdir, tmp_path_factory) -> None:
    out = _cli(["export-proof", "--session", "other", "--session-chain", "session_chain.json"], workdir)
    assert out.returncode == 1 and "not 'other'" in out.stderr
    empty = tmp_path_factory.mktemp("empty")
    assert _cli(["export-proof"], empty).returncode == 1
    assert _cli(["verify-proof", "missing.json"], empty).returncode == 1
    (empty / "junk.json").write_text("{not json")
    assert _cli(["verify-proof", "junk.json"], empty).returncode == 1


def test_cli_unsigned_default_export_is_integrity_only(workdir) -> None:
    out = _cli(["export-proof", "--json"], workdir)
    assert out.returncode == 0, out.stderr
    summary = json.loads(out.stdout)
    assert summary["status"] == "integrity_only" and summary["signed"] is False
    assert Path(workdir / summary["bundle"]).exists()
    strict = _cli(["verify-proof", summary["bundle"], "--require-signature"], workdir)
    assert strict.returncode == 1
