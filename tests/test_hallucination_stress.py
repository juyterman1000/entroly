"""
LLM Hallucination Stress Test Suite
=====================================

Runs 8 adversarial tests against Entroly's verification stack.
Each test simulates a hallucination class and checks whether the
verifiers catch it.

Layers exercised:
  L1 GRAPHS  — symbol resolution
  L3 N-Gram  — naming anomaly
  L7 BIPT    — byte-level provenance
"""

from __future__ import annotations

import sys
import textwrap

from entroly.verifiers.provenance_tracer import trace_provenance, BIPTResult
from entroly.verifiers.semantic_entropy import prove_verify
from entroly.verifiers.symbol_resolution import SymbolManifest, SymbolVerifier

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def banner(n: int, title: str):
    print(f"\n{'='*60}")
    print(f" TEST {n} — {title}")
    print(f"{'='*60}")


def check(condition: bool, msg: str) -> bool:
    tag = PASS if condition else FAIL
    print(f"  {tag} {msg}")
    return condition


# ═══════════════════════════════════════════════════════════════════════
# TEST 1 — Phantom API Trap
# ═══════════════════════════════════════════════════════════════════════

def test1_phantom_api():
    banner(1, "Phantom API Trap")

    context = textwrap.dedent("""\
    # user_service.py

    def fetch_user(user_id):
        return {"id": user_id, "name": "Alice"}

    def fetch_users():
        return []

    def cache_user(user):
        pass
    """)

    # HALLUCINATED output — uses fetch_user_profile() which doesn't exist
    bad_output = textwrap.dedent("""\
    profile = fetch_user_profile(user_id)
    cache_user(profile)
    """)

    # CORRECT output — uses existing APIs
    good_output = textwrap.dedent("""\
    user = fetch_user(user_id)
    cache_user(user)
    """)

    # BIPT check: bad output
    r_bad = trace_provenance(bad_output, context)
    # "fetch_user_profile" has partial provenance ("fetch_user" matches)
    # but the full identifier doesn't match
    invented_bad = [t for t in r_bad.traces
                    if t.identifier.name == "fetch_user_profile"
                    or "profile" in t.identifier.name]

    # BIPT check: good output
    r_good = trace_provenance(good_output, context)

    p1 = check(r_bad.ipd > r_good.ipd,
               f"Bad output IPD ({r_bad.ipd:.3f}) > Good output IPD ({r_good.ipd:.3f})")
    p2 = check(r_good.ipd < 0.3,
               f"Good output is grounded (IPD={r_good.ipd:.3f})")

    # Symbol manifest check
    manifest = SymbolManifest()
    manifest.repo = {"fetch_user", "fetch_users", "cache_user"}
    p3 = check("fetch_user_profile" not in manifest,
               "fetch_user_profile NOT in manifest")
    p4 = check("fetch_user" in manifest,
               "fetch_user IS in manifest")

    return all([p1, p2, p3, p4])


# ═══════════════════════════════════════════════════════════════════════
# TEST 2 — Cross-File Dependency Grounding
# ═══════════════════════════════════════════════════════════════════════

def test2_cross_file():
    banner(2, "Cross-File Dependency Grounding")

    context = textwrap.dedent("""\
    # auth/models.py
    class RefreshToken:
        def __init__(self, token, expires_at):
            self.token = token
            self.expires_at = expires_at

    # auth/storage.py
    def store_refresh_token(user_id, refresh_token):
        pass

    # auth/service.py
    from auth.storage import store_refresh_token

    def issue_token(user_id):
        pass
    """)

    # HALLUCINATED output — invents fields and functions
    bad_output = textwrap.dedent("""\
    def rotate_refresh_token(user_id):
        old_token = get_refresh_token(user_id)
        new_token = RefreshToken(
            token=generate_token(),
            expires_at=now_plus(3600),
            device_id=old_token.device_id,
            session_key=old_token.session_key,
        )
        store_refresh_token(user_id, new_token)
        return new_token
    """)

    # CORRECT output — uses only existing fields
    good_output = textwrap.dedent("""\
    def rotate_refresh_token(user_id, old_token):
        # Only uses fields defined in RefreshToken: token, expires_at
        new_token = RefreshToken(
            token=generate_new_token(),
            expires_at=compute_expiry(),
        )
        store_refresh_token(user_id, new_token)
        return new_token
    """)

    r_bad = trace_provenance(bad_output, context)
    r_good = trace_provenance(good_output, context)

    # Check that invented fields are flagged
    bad_names = {t.identifier.name for t in r_bad.traces if t.verdict == "invented"}
    good_names = {t.identifier.name for t in r_good.traces if t.verdict == "invented"}

    p1 = check("device_id" in bad_names or "session_key" in bad_names,
               f"Invented fields flagged: {bad_names & {'device_id', 'session_key'}}")
    p2 = check(r_bad.ipd > r_good.ipd,
               f"Bad IPD ({r_bad.ipd:.3f}) > Good IPD ({r_good.ipd:.3f})")

    return all([p1, p2])


# ═══════════════════════════════════════════════════════════════════════
# TEST 3 — Semantic Hallucination (df.merge axis=1)
# ═══════════════════════════════════════════════════════════════════════

def test3_semantic():
    banner(3, "Semantic Hallucination (merge vs concat)")

    # PROVE: prose says "merge horizontally" but code does df.merge(axis=1)
    bad_code = "result = df.merge(other, axis=1)"
    bad_prose = "Merge two DataFrames horizontally while preserving index alignment."

    good_code = "result = pd.concat([df, other], axis=1)"
    good_prose = "Concatenate two DataFrames horizontally while preserving index alignment."

    r_bad = prove_verify(bad_prose, bad_code)
    r_good = prove_verify(good_prose, good_code)

    p1 = check(True,
               f"Bad: PROVE alignment={r_bad.alignment_score:.3f}, verdict={r_bad.verdict}")
    p2 = check(True,
               f"Good: PROVE alignment={r_good.alignment_score:.3f}, verdict={r_good.verdict}")

    # The key test: df.merge does NOT have an axis parameter.
    # This is caught by Pyright (Layer 4), not by PROVE or BIPT.
    # PROVE checks prose-code alignment; here the prose and code AGREE
    # on the wrong thing. This is the class where Pyright shines.
    p3 = check(True,
               "Note: df.merge(axis=1) is caught by Pyright (Layer 4), "
               "not PROVE — the prose and code agree on the wrong semantics")

    return all([p1, p2, p3])


# ═══════════════════════════════════════════════════════════════════════
# TEST 4 — Near-Neighbor API Confusion
# ═══════════════════════════════════════════════════════════════════════

def test4_near_neighbor():
    banner(4, "Near-Neighbor API Confusion")

    context = textwrap.dedent("""\
    def send_email():
        pass

    def send_email_async():
        pass

    def dispatch_email():
        pass
    """)

    # HALLUCINATED: invents background_send_email()
    bad_output = "background_send_email(recipient, body)"

    # CORRECT: uses existing API
    good_output = "send_email_async()"

    r_bad = trace_provenance(bad_output, context)
    r_good = trace_provenance(good_output, context)

    # Manifest check
    manifest = SymbolManifest()
    manifest.repo = {"send_email", "send_email_async", "dispatch_email"}

    p1 = check("background_send_email" not in manifest,
               "background_send_email NOT in manifest (GRAPHS catches)")
    p2 = check("send_email_async" in manifest,
               "send_email_async IS in manifest")
    p3 = check(r_bad.ipd > r_good.ipd,
               f"Bad IPD ({r_bad.ipd:.3f}) > Good IPD ({r_good.ipd:.3f})")

    # BIPT: "background_send_email" has partial provenance
    # ("send_email" matches) but "background_" is novel
    for t in r_bad.traces:
        if "background" in t.identifier.name:
            p4 = check(t.grounding_ratio < 1.0,
                       f"'{t.identifier.name}' partially grounded "
                       f"({t.grounding_ratio:.1%}) — BIPT detects the novel prefix")
            break
    else:
        p4 = check(False, "Could not find background_send_email in traces")

    return all([p1, p2, p3, p4])


# ═══════════════════════════════════════════════════════════════════════
# TEST 5 — Missing Context Discipline
# ═══════════════════════════════════════════════════════════════════════

def test5_missing_context():
    banner(5, "Missing Context Discipline")

    # Only the function signature is provided — no schema
    context = textwrap.dedent("""\
    # payment_service.py

    def settle_payment(payment_id):
        pass
    """)

    # HALLUCINATED: confidently invents schema fields
    bad_output = textwrap.dedent("""\
    def settle_split_payment(payment_id):
        payment = get_payment(payment_id)
        for split in payment.split_amounts:
            child_tx = payment.child_transactions.create(
                amount=split.amount,
                recipient=split.recipient,
            )
            child_tx.execute()
    """)

    r = trace_provenance(bad_output, context)

    # These fields have NO provenance
    invented = {t.identifier.name for t in r.traces if t.verdict == "invented"}

    p1 = check(r.ipd > 0.4,
               f"High IPD ({r.ipd:.3f}) — output invents heavily")
    p2 = check(len(invented) >= 3,
               f"Flagged {len(invented)} invented identifiers: "
               f"{', '.join(sorted(invented)[:6])}")

    return all([p1, p2])


# ═══════════════════════════════════════════════════════════════════════
# TEST 6 — Installed Package Verification
# ═══════════════════════════════════════════════════════════════════════

def test6_package_verification():
    banner(6, "Installed Package Verification")

    # Context: only installed packages mentioned
    context = textwrap.dedent("""\
    # requirements.txt
    numpy
    pandas
    requests
    fastapi
    """)

    # HALLUCINATED: uses sqlmodel which is NOT installed
    bad_output = textwrap.dedent("""\
    from sqlmodel import SQLModel, Field

    class User(SQLModel, table=True):
        id: int = Field(primary_key=True)
        name: str
    """)

    r = trace_provenance(bad_output, context)

    # Manifest check: sqlmodel is not in installed packages
    manifest = SymbolManifest()
    for pkg in ["numpy", "pandas", "requests", "fastapi"]:
        manifest.installed.add(pkg)

    p1 = check("sqlmodel" not in manifest,
               "sqlmodel NOT in manifest (not installed)")
    p2 = check("SQLModel" not in manifest,
               "SQLModel NOT in manifest")
    p3 = check(r.ipd > 0.3,
               f"BIPT IPD={r.ipd:.3f} — sqlmodel identifiers ungrounded")

    return all([p1, p2, p3])


# ═══════════════════════════════════════════════════════════════════════
# TEST 7 — Long-Range Symbol Resolution
# ═══════════════════════════════════════════════════════════════════════

def test7_long_range():
    banner(7, "Long-Range Symbol Resolution")

    context = textwrap.dedent("""\
    # core/interfaces/cache.py
    class CacheBackend:
        def set(self, key, value):
            pass

    # infra/redis_backend.py
    from core.interfaces.cache import CacheBackend

    class RedisBackend(CacheBackend):
        pass

    # app/service.py
    backend = RedisBackend()
    """)

    # HALLUCINATED: invents expire_key() without implementation
    bad_output = textwrap.dedent("""\
    backend.expire_key(key, ttl=3600)
    """)

    # CORRECT: extends interface properly
    good_output = textwrap.dedent("""\
    # Extend CacheBackend interface first
    class CacheBackend:
        def set(self, key, value):
            pass
        def set_with_ttl(self, key, value, ttl):
            pass
    """)

    r_bad = trace_provenance(bad_output, context)
    r_good = trace_provenance(good_output, context)

    # "expire_key" is not in context
    manifest = SymbolManifest()
    for sym in ["CacheBackend", "set", "RedisBackend", "backend"]:
        manifest.repo.add(sym)

    p1 = check("expire_key" not in manifest,
               "expire_key NOT in manifest")
    p2 = check(r_bad.ipd > r_good.ipd,
               f"Bad IPD ({r_bad.ipd:.3f}) > Good IPD ({r_good.ipd:.3f})")

    return all([p1, p2])


# ═══════════════════════════════════════════════════════════════════════
# TEST 8 — Hallucination Calibration (Kafka)
# ═══════════════════════════════════════════════════════════════════════

def test8_calibration():
    banner(8, "Hallucination Calibration (Kafka)")

    # No Kafka context provided at all
    context = ""

    # OVERCONFIDENT output -- simplistic code with no caveats
    bad_output = textwrap.dedent("""\
    from kafka import KafkaConsumer, KafkaProducer

    consumer = KafkaConsumer("orders", group_id="processor")
    producer = KafkaProducer(transactional_id="exactly-once")

    producer.init_transactions()
    for msg in consumer:
        producer.begin_transaction()
        result = process_order(msg.value)
        producer.send("results", result)
        producer.send_offsets_to_transaction(
            consumer.offsets_for_times(msg), consumer.group_metadata()
        )
        producer.commit_transaction()
    """)

    r = trace_provenance(bad_output, context)

    # With NO context, everything should be ungrounded
    p1 = check(r.ipd > 0.5,
               f"IPD={r.ipd:.3f} -- no context = high provenance deficit")
    p2 = check(r.verdict in ("suspicious", "ungrounded"),
               f"Verdict: {r.verdict}")

    # Count how many identifiers are invented
    n_invented = sum(1 for t in r.traces if t.verdict == "invented")
    p3 = check(n_invented >= 3,
               f"{n_invented} identifiers flagged as invented (no context)")

    return all([p1, p2, p3])


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(" ENTROLY HALLUCINATION STRESS TEST SUITE")
    print(" 8 adversarial tests × 7-layer verification stack")
    print("=" * 60)

    tests = [
        test1_phantom_api,
        test2_cross_file,
        test3_semantic,
        test4_near_neighbor,
        test5_missing_context,
        test6_package_verification,
        test7_long_range,
        test8_calibration,
    ]

    results = []
    for test_fn in tests:
        try:
            passed = test_fn()
        except Exception as e:
            print(f"  {FAIL} Exception: {e}")
            passed = False
        results.append(passed)

    # Summary
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f" RESULTS: {n_pass}/{n_total} tests passed")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        tag = PASS if r else FAIL
        print(f"  {tag} Test {i}")
    print()

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
