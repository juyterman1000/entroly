"""Full CogOps Rust engine integration test."""
import tempfile
import json
from entroly_core import CogOpsEngine

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS: {name}")
    else: FAIL += 1; print(f"  FAIL: {name}")

with tempfile.TemporaryDirectory() as td:
    e = CogOpsEngine(td, miss_threshold=3)

    # ── Intent Classification ──
    print("=== Intent Classification (Rust) ===")
    check("architecture", e.classify_intent("how does the auth module work?") == "architecture")
    check("pr_brief", e.classify_intent("review this PR diff") == "pr_brief")
    check("incident", e.classify_intent("production outage on API") == "incident")
    check("audit", e.classify_intent("check for security vulnerabilities") == "audit")
    check("onboarding", e.classify_intent("explain this to a new engineer") == "onboarding")
    check("repair", e.classify_intent("fix the broken login") == "repair")
    check("general", e.classify_intent("hello world") == "general")

    # ── Entity Extraction ──
    print("\n=== Entity Extraction (Rust) ===")
    py_code = '''
class AuthService:
    """Handles authentication."""
    def verify_token(self, token: str) -> bool:
        pass
    def rotate_keys(self):
        pass
'''
    entities = e.extract_entities(py_code, "auth_service.py")
    check("Python extraction", len(entities) >= 2)
    names = [x["name"] for x in entities]
    check("Found AuthService", "AuthService" in names)
    check("Found verify_token", "verify_token" in names)

    rs_code = '''
/// Token budget allocator
pub struct NkbeAllocator {
    budget: usize,
}
pub trait BudgetPolicy {
    fn evaluate(&self) -> f64;
}
pub fn allocate(budget: usize) -> Vec<usize> { vec![] }
'''
    rs_ents = e.extract_entities(rs_code, "nkbe.rs")
    check("Rust extraction", len(rs_ents) >= 2)
    rs_names = [x["name"] for x in rs_ents]
    check("Found NkbeAllocator", "NkbeAllocator" in rs_names)
    check("Found BudgetPolicy", "BudgetPolicy" in rs_names)

    # ── Belief Compilation ──
    print("\n=== Belief Compilation (Rust) ===")
    import os
    src = os.path.join(td, "src")
    os.makedirs(src)
    with open(os.path.join(src, "auth.py"), "w") as f: f.write(py_code)
    with open(os.path.join(src, "nkbe.rs"), "w") as f: f.write(rs_code)

    result = e.compile_beliefs(src, 200)
    check("Files processed", result["files_processed"] == 2)
    check("Beliefs written", result["beliefs_written"] >= 2)
    check("Entities extracted", result["entities_extracted"] >= 4)

    # ── Verification ──
    print("\n=== Verification (Rust) ===")
    vr = e.verify_beliefs()
    check("Beliefs checked", vr["total_beliefs_checked"] >= 2)
    check("Mean confidence > 0", vr["mean_confidence"] > 0)

    # ── Blast Radius ──
    print("\n=== Blast Radius (Rust) ===")
    br = e.blast_radius(["auth.py"])
    check("Blast radius runs", "risk_level" in br)

    # ── Routing ──
    print("\n=== Routing (Rust) ===")
    r1 = e.route("how does auth_service work?", False, "")
    check(f"Route flow: {r1['flow']}", r1["flow"] in ("fast_answer", "verify_before_answer", "compile_on_demand"))
    check("Has reasoning", len(r1["reasoning"]) > 0)

    r2 = e.route("PR opened: fix auth", True, "pr")
    check("Change-driven", r2["flow"] == "change_driven")

    # ── Change Pipeline ──
    print("\n=== Change Pipeline (Rust) ===")
    diff = """diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,5 @@
 class AuthService:
+    password = "hardcoded123"
     def verify_token(self, token):
+        # TODO: implement
         pass
"""
    cr = e.process_change(diff, "fix: auth", "Fix hardcoded password")
    check("Has title", len(cr["title"]) > 0)
    check("Has intent", cr["intent"] in ("bugfix", "feature", "security"))
    check("Lines added", cr["lines_added"] > 0)
    check("Findings found", cr["findings_count"] >= 1)

    # ── Skill Engine ──
    print("\n=== Skill Engine (Rust) ===")
    sk = e.create_skill("protobuf_analysis", ["analyze proto", "proto compat"])
    check("Skill created", sk["status"] == "created")
    check("Has skill_id", len(sk["skill_id"]) > 0)

    skills = e.list_skills()
    check("Skills listed", len(skills) >= 1)

    # ── Vault Status ──
    print("\n=== Vault Status (Rust) ===")
    vs = e.vault_status()
    check("Total beliefs", vs["total_beliefs"] >= 2)
    check("Has entities", len(vs["entities"]) >= 2)

    # ── Write Belief ──
    print("\n=== Write Belief (Rust) ===")
    wb = e.write_belief("test_entity", "Test Belief", "This is a test.", 0.9, "verified", ["manual"])
    check("Written", wb["status"] == "written")
    check("Has claim_id", len(wb["claim_id"]) > 0)

print(f"\n{'='*50}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
if FAIL == 0: print("ALL RUST COGOPS TESTS PASSED")
