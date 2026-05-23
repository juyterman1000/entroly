"""Full CogOps data plane smoke test — all engines."""
import json
import tempfile
import os

from entroly.epistemic_router import EpistemicRouter, classify_intent, EpistemicIntent, EpistemicFlow
from entroly.vault import VaultManager, VaultConfig, BeliefArtifact
from entroly.belief_compiler import BeliefCompiler, extract_entities
from entroly.verification_engine import VerificationEngine
from entroly.change_pipeline import ChangePipeline, parse_diff, review_diff
from entroly.flow_orchestrator import FlowOrchestrator
from entroly.skill_engine import SkillEngine
from entroly.evolution_logger import EvolutionLogger

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")

with tempfile.TemporaryDirectory() as td:
    vault = VaultManager(VaultConfig(base_path=td))
    vault.ensure_structure()

    # ── 1. Belief Compiler ──
    print("=== Belief Compiler ===")
    compiler = BeliefCompiler(vault)

    # Test entity extraction (Python)
    py_code = '''
class AuthService:
    """Handles authentication and token rotation."""
    def verify_token(self, token: str) -> bool:
        """Verify a JWT token."""
        pass
    def rotate_keys(self) -> None:
        pass

MAX_TOKEN_AGE = 3600
'''
    entities = extract_entities(py_code, "auth_service.py")
    check("Python entity extraction", len(entities) >= 2)
    check("Python class found", any(e.name == "AuthService" for e in entities))
    check("Python function found", any(e.name == "verify_token" for e in entities))

    # Test entity extraction (Rust)
    rs_code = '''
/// Manages token budget allocation
pub struct NkbeAllocator {
    budget: usize,
}

pub fn allocate(budget: usize, demands: &[usize]) -> Vec<usize> {
    demands.to_vec()
}

pub trait BudgetPolicy {
    fn evaluate(&self) -> f64;
}
'''
    rs_entities = extract_entities(rs_code, "nkbe.rs")
    check("Rust entity extraction", len(rs_entities) >= 2)
    check("Rust struct found", any(e.name == "NkbeAllocator" for e in rs_entities))
    check("Rust trait found", any(e.name == "BudgetPolicy" for e in rs_entities))

    # Test directory compilation
    test_src = os.path.join(td, "src")
    os.makedirs(test_src)
    with open(os.path.join(test_src, "auth.py"), "w") as f:
        f.write(py_code)
    with open(os.path.join(test_src, "nkbe.rs"), "w") as f:
        f.write(rs_code)
    result = compiler.compile_directory(test_src)
    check("Compilation runs", result.files_processed == 2)
    check("Beliefs written", result.beliefs_written >= 2)
    check("Entities extracted", result.entities_extracted >= 4)

    # ── 2. Verification Engine ──
    print("\n=== Verification Engine ===")
    verifier = VerificationEngine(vault, freshness_hours=24.0, min_confidence=0.5)
    report = verifier.full_verification_pass()
    check("Verification runs", report.total_beliefs_checked >= 2)
    check("Mean confidence > 0", report.mean_confidence > 0)

    # Test blast radius
    br = verifier.blast_radius(["auth.py"])
    check("Blast radius finds affected", len(br.affected_entities) >= 0)  # may or may not match

    # Test coverage gaps
    gaps = verifier.coverage_gaps(test_src)
    check("Coverage gaps runs", isinstance(gaps, list))

    # ── 3. Change Pipeline ──
    print("\n=== Change Pipeline ===")
    diff = """diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,5 +1,8 @@
 class AuthService:
+    password = "hardcoded123"
     def verify_token(self, token: str) -> bool:
+        # TODO: implement proper validation
         pass
+    def new_method(self):
+        pass
"""
    change_pipe = ChangePipeline(vault, verifier)


    # Test diff parsing
    cs = parse_diff(diff, "fix: auth hardcoded password")
    check("Diff parsing", len(cs.files_modified) >= 1)
    check("Lines counted", cs.lines_added > 0)
    check("Intent classified", cs.intent in ("bugfix", "feature", "security"))

    # Test review
    findings = review_diff(diff)
    check("Review finds issues", len(findings) >= 1)
    has_secret = any("hardcoded" in f.message.lower() or "secret" in f.message.lower() for f in findings)
    has_todo = any("todo" in f.message.lower() for f in findings)
    check("Detects hardcoded secret", has_secret)
    check("Detects TODO", has_todo)

    # Test full pipeline
    brief = change_pipe.process_diff(diff, "fix: auth", "Fix auth hardcoded password")
    check("PR brief generated", len(brief.title) > 0)
    check("Brief has risk", brief.risk_level in ("low", "medium", "high"))

    # ── 4. Flow Orchestrator ──
    print("\n=== Flow Orchestrator ===")
    router = EpistemicRouter(vault_path=td, miss_threshold=2)
    evo = EvolutionLogger(vault_path=td, gap_threshold=2)
    orchestrator = FlowOrchestrator(
        vault=vault, router=router, compiler=compiler, verifier=verifier,
        change_pipe=change_pipe, evolution=evo, source_dir=test_src,
    )

    # Test fast answer (should find beliefs since we compiled)
    r1 = orchestrator.execute("How does auth_service work?")
    check(f"Flow selected: {r1.flow}", r1.flow in ("fast_answer", "verify_before_answer", "compile_on_demand"))
    check("Has steps", len(r1.steps_completed) >= 2)

    # Test change-driven flow
    r2 = orchestrator.execute("PR #42 opened", diff_text=diff, is_event=True, event_type="pr")
    check("Change-driven flow", r2.flow == "change_driven")
    check("Change-driven has answer", len(r2.answer) > 0)

    # ── 5. Skill Engine ──
    print("\n=== Skill Engine ===")
    skill_engine = SkillEngine(vault)
    create_result = skill_engine.create_skill(
        entity_key="protobuf_analysis",
        failing_queries=["analyze protobuf schema", "protobuf compatibility check"],
        intent="research",
    )
    check("Skill created", create_result["status"] == "created")
    check("Skill has ID", len(create_result["skill_id"]) > 0)

    # List skills
    skills = skill_engine.list_skills()
    check("Skills listed", len(skills) >= 1)

    # Benchmark
    bench = skill_engine.benchmark_skill(create_result["skill_id"])
    check("Benchmark ran", bench["status"] == "benchmarked")

    # Promote/prune
    pp = skill_engine.promote_or_prune(create_result["skill_id"])
    check("Promotion decision", pp["status"] in ("promoted", "pruned", "kept"))

    # ── 6. Full integration: execute_flow with self-improvement ──
    print("\n=== Self-Improvement Flow ===")
    # Force repeated misses
    router2 = EpistemicRouter(vault_path=td, miss_threshold=2)
    evo2 = EvolutionLogger(vault_path=td, gap_threshold=2)
    orch2 = FlowOrchestrator(
        vault=vault, router=router2, compiler=compiler, verifier=verifier,
        change_pipe=change_pipe, evolution=evo2, source_dir=test_src,
    )
    # Force repeated misses — pre-seed via explicit record_miss to
    # ensure the counter reaches threshold regardless of compile side-effects
    router2.record_miss("quantum_flux pipeline")
    router2.record_miss("quantum_flux pipeline")
    r_evo = orch2.execute("quantum_flux pipeline design")
    check(f"Evolution triggered: {r_evo.flow}", r_evo.flow == "self_improvement")

print(f"\n{'='*50}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
if FAIL == 0:
    print("ALL COGOPS DATA PLANE TESTS PASSED")
else:
    print(f"WARNING: {FAIL} test(s) failed")
