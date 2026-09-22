"""Promotion must follow frozen, independently supplied benchmark evidence."""

import json
from pathlib import Path

from entroly.fast_path import FastPathRouter
from entroly.evolution_daemon import EvolutionDaemon
from entroly.integrations.agentskills import export_promoted
from entroly.skill_engine import SkillBenchmark, SkillEngine
from entroly.vault import VaultConfig, VaultManager


def _candidate(tmp_path):
    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    vault.ensure_structure()
    engine = SkillEngine(vault)
    created = engine.create_skill("checks", ["training example"])
    skill_id = created["skill_id"]
    directory = Path(created["path"])
    (directory / "tool.py").write_text(
        "def execute(query, context):\n"
        "    return {'answer': 'verified', 'query': query}\n",
        encoding="utf-8",
    )
    (directory / "tests" / "test_cases.json").write_text(
        json.dumps([{
            "input": "training example", "expected": {"answer": "verified"},
        }]),
        encoding="utf-8",
    )
    return engine, skill_id, directory


def _holdout():
    return [
        {
            "input": f"unseen validation example {index}",
            "expected": {
                "answer": "verified", "query": f"unseen validation example {index}",
            },
        }
        for index in range(10)
    ]


def test_self_generated_benchmark_and_metrics_cannot_promote(tmp_path):
    engine, skill_id, directory = _candidate(tmp_path)
    benchmark = engine.benchmark_skill(skill_id)
    assert benchmark["status"] == "benchmarked"
    assert benchmark["fitness"] == 1.0
    assert benchmark["evaluation_scope"] == "development"

    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["fitness_score"] = 1.0
    metrics["benchmark_runs"] = 999
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    decision = engine.promote_or_prune(skill_id)
    assert decision["status"] == "kept"
    assert "development-only" in decision["evidence_status"]
    assert engine.load_promoted_skill(skill_id) is None


def test_holdout_promotion_revoked_when_code_or_tests_change(tmp_path):
    engine, skill_id, directory = _candidate(tmp_path)
    assert engine.benchmark_skill(
        skill_id, validation_cases=_holdout()
    )["status"] == "benchmarked"
    assert engine.promote_or_prune(skill_id)["status"] == "promoted"
    assert engine.load_promoted_skill(skill_id) is not None

    tool = directory / "tool.py"
    original = tool.read_text(encoding="utf-8")
    tool.write_text(original + "\n# modified after evaluation\n", encoding="utf-8")
    assert engine.load_promoted_skill(skill_id) is None
    listed = next(s for s in engine.list_skills() if s["skill_id"] == skill_id)
    assert listed["status"] == "testing"
    assert "stale" in listed["promotion_integrity"]

    tool.write_text(original, encoding="utf-8")
    cases = directory / "tests" / "test_cases.json"
    cases.write_text(cases.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert engine.load_promoted_skill(skill_id) is None
    assert engine.promote_or_prune(skill_id)["status"] == "kept"


def test_altered_evidence_and_missing_key_fail_closed(tmp_path):
    engine, skill_id, directory = _candidate(tmp_path)
    benchmark = engine.benchmark_skill(skill_id, validation_cases=_holdout())
    assert engine.promote_or_prune(skill_id)["status"] == "promoted"
    evidence = (
        directory.parent.parent / "benchmark-evidence" / skill_id
        / f"{benchmark['evidence_id']}.json"
    )
    record = json.loads(evidence.read_text(encoding="utf-8"))
    record["payload"]["fitness_score"] = 0.0
    evidence.write_text(json.dumps(record), encoding="utf-8")
    assert engine.load_promoted_skill(skill_id) is None
    assert engine.promote_or_prune(skill_id)["status"] == "kept"

    # A missing signing key never silently creates a new authority on read.
    key = directory.parent.parent / "benchmark.key"
    key.unlink()
    assert engine.load_promoted_skill(skill_id) is None
    assert not key.exists()


def test_holdout_must_be_unseen_and_nonempty(tmp_path):
    engine, skill_id, _ = _candidate(tmp_path)
    overlap = engine.benchmark_skill(
        skill_id,
        validation_cases=[{"input": "training example", "expected": {"answer": "verified"}}],
    )
    assert overlap["status"] == "invalid_contract"
    assert engine.benchmark_skill(skill_id, validation_cases=[])["status"] == "invalid_contract"
    assert engine.benchmark_skill(
        skill_id,
        validation_cases=[{"input": "unseen", "expected": {}}],
    )["status"] == "invalid_contract"
    assert engine.benchmark_skill(
        skill_id,
        validation_cases=[{"input": "unseen", "expected": {"answer": "should_work"}}],
    )["status"] == "invalid_contract"
    assert engine.benchmark_skill(
        skill_id,
        validation_cases=[{"input": "unseen", "expected": {"$min_items": 0}}],
    )["status"] == "invalid_contract"
    assert engine.benchmark_skill(
        skill_id,
        validation_cases=[{"input": "unseen", "expected": {"answer": "verified"}}] * 10,
    )["status"] == "invalid_contract"
    assert engine.promote_or_prune(skill_id)["status"] == "kept"


def test_one_heldout_success_is_insufficient_for_promotion(tmp_path):
    engine, skill_id, _ = _candidate(tmp_path)
    only_one = engine.benchmark_skill(skill_id, validation_cases=_holdout()[:1])
    assert only_one["fitness"] == 1.0
    decision = engine.promote_or_prune(skill_id)
    assert decision["status"] == "kept"
    assert decision["fitness_lower_bound"] < engine.PROMOTION_THRESHOLD


def test_inflight_skill_edit_does_not_record_evidence(tmp_path):
    engine, skill_id, directory = _candidate(tmp_path)

    class EditingRunner:
        def run_tool(self, tool_code, query):
            with (directory / "tool.py").open("a", encoding="utf-8") as handle:
                handle.write("\n# changed while evaluating\n")
            return {"status": "success", "result": {"answer": "verified", "query": query}}

    engine._benchmark._runner = EditingRunner()
    result = engine.benchmark_skill(skill_id, validation_cases=_holdout())
    assert result["status"] == "stale_inputs"
    assert engine.promote_or_prune(skill_id)["status"] == "kept"


def test_metrics_cannot_rewind_to_older_passing_evaluation(tmp_path):
    engine, skill_id, directory = _candidate(tmp_path)
    passing = engine.benchmark_skill(skill_id, validation_cases=_holdout())
    assert engine.promote_or_prune(skill_id)["status"] == "promoted"

    failing = engine.benchmark_skill(skill_id, validation_cases=[
        {"input": f"later unseen regression {index}", "expected": {"answer": "different"}}
        for index in range(10)
    ])
    assert failing["fitness"] == 0.0
    assert engine.load_promoted_skill(skill_id) is None
    assert next(s for s in engine.list_skills() if s["skill_id"] == skill_id)["status"] == "testing"
    assert engine.promote_or_prune(skill_id)["status"] == "pruned"

    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["benchmark_evidence_id"] = passing["evidence_id"]
    metrics["fitness_score"] = 1.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    decision = engine.promote_or_prune(skill_id)
    assert decision["status"] == "kept"
    assert "newer benchmark" in decision["evidence_status"]
    assert engine.load_promoted_skill(skill_id) is None


def test_fast_path_rechecks_promotion_after_cache_load(tmp_path):
    directory = tmp_path / "cached-skill"
    directory.mkdir()
    (directory / "tool.py").write_text(
        "import re\n"
        "TRIGGER_PATTERN = re.compile('auth')\n"
        "FRAGMENT_RECIPE = ['fragment-1']\n"
        "WEIGHT_PROFILE = {}\n",
        encoding="utf-8",
    )
    valid = {"value": True}
    router = FastPathRouter(
        skill_lister=lambda: [{
            "skill_id": "cached-skill", "status": "promoted",
            "path": str(directory), "metrics": {"fitness_score": 1.0},
        }],
        fragment_lookup=lambda _: {"id": "fragment-1", "token_count": 3},
        promotion_check=lambda _: valid["value"],
    )
    assert router.try_match("auth") is not None
    valid["value"] = False
    assert router.try_match("auth") is None


def test_portable_export_removes_revoked_skill(tmp_path):
    engine, skill_id, directory = _candidate(tmp_path)
    engine.benchmark_skill(skill_id, validation_cases=_holdout())
    assert engine.promote_or_prune(skill_id)["status"] == "promoted"
    target_root = tmp_path / "export"
    first = export_promoted(directory.parent.parent.parent, target_root)
    assert skill_id in first["exported"]
    assert (target_root / skill_id / "tool.py").is_file()

    (directory / "tool.py").write_text("def execute(query, context): return {}\n")
    second = export_promoted(directory.parent.parent.parent, target_root)
    assert skill_id in second["skipped"]
    assert not (target_root / skill_id).exists()


def test_changed_evaluator_revokes_existing_promotion(tmp_path, monkeypatch):
    engine, skill_id, _ = _candidate(tmp_path)
    engine.benchmark_skill(skill_id, validation_cases=_holdout())
    assert engine.promote_or_prune(skill_id)["status"] == "promoted"

    def accept_everything(cls, actual, expected, path="result"):
        return True, "matched"

    monkeypatch.setattr(
        SkillBenchmark, "_matches_expected", classmethod(accept_everything)
    )
    assert engine.load_promoted_skill(skill_id) is None
    assert engine.promote_or_prune(skill_id)["status"] == "kept"


def test_daemon_never_promotes_after_failed_benchmark():
    class FailingBenchmark:
        def benchmark_skill(self, skill_id):
            return {"status": "stale_inputs", "reason": "candidate changed"}

        def promote_or_prune(self, skill_id):
            raise AssertionError("promotion must not run")

    daemon = object.__new__(EvolutionDaemon)
    daemon._skill_engine = FailingBenchmark()
    result = daemon._benchmark_and_promote("candidate")
    assert result["status"] == "benchmark_failed"
    assert result["benchmark_status"] == "stale_inputs"
