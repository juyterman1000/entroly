"""Counterexamples and real lifecycle tests for the experimental primitives."""
import itertools
import json
import random

import pytest

from entroly.sufficiency import Candidate, build_obligation_budget_witness
from entroly.relate.coverage_verification import audit_evidence_boundary
from entroly.relate.compression_residual import measure_context_drift
from entroly.vault import (
    BeliefArtifact,
    VaultConfig,
    VaultManager,
    classify_source_boundary,
)


def test_shared_cover_invalidates_cheapest_per_obligation_lower_bound():
    candidates = [Candidate('auth', 1, 6, False), Candidate('cache', 1, 6, False),
                  Candidate('auth_cache', 1, 10, False)]
    result = build_obligation_budget_witness(candidates, ['auth', 'cache'], 10)
    assert result.sufficient
    assert result.minimum_cover_cost == 10
    assert result.selected_ids == ('auth_cache',)


def test_bounded_search_abstains_instead_of_using_upper_bound_as_proof():
    candidates = [Candidate('auth', 1, 6, False), Candidate('cache', 1, 6, False),
                  Candidate('auth_cache', 1, 10, False)]
    result = build_obligation_budget_witness(
        candidates, ['auth', 'cache'], 10, max_states=1,
    )
    assert result.verdict == 'unknown'
    assert result.minimum_cover_cost is None
    assert result.lower_bound == 6 and result.upper_bound == 12
    assert result.deficit == 0


def test_set_cover_against_exhaustive_oracle():
    rng = random.Random(241)
    for _ in range(60):
        candidates = [Candidate(str(i), 1, rng.randrange(0, 15), False) for i in range(6)]
        obligations = ['a', 'b', 'c', 'd']
        coverage = {o: [c.unit_id for c in candidates if rng.random() < .5]
                    for o in obligations}
        costs = []
        for size in range(7):
            for chosen in itertools.combinations(candidates, size):
                ids = {c.unit_id for c in chosen}
                if all(ids.intersection(coverage[o]) for o in obligations):
                    costs.append(sum(c.cost for c in chosen))
        result = build_obligation_budget_witness(
            candidates, obligations, 15, coverage=coverage,
        )
        assert result.minimum_cover_cost == (min(costs) if costs else None)
        assert result.sufficient == bool(costs and min(costs) <= 15)
        if result.sufficient:
            assert all(set(result.selected_ids).intersection(coverage[o]) for o in obligations)


@pytest.mark.parametrize('budget', [-1, 1.5, True])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        build_obligation_budget_witness([], [], budget)


def test_missing_and_stopword_obligations_cannot_certify():
    result = build_obligation_budget_witness(
        [Candidate('auth', 1, 1, False)], ['the'], 100,
    )
    assert result.verdict == 'missing_evidence'
    assert result.minimum_cover_cost is None


@pytest.mark.parametrize('source', [[], [{'text': 'ImaginaryManagerExtra'}],
                                 [{'text': 'imaginarymanager'}]])
def test_missing_reference_never_counts_as_sourced(source):
    result = audit_evidence_boundary('`ImaginaryManager` handles tokens.', source, [])
    assert result.honest is False
    assert result.entities_sourced == 0
    assert result.entities_unsourced == 1
    assert result.coverage_ratio == 0


def test_mention_coverage_does_not_verify_assertion():
    result = audit_evidence_boundary(
        '`AuthManager` is perfectly secure.',
        [{'text': 'class AuthManager: pass'}], [],
    )
    assert result.entities_sourced == 1
    assert result.to_dict()['claims_verified'] is False


def test_path_prefix_is_not_path_evidence():
    result = audit_evidence_boundary(
        'See `src/auth.py`.', [{'source_path': 'src/auth.py.bak'}], [],
    )
    assert result.honest is False
    assert audit_evidence_boundary(
        'See `src/auth.py`.', [{'text': 'Look in src/auth.py.bak'}], [],
    ).honest is False


@pytest.mark.parametrize('fragment,retained,update', [
    ('critical facts', '', 'update'), ('critical facts', 'unrelated', ''),
    ('rate limit 100', 'rate limit 100', 'rate limit 200'),
])
def test_codec_diagnostic_cannot_certify_semantic_stability(fragment, retained, update):
    result = measure_context_drift(fragment, retained, update)
    assert result.stable is None and result.requires_reverification
    assert len(result.input_sha256) == 3


@pytest.mark.parametrize('kwargs', [{'compressors': ()}, {'compressors': ('invalid',)},
                                   {'shift_threshold': float('nan')}])
def test_invalid_codec_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        measure_context_drift('f', 'r', 'u', **kwargs)


def test_trust_does_not_follow_directory_names_or_prefixes(tmp_path):
    root = tmp_path / 'repo'
    sibling = tmp_path / 'repo-malicious'
    root.mkdir()
    sibling.mkdir()
    outside = sibling / 'payload.py'
    outside.write_text('pass')
    assert classify_source_boundary('entroly/../../outside.py') == 'unknown'
    assert classify_source_boundary(str(outside), root) == 'untrusted'
    assert classify_source_boundary('../repo-malicious/payload.py', root) == 'untrusted'


def test_symlink_escape_does_not_inherit_root_trust(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    outside = tmp_path / 'payload.py'
    outside.write_text('pass')
    try:
        (root / 'link.py').symlink_to(outside)
    except OSError:
        pytest.skip('OS does not grant symlink creation')
    assert classify_source_boundary('link.py', root) == 'untrusted'


def test_vault_trust_survives_write_read_and_filters_actual_projection(tmp_path):
    from entroly.coupling import project_beliefs
    root = tmp_path / 'repo'
    root.mkdir()
    source = root / 'auth.py'
    source.write_text('pass')
    vault = VaultManager(VaultConfig(base_path=str(tmp_path / 'vault'),
                                    trusted_source_root=str(root)))
    for entity, sources in [('good', ['auth.py']), ('bad', ['../external.py']),
                            ('mixed', ['auth.py', '../external.py'])]:
        vault.write_belief(BeliefArtifact(entity=entity, sources=sources,
                          trust_class='trusted', confidence=.99, status='verified',
                          body='Authentication token validation logic'))
    assert vault.read_belief('bad')['frontmatter']['trust_class'] == 'untrusted'
    assert vault.read_belief('mixed')['selection_eligible'] is False
    vault.write_belief(BeliefArtifact(entity='external-origin', sources=['auth.py'],
                      source_root=str(tmp_path / 'external'), trust_class='trusted',
                      confidence=.99, status='verified', body='Authentication token'))
    assert vault.read_belief('external-origin')['selection_eligible'] is False
    assert [b.entity for b in project_beliefs(vault, 'authentication token')] == ['good']
    source.unlink()
    assert project_beliefs(vault, 'authentication token') == []


def _benchmarked_engine(tmp_path):
    from entroly.skill_engine import SkillEngine
    vault = VaultManager(VaultConfig(base_path=str(tmp_path / 'vault')))
    engine = SkillEngine(vault)
    created = engine.create_skill('auth', ['check authentication'])
    folder = engine._skill_dir(created['skill_id'])
    (folder / 'tool.py').write_text("def execute(query, context=None):\n    return {'answer': 'ok'}\n")
    (folder / 'tests' / 'test_cases.json').write_text(json.dumps([
        {'input': 'question', 'expected': {'answer': 'ok'}}]))
    result = engine.benchmark_skill(created['skill_id'], validation_cases=[
        {'input': f'held-out question {index}', 'expected': {'answer': 'ok'}}
        for index in range(10)
    ])
    assert result['fitness'] == 1.0, result
    return engine, created['skill_id'], folder


@pytest.mark.parametrize('mutation', ['tool', 'cases', 'evidence', 'key',
                                     'evidence_id', 'shape'])
def test_real_benchmark_promotion_rejects_changed_inputs(tmp_path, mutation):
    engine, skill_id, folder = _benchmarked_engine(tmp_path)
    assert engine.promote_or_prune(skill_id)['status'] == 'promoted'
    assert engine.validated_promoted_skill(skill_id) is not None
    if mutation == 'tool':
        (folder / 'tool.py').write_text('def execute(query): return None')
    elif mutation == 'cases':
        (folder / 'tests' / 'test_cases.json').write_text('[]')
    elif mutation == 'evidence':
        metrics = json.loads((folder / 'metrics.json').read_text())
        path = (folder.parent.parent / 'benchmark-evidence' / skill_id /
                f"{metrics['benchmark_evidence_id']}.json")
        record = json.loads(path.read_text())
        record['payload']['passed'] = 0
        path.write_text(json.dumps(record))
    elif mutation == 'key':
        (folder.parent.parent / 'benchmark.key').unlink()
    else:
        file = folder / 'metrics.json'
        metrics = json.loads(file.read_text())
        if mutation == 'shape':
            metrics = []
        else:
            metrics['benchmark_evidence_id'] = '0' * 64
        file.write_text(json.dumps(metrics))
    assert engine.validated_promoted_skill(skill_id) is None
    result = engine.promote_or_prune(skill_id)
    assert result['evidence_status'] != 'verified'
    assert result['new_status'] == 'testing'
    assert 'status: testing' in (folder / 'SKILL.md').read_text()


def test_mutable_fitness_and_trust_labels_cannot_override_signed_result(tmp_path):
    engine, skill_id, folder = _benchmarked_engine(tmp_path)
    assert engine.promote_or_prune(skill_id)['status'] == 'promoted'
    metrics_file = folder / 'metrics.json'
    metrics = json.loads(metrics_file.read_text())
    metrics.update(fitness_score=0.0, benchmark_evidence_class='untrusted',
                   benchmark_runs='not a number')
    metrics_file.write_text(json.dumps(metrics))
    assert engine.validated_promoted_skill(skill_id) is not None
    assert engine.promote_or_prune(skill_id)['status'] == 'promoted'
