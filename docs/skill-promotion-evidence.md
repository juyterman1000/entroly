# Skill promotion evidence

Entroly treats a generated skill as a candidate until it passes a separate
validation run. The automatic daemon may run development tests and retain the
result, but those tests come from the same examples that created the skill.
They cannot authorize promotion.

An operator can supply held-out cases through the Python API:

```python
from entroly.skill_engine import SkillEngine

result = engine.benchmark_skill(
    skill_id,
    validation_cases=held_out_cases,
)
decision = engine.promote_or_prune(skill_id)
```

Validation inputs must differ from development inputs. The caller is
responsible for keeping the cases genuinely independent and for checking that
the expected outputs represent useful work. Entroly cannot infer this from a
JSON test case, and a passing local test is not an externally verified task
outcome.

The promotion threshold applies to the lower end of a 95% Wilson interval
over held-out case outcomes, not to raw pass rate. The pruning threshold
applies to its upper end. A single passing case therefore cannot promote a
skill; ten of ten passes clear the current 0.7 promotion threshold. These
cases still need to represent the actual task distribution. The interval
does not correct biased sampling, correlated cases, or benchmark gaming.

Each run records an immutable, keyed integrity record containing hashes of the
candidate code, development tests, evaluation cases, evaluator source, and
per-case observations. Raw successful outputs and held-out expectations are
not written to the record. The current promotion decision reads the latest
valid record; mutable `metrics.json` values are diagnostic and cannot override
it. Changes to code, development tests, the evaluator, the record, or its key
make promotion and opted-in execution fail closed until a new valid run.

The key and candidate execute under the same local operating-system account.
This protects against accidental edits, stale results, and ordinary artifact
tampering. It is **not** an isolation boundary against malicious Python code
with access to the user's files. The runner enforces a timeout but does not
sandbox filesystem, network, or process permissions. Only benchmark or enable
execution of code trusted for that account; stronger adversarial assurance
requires an isolated runner and an evaluator authority unavailable to the
candidate process.

If an older promoted skill has no matching evidence record, the effective
status reported by `list_skills()` becomes `testing`. Its source files remain
intact so the operator can evaluate it again or inspect the prior state.
