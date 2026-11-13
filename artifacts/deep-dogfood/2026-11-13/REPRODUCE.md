# Reproduce / runbook

Baseline: `1de499a58835f13eba923a89a8020a9e43eed92e`.
The remote GitHub connector supplied source and Actions evidence; the analysis container could not resolve the public GitHub host for cloning. Tests below should run in CI or in an authorized checkout.

## Minimal Python demonstration of old parser behavior

```bash
python - <<'PY'
from decimal import Decimal
for value in ("NaN", "not-a-number", "Infinity"):
    try:
        rate = Decimal(str(value))
        if rate < 0: raise ValueError("negative")
        print(value, "accepted:", rate)
    except Exception as exc:
        print(value, "raised:", type(exc).__name__)
PY
```

Expected under old parser: `NaN` and `not-a-number` raise `InvalidOperation`; `Infinity` is accepted. Under the fix, `net_savings._rate` raises a caught `ValueError` for each.

## Targeted tests

```bash
python -m pytest -q tests/test_net_savings.py
python -m pytest -q tests/test_net_saving_receipts_e2e.py tests/test_proxy_traffic_value.py
ruff check entroly/net_savings.py
```

## Existing project gates

```bash
python -m pytest tests/ -v --tb=short --timeout=60
python tests/functional_test.py
cd entroly-core && cargo test --lib && cargo clippy --all-targets -- -D warnings
```

For packaging changes, also follow `AGENTS.md`/ `CLAUDE.md` release instructions. This fix changes no Rust or packaging surface.

## CI evidence

- [Baseline post-merge CI](https://github.com/juyterman1000/entroly/actions/runs/43425789632)
- [Baseline deep dogfood](https://github.com/juyterman1000/entroly/actions/runs/43419434393)
- This PR's checks must be evaluated on **this PR's head SHA**, not inferred from a previous run.

## Honest completion rule

Do not mark the live-provider A/B/C quality, cost or latency gates complete until a trace and independent verification exist. Do not commit API keys, prompts, private source content or customer traces.
