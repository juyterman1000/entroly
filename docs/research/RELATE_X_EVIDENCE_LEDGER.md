# RELATE-X Evidence Ledger

## Remote branch reconstruction checkpoint

Branch: `research/relate-x-omission-safety-checkpoint`
Base: `e61a60b7e5e6822d6985e62dbe019186f163a07b`

This branch is a remote research checkpoint, not a production integration and not a breakthrough claim.

Included scope:

- isolated `entroly/relate/` RELATE-X research primitives;
- conservative query constraint compiler;
- semantic collision detector;
- differential span extraction;
- explicit local-NLI adapter that never downloads silently;
- fail-closed pre-calibration policy;
- deterministic research selector;
- raw action normalizer;
- conservative omission-safety witness;
- frozen NevIR loader stub;
- unit/invariant tests in `tests/test_relate_x.py`.

Local sandbox verification before remote branch push:

```text
PYTHONPATH=. pytest -q tests/test_relate_x.py
9 passed
```

Scientific status: `CONTINUE RESEARCH`.

Known limitations:

- no NevIR neural benchmark has been run in this checkpoint;
- no ExcluIR benchmark has been run in this checkpoint;
- no Jev comparison has been independently reproduced in this checkpoint;
- this branch must not be merged or marketed as a breakthrough until frozen promotion gates pass.

Decision: we have not lost, but we have not won. The architecture is staged for falsification. The next win condition is a frozen omission-safety benchmark and external semantic retrieval gates.
