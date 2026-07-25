# Experiment 1 — Ordered-selection reproducibility

**Status: pending a clean v2 rerun. No reproducibility result is currently
accepted or published.**

## Question

Given the same corpus, query, policy, budget, Entroly package, native module,
operating system, and architecture, does Entroly produce the same ordered
selection byte-for-byte?

This is a falsifiable engineering question. Determinism is not assumed.

## Why the earlier observation was retired

An exploratory v1 run appeared stable on one Windows machine, but its evidence
contract was insufficient for a public claim:

- the generated frozen corpus was not retained, so another maintainer could not
  replay the exact input;
- the artifact recorded a native-module hash but the capture process did not
  enforce that hash;
- the selection digest omitted `source_fragment_ids`;
- Jaccard and Kendall metrics used only source paths, collapsing distinct
  fragments from the same file;
- the corpus freezer could reuse ambient persistent index state rather than
  constructing a fresh index from an explicit clean checkout.

Those gaps do not prove Entroly is nondeterministic. They mean the prior run
does not prove that it is deterministic. Its tables and headline conclusion
were therefore removed instead of being promoted as evidence.

## v2 evidence contract

### Corpus construction

[`freeze_corpus.py`](exp1/freeze_corpus.py):

1. requires an explicit, clean Git checkout;
2. records full 40-character commit and tree identities;
3. creates a fresh non-persistent Entroly engine;
4. fails if indexing skips oversized or unreadable files;
5. binds the exact fragment list, Entroly version, native-core version, native
   module SHA-256, Python version, OS, and architecture.

The frozen corpus remains a generated artifact because it can contain repository
source. A valid published result must attach it as a release/CI artifact with its
SHA-256 and applicable source-license notice; a summary without that artifact is
not independently reproducible.

### Ordered selection identity

[`capture_selection.py`](exp1/capture_selection.py) refuses to run unless the
installed Entroly package and native module exactly match the frozen artifact.
For every selected fragment it records:

```text
(rank, source, content_sha256, content_byte_length, source_fragment_ids)
```

The result digest covers all five fields in order. Changing a source fragment's
origin identity, content, or rank changes the digest.

### Perturbation matrix

[`repro_harness.py`](exp1/repro_harness.py) runs each condition in a fresh
subprocess:

- a repeated baseline;
- `PYTHONHASHSEED ∈ {0, 1, 42, random}`;
- `RAYON_NUM_THREADS / OMP_NUM_THREADS ∈ {1, 2, 8}`;
- three seeded input permutations;
- full input reversal.

Set overlap and rank agreement use full fragment identities, including duplicate
occurrences. A failed, timed-out, malformed, runtime-mismatched, or incomplete
condition makes the run invalid and returns non-zero.

## Acceptance rule

The narrow statement “strictly deterministic on the tested runtime and
architecture” is allowed only when:

- every declared condition completes;
- every ordered-selection digest is byte-identical to baseline;
- fragment-identity Jaccard is exactly `1.0`;
- the corpus and full machine-readable result are retained.

This does **not** establish cross-platform determinism, semantic retrieval
quality, task success, or superiority over another product.

## Reproduce

Run from a clean checkout of the exact commit under test:

```bash
python docs/research/exp1/freeze_corpus.py \
  docs/research/exp1/frozen_corpus.json --source .
python docs/research/exp1/repro_harness.py \
  --output docs/research/exp1/repro_result.json
python -m pytest \
  tests/test_exp1_repro_contract.py \
  tests/test_index_determinism.py
```

Do not publish a result from console text alone. Retain the corpus, exact command,
stdout/stderr, package lock, native module, machine metadata, and exit status.
