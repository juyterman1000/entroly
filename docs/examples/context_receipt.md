# Context Receipt cr_049a1d7b4ffb

Query: `Does this contract have a change-of-control clause?`

## Token Budget

- Budget: 45
- Source tokens: 78
- Selected tokens: 39
- Reduction: 50.0%
- Source-to-selected ratio: 2.00:1

## Coverage And Risk Controls

- Coverage score: 0.598
- Review level: medium
- Dependency closure: partial
- Omitted evidence pressure: medium
- Replayable fingerprints: True

## Included Context

### chk_3edf66e42227
- Source: `docs/examples/context_receipt_sources/schedule-a.md` - Schedule A
- Tokens: 16; score: 0.7536
- Why: lexical match: change, control, of
- Fingerprint: `sha256:b076f577efa5447333dc56e47d9dc2b89d3ac6fe16bef9e3d5dfe1c186c9c292`
- Dependencies included: chk_c69fe14d7aae
- Dependencies missing or unresolved: chk_2c47a62f27b9

### chk_c69fe14d7aae
- Source: `docs/examples/context_receipt_sources/addendum.md` - Addendum A
- Tokens: 23; score: 0.6957
- Why: lexical match: change, control, of; contains explicit dependency/reference language
- Fingerprint: `sha256:d66a9e964fcd979abb29d5d660d7015bef4ccc74269e674b6c62563d73be4c7f`
- Dependencies included: chk_3edf66e42227, chk_3edf66e42227
- Dependencies missing or unresolved: chk_2c47a62f27b9, chk_2c47a62f27b9, Addendum A, chk_2c47a62f27b9

## Omitted Context

### chk_2c47a62f27b9
- Source: `docs/examples/context_receipt_sources/master.md` - Section 1 Definitions
- Tokens: 39; score: 0.7278
- Why omitted: dependency_not_included_due_to_budget
- Ranking reason: lexical match: change, control, of
- Preview: # Section 1 Definitions "Change of Control" means a merger, sale of substantially all assets, or replacement of control. # Section 2 Assignment Neither party may assign this Agreement without prior written consent.

## Dependency Graph Summary

- `chk_c69fe14d7aae` -> `chk_2c47a62f27b9` (defined_term, resolved): change of control
- `chk_c69fe14d7aae` -> `chk_2c47a62f27b9` (pursuant_to, resolved): Section 1
- `chk_c69fe14d7aae` -> `chk_3edf66e42227` (see_reference, resolved): Schedule A.
- `chk_c69fe14d7aae` -> `UNRESOLVED` (structural_reference, unresolved): Addendum A
- `chk_c69fe14d7aae` -> `chk_2c47a62f27b9` (structural_reference, resolved): Section 1
- `chk_c69fe14d7aae` -> `chk_3edf66e42227` (structural_reference, resolved): Schedule A.
- `chk_2c47a62f27b9` -> `chk_c69fe14d7aae` (structural_reference, resolved): Section 1
- `chk_2c47a62f27b9` -> `UNRESOLVED` (structural_reference, unresolved): Section 2
- `chk_3edf66e42227` -> `chk_2c47a62f27b9` (defined_term, resolved): change of control
- `chk_3edf66e42227` -> `chk_c69fe14d7aae` (structural_reference, resolved): Schedule A

## Risks And Warnings

- Dependency not included due to budget: chk_3edf66e42227 -> chk_2c47a62f27b9 (defined_term)
- 2 dependency reference(s) could not be resolved to an ingested chunk.
- 1 relevant chunk(s) were omitted; inspect omitted_context.

## Reproducibility

- Reproducibility hash: `049a1d7b4ffb61fccae5d001cf662aed269b78b5fc55c8b18f0e7cbf36079642`
- Schema: `context-receipt.v1`
