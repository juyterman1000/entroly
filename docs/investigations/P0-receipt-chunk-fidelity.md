# P0 — Context Receipt fragments are not byte-faithful to their source

**Status:** fixed and independently remeasured in `a02819d`; the original
baseline diagnosis is retained below
**Class:** product defect, compounded by a test defect
**Baseline measured:** `1ecf1e093348068539f9e1463826209c966ed535` (entroly 1.0.69, pinned)
**Environment:** CPython 3.10.0, Windows-10-10.0.26200-SP0 — recorded for provenance only; the corpus is read from the pinned ref, so the result does not depend on the checkout or platform
**Surfaces affected:** SDK, MCP server, proxy, CLI, context-commits, receipt attestation/merkle/witness/disclosure

## Remediation and release gate

The repair stopped rebuilding chunks from stripped/rejoined text and now
addresses exact UTF-8 source spans. Public selected and omitted receipt rows
carry `source_sha256`, `byte_start`, `byte_end`, and `fragment_sha256`. Recovery
schema v2 checks the receipt-owned exact-byte digest and labels old normalized
fingerprints as `legacy_normalized_text`; it no longer gives them the stronger
exact-byte description.

The fixed implementation was measured on the same pinned corpus:

| Shipped path | Files | Fragments | Verbatim | Byte range | Source SHA | Fragment SHA |
|---|---:|---:|---:|---:|---:|---:|
| Default installed/native | 1,104 | 5,117 | 5,117 | 5,117 | 5,117 | 5,117 |
| Pure-Python fallback | 1,104 | 11,986 | 11,986 | 11,986 | 11,986 | 11,986 |

The different fragment counts reflect backend chunk granularity, not exclusions;
both use the same 1,104-file manifest. A separate public-SDK probe recovered
13/13 omitted fragments from two pinned files, and every returned byte string
matched both its recorded source span and receipt-owned digest.

- [Default-path artifact](../../benchmarks/results/receipt_fragment_fidelity_default.json)
- [Pure-Python artifact](../../benchmarks/results/receipt_fragment_fidelity_python.json)
- [Public SDK recovery artifact](../../benchmarks/results/receipt_public_integrity.json)

Each cited artifact records its claim scope, denominator, pinned input ref,
implementation commit, benchmark-harness SHA-256, limitations, exclusions, and
a checked `.sha256` sidecar. `scripts/verify_readme_claims.py` prevents a
prominent README citation when that metadata or checksum is absent.

> **The corpus is pinned to a git ref, and why that matters.** An earlier
> version of this benchmark measured the *working tree*, taking "every tracked
> file matching the language list" with no exclusions in order to avoid
> cherry-picking. That was wrong twice over. The benchmark's own 281 KB result
> JSON is itself a tracked `.json` file whose content is the aggregate being
> measured, so it entered its own corpus and created a feedback loop — the
> published figure moved three times across documentation-only commits with no
> product change at all. And reading the working tree made the result
> platform-dependent, because this repository uses `core.autocrlf=true`.
>
> Both problems dissolve by reading blobs from a pinned commit
> (`BASELINE_REF = 1ecf1e0`, entroly 1.0.69) instead of from disk. Nothing added
> after that commit — this document, the benchmark, its artifacts — can exist in
> the corpus, and git stores LF, so a Windows and a Linux checkout produce
> byte-identical inputs and therefore identical numbers. `verify` reproduces the
> artifact from the same ref and is independent of the current checkout.
> `tests/test_receipt_fragment_fidelity_benchmark.py` asserts the ref is a
> pinned 40-character SHA and that the benchmark's own files cannot appear in
> its corpus.

## Claim under test

Exact recovery is advertised as a primary differentiator:

- `README.md:8` — "content-addressed evidence, exact recovery, and auditable receipts"
- `README.md:505` — "Exact recovery of omitted chunks | **576 / 576**"
- `docs/agent-integrations.html:110` — "Exact recovery uses the content handle, not fuzzy search or a generated summary."

The testable form: for any fragment a receipt describes as recoverable,
`source_bytes[byte_start:byte_end]` must equal that fragment's bytes, and the
recorded fingerprint must be recomputable by a third party.

## Result

Both properties fail, and they fail worst on source code — the product's
primary input class.

Across **11,835 fragments ingested from 1,104 files in 13 languages**, read from
the pinned baseline `1ecf1e0`:

| Metric | Result |
|---|---|
| Fragment text appears verbatim in its source file | **5,129 / 11,835 (43.3%)** |
| Fragment's own byte range slices it back out of the source | **2,479 / 11,835 (20.9%)** |

A failed `verbatim` check specifically indicates **interior** corruption. A
fragment that had merely been trimmed at its edges would still be a substring of
its source; only injected or altered bytes *inside* the fragment break the
check. So 56.7% of fragments contain text that does not exist in the repository
at all — this is the more serious of the two metrics. `byte_span_exact` can fail
for the milder reason of offset skew alone, which is why it is far lower: **79.1%
of fragments cannot be recovered from their own recorded byte range.**

Per language, sorted by byte-exactness:

| Language | Files | Fragments | Verbatim | Byte-exact |
|---|---:|---:|---:|---:|
| Ruby | 1 | 2 | 0.0% | 0.0% |
| TypeScript | 3 | 5 | 100.0% | 0.0% |
| YAML | 28 | 125 | 23.2% | 1.6% |
| **Python** | 630 | 6109 | **19.1%** | **3.8%** |
| Rust | 86 | 2509 | 42.5% | 5.6% |
| Shell | 6 | 30 | 60.0% | 6.7% |
| JavaScript | 44 | 340 | 63.2% | 9.1% |
| Markdown | 168 | 541 | 92.6% | 9.6% |
| TOML | 7 | 15 | 73.3% | 26.7% |
| HTML | 24 | 171 | 75.4% | 34.5% |
| CSS | 1 | 15 | 100.0% | 40.0% |
| JSON | 104 | 1969 | 99.9% | 98.9% |
| XML | 2 | 4 | 100.0% | 100.0% |
| **TOTAL** | **1104** | **11835** | **43.3%** | **20.9%** |

Markdown survives at 92.6% verbatim while Python sits at 19.1%: prose has real
blank-line paragraphs, so splitting on them is close to lossless. Source code
does not, and `byte_span_exact` collapses for *every* language except JSON and
XML — the two that rarely contain blank lines at all.

### CRLF makes it worse

Newline form is an independent degrader. Taking the same Python, Markdown and
Rust files and changing only the line endings:

| Line endings | Verbatim |
|---|---|
| LF (as stored in git) | 2,736 / 9,159 (**29.9%**) |
| CRLF (a default Windows checkout) | 1,870 / 9,159 (**20.4%**) |

The stray `\r` is consumed by the same `.strip()` calls, so a Windows developer
gets materially worse fidelity than the headline figure from identical content.

JSON and XML survive because they rarely contain blank lines or `#` comments.
Everything the product is actually pointed at does not.

### Public-path confirmation

Through the public SDK (`create_context_receipt` → `recover_receipt_omission`),
on two indentation-heavy Python files at budget 900:

| Property | Result |
|---|---|
| Recovered text appears verbatim in source | **0 / 13** |
| Receipt fingerprint == sha256(recovered bytes) | **0 / 13** |
| Receipt fingerprint == sha256(original byte span) | **0 / 13** |

No fragment is byte-exact, and the fingerprint verifies neither the returned
bytes nor the original span — so a receipt holder cannot check anything.

### Superseded figures

Two earlier readings should not be cited:

- An exploratory pass that capped the corpus at 40 files per language reported
  51.8% verbatim / 14.0% byte-exact. Superseded by measuring every file matching
  the declared rules with no sampling.
- Working-tree runs reported 36.1%/22.1% and then 35.1%/20.9%. Those measured a
  CRLF Windows checkout and included the benchmark's own result JSON in its own
  corpus, so they were both platform-specific and self-referential. Superseded
  by reading blobs from the pinned baseline ref.

## Corpus rules

Inclusion — deterministic, no sampling:

- tracked at the pinned baseline (`git ls-tree -r 1ecf1e0`), sorted lexicographically
- suffix in the declared `LANGUAGES` map (28 suffixes → 13 languages present)
- decodes as **strict** UTF-8 (no replacement characters)
- non-empty after strip
- at most 400,000 bytes

Exclusions are enumerated with reasons in the artifact (7 files: 1 empty, 6 over
the size cap). Per-file SHA-256 and byte length are recorded for every included
file, so `python -m benchmarks.receipt_fragment_fidelity verify` re-reads each
blob from the ref, checks its hash, and re-measures before trusting the stored
totals. Any single-byte change to the baseline would fail verification.

## Root cause

Two independent bugs in `entroly/context_receipts/ingest.py` compound.

**1. `HEADING_RE` classifies every `#` comment as a Markdown heading.**
The alternation at `ingest.py:108-113` begins with `#{1,6}\s+.+`, which matches
Python, Ruby, Shell, YAML, TOML, R, Perl and Makefile comments:

```python
>>> HEADING_RE.match('# IDF-weighted Jaccard over content words')   # match
>>> HEADING_RE.match('    # Entity precision of claim_atom')        # match
```

A heading match forces `flush()` in `_paragraph_blocks` (`ingest.py:291-293`),
so **every comment line becomes a block boundary**.

**2. Block text is stripped, then blocks are rejoined with a blank line.**
`_paragraph_blocks.flush()` does `"".join(current).strip()` (`ingest.py:266`),
removing the leading indentation of the block's first line.
`_chunk_document.flush()` then does `"\n\n".join(pending)` (`ingest.py:369`),
inserting a blank line between blocks that were adjacent in the original.

```python
>>> _paragraph_blocks('def f():\n    x = 1\n    # one\n    # two\n    y = 2\n')
[{'text': 'def f():\n    x = 1'}, {'text': '# one'}, {'text': '# two\n    y = 2'}]
```

Observed corruption in a recovered fragment of `entroly/esg.py`:

```diff
-    # Entity precision of claim_atom against ev_atom.
-    # Penalty -0.35 / bonus +0.15 — values selected by group-DRO calibration
+# Entity precision of claim_atom against ev_atom.
+
+# Penalty -0.35 / bonus +0.15 — values selected by group-DRO calibration
```

Indentation is destroyed and blank lines are injected. The result is not valid
Python, and this text is what both `selected_context` (sent to the model) and
`recover_receipt_omission` (the audit path) return.

**3. Byte offsets are skewed by the same strip.**
`block["start"]` is captured before `.strip()` runs, so offsets point a few
bytes ahead of where the stored text begins. `_split_large_block` then adds
offsets computed against the *stripped* text to that already-skewed base. This
is why byte-exactness (20.9%) is far lower than verbatim presence (43.3%): the
range is wrong independently of the text being altered.

**4. The path-based API normalizes newlines before chunking even starts.**
`read_documents_from_path` (`ingest.py:244`) uses `Path.read_text()`, which
applies universal-newline translation. On a CRLF file the indexed text differs
from the bytes on disk before any chunking happens, so a byte range into the
real file cannot match the indexed text under any chunker:

```python
>>> raw = Path('entroly/esg.py').read_bytes()
>>> raw.count(b'\r\n')                                    # 651
>>> Path('entroly/esg.py').read_text(encoding='utf-8').count('\r\n')   # 0
>>> read_documents_from_path('entroly/esg.py')[0][1] == raw.decode('utf-8')
False
```

This repository is checked out with `core.autocrlf=true`, so **every** tracked
file on this machine is CRLF — the default state of a Windows clone. This defect
is independent of the chunker: fixing block boundaries alone would not make
path-based recovery byte-exact, because the text being chunked is already not
the file. It compounds with the CRLF degradation measured above (29.9% → 20.4%
verbatim), which is why a Windows user is worse off on both counts.

**5. The fingerprint attests to the corrupted text.**
`ingest.py:414` computes
`text_fingerprint(f"{document_fingerprint}\n{start}:{end}\n{chunk_text}")` — a
composite over the already-corrupted `chunk_text`. It therefore matches neither
`sha256(recovered)` nor `sha256(original_span)`. The field records internal
consistency, not fidelity to the source, and cannot be recomputed by a receipt
holder.

## Why CI is green

`tests/test_recoverable_receipts.py` asserts the guarantee in its docstring
("recovered byte-exact") but cannot observe the defect:

- **Unrepresentative input** (line 19):
  `_TEXT = " ".join(f"token{i}" for i in range(800))` — one long line with no
  indentation, comments or blank lines. None of the structures that get
  corrupted are present in the fixture.
- **Self-referential oracle** (line 37):
  `original = {c["chunk_id"]: c["text"] for c in index["chunks"]}` — "original"
  is the chunker's *own output*. Line 43's
  `assert r["text"] == original[r["chunk_id"]]  # byte-exact` proves the store
  round-trips a fragment, not that the fragment matches the file on disk.

The test can never fail on this defect regardless of how corrupt the chunker is.

## Reproduction

```bash
# reproduce the committed corpus artifact from the pinned baseline (~1 min).
# Independent of your checkout, working tree and platform.
python -m benchmarks.receipt_fragment_fidelity verify \
    benchmarks/results/receipt_fragment_fidelity_prefix.json

# public SDK end-to-end probe
python -m benchmarks.receipt_fragment_fidelity sdk-probe \
    --out /tmp/sdk_probe.json

# minimal single-file demonstration, reading the exact baseline bytes
python - <<'PY'
import subprocess
from entroly.context_receipts.ingest import ingest_documents
raw = subprocess.run(
    ["git", "cat-file", "blob", "1ecf1e0:entroly/esg.py"],
    capture_output=True, check=True).stdout
text = raw.decode("utf-8")
idx = ingest_documents([("entroly/esg.py", text)])
print(f"verbatim: {sum(1 for c in idx.chunks if c.text in text)}/{len(idx.chunks)}")
PY
```

`verify` re-reads every blob listed in the artifact from `BASELINE_REF`, checks
each SHA-256, re-measures, and fails if any per-file or aggregate count differs.
Because the corpus is a pinned commit rather than the working tree, the
published numbers cannot drift as the repository grows.

## Raw artifacts

| Artifact | Contents |
|---|---|
| `benchmarks/results/receipt_fragment_fidelity_prefix.json` | Corpus manifest with per-file SHA-256, per-file and per-language counts, exclusions with reasons, environment |
| `benchmarks/results/receipt_fragment_fidelity_sdk_prefix.json` | Public SDK probe: per-fragment verbatim and fingerprint results, input file hashes |

## Fix direction (not implemented here)

Recovery must return **original bytes**, never a re-rendered fragment.
The invariant to establish:

> Any recoverable source fragment maps to an exact byte range in an immutable
> source object, and recovery returns precisely those bytes.

This requires separating three representations that are currently conflated:

1. **Source object** — exact original bytes, content-addressed, with whole-file
   SHA-256, byte length, encoding and newline form.
2. **Selection representation** — decoded, normalized, tokenized text used for
   retrieval only. Never the recovery authority.
3. **Recovery descriptor** — source object ID, byte start/end, fragment SHA-256,
   source-file SHA-256, schema version.

Reversing the destructive transformation is not an acceptable fix. The chunker
must stop producing recovery-authoritative text at all: boundaries become byte
offsets into the immutable source, and `.strip()` / `"\n\n".join(...)` are
removed from any recoverable path.

Verification must be executable by a party holding only the receipt, the
addressed source object, and a documented hash algorithm — and must fail if the
source changed, a single byte differs, the range is invalid, the object is
missing, or the fragment hash was computed from normalized text.
