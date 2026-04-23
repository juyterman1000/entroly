---
claim_id: e67579e3-d71e-4411-b729-1494522dd1f8
entity: sast
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\sast.rs:53
  - entroly-wasm\src\sast.rs:74
  - entroly-wasm\src\sast.rs:91
  - entroly-wasm\src\sast.rs:30
  - entroly-wasm\src\sast.rs:1858
  - entroly-wasm\src\sast.rs:1883
  - entroly-wasm\src\sast.rs:1908
  - entroly-wasm\src\sast.rs:1966
  - entroly-wasm\src\sast.rs:1989
  - entroly-wasm\src\sast.rs:2031
last_checked: 2026-04-23T03:07:07.909218+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: sast

**Language:** rust
**Lines of code:** 2853

## Types
- `pub struct SastRule` — A single SAST rule.
- `pub struct SastFinding` — A single SAST finding.
- `pub struct SastReport` — The full result of scanning a fragment.
- `pub enum Severity`

## Functions
- `fn detect_lang(source: &str) -> Option<&'static str>`
- `fn rule_applies(rule: &SastRule, lang: Option<&str>) -> bool`
- `fn is_non_code_file(source: &str) -> bool` — Detect non-code files where structural security rules are meaningless. Markdown relative links (`../guide.md`), HTML hrefs, CSS paths, etc. are NOT security vulnerabilities — scanning them produces fa
- `fn collect_taint_sources(lines: &[&str]) -> HashMap<usize, Vec<String>>` — Collect variable names that hold tainted (user-controlled) values.
- `fn extract_assignment_lhs(line: &str) -> Option<String>` — Extract the left-hand side variable name from a simple assignment. Works for: `var_name = ...`, `var_name: Type = ...`
- `fn propagate_taint(lines: &[&str], direct_sources: &HashMap<usize, Vec<String>>) -> HashSet<String>` — Given source lines and the set of taint sources, propagate taint through assignments. Returns a set of tainted variable names as of each line, plus which line they were last updated.  Algorithm: singl
- `fn line_is_tainted(line_lower: &str, tainted_vars: &HashSet<String>, direct_sources: &HashMap<usize, Vec<String>>, line_idx: usize) -> bool` — Check if a line refers to any tainted variable.
- `fn confidence_for_context(source: &str, line: &str, rule: &SastRule) -> f64` — Confidence modifier based on context.
- `fn compute_risk_score(findings: &[SastFinding]) -> f64`
- `fn scan_content(content: &str, source: &str) -> SastReport` — Scan `content` from `source` file and return a full SastReport.  This is the primary entry point. Call once per `ingest()`.
- `fn dangerous()`
- `fn read_aligned(ptr: *const u8) -> u8`

## Dependencies
- `serde::`
- `std::collections::`
