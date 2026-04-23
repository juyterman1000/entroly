---
claim_id: b7bc02e8-77b0-472e-8830-81dd75c060da
entity: depgraph
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\depgraph.rs:32
  - entroly-wasm\src\depgraph.rs:61
  - entroly-wasm\src\depgraph.rs:44
  - entroly-wasm\src\depgraph.rs:763
  - entroly-wasm\src\depgraph.rs:794
  - entroly-wasm\src\depgraph.rs:1011
  - entroly-wasm\src\depgraph.rs:1043
  - entroly-wasm\src\depgraph.rs:1054
  - entroly-wasm\src\depgraph.rs:1069
  - entroly-wasm\src\depgraph.rs:1253
last_checked: 2026-04-23T03:07:07.894159+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: depgraph

**Language:** rust
**Lines of code:** 1434

## Types
- `pub struct Dependency` — A directed dependency between two fragments.
- `pub struct DepGraph` — The dependency graph.
- `pub enum DepType`

## Functions
- `fn extract_identifiers(content: &str) -> Vec<String>` — Extract identifiers (function names, variable names) from source code. Fast, regex-free extraction for supported languages.
- `fn extract_definitions(content: &str) -> Vec<String>` — Extract symbol definitions (def, class, fn, struct, etc.)
- `fn extract_require_lhs(line: &str) -> Option<String>` — Extract the LHS variable name from `const foo = require('...')`.
- `fn extract_fn_name_from_line(line: &str) -> Option<String>` — Extract function name from a line like "fn process(" or "pub fn process(".
- `fn extract_struct_name_from_line(line: &str) -> Option<String>` — Extract struct/class name from a line like "struct Engine" or "pub struct Engine {".
- `fn is_keyword(word: &str) -> bool` — Check if an identifier is a language keyword (ignore these).
- `fn process_data(input: &str) -> PyResult<String>`
- `fn greet(name: &str) -> String`

## Dependencies
- `pyo3::prelude::`
- `serde::`
- `std::collections::`
- `wasm_bindgen::prelude::`
