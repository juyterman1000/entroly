---
claim_id: 43ad9b47-9edf-45dd-bea8-898619878c04
entity: query
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\query.rs:58
  - entroly-wasm\src\query.rs:74
  - entroly-wasm\src\query.rs:81
  - entroly-wasm\src\query.rs:86
  - entroly-wasm\src\query.rs:99
  - entroly-wasm\src\query.rs:118
  - entroly-wasm\src\query.rs:170
  - entroly-wasm\src\query.rs:223
  - entroly-wasm\src\query.rs:250
last_checked: 2026-04-23T03:07:07.906541+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: query

**Language:** rust
**Lines of code:** 394

## Types
- `pub struct QueryAnalysis`

## Functions
- `fn tokenize(text: &str) -> Vec<String>` — Tokenize text into lowercase words, removing stop words and punctuation.
- `fn is_stop_word(word: &str) -> bool`
- `fn tf(tokens: &[String]) -> HashMap<String, f64>` — Compute TF (term frequency) for a token list.
- `fn idf(corpus: &[Vec<String>]) -> HashMap<String, f64>` — Compute IDF (inverse document frequency) given a corpus of token lists. Uses log(N / (1 + df)) + 1 (smooth IDF).
- `fn extract_key_terms(
    query: &str,
    fragment_summaries: &[String],
    top_n: usize,
) -> Vec<String>` — Extract top-N key terms from a query using TF-IDF over the fragment corpus.  If `fragment_summaries` is empty, falls back to TF-only ranking.  Returns sorted (descending score) vector of term strings.
- `fn compute_vagueness(query: &str) -> (f64, String)` —  Algorithm: vagueness = generic_verb_ratio × 0.5 + short_penalty × 0.3 − specificity_bonus × 0.2  - `generic_verb_ratio`: fraction of tokens that are generic verbs ("fix", "help", "add") - `short_pena
- `fn analyze_query(query: &str, fragment_summaries: &[String]) -> QueryAnalysis` — Full query analysis: vagueness + key term extraction.
- `fn refine_heuristic(query: &str, fragment_summaries: &[String]) -> String` —  Algorithm: 1. Extract key terms from query 2. Find which fragments have highest vocabulary overlap with query terms 3. Inject top matching fragments' unique vocabulary into query 4. Reconstruct a mor

## Dependencies
- `serde::Serialize`
- `std::collections::HashMap`
