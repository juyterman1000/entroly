//! QCCR wasm-bindgen bindings — thin wrappers over the shared `entroly-qccr`
//! crate (the single source of truth, shared verbatim with the PyO3 build, so
//! npm and pip/MCP/SDK rank and select identically). All logic lives in
//! `entroly-qccr`; this file only marshals JSON across the JS boundary.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

/// Rank files; returns a JSON array of `[index, score]` pairs, best-first.
#[wasm_bindgen]
pub fn qccr_rank_files(
    sources: Vec<String>,
    texts: Vec<String>,
    query: &str,
    overrides_json: &str,
) -> String {
    let ov: HashMap<String, f64> = serde_json::from_str(overrides_json).unwrap_or_default();
    let ranked = entroly_qccr::rank_files(&sources, &texts, query, &ov);
    serde_json::to_string(&ranked).unwrap_or_else(|_| "[]".to_string())
}

/// Expand a query into its retrieval vocabulary; returns a sorted JSON array.
#[wasm_bindgen]
pub fn qccr_expand_query(query: &str) -> String {
    serde_json::to_string(&entroly_qccr::expand_query_sorted(query))
        .unwrap_or_else(|_| "[]".to_string())
}

/// Full QCCR selection (rank + sentence-MMR + emit + trim). `fragments_json`
/// is a JSON array of `{source, content}`; `preferred_json` is the optional
/// file order (or "[]"); returns a JSON array of selected fragments.
#[wasm_bindgen]
pub fn qccr_select(
    fragments_json: &str,
    token_budget: i32, // i32 -> JS number (i64 would surface as BigInt to callers)
    query: &str,
    overrides_json: &str,
    preferred_json: &str,
) -> String {
    entroly_qccr::select_json(
        fragments_json,
        token_budget as i64,
        query,
        overrides_json,
        preferred_json,
    )
}
