//! QCCR PyO3 bindings — thin wrappers over the shared `entroly-qccr` crate
//! (the single source of truth, shared verbatim with the WASM build, so Python
//! and npm rank and select identically). All ranking/selection logic lives in
//! `entroly-qccr`; this file only marshals types across the Python boundary.

use std::collections::HashMap;

use pyo3::prelude::*;

/// Rank files; returns `(index, score)` pairs best-first.
#[pyfunction]
#[pyo3(signature = (sources, texts, query, overrides=None))]
pub fn py_qccr_rank_files(
    sources: Vec<String>,
    texts: Vec<String>,
    query: String,
    overrides: Option<HashMap<String, f64>>,
) -> Vec<(usize, f64)> {
    entroly_qccr::rank_files(&sources, &texts, &query, &overrides.unwrap_or_default())
}

/// Expand a query into its retrieval vocabulary (sorted, deterministic).
#[pyfunction]
pub fn py_qccr_expand_query(query: String) -> Vec<String> {
    entroly_qccr::expand_query_sorted(&query)
}

/// Full QCCR selection (rank + sentence-MMR + emit + trim). `fragments_json`
/// is a JSON array of `{source, content}`; `preferred_json` is the optional
/// post-localizer file order (or "[]"); returns a JSON array of fragments.
#[pyfunction]
#[pyo3(signature = (fragments_json, token_budget, query, overrides_json="{}".to_string(), preferred_json="[]".to_string()))]
pub fn py_qccr_select(
    fragments_json: String,
    token_budget: i64,
    query: String,
    overrides_json: String,
    preferred_json: String,
) -> String {
    entroly_qccr::select_json(
        &fragments_json,
        token_budget,
        &query,
        &overrides_json,
        &preferred_json,
    )
}
