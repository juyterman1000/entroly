//! QCCR PyO3 bindings — thin wrappers over the shared Rust selector crates.
//!
//! `entroly-qccr` remains the compatibility selector SSOT. The optional audited
//! path uses `entroly-qccr-audit`, which layers exact span/candidate receipts and
//! atomic-unit packing over the same file ranker. Python and WASM therefore
//! cannot drift in either ranking or audit semantics.

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

/// Full QCCR selection.
///
/// The five-argument form is byte-compatible with the historical API and
/// returns the legacy selected-fragment array. Passing `with_audit=true`
/// returns an envelope containing `selected`, every considered candidate,
/// exact UTF-8 spans, optimizer residuals and a scope-bounded certificate.
#[pyfunction]
#[pyo3(signature = (
    fragments_json,
    token_budget,
    query,
    overrides_json="{}".to_string(),
    preferred_json="[]".to_string(),
    with_audit=false
))]
pub fn py_qccr_select(
    fragments_json: String,
    token_budget: i64,
    query: String,
    overrides_json: String,
    preferred_json: String,
    with_audit: bool,
) -> String {
    if with_audit {
        entroly_qccr_audit::select_with_audit_json(
            &fragments_json,
            token_budget,
            &query,
            &overrides_json,
            &preferred_json,
        )
    } else {
        entroly_qccr::select_json(
            &fragments_json,
            token_budget,
            &query,
            &overrides_json,
            &preferred_json,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn fixture_json() -> String {
        let first = "Résumé background. The Dutch name is:\nRhijn.";
        let second = "Authentication tokens rotate. Authentication failures are logged.";
        json!([
            {
                "fragment_id": "unicode-1",
                "source": "doc.txt",
                "content": first,
                "start_byte": 0,
                "end_byte": first.len(),
                "token_count": 20
            },
            {
                "fragment_id": "other-1",
                "source": "other.txt",
                "content": second,
                "start_byte": 0,
                "end_byte": second.len(),
                "token_count": 20
            }
        ])
        .to_string()
    }

    #[test]
    fn audited_adapter_matches_cross_runtime_golden() {
        let actual: Value = serde_json::from_str(&py_qccr_select(
            fixture_json(),
            12,
            "What is the Dutch name?".to_string(),
            "{}".to_string(),
            "[]".to_string(),
            true,
        ))
        .expect("native audited QCCR must return JSON");
        let expected: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/audited_qccr_runtime_golden.json"
        ))
        .expect("golden fixture must be valid JSON");
        assert_eq!(actual, expected);
    }
}
