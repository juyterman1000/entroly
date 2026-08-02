//! QCCR wasm-bindgen bindings over the shared Rust selector crates.
//!
//! The compatibility selector and the audited atomic selector are both shared
//! with PyO3, so npm and pip/MCP/SDK cannot drift.

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

/// Compatibility QCCR selection; returns a JSON array of selected fragments.
#[wasm_bindgen]
pub fn qccr_select(
    fragments_json: &str,
    token_budget: i32,
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

/// Audited atomic selection; returns selected fragments, exact source spans,
/// every candidate and a scope-bounded structural sufficiency certificate.
#[wasm_bindgen]
pub fn qccr_select_with_audit(
    fragments_json: &str,
    token_budget: i32,
    query: &str,
    overrides_json: &str,
    preferred_json: &str,
) -> String {
    entroly_qccr_audit::select_with_audit_json(
        fragments_json,
        token_budget as i64,
        query,
        overrides_json,
        preferred_json,
    )
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
        let actual: Value = serde_json::from_str(&qccr_select_with_audit(
            &fixture_json(),
            12,
            "What is the Dutch name?",
            "{}",
            "[]",
        ))
        .expect("WASM audited QCCR must return JSON");
        let expected: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/audited_qccr_runtime_golden.json"
        ))
        .expect("golden fixture must be valid JSON");
        assert_eq!(actual, expected);
    }
}
