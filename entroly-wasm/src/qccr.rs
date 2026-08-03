//! QCCR wasm-bindgen bindings — thin wrappers over the shared `entroly-qccr`
//! crate. The tests in this module execute the same frozen parity fixture as
//! the Python/PyO3 surface while compiling `entroly-qccr` with `regex-lite`.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn qccr_rank_files(
    sources: Vec<String>,
    texts: Vec<String>,
    query: &str,
    overrides_json: &str,
) -> String {
    let overrides: HashMap<String, f64> =
        serde_json::from_str(overrides_json).unwrap_or_default();
    let ranked = entroly_qccr::rank_files(&sources, &texts, query, &overrides);
    serde_json::to_string(&ranked).unwrap_or_else(|_| "[]".to_string())
}

#[wasm_bindgen]
pub fn qccr_expand_query(query: &str) -> String {
    serde_json::to_string(&entroly_qccr::expand_query_sorted(query))
        .unwrap_or_else(|_| "[]".to_string())
}

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

#[cfg(test)]
mod parity_tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/qccr_parity.json"))
            .expect("QCCR parity fixture must be valid JSON")
    }

    #[test]
    fn regex_lite_surface_matches_frozen_rank_order() {
        let fixture = fixture();
        assert_eq!(
            fixture["schema_version"].as_str(),
            Some("entroly.qccr-parity.v1")
        );
        let cases = fixture["cases"]
            .as_array()
            .expect("QCCR parity fixture requires cases");
        assert!(!cases.is_empty(), "QCCR parity fixture must not be empty");

        for case in cases {
            let files = case["files"]
                .as_object()
                .expect("case files must be an object");
            let sources: Vec<String> = files.keys().cloned().collect();
            let texts: Vec<String> = sources
                .iter()
                .map(|source| {
                    files[source]
                        .as_str()
                        .expect("file content must be text")
                        .to_string()
                })
                .collect();
            let query = case["query"].as_str().expect("query must be text");
            let mut ranked = entroly_qccr::rank_files(
                &sources,
                &texts,
                query,
                &HashMap::new(),
            );
            ranked.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let actual: Vec<&str> = ranked
                .iter()
                .map(|(index, _)| sources[*index].as_str())
                .collect();
            let expected: Vec<&str> = case["expected_rank_order"]
                .as_array()
                .expect("expected rank order must be an array")
                .iter()
                .map(|value| value.as_str().expect("rank entries must be text"))
                .collect();
            assert_eq!(actual, expected, "case {} drifted", case["id"]);
        }
    }

    #[test]
    fn regex_lite_expansion_size_matches_frozen_fixture() {
        let fixture = fixture();
        for case in fixture["cases"]
            .as_array()
            .expect("QCCR parity fixture requires cases")
        {
            let query = case["query"].as_str().expect("query must be text");
            let expected = case["expansion_terms"]
                .as_u64()
                .expect("expansion_terms must be an integer") as usize;
            let actual = entroly_qccr::expand_query_sorted(query).len();
            assert_eq!(actual, expected, "case {} expansion drifted", case["id"]);
        }
    }
}
