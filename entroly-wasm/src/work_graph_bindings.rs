//! Thin wasm-bindgen boundary over the shared Rust AI Work Graph.
//!
//! No Work Graph inference, trust, coordination, or handoff logic belongs here.
//! JavaScript submits canonical JSON to `entroly-engine` and receives the same
//! canonical JSON that Python receives through PyO3.

use entroly_engine::work_graph::{
    stable_edge_id_for_token, stable_node_id_for_token, HandoffReceipt, WorkGraph,
};
use wasm_bindgen::prelude::*;

fn js_err(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn parse_receipt(json_text: &str) -> Result<HandoffReceipt, JsValue> {
    serde_json::from_str(json_text).map_err(js_err)
}

/// Canonical node identity, so Node addresses artifacts exactly as Python and
/// the graph itself do. Same engine function behind all three.
#[wasm_bindgen(js_name = workGraphNodeId)]
pub fn work_graph_node_id(kind: &str, repo_id: &str, key: &str) -> Result<String, JsValue> {
    stable_node_id_for_token(kind, repo_id, key).map_err(js_err)
}

/// Canonical edge identity. See `workGraphNodeId`.
#[wasm_bindgen(js_name = workGraphEdgeId)]
pub fn work_graph_edge_id(from: &str, kind: &str, to: &str) -> Result<String, JsValue> {
    stable_edge_id_for_token(from, kind, to).map_err(js_err)
}

fn js_safe_i64(value: f64, name: &str) -> Result<i64, JsValue> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > MAX_SAFE_INTEGER {
        return Err(JsValue::from_str(&format!(
            "{name} must be a finite JavaScript-safe integer"
        )));
    }
    Ok(value as i64)
}

#[wasm_bindgen]
pub struct WasmWorkGraph {
    inner: WorkGraph,
}

#[wasm_bindgen]
impl WasmWorkGraph {
    #[wasm_bindgen(constructor)]
    pub fn new(repo_id: &str) -> Result<WasmWorkGraph, JsValue> {
        Ok(Self {
            inner: WorkGraph::new(repo_id).map_err(js_err)?,
        })
    }

    #[wasm_bindgen(js_name = fromJSON)]
    pub fn from_json(json_text: &str) -> Result<WasmWorkGraph, JsValue> {
        Ok(Self {
            inner: WorkGraph::from_json(json_text).map_err(js_err)?,
        })
    }

    #[wasm_bindgen(getter, js_name = repoId)]
    pub fn repo_id(&self) -> String {
        self.inner.repo_id().to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn revision(&self) -> u64 {
        self.inner.revision()
    }

    #[wasm_bindgen(getter, js_name = graphCommitment)]
    pub fn graph_commitment(&self) -> String {
        self.inner.graph_commitment().to_owned()
    }

    #[wasm_bindgen(getter, js_name = eventCount)]
    pub fn event_count(&self) -> usize {
        self.inner.event_count()
    }

    #[wasm_bindgen(js_name = applyEventJSON)]
    pub fn apply_event_json(&mut self, json_text: &str) -> Result<String, JsValue> {
        self.inner.apply_event_json(json_text).map_err(js_err)
    }

    #[wasm_bindgen(js_name = observeRepositoryJSON)]
    pub fn observe_repository_json(&mut self, json_text: &str) -> Result<String, JsValue> {
        self.inner
            .observe_repository_json(json_text)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = mergeJSON)]
    pub fn merge_json(&mut self, json_text: &str) -> Result<usize, JsValue> {
        self.inner.merge_json(json_text).map_err(js_err)
    }

    #[wasm_bindgen(js_name = exportJSON)]
    pub fn export_json(&self, pretty: bool) -> Result<String, JsValue> {
        self.inner.export_json(pretty).map_err(js_err)
    }

    #[wasm_bindgen(js_name = summaryJSON)]
    pub fn summary_json(&self) -> Result<String, JsValue> {
        self.inner.summary_json().map_err(js_err)
    }

    #[wasm_bindgen(js_name = snapshotJSON)]
    pub fn snapshot_json(&self, pretty: bool) -> Result<String, JsValue> {
        self.inner.snapshot_json(pretty).map_err(js_err)
    }

    #[wasm_bindgen(js_name = unfinishedJSON)]
    pub fn unfinished_json(&self, pretty: bool) -> Result<String, JsValue> {
        self.inner.unfinished_json(pretty).map_err(js_err)
    }

    #[wasm_bindgen(js_name = resumeJSON)]
    pub fn resume_json(
        &self,
        workstream_id: Option<String>,
        max_evidence: usize,
        pretty: bool,
    ) -> Result<String, JsValue> {
        self.inner
            .resume_json(workstream_id.as_deref(), max_evidence, pretty)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = coordinationJSON)]
    pub fn coordination_json(&self, now_ms: f64, pretty: bool) -> Result<String, JsValue> {
        let now_ms = js_safe_i64(now_ms, "now_ms")?;
        self.inner.coordination_json(now_ms, pretty).map_err(js_err)
    }

    #[wasm_bindgen(js_name = handoffJSON)]
    pub fn handoff_json(
        &self,
        workstream_id: &str,
        from_agent: &str,
        to_agent: &str,
        generated_at_ms: f64,
        pretty: bool,
    ) -> Result<String, JsValue> {
        let generated_at_ms = js_safe_i64(generated_at_ms, "generated_at_ms")?;
        self.inner
            .handoff_json(workstream_id, from_agent, to_agent, generated_at_ms, pretty)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = verifyHandoffJSON)]
    pub fn verify_handoff_json(&self, receipt_json: &str) -> Result<bool, JsValue> {
        let receipt = parse_receipt(receipt_json)?;
        self.inner
            .verify_handoff_receipt_against_graph(&receipt)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = verifyHandoffIntegrityJSON)]
    pub fn verify_handoff_integrity_json(receipt_json: &str) -> Result<bool, JsValue> {
        let receipt = parse_receipt(receipt_json)?;
        WorkGraph::verify_handoff_receipt(&receipt).map_err(js_err)
    }
}
