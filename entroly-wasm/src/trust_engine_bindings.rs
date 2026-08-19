//! Thin wasm-bindgen boundary over `entroly_engine::trust_engine`.

use entroly_engine::trust_engine::TrustEngine;
use wasm_bindgen::prelude::*;

fn js_err(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
pub struct WasmTrustEngine {
    inner: TrustEngine,
}

#[wasm_bindgen]
impl WasmTrustEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(profile: &str) -> Result<WasmTrustEngine, JsValue> {
        Ok(Self {
            inner: TrustEngine::try_new(profile).map_err(js_err)?,
        })
    }

    #[wasm_bindgen(js_name = assessClaimJSON)]
    pub fn assess_claim_json(&self, evidence: &str, claim: &str) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.assess_claim_support(evidence, claim)).map_err(js_err)
    }

    #[wasm_bindgen(js_name = fileCriticality)]
    pub fn file_criticality(&self, path: &str) -> String {
        self.inner.file_criticality_label(path).to_string()
    }

    #[wasm_bindgen(js_name = hasSafetySignal)]
    pub fn has_safety_signal(&self, content: &str) -> bool {
        self.inner.has_safety_signal(content)
    }
}
