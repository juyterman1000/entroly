//! Thin wasm-bindgen boundary over the shared NKBE budget allocator.
//!
//! The allocation semantics — dual ascent on the shared budget, the Nash
//! bargaining iterations, temperature and dual-gap bookkeeping — live in
//! `entroly_engine::nkbe`. This file converts to and from JS and nothing more.
//!
//! `allocateJSON` and `statsJSON` return the engine's own serialized objects,
//! so a Node caller and a Python caller observe the same allocation for the
//! same registered agents and fragments.

use entroly_engine::nkbe::NkbeAllocator;
use wasm_bindgen::prelude::*;

fn js_err(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn to_json_string(value: &serde_json::Value) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(js_err)
}

#[wasm_bindgen]
pub struct WasmNkbeAllocator {
    inner: NkbeAllocator,
}

#[wasm_bindgen]
impl WasmNkbeAllocator {
    /// Defaults mirror the PyO3 constructor exactly: 128000 / 0.1 / 1e-4 / 30 /
    /// 5 / 0.01. Divergent defaults would be a parity break that no test on
    /// either side would notice.
    #[wasm_bindgen(constructor)]
    pub fn new(
        global_budget: Option<u32>,
        tau: Option<f64>,
        epsilon: Option<f64>,
        max_iter: Option<u32>,
        nash_iterations: Option<u32>,
        learning_rate: Option<f64>,
    ) -> WasmNkbeAllocator {
        Self {
            inner: NkbeAllocator::new(
                global_budget.unwrap_or(128_000),
                tau.unwrap_or(0.1),
                epsilon.unwrap_or(1e-4),
                max_iter.unwrap_or(30),
                nash_iterations.unwrap_or(5),
                learning_rate.unwrap_or(0.01),
            ),
        }
    }

    #[wasm_bindgen(js_name = registerAgent)]
    pub fn register_agent(&mut self, name: &str, weight: f64, min_budget: u32) {
        self.inner.register_agent(name, weight, min_budget);
    }

    /// Returns false when the agent is unknown, matching PyO3.
    #[wasm_bindgen(js_name = addFragment)]
    pub fn add_fragment(
        &mut self,
        agent_name: &str,
        fragment_id: &str,
        relevance: f64,
        token_cost: u32,
    ) -> bool {
        self.inner
            .add_fragment(agent_name, fragment_id, relevance, token_cost)
    }

    /// `{agent_name: {budget, weight, utility}}`, as the engine builds it.
    #[wasm_bindgen(js_name = allocateJSON)]
    pub fn allocate_json(&mut self) -> Result<String, JsValue> {
        to_json_string(&self.inner.allocate())
    }

    #[wasm_bindgen(js_name = reinforce)]
    pub fn reinforce(&mut self, outcomes_json: &str) -> bool {
        self.inner.reinforce(outcomes_json)
    }

    #[wasm_bindgen(js_name = statsJSON)]
    pub fn stats_json(&self) -> Result<String, JsValue> {
        to_json_string(&self.inner.stats())
    }
}
