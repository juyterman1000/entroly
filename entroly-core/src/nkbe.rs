//! PyO3 binding for the NKBE budget allocator.
//!
//! The allocation semantics — dual-ascent on the shared budget, the Nash
//! bargaining iterations, the temperature and dual-gap bookkeeping — live in
//! `entroly_engine::nkbe`. This file is transport only.
//!
//! Before this, the implementation lived here (506 lines) and a second,
//! pure-Rust copy sat unused in `entroly-wasm/src/nkbe.rs` (486 lines, declared
//! by `mod nkbe;` and referenced nowhere, never bound to wasm, never exported to
//! npm). Same shape as the cognitive bus: live semantics in a binding crate,
//! a dead twin in the other one, and the engine owning neither.

use pyo3::prelude::*;

use entroly_engine::nkbe::NkbeAllocator as EngineAllocator;

use crate::py_json::json_to_py;

#[pyclass]
pub struct NkbeAllocator {
    inner: EngineAllocator,
}

#[pymethods]
impl NkbeAllocator {
    #[new]
    #[pyo3(signature = (global_budget=128000, tau=0.1, epsilon=1e-4, max_iter=30, nash_iterations=5, learning_rate=0.01))]
    pub fn new(
        global_budget: u32,
        tau: f64,
        epsilon: f64,
        max_iter: u32,
        nash_iterations: u32,
        learning_rate: f64,
    ) -> Self {
        Self {
            inner: EngineAllocator::new(
                global_budget,
                tau,
                epsilon,
                max_iter,
                nash_iterations,
                learning_rate,
            ),
        }
    }

    pub fn register_agent(&mut self, name: &str, weight: f64, min_budget: u32) {
        self.inner.register_agent(name, weight, min_budget);
    }

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

    /// Returns `{agent_name: {budget, weight, utility}}`, unchanged from the
    /// previous implementation -- the engine builds the same object.
    pub fn allocate(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.allocate())
    }

    pub fn reinforce(&mut self, outcomes_json: &str) -> bool {
        self.inner.reinforce(outcomes_json)
    }

    pub fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.stats())
    }
}
