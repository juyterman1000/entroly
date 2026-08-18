//! PyO3 binding for the Cognitive Bus.
//!
//! The routing semantics — ISA priority, Poisson rate estimation, Welford spike
//! detection, novelty scoring, per-agent queue caps and the hippocampus bridge —
//! live in `entroly_engine::cognitive_bus`. This file is transport only: it
//! converts between Python objects and the engine's types and adds nothing to
//! the model.
//!
//! Before this, the implementation lived here (966 lines) and a second, pure-Rust
//! copy sat unused in `entroly-wasm/src/cognitive_bus.rs` (946 lines, declared by
//! `mod cognitive_bus;` and referenced nowhere, never bound to wasm, never
//! exported to npm). Shared product semantics live once in Rust; the engine is
//! where "once" is, and a binding crate is not.

use pyo3::prelude::*;

use entroly_engine::cognitive_bus::{BusEvent, CognitiveBus as EngineBus};

use crate::py_json::json_to_py;

// The two drain methods deliberately do NOT build their dicts here.
//
// `drain()` and `drain_memory_bridge()` return *different* shapes, and that is
// intentional: `drain` yields the full event (8 keys, `source_agent`) while the
// hippocampus bridge yields the 5 keys `hippocampus.remember()` actually
// consumes, under the key `source`. An earlier version of this binding
// serialized both through one shared helper, which silently widened the bridge
// payload to the full event shape and renamed `source` to `source_agent` -- a
// public API break, and one that would have put Python and npm on different
// shapes the moment the WASM binding landed.
//
// Both now delegate to `entroly_engine::cognitive_bus`, which already emits the
// canonical JSON for each, so the two runtimes cannot disagree and neither can
// drift from the engine.

#[pyclass]
pub struct CognitiveBus {
    inner: EngineBus,
}

#[pymethods]
impl CognitiveBus {
    #[new]
    #[pyo3(signature = (memory_salience_threshold=50.0))]
    pub fn new(memory_salience_threshold: f64) -> Self {
        Self {
            inner: EngineBus::new(memory_salience_threshold),
        }
    }

    pub fn subscribe(&mut self, agent_id: &str, event_types: Vec<String>) {
        self.inner.subscribe(agent_id, event_types);
    }

    pub fn unsubscribe(&mut self, agent_id: &str) {
        self.inner.unsubscribe(agent_id);
    }

    pub fn set_task_context(&mut self, agent_id: &str, task_text: &str) {
        self.inner.set_task_context(agent_id, task_text);
    }

    pub fn tick(&mut self) {
        self.inner.tick();
    }

    pub fn set_tick(&mut self, tick: f64) {
        self.inner.set_tick(tick);
    }

    #[pyo3(signature = (source_agent, event_type, content, emotional_tag=0, salience=0.0))]
    pub fn publish(
        &mut self,
        source_agent: &str,
        event_type: &str,
        content: &str,
        emotional_tag: u8,
        salience: f64,
    ) -> usize {
        self.inner
            .publish(source_agent, event_type, content, emotional_tag, salience)
    }

    /// Full event shape: `id`, `source_agent`, `event_type`, `content`,
    /// `timestamp`, `emotional_tag`, `salience`, `is_spike`.
    #[pyo3(signature = (agent_id, limit=10))]
    pub fn drain(&mut self, py: Python<'_>, agent_id: &str, limit: usize) -> PyResult<PyObject> {
        json_to_py(py, &serde_json::Value::Array(self.inner.drain(agent_id, limit)))
    }

    /// Hippocampus bridge shape: `content`, `source`, `salience`,
    /// `emotional_tag`, `event_type`. Narrower than `drain()` on purpose -- see
    /// the note at the top of this file.
    pub fn drain_memory_bridge(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(
            py,
            &serde_json::Value::Array(self.inner.drain_memory_bridge()),
        )
    }

    pub fn queue_depth(&self, agent_id: &str) -> usize {
        self.inner.queue_depth(agent_id)
    }

    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        json_to_py(py, &self.inner.stats())
    }
}

impl CognitiveBus {
    /// Raw event access for other Rust modules in this crate; not exposed to
    /// Python. Kept because the previous implementation exposed it and internal
    /// callers rely on the un-serialized form.
    pub fn drain_raw(&mut self, agent_id: &str, limit: usize) -> Vec<BusEvent> {
        self.inner.drain_raw(agent_id, limit)
    }

    pub fn drain_memory_bridge_raw(&mut self) -> Vec<BusEvent> {
        self.inner.drain_memory_bridge_raw()
    }
}
