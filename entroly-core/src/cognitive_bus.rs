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
use pyo3::types::PyDict;

use entroly_engine::cognitive_bus::{BusEvent, CognitiveBus as EngineBus};

use crate::py_json::json_to_py;

/// Serialize one bus event into the dict shape Python callers already expect.
///
/// The key set and ordering are unchanged from the previous implementation:
/// renaming or dropping a key here is a public API break for every consumer of
/// `drain()` and `drain_memory_bridge()`.
fn event_dict<'py>(py: Python<'py>, event: &BusEvent) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", event.id)?;
    dict.set_item("source_agent", &event.source_agent)?;
    dict.set_item("event_type", event.event_type.as_str())?;
    dict.set_item("content", &event.content)?;
    dict.set_item("timestamp", event.timestamp)?;
    dict.set_item("emotional_tag", event.emotional_tag)?;
    dict.set_item("salience", event.salience)?;
    dict.set_item("is_spike", event.is_spike)?;
    Ok(dict)
}

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

    #[pyo3(signature = (agent_id, limit=10))]
    pub fn drain<'py>(
        &mut self,
        py: Python<'py>,
        agent_id: &str,
        limit: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .drain_raw(agent_id, limit)
            .iter()
            .map(|event| event_dict(py, event))
            .collect()
    }

    pub fn drain_memory_bridge<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .drain_memory_bridge_raw()
            .iter()
            .map(|event| event_dict(py, event))
            .collect()
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
