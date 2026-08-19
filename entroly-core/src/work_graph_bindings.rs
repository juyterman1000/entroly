//! Thin PyO3 binding over `entroly_engine::work_graph`.
//!
//! All Work Graph semantics live in `entroly-engine`. This module deliberately
//! performs only boundary conversion and error mapping so Python and npm cannot
//! drift in task-state inference, trust handling, coordination, or handoff rules.

use entroly_engine::work_graph::{
    stable_edge_id_for_token, stable_node_id_for_token, HandoffReceipt, WorkGraph,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_err(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_receipt(json_text: &str) -> PyResult<HandoffReceipt> {
    serde_json::from_str(json_text).map_err(py_err)
}

/// Canonical node identity, exposed so Python addresses artifacts the same way
/// the graph does.
///
/// Without this, code that wants to refer to a file or symbol has to invent an
/// id, which is how `entroly/repository_intelligence` ended up with a
/// free-form `symbol_id` that cannot be matched against a `NodeKind::File` in
/// the work graph. One artifact, one id, computed in one place.
#[pyfunction]
pub(crate) fn work_graph_node_id(kind: &str, repo_id: &str, key: &str) -> PyResult<String> {
    stable_node_id_for_token(kind, repo_id, key).map_err(py_err)
}

/// Canonical edge identity. See `work_graph_node_id`.
#[pyfunction]
pub(crate) fn work_graph_edge_id(from: &str, kind: &str, to: &str) -> PyResult<String> {
    stable_edge_id_for_token(from, kind, to).map_err(py_err)
}

#[pyclass(name = "WorkGraph", module = "entroly_core")]
pub(crate) struct PyWorkGraph {
    inner: WorkGraph,
}

#[pymethods]
impl PyWorkGraph {
    #[new]
    fn new(repo_id: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WorkGraph::new(repo_id).map_err(py_err)?,
        })
    }

    #[staticmethod]
    fn from_json(json_text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WorkGraph::from_json(json_text).map_err(py_err)?,
        })
    }

    #[getter]
    fn repo_id(&self) -> String {
        self.inner.repo_id().to_owned()
    }

    #[getter]
    fn revision(&self) -> u64 {
        self.inner.revision()
    }

    #[getter]
    fn graph_commitment(&self) -> String {
        self.inner.graph_commitment().to_owned()
    }

    #[getter]
    fn event_count(&self) -> usize {
        self.inner.event_count()
    }

    fn apply_event_json(&mut self, json_text: &str) -> PyResult<String> {
        self.inner.apply_event_json(json_text).map_err(py_err)
    }

    fn observe_repository_json(&mut self, json_text: &str) -> PyResult<String> {
        self.inner
            .observe_repository_json(json_text)
            .map_err(py_err)
    }

    fn merge_json(&mut self, json_text: &str) -> PyResult<usize> {
        self.inner.merge_json(json_text).map_err(py_err)
    }

    #[pyo3(signature = (pretty = false))]
    fn export_json(&self, pretty: bool) -> PyResult<String> {
        self.inner.export_json(pretty).map_err(py_err)
    }

    fn summary_json(&self) -> PyResult<String> {
        self.inner.summary_json().map_err(py_err)
    }

    #[pyo3(signature = (pretty = false))]
    fn snapshot_json(&self, pretty: bool) -> PyResult<String> {
        self.inner.snapshot_json(pretty).map_err(py_err)
    }

    #[pyo3(signature = (pretty = false))]
    fn unfinished_json(&self, pretty: bool) -> PyResult<String> {
        self.inner.unfinished_json(pretty).map_err(py_err)
    }

    #[pyo3(signature = (workstream_id = None, max_evidence = 128, pretty = false))]
    fn resume_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> PyResult<String> {
        self.inner
            .resume_json(workstream_id, max_evidence, pretty)
            .map_err(py_err)
    }

    #[pyo3(signature = (workstream_id = None, max_evidence = 128, pretty = false))]
    fn context_scope_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> PyResult<String> {
        self.inner
            .context_scope_json(workstream_id, max_evidence, pretty)
            .map_err(py_err)
    }

    #[pyo3(signature = (now_ms, pretty = false))]
    fn coordination_json(&self, now_ms: i64, pretty: bool) -> PyResult<String> {
        self.inner.coordination_json(now_ms, pretty).map_err(py_err)
    }

    #[pyo3(signature = (workstream_id, from_agent, to_agent, generated_at_ms, pretty = false))]
    fn handoff_json(
        &self,
        workstream_id: &str,
        from_agent: &str,
        to_agent: &str,
        generated_at_ms: i64,
        pretty: bool,
    ) -> PyResult<String> {
        self.inner
            .handoff_json(workstream_id, from_agent, to_agent, generated_at_ms, pretty)
            .map_err(py_err)
    }

    fn verify_handoff_json(&self, receipt_json: &str) -> PyResult<bool> {
        let receipt = parse_receipt(receipt_json)?;
        self.inner
            .verify_handoff_receipt_against_graph(&receipt)
            .map_err(py_err)
    }

    #[staticmethod]
    fn verify_handoff_integrity_json(receipt_json: &str) -> PyResult<bool> {
        let receipt = parse_receipt(receipt_json)?;
        WorkGraph::verify_handoff_receipt(&receipt).map_err(py_err)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkGraph>()
}
