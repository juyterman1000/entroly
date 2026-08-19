//! Thin PyO3 boundary over `entroly_engine::trust_engine`.
//!
//! Claim support, policy validation, commitments, and guardrail semantics stay
//! in Rust. This file only converts transport values and serializes results.

use entroly_engine::trust_engine::TrustEngine;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_err(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "TrustEngine", module = "entroly_core")]
pub(crate) struct PyTrustEngine {
    inner: TrustEngine,
}

#[pymethods]
impl PyTrustEngine {
    #[new]
    #[pyo3(signature = (profile = "rag"))]
    fn new(profile: &str) -> PyResult<Self> {
        Ok(Self {
            inner: TrustEngine::try_new(profile).map_err(py_err)?,
        })
    }

    fn assess_claim_json(&self, evidence: &str, claim: &str) -> PyResult<String> {
        serde_json::to_string(&self.inner.assess_claim_support(evidence, claim)).map_err(py_err)
    }

    fn file_criticality(&self, path: &str) -> String {
        self.inner.file_criticality_label(path).to_string()
    }

    fn has_safety_signal(&self, content: &str) -> bool {
        self.inner.has_safety_signal(content)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrustEngine>()
}
