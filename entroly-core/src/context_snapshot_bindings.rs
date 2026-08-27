//! PyO3 transport for the shared verified-context snapshot verifier.
//!
//! Exact byte commitment semantics live in `entroly-engine`; this module only
//! converts a Rust verification error into a Python `ValueError` and registers
//! the function on the extension module.

use entroly_engine::verified_context_snapshot::verify_verified_context_snapshot_bytes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn verified_context_snapshot_verify_bytes(
    payload: &[u8],
    expected_commitment: &str,
) -> PyResult<String> {
    verify_verified_context_snapshot_bytes(payload, expected_commitment)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(verified_context_snapshot_verify_bytes, module)?)?;
    Ok(())
}
