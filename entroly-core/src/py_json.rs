//! One conversion from `serde_json::Value` to Python objects.
//!
//! The engine returns JSON values so it never has to know about a host runtime.
//! Each binding renders those into its own types; on the Python side that
//! rendering is this function, written once rather than once per binding module.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<PyObject> {
    Ok(match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.unbind().into()
            } else {
                n.as_f64().unwrap_or(0.0).into_pyobject(py)?.unbind().into()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py)?.unbind().into(),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.unbind().into()
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.unbind().into()
        }
    })
}
