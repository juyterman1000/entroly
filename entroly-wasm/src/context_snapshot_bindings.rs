//! wasm-bindgen transport for the shared verified-context snapshot verifier.
//!
//! No commitment semantics live here. The binding passes exact bytes and the
//! independently trusted commitment into `entroly-engine` and returns the
//! verified commitment or a fail-closed JavaScript exception.

use entroly_engine::verified_context_snapshot::verify_verified_context_snapshot_bytes;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = verifiedContextSnapshotVerifyBytes)]
pub fn verified_context_snapshot_verify_bytes(
    payload: &[u8],
    expected_commitment: &str,
) -> Result<String, JsValue> {
    verify_verified_context_snapshot_bytes(payload, expected_commitment)
        .map_err(|error| JsValue::from_str(&error.to_string()))
}
