//! Shared verifier for persisted verified-code-context snapshot bytes.
//!
//! The v1 verified-code-context producer predates the Rust semantic kernel and
//! commits Python's compact, sorted, ensure-ASCII JSON byte representation.
//! Re-serializing that JSON in another runtime is not equivalent: for example,
//! JavaScript turns the valid Python number lexeme `1.0` into `1`.  Therefore
//! cross-runtime verification works on the exact stored bytes and never asks a
//! host runtime to reproduce Python's JSON serializer.
//!
//! This module owns the verification semantics. Python and Node bindings are
//! transport only; they pass canonical bytes plus the independently trusted
//! context commitment and receive either that commitment back or a fail-closed
//! error.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;

pub const VERIFIED_CONTEXT_SCHEMA_VERSION: &str = "entroly.verified-code-context.v1";
pub const VERIFIED_CONTEXT_COMMITMENT_SCOPE: &str =
    "payload-excluding-generation-command-and-context-sha256";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedContextSnapshotError {
    NonAscii,
    InvalidJson(String),
    InvalidRoot,
    UnsupportedSchema,
    VolatileMetadata,
    MissingReceipt,
    UnsupportedCommitmentScope,
    InvalidCommitment,
    ExpectedCommitmentMismatch,
    AmbiguousCommitmentField,
    NonCanonicalCommitmentField,
    CommitmentMismatch,
}

impl fmt::Display for VerifiedContextSnapshotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonAscii => write!(formatter, "context snapshot is not canonical ASCII JSON"),
            Self::InvalidJson(detail) => {
                write!(formatter, "context snapshot is not valid JSON: {detail}")
            }
            Self::InvalidRoot => write!(formatter, "context snapshot root must be an object"),
            Self::UnsupportedSchema => write!(formatter, "unsupported context snapshot schema"),
            Self::VolatileMetadata => {
                write!(
                    formatter,
                    "context snapshot contains volatile host metadata"
                )
            }
            Self::MissingReceipt => write!(formatter, "context snapshot is missing its receipt"),
            Self::UnsupportedCommitmentScope => {
                write!(formatter, "unsupported context snapshot commitment scope")
            }
            Self::InvalidCommitment => {
                write!(
                    formatter,
                    "context snapshot is missing a valid context commitment"
                )
            }
            Self::ExpectedCommitmentMismatch => write!(
                formatter,
                "context snapshot does not match the expected commitment"
            ),
            Self::AmbiguousCommitmentField => {
                write!(
                    formatter,
                    "context snapshot has an ambiguous context_sha256 field"
                )
            }
            Self::NonCanonicalCommitmentField => {
                write!(
                    formatter,
                    "context snapshot commitment field is not canonical"
                )
            }
            Self::CommitmentMismatch => write!(formatter, "context snapshot commitment is invalid"),
        }
    }
}

impl std::error::Error for VerifiedContextSnapshotError {}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, window)| (window == needle).then_some(index))
        .collect()
}

/// Verify one stored v1 verified-code-context snapshot against an independently
/// supplied context commitment.
///
/// The commitment is *not* trusted merely because the snapshot carries it. The
/// caller must supply the commitment obtained from a Context Receipt / snapshot
/// token / other trusted continuity state. This prevents a mutated snapshot
/// from self-signing a new payload and calling itself verified.
pub fn verify_verified_context_snapshot_bytes(
    bytes: &[u8],
    expected_commitment: &str,
) -> Result<String, VerifiedContextSnapshotError> {
    if !is_digest(expected_commitment) {
        return Err(VerifiedContextSnapshotError::InvalidCommitment);
    }
    if bytes.iter().any(|byte| !byte.is_ascii()) {
        return Err(VerifiedContextSnapshotError::NonAscii);
    }

    let payload: Value = serde_json::from_slice(bytes)
        .map_err(|error| VerifiedContextSnapshotError::InvalidJson(error.to_string()))?;
    let root = payload
        .as_object()
        .ok_or(VerifiedContextSnapshotError::InvalidRoot)?;

    if root.get("schema_version").and_then(Value::as_str) != Some(VERIFIED_CONTEXT_SCHEMA_VERSION) {
        return Err(VerifiedContextSnapshotError::UnsupportedSchema);
    }
    if root.contains_key("generation") || root.contains_key("command") {
        return Err(VerifiedContextSnapshotError::VolatileMetadata);
    }

    let receipt = root
        .get("receipt")
        .and_then(Value::as_object)
        .ok_or(VerifiedContextSnapshotError::MissingReceipt)?;
    if receipt.get("commitment_scope").and_then(Value::as_str)
        != Some(VERIFIED_CONTEXT_COMMITMENT_SCOPE)
    {
        return Err(VerifiedContextSnapshotError::UnsupportedCommitmentScope);
    }
    let digest = receipt
        .get("context_sha256")
        .and_then(Value::as_str)
        .filter(|value| is_digest(value))
        .ok_or(VerifiedContextSnapshotError::InvalidCommitment)?;
    if digest != expected_commitment {
        return Err(VerifiedContextSnapshotError::ExpectedCommitmentMismatch);
    }

    // Python's canonical snapshot has exactly one literal key with this name.
    // Counting the raw key token catches duplicate-key ambiguity that a normal
    // JSON parser would otherwise collapse before we could reject it.
    let key = br#""context_sha256""#;
    let key_positions = find_all(bytes, key);
    if key_positions.len() != 1 {
        return Err(VerifiedContextSnapshotError::AmbiguousCommitmentField);
    }

    let field = format!("\"context_sha256\":\"{digest}\"");
    let field_positions = find_all(bytes, field.as_bytes());
    if field_positions.len() != 1 {
        return Err(VerifiedContextSnapshotError::NonCanonicalCommitmentField);
    }
    let mut remove_start = field_positions[0];
    let mut remove_end = remove_start + field.len();

    // `context_sha256` is excluded from its own hash, so moving only this
    // field could otherwise preserve the preimage. Python's producer writes
    // sorted object keys; enforce the exact lexicographic successor (or final
    // object position) so Node cannot accept a byte layout Python rejects.
    let successor = receipt
        .keys()
        .filter(|key| key.as_str() > "context_sha256")
        .min();
    match successor {
        Some(key) => {
            if !key.is_ascii() || key.contains('"') || key.contains('\\') {
                return Err(VerifiedContextSnapshotError::NonCanonicalCommitmentField);
            }
            let expected_suffix = format!(",\"{key}\":");
            if !bytes[remove_end..].starts_with(expected_suffix.as_bytes()) {
                return Err(VerifiedContextSnapshotError::NonCanonicalCommitmentField);
            }
        }
        None => {
            if bytes.get(remove_end) != Some(&b'}') {
                return Err(VerifiedContextSnapshotError::NonCanonicalCommitmentField);
            }
        }
    }

    if bytes.get(remove_end) == Some(&b',') {
        remove_end += 1;
    } else if remove_start > 0 && bytes.get(remove_start - 1) == Some(&b',') {
        remove_start -= 1;
    } else {
        return Err(VerifiedContextSnapshotError::NonCanonicalCommitmentField);
    }

    let mut hasher = Sha256::new();
    hasher.update(&bytes[..remove_start]);
    hasher.update(&bytes[remove_end..]);
    let actual = format!("{:x}", hasher.finalize());
    if actual != digest {
        return Err(VerifiedContextSnapshotError::CommitmentMismatch);
    }
    Ok(digest.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (String, Vec<u8>) {
        let preimage = format!(
            "{{\"payload\":{{\"score\":1.0,\"text\":\"snowman \\u2603\"}},\"receipt\":{{\"commitment_scope\":\"{}\"}},\"schema_version\":\"{}\"}}",
            VERIFIED_CONTEXT_COMMITMENT_SCOPE,
            VERIFIED_CONTEXT_SCHEMA_VERSION,
        );
        let digest = format!("{:x}", Sha256::digest(preimage.as_bytes()));
        let sealed = format!(
            "{{\"payload\":{{\"score\":1.0,\"text\":\"snowman \\u2603\"}},\"receipt\":{{\"commitment_scope\":\"{}\",\"context_sha256\":\"{}\"}},\"schema_version\":\"{}\"}}",
            VERIFIED_CONTEXT_COMMITMENT_SCOPE,
            digest,
            VERIFIED_CONTEXT_SCHEMA_VERSION,
        );
        (digest, sealed.into_bytes())
    }

    #[test]
    fn verifies_python_v1_bytes_without_reserializing_numbers_or_unicode() {
        let (digest, bytes) = fixture();
        assert!(bytes.windows(11).any(|window| window == b"\"score\":1.0"));
        assert!(bytes.windows(6).any(|window| window == br#"\u2603"#));
        assert_eq!(
            verify_verified_context_snapshot_bytes(&bytes, &digest).unwrap(),
            digest
        );
    }

    #[test]
    fn rejects_reordered_context_hash_field_even_when_preimage_is_unchanged() {
        let (digest, bytes) = fixture();
        let text = String::from_utf8(bytes).unwrap();
        let canonical = format!(
            "\"commitment_scope\":\"{}\",\"context_sha256\":\"{}\"",
            VERIFIED_CONTEXT_COMMITMENT_SCOPE, digest,
        );
        let reordered = format!(
            "\"context_sha256\":\"{}\",\"commitment_scope\":\"{}\"",
            digest, VERIFIED_CONTEXT_COMMITMENT_SCOPE,
        );
        let mutated = text.replace(&canonical, &reordered);
        assert_ne!(mutated, text);
        assert_eq!(
            verify_verified_context_snapshot_bytes(mutated.as_bytes(), &digest),
            Err(VerifiedContextSnapshotError::NonCanonicalCommitmentField)
        );
    }

    #[test]
    fn fails_closed_on_tamper_or_wrong_external_commitment() {
        let (digest, mut bytes) = fixture();
        let position = bytes
            .windows(b"snowman".len())
            .position(|window| window == b"snowman")
            .unwrap();
        bytes[position] = b'S';
        assert_eq!(
            verify_verified_context_snapshot_bytes(&bytes, &digest),
            Err(VerifiedContextSnapshotError::CommitmentMismatch)
        );

        let (_digest, bytes) = fixture();
        assert_eq!(
            verify_verified_context_snapshot_bytes(&bytes, &"0".repeat(64)),
            Err(VerifiedContextSnapshotError::ExpectedCommitmentMismatch)
        );
    }

    #[test]
    fn rejects_unknown_schema_scope_volatile_fields_and_duplicate_commitment_keys() {
        let (digest, bytes) = fixture();
        let text = String::from_utf8(bytes).unwrap();

        let wrong_schema = text.replace(VERIFIED_CONTEXT_SCHEMA_VERSION, "future.schema.v2");
        assert_eq!(
            verify_verified_context_snapshot_bytes(wrong_schema.as_bytes(), &digest),
            Err(VerifiedContextSnapshotError::UnsupportedSchema)
        );

        let wrong_scope = text.replace(VERIFIED_CONTEXT_COMMITMENT_SCOPE, "other-scope");
        assert_eq!(
            verify_verified_context_snapshot_bytes(wrong_scope.as_bytes(), &digest),
            Err(VerifiedContextSnapshotError::UnsupportedCommitmentScope)
        );

        let volatile = text.replacen('{', "{\"generation\":1,", 1);
        assert_eq!(
            verify_verified_context_snapshot_bytes(volatile.as_bytes(), &digest),
            Err(VerifiedContextSnapshotError::VolatileMetadata)
        );

        let duplicate = text.replace(
            "\"context_sha256\":",
            &format!("\"context_sha256\":\"{}\",\"context_sha256\":", digest),
        );
        assert_eq!(
            verify_verified_context_snapshot_bytes(duplicate.as_bytes(), &digest),
            Err(VerifiedContextSnapshotError::AmbiguousCommitmentField)
        );
    }
}
