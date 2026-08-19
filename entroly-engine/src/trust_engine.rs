//! Product-level evidence-bounded Trust Engine facade.
//!
//! This module deliberately does not claim universal hallucination detection.
//! It composes existing shared-Rust evidence assessment and guardrail
//! primitives, validates policy selection explicitly, and exposes conservative
//! product vocabulary about support *against the supplied evidence*.

use crate::eicv::EicvAnalyzer;
use crate::guardrails::{file_criticality, has_safety_signal, Criticality};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

/// Profiles whose thresholds are intentionally accepted by the product-level
/// Trust Engine. The legacy EICV constructor remains permissive for backwards
/// compatibility, but new trust-sensitive callers fail closed on typos.
pub const TRUST_PROFILES: &[&str] = &[
    "rag",
    "qa",
    "summarization",
    "dialogue",
    "fact_check",
    "default",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidTrustProfile {
    profile: String,
}

impl InvalidTrustProfile {
    pub fn profile(&self) -> &str {
        &self.profile
    }
}

impl fmt::Display for InvalidTrustProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsupported Trust Engine profile {:?}; expected one of {}",
            self.profile,
            TRUST_PROFILES.join(", ")
        )
    }
}

impl std::error::Error for InvalidTrustProfile {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSupportStatus {
    Supported,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClaimEvidenceAssessment {
    pub status: EvidenceSupportStatus,
    pub support_density: f64,
    pub unsupported_fraction: f64,
    pub contradiction_fraction: f64,
    /// Cryptographic commitment to the exact evidence bytes supplied to this
    /// assessment. Format: `sha256:<64 lowercase hex characters>`.
    pub evidence_commitment: String,
}

/// Stable facade over Entroly's shared-Rust trust primitives.
pub struct TrustEngine {
    analyzer: EicvAnalyzer,
}

impl TrustEngine {
    /// Construct the product-level trust facade with an explicit supported
    /// profile. Unlike the legacy low-level EICV constructor, this fails closed
    /// on an unknown profile so a typo cannot silently select default policy.
    pub fn try_new(profile: &str) -> Result<Self, InvalidTrustProfile> {
        if !TRUST_PROFILES.contains(&profile) {
            return Err(InvalidTrustProfile {
                profile: profile.to_string(),
            });
        }
        Ok(Self {
            analyzer: EicvAnalyzer::new(profile),
        })
    }

    /// Assess how well a claim is supported by the evidence supplied to this
    /// call. The result is scoped to that evidence and is not a universal truth
    /// or hallucination oracle.
    pub fn assess_claim_support(&self, evidence: &str, claim: &str) -> ClaimEvidenceAssessment {
        let certificate = self.analyzer.verify(evidence, claim);
        let status = match certificate.decision.as_str() {
            "supported" => EvidenceSupportStatus::Supported,
            "hallucinated" => EvidenceSupportStatus::Unsupported,
            _ => EvidenceSupportStatus::Unknown,
        };
        let digest = Sha256::digest(evidence.as_bytes());

        ClaimEvidenceAssessment {
            status,
            support_density: certificate.phi,
            unsupported_fraction: certificate.unsupported_fraction,
            contradiction_fraction: certificate.contradiction_fraction,
            evidence_commitment: format!("sha256:{digest:x}"),
        }
    }

    /// Classify a repository path using Entroly's existing critical-file
    /// policy. This facade does not duplicate or reinterpret that policy.
    pub fn file_criticality(&self, path: &str) -> Criticality {
        file_criticality(path)
    }

    /// Detect content that the existing guardrail policy says must not be
    /// silently stripped from context.
    pub fn has_safety_signal(&self, content: &str) -> bool {
        has_safety_signal(content)
    }
}

impl Default for TrustEngine {
    fn default() -> Self {
        Self::try_new("rag").expect("built-in rag profile must remain valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_profile_fails_closed() {
        let error = TrustEngine::try_new("rga").err().expect("must reject typo");
        assert_eq!(error.profile(), "rga");
        assert!(error.to_string().contains("unsupported Trust Engine profile"));
    }

    #[test]
    fn default_profile_is_explicitly_supported() {
        assert!(TrustEngine::try_new("default").is_ok());
    }

    #[test]
    fn assessment_is_evidence_bounded_and_cryptographically_committed() {
        let evidence =
            "The payment service retries a request three times before returning an error.";
        let claim = "The payment service retries a request three times.";
        let assessment = TrustEngine::try_new("rag")
            .unwrap()
            .assess_claim_support(evidence, claim);
        let direct = EicvAnalyzer::new("rag").verify(evidence, claim);

        let expected = match direct.decision.as_str() {
            "supported" => EvidenceSupportStatus::Supported,
            "hallucinated" => EvidenceSupportStatus::Unsupported,
            _ => EvidenceSupportStatus::Unknown,
        };
        assert_eq!(assessment.status, expected);
        assert_eq!(assessment.support_density, direct.phi);
        assert_eq!(assessment.unsupported_fraction, direct.unsupported_fraction);
        assert_eq!(assessment.contradiction_fraction, direct.contradiction_fraction);
        assert!(assessment.evidence_commitment.starts_with("sha256:"));
        assert_eq!(assessment.evidence_commitment.len(), 71);
    }

    #[test]
    fn sha256_commitment_matches_known_vector() {
        let assessment = TrustEngine::default().assess_claim_support("abc", "some claim text");
        assert_eq!(
            assessment.evidence_commitment,
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn guardrail_facade_preserves_existing_policy() {
        let engine = TrustEngine::default();
        assert_eq!(
            engine.file_criticality("file:SECURITY.md"),
            file_criticality("file:SECURITY.md")
        );
        assert_eq!(
            engine.has_safety_signal("AWS_SECRET_ACCESS_KEY=example"),
            has_safety_signal("AWS_SECRET_ACCESS_KEY=example")
        );
    }
}
