//! Stable product-level facade for Entroly trust decisions.
//!
//! The facade composes existing shared-Rust verification and guardrail
//! primitives. It deliberately does not invent a second hallucination detector,
//! evidence model, or safety classifier.

use crate::eicv::{EicvAnalyzer, EicvCertificate};
use crate::guardrails::{file_criticality, has_safety_signal, Criticality};

/// Stable facade over Entroly's shared-Rust trust primitives.
pub struct TrustEngine {
    analyzer: EicvAnalyzer,
}

impl TrustEngine {
    /// Construct a trust engine using an existing EICV profile such as `rag`,
    /// `qa`, `summarization`, `dialogue`, or `fact_check`.
    pub fn new(profile: &str) -> Self {
        Self {
            analyzer: EicvAnalyzer::new(profile),
        }
    }

    /// Verify a claim against supplied evidence using the canonical EICV
    /// implementation. `abstain` remains distinct from both supported and
    /// hallucinated; this facade never upgrades uncertainty into truth.
    pub fn verify(&self, evidence: &str, claim: &str) -> EicvCertificate {
        self.analyzer.verify(evidence, claim)
    }

    /// Classify a repository path using Entroly's existing critical-file policy.
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
        Self::new("rag")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_is_exactly_the_existing_eicv_analyzer() {
        let evidence = "The payment service retries a request three times before returning an error.";
        let claim = "The payment service retries a request three times.";

        let facade = TrustEngine::new("rag").verify(evidence, claim);
        let direct = EicvAnalyzer::new("rag").verify(evidence, claim);

        assert_eq!(facade.decision, direct.decision);
        assert_eq!(facade.phi, direct.phi);
        assert_eq!(facade.hallucination_score, direct.hallucination_score);
        assert_eq!(facade.unsupported_fraction, direct.unsupported_fraction);
        assert_eq!(facade.contradiction_fraction, direct.contradiction_fraction);
        assert_eq!(facade.evidence_hash, direct.evidence_hash);
    }

    #[test]
    fn guardrail_facade_does_not_change_existing_policy() {
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
