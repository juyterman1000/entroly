from __future__ import annotations

from pathlib import Path


eicv_path = Path("entroly-engine/src/eicv.rs")
text = eicv_path.read_text(encoding="utf-8")

old_intro = "//! Deterministic hallucination detection pipeline. No neural model, no LLM calls.\n"
new_intro = "//! Deterministic evidence-support assessment pipeline. No neural model, no LLM calls.\n"
if text.count(old_intro) != 1:
    raise SystemExit("EICV intro anchor changed")
text = text.replace(old_intro, new_intro, 1)

old_imports = "use crate::rnr::rnr_score;\nuse serde::Serialize;\nuse std::collections::{HashMap, HashSet};\n"
new_imports = "use crate::rnr::rnr_score;\nuse serde::Serialize;\nuse sha2::{Digest, Sha256};\nuse std::collections::{HashMap, HashSet};\nuse std::fmt;\n"
if text.count(old_imports) != 1:
    raise SystemExit("EICV import anchor changed")
text = text.replace(old_imports, new_imports, 1)

old_hash_field = '''    /// Truncated SHA-256 hex of the evidence (for cache keying / audit).\n    pub evidence_hash: String,\n'''
new_hash_field = '''    /// Legacy 64-bit DJB2 evidence fingerprint retained for compatibility.\n    ///\n    /// This is suitable only for cache-key continuity. It is NOT an integrity\n    /// commitment and must never be used to prove evidence identity.\n    pub evidence_hash: String,\n    /// Cryptographic commitment to the exact supplied evidence bytes.\n    ///\n    /// Format: `sha256:<64 lowercase hex characters>`. New trust/integrity\n    /// surfaces must use this field rather than the legacy cache fingerprint.\n    pub evidence_commitment: String,\n'''
if text.count(old_hash_field) != 1:
    raise SystemExit("EICV evidence hash field anchor changed")
text = text.replace(old_hash_field, new_hash_field, 1)

old_profiles = '''fn profile_thresholds(profile: &str) -> ProfileThresholds {\n    match profile {\n        "rag" => ProfileThresholds {\n            supported: 0.65,\n            hallucinated: 0.35,\n        },\n        "qa" => ProfileThresholds {\n            supported: 0.60,\n            hallucinated: 0.30,\n        },\n        "summarization" => ProfileThresholds {\n            supported: 0.55,\n            hallucinated: 0.25,\n        },\n        "dialogue" => ProfileThresholds {\n            supported: 0.50,\n            hallucinated: 0.20,\n        },\n        "fact_check" => ProfileThresholds {\n            supported: 0.75,\n            hallucinated: 0.45,\n        },\n        _ => ProfileThresholds {\n            supported: 0.60,\n            hallucinated: 0.35,\n        }, // default\n    }\n}\n'''
new_profiles = '''/// Profiles whose decision thresholds are part of the public EICV contract.\npub const EICV_PROFILES: &[&str] = &[\n    "rag",\n    "qa",\n    "summarization",\n    "dialogue",\n    "fact_check",\n    "default",\n];\n\n#[derive(Debug, Clone, PartialEq, Eq)]\npub struct InvalidEicvProfile {\n    profile: String,\n}\n\nimpl InvalidEicvProfile {\n    pub fn profile(&self) -> &str {\n        &self.profile\n    }\n}\n\nimpl fmt::Display for InvalidEicvProfile {\n    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {\n        write!(\n            formatter,\n            "unsupported EICV profile {:?}; expected one of {}",\n            self.profile,\n            EICV_PROFILES.join(", ")\n        )\n    }\n}\n\nimpl std::error::Error for InvalidEicvProfile {}\n\nfn profile_thresholds(profile: &str) -> Option<ProfileThresholds> {\n    let thresholds = match profile {\n        "rag" => ProfileThresholds {\n            supported: 0.65,\n            hallucinated: 0.35,\n        },\n        "qa" => ProfileThresholds {\n            supported: 0.60,\n            hallucinated: 0.30,\n        },\n        "summarization" => ProfileThresholds {\n            supported: 0.55,\n            hallucinated: 0.25,\n        },\n        "dialogue" => ProfileThresholds {\n            supported: 0.50,\n            hallucinated: 0.20,\n        },\n        "fact_check" => ProfileThresholds {\n            supported: 0.75,\n            hallucinated: 0.45,\n        },\n        "default" => ProfileThresholds {\n            supported: 0.60,\n            hallucinated: 0.35,\n        },\n        _ => return None,\n    };\n    Some(thresholds)\n}\n\nfn legacy_profile_thresholds(profile: &str) -> ProfileThresholds {\n    // `new()` historically accepted arbitrary strings and used these default\n    // thresholds. Keep that compatibility for existing low-level callers, but\n    // all new product/trust surfaces must use `try_new()` and fail closed.\n    profile_thresholds(profile).unwrap_or(ProfileThresholds {\n        supported: 0.60,\n        hallucinated: 0.35,\n    })\n}\n'''
if text.count(old_profiles) != 1:
    raise SystemExit("EICV profile anchor changed")
text = text.replace(old_profiles, new_profiles, 1)

old_constructor = '''impl EicvAnalyzer {\n    pub fn new(profile: &str) -> Self {\n        Self {\n            thresholds: profile_thresholds(profile),\n        }\n    }\n'''
new_constructor = '''impl EicvAnalyzer {\n    /// Compatibility constructor for existing callers.\n    ///\n    /// Unknown profile names retain the historical default-threshold behavior.\n    /// New trust-sensitive code must use [`Self::try_new`] so configuration\n    /// mistakes cannot silently select a different trust policy.\n    pub fn new(profile: &str) -> Self {\n        Self {\n            thresholds: legacy_profile_thresholds(profile),\n        }\n    }\n\n    /// Construct an analyzer only when the requested profile is explicit and\n    /// supported. This is the constructor used by product-level Trust Engine\n    /// APIs.\n    pub fn try_new(profile: &str) -> Result<Self, InvalidEicvProfile> {\n        let thresholds = profile_thresholds(profile).ok_or_else(|| InvalidEicvProfile {\n            profile: profile.to_string(),\n        })?;\n        Ok(Self { thresholds })\n    }\n'''
if text.count(old_constructor) != 1:
    raise SystemExit("EICV constructor anchor changed")
text = text.replace(old_constructor, new_constructor, 1)

old_hash_compute = '''        // Evidence hash (simple djb2 to avoid pulling sha2 crate)\n        let evidence_hash = format!("{:016x}", djb2_hash(evidence));\n'''
new_hash_compute = '''        // Keep the legacy DJB2 value for cache compatibility, but pair it with\n        // a cryptographic commitment for integrity-sensitive consumers.\n        let evidence_hash = format!("{:016x}", djb2_hash(evidence));\n        let evidence_digest = Sha256::digest(evidence.as_bytes());\n        let evidence_commitment = format!("sha256:{evidence_digest:x}");\n'''
if text.count(old_hash_compute) != 1:
    raise SystemExit("EICV hash computation anchor changed")
text = text.replace(old_hash_compute, new_hash_compute, 1)

old_construct_tail = '''            claim: claim.to_string(),\n            evidence_hash,\n        }\n'''
new_construct_tail = '''            claim: claim.to_string(),\n            evidence_hash,\n            evidence_commitment,\n        }\n'''
if text.count(old_construct_tail) != 1:
    raise SystemExit("EICV certificate construction anchor changed")
text = text.replace(old_construct_tail, new_construct_tail, 1)

old_djb2_doc = "/// Simple DJB2 hash (avoids pulling sha2 crate).\n"
new_djb2_doc = "/// Legacy DJB2 cache fingerprint retained for compatibility only.\n"
if text.count(old_djb2_doc) != 1:
    raise SystemExit("DJB2 documentation anchor changed")
text = text.replace(old_djb2_doc, new_djb2_doc, 1)

insert_anchor = '''    #[test]\n    fn profiles_affect_decision() {\n'''
if text.count(insert_anchor) != 1:
    raise SystemExit("EICV test insertion anchor changed")
new_tests = '''    #[test]\n    fn trust_sensitive_constructor_rejects_unknown_profile() {\n        let error = EicvAnalyzer::try_new("rag_typo").err().expect("must reject typo");\n        assert_eq!(error.profile(), "rag_typo");\n        assert!(error.to_string().contains("unsupported EICV profile"));\n    }\n\n    #[test]\n    fn compatibility_constructor_keeps_legacy_unknown_profile_behavior() {\n        let evidence = "The service retries a request three times.";\n        let claim = "The service retries a request three times.";\n        let legacy = EicvAnalyzer::new("legacy-custom-name").verify(evidence, claim);\n        let explicit = EicvAnalyzer::try_new("default")\n            .unwrap()\n            .verify(evidence, claim);\n        assert_eq!(legacy.decision, explicit.decision);\n        assert_eq!(legacy.phi, explicit.phi);\n    }\n\n    #[test]\n    fn evidence_commitment_is_real_sha256_and_legacy_hash_is_stable() {\n        let cert = verify("abc", "some claim text");\n        assert_eq!(cert.evidence_hash.len(), 16);\n        assert_eq!(\n            cert.evidence_commitment,\n            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"\n        );\n    }\n\n'''
text = text.replace(insert_anchor, new_tests + insert_anchor, 1)
eicv_path.write_text(text, encoding="utf-8")

trust = r'''//! Product-level evidence-bounded Trust Engine facade.\n//!\n//! This module does not claim universal hallucination detection. It composes\n//! existing shared-Rust evidence assessment and guardrail primitives and\n//! exposes conservative product vocabulary: Supported, Unsupported, or Unknown\n//! *against the supplied evidence*.\n\nuse crate::eicv::{EicvAnalyzer, InvalidEicvProfile};\nuse crate::guardrails::{file_criticality, has_safety_signal, Criticality};\nuse serde::Serialize;\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]\n#[serde(rename_all = "snake_case")]\npub enum EvidenceSupportStatus {\n    Supported,\n    Unsupported,\n    Unknown,\n}\n\n#[derive(Debug, Clone, Serialize)]\npub struct ClaimEvidenceAssessment {\n    pub status: EvidenceSupportStatus,\n    pub support_density: f64,\n    pub unsupported_fraction: f64,\n    pub contradiction_fraction: f64,\n    pub evidence_commitment: String,\n}\n\n/// Stable facade over Entroly's shared-Rust trust primitives.\npub struct TrustEngine {\n    analyzer: EicvAnalyzer,\n}\n\nimpl TrustEngine {\n    /// Construct a trust engine with an explicit supported EICV profile.\n    ///\n    /// Unlike the low-level compatibility constructor, this fails closed on an\n    /// unknown profile so a typo cannot silently select another trust policy.\n    pub fn try_new(profile: &str) -> Result<Self, InvalidEicvProfile> {\n        Ok(Self {\n            analyzer: EicvAnalyzer::try_new(profile)?,\n        })\n    }\n\n    /// Assess how well a claim is supported by the evidence supplied to this\n    /// call. This is not a universal truth or hallucination oracle.\n    pub fn assess_claim_support(&self, evidence: &str, claim: &str) -> ClaimEvidenceAssessment {\n        let certificate = self.analyzer.verify(evidence, claim);\n        let status = match certificate.decision.as_str() {\n            "supported" => EvidenceSupportStatus::Supported,\n            "hallucinated" => EvidenceSupportStatus::Unsupported,\n            _ => EvidenceSupportStatus::Unknown,\n        };\n        ClaimEvidenceAssessment {\n            status,\n            support_density: certificate.phi,\n            unsupported_fraction: certificate.unsupported_fraction,\n            contradiction_fraction: certificate.contradiction_fraction,\n            evidence_commitment: certificate.evidence_commitment,\n        }\n    }\n\n    /// Classify a repository path using Entroly's existing critical-file policy.\n    pub fn file_criticality(&self, path: &str) -> Criticality {\n        file_criticality(path)\n    }\n\n    /// Detect content that the existing guardrail policy says must not be\n    /// silently stripped from context.\n    pub fn has_safety_signal(&self, content: &str) -> bool {\n        has_safety_signal(content)\n    }\n}\n\nimpl Default for TrustEngine {\n    fn default() -> Self {\n        Self::try_new("rag").expect("built-in rag profile must remain valid")\n    }\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn invalid_profile_fails_closed() {\n        assert!(TrustEngine::try_new("rga").is_err());\n    }\n\n    #[test]\n    fn assessment_delegates_to_eicv_without_overclaiming_truth() {\n        let evidence = "The payment service retries a request three times before returning an error.";\n        let claim = "The payment service retries a request three times.";\n        let assessment = TrustEngine::try_new("rag")\n            .unwrap()\n            .assess_claim_support(evidence, claim);\n        let direct = EicvAnalyzer::try_new("rag").unwrap().verify(evidence, claim);\n\n        let expected = match direct.decision.as_str() {\n            "supported" => EvidenceSupportStatus::Supported,\n            "hallucinated" => EvidenceSupportStatus::Unsupported,\n            _ => EvidenceSupportStatus::Unknown,\n        };\n        assert_eq!(assessment.status, expected);\n        assert_eq!(assessment.support_density, direct.phi);\n        assert_eq!(assessment.unsupported_fraction, direct.unsupported_fraction);\n        assert_eq!(assessment.contradiction_fraction, direct.contradiction_fraction);\n        assert_eq!(assessment.evidence_commitment, direct.evidence_commitment);\n        assert!(assessment.evidence_commitment.starts_with("sha256:"));\n        assert_eq!(assessment.evidence_commitment.len(), 71);\n    }\n\n    #[test]\n    fn guardrail_facade_does_not_change_existing_policy() {\n        let engine = TrustEngine::default();\n        assert_eq!(\n            engine.file_criticality("file:SECURITY.md"),\n            file_criticality("file:SECURITY.md")\n        );\n        assert_eq!(\n            engine.has_safety_signal("AWS_SECRET_ACCESS_KEY=example"),\n            has_safety_signal("AWS_SECRET_ACCESS_KEY=example")\n        );\n    }\n}\n'''.replace('\\n', '\n')
trust_path = Path("entroly-engine/src/trust_engine.rs")
if trust_path.exists():
    raise SystemExit("trust_engine.rs already exists on target branch")
trust_path.write_text(trust, encoding="utf-8")

lib_path = Path("entroly-engine/src/lib.rs")
lib = lib_path.read_text(encoding="utf-8")
anchor = "pub mod trajectory;\npub mod utilization;\n\npub mod work_graph;\n"
replacement = "pub mod trajectory;\npub mod trust_engine;\npub mod utilization;\n\npub mod work_graph;\n"
if lib.count(anchor) != 1:
    raise SystemExit("lib.rs trust module anchor changed")
lib_path.write_text(lib.replace(anchor, replacement, 1), encoding="utf-8")

print("trust engine hardening patch applied")
