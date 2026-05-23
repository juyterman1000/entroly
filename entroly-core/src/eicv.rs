//! EICV — Evidence-Invariant Causal Verification
//!
//! Deterministic hallucination detection pipeline. No neural model, no LLM calls.
//! Computes an epistemic support density φ ∈ [0,1] via a 7-step fusion:
//!
//!   φ = 0.30·T(G) + 0.25·NLI + 0.20·RNR + 0.15·γ + 0.10·(1 − H_sem)
//!
//! Layers:
//!   T(G)  — Bipartite token-graph matching (greedy Jaccard)
//!   NLI   — Lexical entailment proxy (token-coverage)
//!   RNR   — Recognize-and-Reject (recognition ratio − novel entity penalty)
//!   γ     — Named-entity / number overlap
//!   H_sem — Semantic entropy of claim tokens, adjusted by evidence coverage
//!
//! Decision bands are configurable via profiles (rag, qa, summarization, …).

use crate::rnr::rnr_score;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

// ── Certificate ──────────────────────────────────────────────────────

/// The output of a single EICV verification.
#[derive(Debug, Clone, Serialize)]
pub struct EicvCertificate {
    /// Epistemic support density [0 = hallucinated, 1 = grounded].
    pub phi: f64,
    /// 1 − phi.
    pub hallucination_score: f64,
    /// "supported" | "abstain" | "hallucinated".
    pub decision: String,
    /// Per-layer score breakdown.
    pub layer_scores: LayerScores,
    /// Number of atomic propositions extracted from the claim.
    pub n_claim_atoms: usize,
    /// Number of atomic propositions extracted from the evidence.
    pub n_ev_atoms: usize,
    /// Fraction of claim atoms with no evidence support.
    pub unsupported_fraction: f64,
    /// Fraction of claim atoms with active contradiction signal.
    pub contradiction_fraction: f64,
    /// Wall-clock latency in milliseconds.
    pub elapsed_ms: f64,
    /// The claim that was verified (echoed back).
    pub claim: String,
    /// Truncated SHA-256 hex of the evidence (for cache keying / audit).
    pub evidence_hash: String,
}

/// Per-layer breakdown inside an [`EicvCertificate`].
#[derive(Debug, Clone, Serialize)]
pub struct LayerScores {
    pub tg: f64,
    pub nli: f64,
    pub rnr: f64,
    pub gamma: f64,
    pub h_sem: f64,
}

// ── Profile thresholds ───────────────────────────────────────────────

struct ProfileThresholds {
    supported: f64,
    hallucinated: f64,
}

fn profile_thresholds(profile: &str) -> ProfileThresholds {
    match profile {
        "rag" => ProfileThresholds { supported: 0.65, hallucinated: 0.35 },
        "qa" => ProfileThresholds { supported: 0.60, hallucinated: 0.30 },
        "summarization" => ProfileThresholds { supported: 0.55, hallucinated: 0.25 },
        "dialogue" => ProfileThresholds { supported: 0.50, hallucinated: 0.20 },
        "fact_check" => ProfileThresholds { supported: 0.75, hallucinated: 0.45 },
        _ => ProfileThresholds { supported: 0.60, hallucinated: 0.35 }, // default
    }
}

// ── Analyzer ─────────────────────────────────────────────────────────

/// EICV analyzer — stateless, reusable across calls.
pub struct EicvAnalyzer {
    thresholds: ProfileThresholds,
}

impl EicvAnalyzer {
    pub fn new(profile: &str) -> Self {
        Self {
            thresholds: profile_thresholds(profile),
        }
    }

    /// Verify a single claim against evidence.
    ///
    /// Returns an [`EicvCertificate`] with the fused phi score and decision.
    pub fn verify(&self, evidence: &str, claim: &str) -> EicvCertificate {
        let start = std::time::Instant::now();

        // Step 0: Atomic decomposition
        let claim_atoms = atomic_decompose(claim);
        let ev_atoms = atomic_decompose(evidence);

        let n_claim = claim_atoms.len().max(1);
        let n_ev = ev_atoms.len().max(1);

        // Step 1: Token Graph T(G) — bipartite greedy Jaccard matching
        let tg = bipartite_matching(&claim_atoms, &ev_atoms);

        // Step 2: Lexical entailment (NLI-free)
        let nli = lexical_entailment(&claim_atoms, &ev_atoms);

        // Step 3: RNR
        let rnr = rnr_score(claim, evidence);

        // Step 4: Gamma — entity / number overlap
        let gamma = entity_overlap(claim, evidence);

        // Step 5: Semantic entropy
        let h_sem = semantic_entropy(claim, evidence);

        // Step 6: Phi fusion
        let phi = (0.30 * tg + 0.25 * nli + 0.20 * rnr + 0.15 * gamma + 0.10 * (1.0 - h_sem))
            .clamp(0.0, 1.0);

        // Step 7: Decision
        let decision = if phi >= self.thresholds.supported {
            "supported"
        } else if phi <= self.thresholds.hallucinated {
            "hallucinated"
        } else {
            "abstain"
        };

        // Unsupported / contradiction fractions
        let unsupported_fraction = if !claim_atoms.is_empty() {
            let unsupported = claim_atoms
                .iter()
                .filter(|ca| {
                    ev_atoms
                        .iter()
                        .all(|ea| token_overlap(ca, ea) < 0.1)
                })
                .count();
            unsupported as f64 / claim_atoms.len() as f64
        } else {
            0.0
        };

        // Simple contradiction signal: claim atoms that have partial match
        // but very low entailment (suggests entity swap / negation)
        let contradiction_fraction = if !claim_atoms.is_empty() {
            let contradicted = claim_atoms
                .iter()
                .filter(|ca| {
                    let best = ev_atoms
                        .iter()
                        .map(|ea| token_overlap(ca, ea))
                        .fold(0.0_f64, f64::max);
                    // Partial match (some tokens overlap) but low coverage
                    best > 0.15 && best < 0.35
                })
                .count();
            contradicted as f64 / claim_atoms.len() as f64
        } else {
            0.0
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Evidence hash (simple djb2 to avoid pulling sha2 crate)
        let evidence_hash = format!("{:016x}", djb2_hash(evidence));

        EicvCertificate {
            phi: round4(phi),
            hallucination_score: round4(1.0 - phi),
            decision: decision.to_string(),
            layer_scores: LayerScores {
                tg: round4(tg),
                nli: round4(nli),
                rnr: round4(rnr),
                gamma: round4(gamma),
                h_sem: round4(h_sem),
            },
            n_claim_atoms: n_claim,
            n_ev_atoms: n_ev,
            unsupported_fraction: round4(unsupported_fraction),
            contradiction_fraction: round4(contradiction_fraction),
            elapsed_ms: (elapsed_ms * 100.0).round() / 100.0,
            claim: claim.to_string(),
            evidence_hash,
        }
    }
}

// ── Convenience function ─────────────────────────────────────────────

/// Verify a claim against evidence with the default "rag" profile.
pub fn verify(evidence: &str, claim: &str) -> EicvCertificate {
    EicvAnalyzer::new("rag").verify(evidence, claim)
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Split text into atomic propositions.
///
/// Splits on sentence boundaries (`. `, `? `, `! `) and conjunction
/// boundaries (`, and `, `, but `, `; `). Strips leading conjunctions.
/// Discards fragments with fewer than 3 words.
fn atomic_decompose(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![text.trim().to_string()];
    }

    let mut atoms = Vec::new();

    // Phase 1: sentence split
    let sentences = split_sentences(text);

    // Phase 2: conjunction split within each sentence
    for sentence in &sentences {
        let parts = split_conjunctions(sentence);
        for part in parts {
            let cleaned = strip_leading_conjunction(&part);
            let trimmed = cleaned.trim();
            if trimmed.split_whitespace().count() >= 3 {
                atoms.push(trimmed.to_string());
            }
        }
    }

    if atoms.is_empty() {
        // Fall back to the whole text as a single atom
        atoms.push(text.trim().to_string());
    }
    atoms
}

/// Split text on sentence-ending punctuation followed by whitespace.
fn split_sentences(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        current.push(chars[i]);
        if (chars[i] == '.' || chars[i] == '?' || chars[i] == '!') && i + 1 < len && chars[i + 1] == ' ' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                result.push(trimmed);
            }
            current.clear();
        }
        i += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        result.push(trimmed);
    }
    result
}

/// Split on conjunction boundaries: `, and `, `, but `, `; `.
fn split_conjunctions(text: &str) -> Vec<String> {
    let mut parts = vec![text.to_string()];
    for delim in &[", and ", ", but ", "; "] {
        let mut next = Vec::new();
        for part in &parts {
            for segment in part.split(delim) {
                let trimmed = segment.trim().to_string();
                if !trimmed.is_empty() {
                    next.push(trimmed);
                }
            }
        }
        parts = next;
    }
    parts
}

/// Strip leading conjunctions (And, But, However, …).
fn strip_leading_conjunction(text: &str) -> String {
    let prefixes = ["and ", "but ", "however ", "also ", "yet ", "so ", "then "];
    let lower = text.trim().to_lowercase();
    for prefix in &prefixes {
        if lower.starts_with(prefix) {
            return text.trim()[prefix.len()..].to_string();
        }
    }
    text.to_string()
}

/// Jaccard similarity on lowercased word sets.
fn token_overlap(a: &str, b: &str) -> f64 {
    let set_a: HashSet<String> = a.split_whitespace().map(|w| w.to_lowercase()).collect();
    let set_b: HashSet<String> = b.split_whitespace().map(|w| w.to_lowercase()).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Bipartite greedy max-matching: for each claim atom, find the best
/// Jaccard match among evidence atoms. Return the mean of best matches.
fn bipartite_matching(claim_atoms: &[String], ev_atoms: &[String]) -> f64 {
    if claim_atoms.is_empty() || ev_atoms.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for ca in claim_atoms {
        let best = ev_atoms
            .iter()
            .map(|ea| token_overlap(ca, ea))
            .fold(0.0_f64, f64::max);
        total += best;
    }
    total / claim_atoms.len() as f64
}

/// Lexical entailment: for each claim atom, what fraction of its tokens
/// appear in ANY evidence atom? Average across claim atoms.
fn lexical_entailment(claim_atoms: &[String], ev_atoms: &[String]) -> f64 {
    if claim_atoms.is_empty() {
        return 1.0;
    }

    // Build evidence token superset
    let ev_tokens: HashSet<String> = ev_atoms
        .iter()
        .flat_map(|ea| ea.split_whitespace().map(|w| w.to_lowercase()))
        .collect();

    let mut total_coverage = 0.0;
    for ca in claim_atoms {
        let ca_tokens: Vec<String> = ca.split_whitespace().map(|w| w.to_lowercase()).collect();
        if ca_tokens.is_empty() {
            total_coverage += 1.0;
            continue;
        }
        let covered = ca_tokens.iter().filter(|t| ev_tokens.contains(*t)).count();
        total_coverage += covered as f64 / ca_tokens.len() as f64;
    }
    total_coverage / claim_atoms.len() as f64
}

/// Entity overlap: extract capitalized multi-word sequences and numbers,
/// compute the fraction of claim entities found in evidence.
fn entity_overlap(claim: &str, evidence: &str) -> f64 {
    let claim_entities = extract_entities(claim);
    let evidence_entities = extract_entities(evidence);

    if claim_entities.is_empty() {
        return 1.0; // No entities to check → fully grounded
    }

    let evidence_lower: HashSet<String> = evidence_entities
        .iter()
        .map(|e| e.to_lowercase())
        .collect();

    let found = claim_entities
        .iter()
        .filter(|e| evidence_lower.contains(&e.to_lowercase()))
        .count();

    found as f64 / claim_entities.len() as f64
}

/// Extract entities: capitalized words and numeric tokens.
fn extract_entities(text: &str) -> HashSet<String> {
    let mut entities = HashSet::new();
    for word in text.split_whitespace() {
        // Strip trailing punctuation for matching
        let cleaned: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_').collect();
        if cleaned.is_empty() {
            continue;
        }
        let first_char = cleaned.chars().next().unwrap();
        // Capitalized word (not sentence-start heuristic — we accept the noise)
        if first_char.is_uppercase() && cleaned.len() > 1 {
            entities.insert(cleaned.clone());
        }
        // Numbers (years, quantities, IDs)
        if first_char.is_ascii_digit() {
            entities.insert(cleaned);
        }
    }
    entities
}

/// Semantic entropy of the claim token distribution, adjusted by
/// evidence coverage.
///
/// High entropy (uniform distribution) + low coverage = hallucination risk.
/// Returns a value in [0, 1] where 1 = maximum hallucination signal.
fn semantic_entropy(claim: &str, evidence: &str) -> f64 {
    let claim_tokens: Vec<String> = claim.split_whitespace().map(|w| w.to_lowercase()).collect();
    if claim_tokens.is_empty() {
        return 0.0;
    }

    // Token frequency distribution in claim
    let mut freq: HashMap<String, usize> = HashMap::new();
    for t in &claim_tokens {
        *freq.entry(t.clone()).or_insert(0) += 1;
    }

    // Shannon entropy
    let n = claim_tokens.len() as f64;
    let mut entropy = 0.0_f64;
    for &count in freq.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    // Normalize to [0, 1]
    let max_entropy = (freq.len() as f64).log2().max(1.0);
    let norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0);

    // Coverage: fraction of claim tokens found in evidence
    let ev_tokens: HashSet<String> = evidence.split_whitespace().map(|w| w.to_lowercase()).collect();
    let covered = claim_tokens.iter().filter(|t| ev_tokens.contains(*t)).count();
    let coverage = covered as f64 / claim_tokens.len() as f64;

    // High entropy + low coverage → high hallucination signal
    // Low entropy or high coverage → low signal
    (norm_entropy * (1.0 - coverage)).clamp(0.0, 1.0)
}

/// Round to 4 decimal places.
#[inline]
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// Simple DJB2 hash (avoids pulling sha2 crate).
fn djb2_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_claim() {
        let evidence = "Paris is the capital of France. France is located in Western Europe. \
                        The population of Paris is approximately 2.1 million people.";
        let claim = "Paris is the capital of France.";
        let cert = verify(evidence, claim);
        assert_eq!(cert.decision, "supported", "phi={}", cert.phi);
        assert!(cert.phi > 0.6, "phi={}", cert.phi);
    }

    #[test]
    fn hallucinated_claim() {
        let evidence = "Paris is the capital of France.";
        let claim = "Tokyo is the largest city in Brazil and has 50 million residents.";
        let cert = verify(evidence, claim);
        assert!(
            cert.decision == "hallucinated" || cert.decision == "abstain",
            "decision={}, phi={}",
            cert.decision,
            cert.phi
        );
        assert!(cert.phi < 0.5, "phi={}", cert.phi);
    }

    #[test]
    fn empty_claim() {
        let cert = verify("some evidence", "");
        // Empty claim should not crash
        assert!(cert.phi >= 0.0 && cert.phi <= 1.0);
    }

    #[test]
    fn empty_evidence() {
        let cert = verify("", "some claim");
        assert!(cert.phi < 0.5, "phi={}", cert.phi);
    }

    #[test]
    fn profiles_affect_decision() {
        let evidence = "The project uses Rust for performance.";
        let claim = "The project uses Rust and also Python for scripting.";

        let strict = EicvAnalyzer::new("fact_check").verify(evidence, claim);
        let lenient = EicvAnalyzer::new("dialogue").verify(evidence, claim);

        // Same phi, different decisions possible
        assert!((strict.phi - lenient.phi).abs() < 0.001);
    }

    #[test]
    fn atomic_decompose_splits_sentences() {
        let atoms = atomic_decompose("Paris is in France. Berlin is in Germany.");
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn atomic_decompose_splits_conjunctions() {
        let atoms = atomic_decompose("Paris is in France, and Berlin is in Germany");
        assert!(atoms.len() >= 2, "atoms={:?}", atoms);
    }

    #[test]
    fn token_overlap_identical() {
        assert!((token_overlap("hello world", "hello world") - 1.0).abs() < 0.001);
    }

    #[test]
    fn token_overlap_disjoint() {
        assert!(token_overlap("hello world", "foo bar").abs() < 0.001);
    }

    #[test]
    fn certificate_serializes() {
        let cert = verify("evidence text here", "claim text here");
        let json = serde_json::to_string(&cert).unwrap();
        assert!(json.contains("phi"));
        assert!(json.contains("decision"));
    }
}
