//! Context Utilization Scoring
//!
//! Closes the optimization loop: after each LLM response, measures how
//! much of the injected context the LLM actually used.
//!
//! Every context-injection tool (RAG, copilot, etc.) is open-loop —
//! stuff context in, hope it helps. This makes Entroly closed-loop.
//!
//! Scoring per fragment fᵢ against response r:
//!   1. Trigram Jaccard: J₃(fᵢ, r) = |trigrams(fᵢ) ∩ trigrams(r)| / |trigrams(fᵢ)|
//!   2. Identifier overlap: I(fᵢ, r) = |idents(fᵢ) ∩ idents(r)| / |idents(fᵢ)|
//!   3. Combined: U(fᵢ) = 0.4 × J₃ + 0.6 × I
//!
//! Identifier overlap is weighted higher (0.6) because it's a stronger
//! signal — if the LLM references a function name from your fragment,
//! that's real utilization, not accidental n-gram overlap.
//!
//! The utilization score feeds back into weight learning: fragments
//! that consistently get ignored get deprioritized in future selection.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use crate::fragment::ContextFragment;
use crate::depgraph::extract_identifiers;

/// Utilization score for a single injected fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentUtilization {
    pub fragment_id: String,
    pub source: String,
    /// Trigram Jaccard overlap [0, 1].
    pub trigram_overlap: f64,
    /// Identifier overlap [0, 1].
    pub identifier_overlap: f64,
    /// Combined score: 0.4 × trigram + 0.6 × identifier.
    pub combined_score: f64,
    /// Whether this fragment was "used" (combined > 0.1).
    pub was_used: bool,
}

/// Session-level utilization report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationReport {
    pub fragments_injected: usize,
    pub fragments_used: usize,
    pub fragments_ignored: usize,
    /// Mean combined score across all fragments.
    pub session_utilization: f64,
    /// Per-fragment breakdown.
    pub per_fragment: Vec<FragmentUtilization>,
    pub summary: String,
}

/// Extract word trigrams from text as a HashSet.
fn trigrams(text: &str) -> HashSet<Vec<String>> {
    let words: Vec<String> = text.split_whitespace()
        .map(|w| w.to_lowercase()
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
            .to_string())
        .filter(|w| !w.is_empty())
        .collect();

    if words.len() < 3 {
        return HashSet::new();
    }

    words.windows(3)
        .map(|w| w.to_vec())
        .collect()
}

/// Extract identifiers from text as a HashSet.
fn identifier_set(text: &str) -> HashSet<String> {
    extract_identifiers(text).into_iter().collect()
}

/// Score how much of each injected fragment the LLM actually used
/// in its response.
///
/// Call this after receiving the LLM response, passing in the
/// fragments that were injected into the prompt context.
pub fn score_utilization(
    fragments: &[&ContextFragment],
    response: &str,
) -> UtilizationReport {
    if fragments.is_empty() {
        return UtilizationReport {
            fragments_injected: 0,
            fragments_used: 0,
            fragments_ignored: 0,
            session_utilization: 0.0,
            per_fragment: vec![],
            summary: "No fragments were injected.".into(),
        };
    }

    let response_trigrams = trigrams(response);
    let response_idents = identifier_set(response);

    let mut per_fragment: Vec<FragmentUtilization> = Vec::with_capacity(fragments.len());

    for frag in fragments {
        let frag_trigrams = trigrams(&frag.content);
        let frag_idents = identifier_set(&frag.content);

        // Trigram Jaccard: |intersection| / |fragment_trigrams|
        let trigram_overlap = if frag_trigrams.is_empty() {
            0.0
        } else {
            let overlap = frag_trigrams.intersection(&response_trigrams).count();
            overlap as f64 / frag_trigrams.len() as f64
        };

        // Identifier overlap: |intersection| / |fragment_idents|
        let identifier_overlap = if frag_idents.is_empty() {
            0.0
        } else {
            let overlap = frag_idents.intersection(&response_idents).count();
            overlap as f64 / frag_idents.len() as f64
        };

        let combined = 0.4 * trigram_overlap + 0.6 * identifier_overlap;
        let was_used = combined > 0.1;

        per_fragment.push(FragmentUtilization {
            fragment_id: frag.fragment_id.clone(),
            source: frag.source.clone(),
            trigram_overlap: (trigram_overlap * 1000.0).round() / 1000.0,
            identifier_overlap: (identifier_overlap * 1000.0).round() / 1000.0,
            combined_score: (combined * 1000.0).round() / 1000.0,
            was_used,
        });
    }

    let fragments_used = per_fragment.iter().filter(|f| f.was_used).count();
    let fragments_ignored = per_fragment.len() - fragments_used;
    let session_utilization = if per_fragment.is_empty() {
        0.0
    } else {
        let total: f64 = per_fragment.iter().map(|f| f.combined_score).sum();
        (total / per_fragment.len() as f64 * 1000.0).round() / 1000.0
    };

    let summary = format!(
        "Entroly injected {} fragments, LLM used {}, ignored {}, context efficiency: {:.0}%.",
        per_fragment.len(),
        fragments_used,
        fragments_ignored,
        session_utilization * 100.0,
    );

    UtilizationReport {
        fragments_injected: per_fragment.len(),
        fragments_used,
        fragments_ignored,
        session_utilization,
        per_fragment,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::ContextFragment;

    fn make_frag(id: &str, content: &str) -> ContextFragment {
        ContextFragment::new(id.into(), content.into(), 50, "test.py".into())
    }

    #[test]
    fn test_full_utilization() {
        // Response directly uses content from the fragment
        let frag = make_frag("a", "def calculate_tax(income, rate):\n    return income * rate");
        let response = "To calculate_tax with the given income and rate, \
                        you can call calculate_tax(income, rate) which returns income * rate.";

        let report = score_utilization(&[&frag], response);

        assert_eq!(report.fragments_injected, 1);
        assert!(report.per_fragment[0].identifier_overlap > 0.3,
            "Identifier overlap should be high when response uses same identifiers: {:.3}",
            report.per_fragment[0].identifier_overlap);
        assert!(report.per_fragment[0].was_used,
            "Fragment should be marked as used");
    }

    #[test]
    fn test_zero_utilization() {
        // Response is completely unrelated to the fragment
        let frag = make_frag("a", "def calculate_tax(income, rate):\n    return income * rate");
        let response = "The weather in San Francisco is sunny today with clear skies.";

        let report = score_utilization(&[&frag], response);

        assert!(!report.per_fragment[0].was_used,
            "Fragment should not be marked as used when response is unrelated");
        assert_eq!(report.fragments_ignored, 1);
    }

    #[test]
    fn test_partial_utilization() {
        // Two fragments: one used, one not
        let frag_used = make_frag("used", "def connect_database(host, port):\n    return Connection(host, port)");
        let frag_not = make_frag("not_used", "def send_email(recipient, body):\n    smtp.send(recipient, body)");
        let response = "Use connect_database with the host and port parameters \
                        to establish a Connection to the database.";

        let report = score_utilization(&[&frag_used, &frag_not], response);

        assert_eq!(report.fragments_injected, 2);
        assert!(report.per_fragment[0].combined_score > report.per_fragment[1].combined_score,
            "Used fragment should score higher than unused one");
    }

    #[test]
    fn test_empty_fragments() {
        let report = score_utilization(&[], "some response");
        assert_eq!(report.fragments_injected, 0);
        assert_eq!(report.session_utilization, 0.0);
    }

    #[test]
    fn test_identifier_overlap_weighted_higher() {
        // Fragment with identifiers that appear in response but different phrasing
        let frag = make_frag("a", "fn process_payment amount currency exchange_rate");
        let response = "The process_payment function handles amount in currency with exchange_rate applied.";

        let report = score_utilization(&[&frag], response);

        // Identifier overlap should be higher than trigram overlap
        // because the sentence structure differs but identifier names are reused
        assert!(report.per_fragment[0].identifier_overlap > 0.0,
            "Should detect identifier reuse even with different sentence structure");
    }
}
