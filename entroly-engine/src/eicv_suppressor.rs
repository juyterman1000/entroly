//! EICV Suppressor — hallucination suppression with per-claim audit trail.
//!
//! Takes an LLM response + grounding context, decomposes the response into
//! claims, verifies each claim via [`EicvAnalyzer`], and applies a graduated
//! suppression policy:
//!
//! | Mode       | Behavior                                              |
//! |------------|-------------------------------------------------------|
//! | `audit`    | No output change; certificates only (for dashboards)  |
//! | `annotate` | Append warning footer listing unverified claims        |
//! | `strict`   | supported→PASS, abstain→HEDGE `[unverified]`, hallucinated→SUPPRESS |
//!
//! 100% deterministic, zero LLM calls.

use crate::eicv::{EicvAnalyzer, EicvCertificate};
use serde::Serialize;

// ── Result types ─────────────────────────────────────────────────────

/// Output of the suppression pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct SuppressionResult {
    /// The (possibly rewritten) response text.
    pub rewritten_output: String,
    /// Total claims extracted.
    pub n_claims: usize,
    /// Claims verified as supported.
    pub n_supported: usize,
    /// Claims in the abstain band.
    pub n_abstained: usize,
    /// Claims flagged as hallucinated.
    pub n_hallucinated: usize,
    /// Number of claims suppressed (removed) in strict mode.
    pub suppressed_count: usize,
    /// Number of claims warned/hedged.
    pub warned_count: usize,
    /// n_hallucinated / n_claims (0 if no claims).
    pub hallucination_rate: f64,
    /// Per-claim certificates.
    pub certificates: Vec<EicvCertificate>,
    /// Total pipeline latency in milliseconds.
    pub latency_ms: f64,
}

/// A claim extracted from the response with its character span.
struct ClaimSpan {
    start: usize,
    end: usize,
    text: String,
}

// ── Suppressor ───────────────────────────────────────────────────────

/// EICV suppressor — verifies and optionally rewrites LLM output.
pub struct EicvSuppressor {
    analyzer: EicvAnalyzer,
    mode: String,
}

impl EicvSuppressor {
    /// Create a new suppressor.
    ///
    /// - `profile`: one of `rag`, `qa`, `summarization`, `dialogue`, `fact_check`, `default`
    /// - `mode`: one of `audit`, `annotate`, `strict`
    pub fn new(profile: &str, mode: &str) -> Self {
        Self {
            analyzer: EicvAnalyzer::new(profile),
            mode: mode.to_lowercase(),
        }
    }

    /// Verify an LLM response against the grounding context and apply
    /// the configured suppression policy.
    pub fn suppress(&self, context: &str, output: &str) -> SuppressionResult {
        let start = std::time::Instant::now();

        // Step 1: Extract claims
        let claims = extract_claims(output);

        // Step 2: Verify each claim
        let mut certificates = Vec::with_capacity(claims.len());
        for claim in &claims {
            let cert = self.analyzer.verify(context, &claim.text);
            certificates.push(cert);
        }

        // Tally
        let n_supported = certificates
            .iter()
            .filter(|c| c.decision == "supported")
            .count();
        let n_abstained = certificates
            .iter()
            .filter(|c| c.decision == "abstain")
            .count();
        let n_hallucinated = certificates
            .iter()
            .filter(|c| c.decision == "hallucinated")
            .count();
        let n_claims = certificates.len();
        let hallucination_rate = if n_claims > 0 {
            n_hallucinated as f64 / n_claims as f64
        } else {
            0.0
        };

        // Step 3: Apply policy
        let (rewritten, suppressed_count, warned_count) = match self.mode.as_str() {
            "strict" => self.apply_strict(output, &claims, &certificates),
            "annotate" => self.apply_annotate(output, &certificates, n_hallucinated, n_abstained),
            _ => (output.to_string(), 0, 0), // audit: no changes
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        SuppressionResult {
            rewritten_output: rewritten,
            n_claims,
            n_supported,
            n_abstained,
            n_hallucinated,
            suppressed_count,
            warned_count,
            hallucination_rate: (hallucination_rate * 10000.0).round() / 10000.0,
            certificates,
            latency_ms: (latency_ms * 100.0).round() / 100.0,
        }
    }

    /// Strict mode: graduated 4-action policy.
    fn apply_strict(
        &self,
        output: &str,
        claims: &[ClaimSpan],
        certs: &[EicvCertificate],
    ) -> (String, usize, usize) {
        if claims.is_empty() {
            return (output.to_string(), 0, 0);
        }

        let mut suppressed = 0usize;
        let mut warned = 0usize;

        // Build replacement list: (start, end, replacement_text)
        // Process in reverse order to preserve character offsets
        let mut replacements: Vec<(usize, usize, String)> = Vec::new();

        for (claim, cert) in claims.iter().zip(certs.iter()) {
            match cert.decision.as_str() {
                "supported" => {
                    // PASS — keep as-is
                }
                "abstain" => {
                    // HEDGE — append [unverified]
                    let trimmed = claim.text.trim_end_matches(['.', '!', '?']);
                    let replacement = format!("{} [unverified].", trimmed);
                    replacements.push((claim.start, claim.end, replacement));
                    warned += 1;
                }
                "hallucinated" => {
                    // SUPPRESS — remove the claim together with the horizontal
                    // whitespace that trailed it, so the removal leaves no gap
                    // and no global cleanup pass is needed afterwards. Newlines
                    // are deliberately not consumed: they carry the paragraph
                    // and code-block structure of the response.
                    //
                    // The extended end can reach at most the first following
                    // non-space byte, which is where the next claim begins, so
                    // spans stay disjoint and the reverse-order application
                    // below remains offset-safe.
                    let bytes = output.as_bytes();
                    let mut end = claim.end.min(output.len());
                    while end < output.len() && (bytes[end] == b' ' || bytes[end] == b'\t') {
                        end += 1;
                    }
                    replacements.push((claim.start, end, String::new()));
                    suppressed += 1;
                }
                _ => {}
            }
        }

        // Apply replacements in reverse order to preserve offsets
        let mut result = output.to_string();
        replacements.sort_by_key(|r| std::cmp::Reverse(r.0));
        for (start, end, replacement) in &replacements {
            let actual_end = (*end).min(result.len());
            let actual_start = (*start).min(actual_end);
            result.replace_range(actual_start..actual_end, replacement);
        }

        // Trim only the document edges. There was a `while
        // result.contains("  ") { result.replace("  ", " ") }` here, which
        // collapsed every run of spaces in the response -- including the
        // indentation of any code block, flattening each nesting level to a
        // single space and turning Python output into a syntax error. It also
        // ran unconditionally: `apply_strict` returns early only when there are
        // no claims at all, so a fully grounded response with nothing
        // suppressed was rewritten too. Removals now consume their own trailing
        // whitespace above, which is what the collapse was reaching for.
        let result = result.trim().to_string();

        (result, suppressed, warned)
    }

    /// Annotate mode: append warning footer.
    fn apply_annotate(
        &self,
        output: &str,
        _certs: &[EicvCertificate],
        n_hallucinated: usize,
        n_abstained: usize,
    ) -> (String, usize, usize) {
        let unverified = n_hallucinated + n_abstained;
        if unverified == 0 {
            return (output.to_string(), 0, 0);
        }
        let warning = format!(
            "\n\n[EICV Warning: {} claim{} could not be verified against provided context]",
            unverified,
            if unverified == 1 { "" } else { "s" }
        );
        (format!("{}{}", output, warning), 0, unverified)
    }
}

// ── Convenience function ─────────────────────────────────────────────

/// Suppress hallucinations with default profile ("rag") and mode ("strict").
pub fn suppress(context: &str, output: &str) -> SuppressionResult {
    EicvSuppressor::new("rag", "strict").suppress(context, output)
}

// ── Claim extraction ─────────────────────────────────────────────────

/// Extract claims from text as sentence spans.
///
/// A claim is a sentence with ≥ 4 words. Returns spans with character
/// offsets into the original text for precise replacement.
fn extract_claims(text: &str) -> Vec<ClaimSpan> {
    let mut claims = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        if (chars[i] == '.' || chars[i] == '?' || chars[i] == '!')
            && (i + 1 >= len || chars[i + 1] == ' ' || chars[i + 1] == '\n')
        {
            let end = i + 1;
            let sentence: String = chars[start..end].iter().collect();
            let trimmed = sentence.trim();
            if trimmed.split_whitespace().count() >= 4 {
                // Find actual start/end in byte offsets
                let byte_start = text[..].char_indices().nth(start).map_or(0, |(idx, _)| idx);
                let byte_end = if end >= chars.len() {
                    text.len()
                } else {
                    text.char_indices()
                        .nth(end)
                        .map_or(text.len(), |(idx, _)| idx)
                };
                claims.push(ClaimSpan {
                    start: byte_start,
                    end: byte_end,
                    text: trimmed.to_string(),
                });
            }
            start = i + 1;
            // Skip whitespace after sentence end
            while start < len && (chars[start] == ' ' || chars[start] == '\n') {
                start += 1;
            }
            i = start;
        } else {
            i += 1;
        }
    }

    // Handle trailing text without sentence-ending punctuation
    if start < len {
        let sentence: String = chars[start..].iter().collect();
        let trimmed = sentence.trim();
        if trimmed.split_whitespace().count() >= 4 {
            let byte_start = text.char_indices().nth(start).map_or(0, |(idx, _)| idx);
            claims.push(ClaimSpan {
                start: byte_start,
                end: text.len(),
                text: trimmed.to_string(),
            });
        }
    }

    claims
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_mode_no_change() {
        let context = "Paris is the capital of France.";
        let output = "Paris is the capital of France. Tokyo is in Brazil.";
        let s = EicvSuppressor::new("rag", "audit");
        let result = s.suppress(context, output);
        assert_eq!(result.rewritten_output, output);
        assert!(result.n_claims >= 2);
    }

    #[test]
    fn strict_mode_suppresses() {
        let context = "Paris is the capital of France.";
        let output = "Paris is the capital of France. Tokyo is the largest city in Brazil with 50 million people.";
        let s = EicvSuppressor::new("rag", "strict");
        let result = s.suppress(context, output);
        // The hallucinated claim about Tokyo should be suppressed or hedged
        assert!(
            result.suppressed_count > 0 || result.warned_count > 0,
            "expected suppression: suppressed={}, warned={}",
            result.suppressed_count,
            result.warned_count
        );
    }

    #[test]
    fn annotate_mode_appends_warning() {
        let context = "Rust is a programming language.";
        let output =
            "Rust was created by Mozilla. Python was invented by Guido van Rossum in Antarctica.";
        let s = EicvSuppressor::new("rag", "annotate");
        let result = s.suppress(context, output);
        if result.n_hallucinated + result.n_abstained > 0 {
            assert!(result.rewritten_output.contains("[EICV Warning:"));
        }
    }

    #[test]
    fn extract_claims_basic() {
        let claims = extract_claims(
            "This is a short sentence. This is another longer sentence here. Two words.",
        );
        // "Two words." has only 2 words → excluded
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn convenience_function() {
        let result = suppress(
            "evidence here for grounding",
            "some claim that is long enough.",
        );
        assert!(result.n_claims >= 1);
    }

    #[test]
    fn strict_mode_preserves_code_indentation() {
        // Strict mode used to run `while result.contains("  ") { replace("  ",
        // " ") }` over the whole response, which flattened every nesting level
        // in a code block to one space -- turning Python output into a syntax
        // error. The cleanup now travels with the removal instead.
        //
        // The code block is present in the grounding context so that it is
        // *supported* and therefore not removed. Without that the block is an
        // unsupported claim, strict mode deletes it wholesale, and the test
        // would pass for the wrong reason.
        let code = "fn handler() {\n    let x = parse()?;\n        emit(x);\n}";
        let context = format!(
            "The handler parses the payload and emits the result. {}",
            code
        );
        let output = format!(
            "The handler parses the payload and emits the result.\n\n{}",
            code
        );

        let s = EicvSuppressor::new("rag", "strict");
        let result = s.suppress(&context, &output);

        assert!(
            result.rewritten_output.contains("\n    let x = parse()?;"),
            "one level of indentation was lost: {:?}",
            result.rewritten_output
        );
        assert!(
            result.rewritten_output.contains("\n        emit(x);"),
            "two levels of indentation were lost: {:?}",
            result.rewritten_output
        );
    }

    #[test]
    fn strict_mode_leaves_a_fully_grounded_response_alone() {
        // `apply_strict` returns early only when there are no claims at all,
        // so with nothing suppressed the response must come back unchanged.
        // Under the old global collapse it did not.
        let context = "Paris is the capital of France and sits on the Seine.";
        let output = "Paris  is the capital of France and sits on the Seine.";
        let s = EicvSuppressor::new("rag", "strict");
        let result = s.suppress(context, output);

        assert_eq!(result.suppressed_count, 0);
        assert_eq!(
            result.rewritten_output, output,
            "a response with nothing suppressed was rewritten anyway"
        );
    }

    #[test]
    fn suppressing_a_claim_consumes_its_own_trailing_space() {
        // The property the global collapse was reaching for, obtained locally.
        let context = "Paris is the capital of France.";
        let output = "Paris is the capital of France. Tokyo is the largest city in Brazil with fifty million people. Lyon sits south of Paris in France.";
        let s = EicvSuppressor::new("rag", "strict");
        let result = s.suppress(context, output);

        if result.suppressed_count > 0 {
            assert!(
                !result.rewritten_output.contains("  "),
                "removal left a gap: {:?}",
                result.rewritten_output
            );
        }
    }

    #[test]
    fn result_serializes() {
        let result = suppress("the sky is blue", "the sky is blue and beautiful today.");
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("rewritten_output"));
        assert!(json.contains("certificates"));
    }
}
