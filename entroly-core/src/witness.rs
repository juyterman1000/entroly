//! WITNESS deterministic core.
//!
//! Python owns API/proxy integration and optional LLM-backed NLI. The production
//! local proof engine lives here so the default path is fast, deterministic, and
//! wheel-packaged with the rest of Entroly's Rust core.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Instant;

static WORD_RE: OnceLock<Regex> = OnceLock::new();
static GATE_WORD_RE: OnceLock<Regex> = OnceLock::new();
static NUMBER_RE: OnceLock<Regex> = OnceLock::new();
static ENTITY_RE: OnceLock<Regex> = OnceLock::new();
static CAPS_RE: OnceLock<Regex> = OnceLock::new();
static QUOTED_RE: OnceLock<Regex> = OnceLock::new();

fn word_re() -> &'static Regex {
    WORD_RE.get_or_init(|| Regex::new(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b").expect("valid word regex"))
}

/// Tokenizer for the QA binding/residual gates ONLY. Mirrors Python's
/// `_WORD_RE = [A-Za-z][A-Za-z0-9_'-]+` *exactly* (≥2 chars, apostrophe
/// kept) so that `bind_q` and the residual-payload set are computed over
/// the same token alphabet in both runtimes. Without this the gates
/// would use the module's ≥4-char `word_re()`, mis-calibrating the
/// data-derived 0.34 threshold (which was fit on the Python tokenizer)
/// and breaking Python↔Rust parity. The surrounding feature code keeps
/// using `word_re()` — only the gate is unified here.
fn gate_word_re() -> &'static Regex {
    GATE_WORD_RE
        .get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z0-9_'-]+").expect("valid gate word regex"))
}

fn number_re() -> &'static Regex {
    NUMBER_RE.get_or_init(|| Regex::new(r"\$?\d[\d,]*(?:\.\d+)?%?").expect("valid number regex"))
}

fn entity_re() -> &'static Regex {
    ENTITY_RE.get_or_init(|| {
        Regex::new(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,4}\b")
            .expect("valid entity regex")
    })
}

fn caps_re() -> &'static Regex {
    CAPS_RE.get_or_init(|| Regex::new(r"\b[A-Z][A-Z0-9]{1,}\b").expect("valid caps regex"))
}

fn quoted_re() -> &'static Regex {
    QUOTED_RE.get_or_init(|| Regex::new(r#""([^"]{2,90})""#).expect("valid quote regex"))
}

fn stopwords() -> &'static HashSet<&'static str> {
    static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOPWORDS.get_or_init(|| {
        [
            "about", "after", "again", "against", "also", "although", "among", "another", "before",
            "being", "between", "both", "could", "does", "doing", "done", "each", "either",
            "every", "first", "from", "have", "into", "just", "last", "like", "made", "make",
            "many", "more", "most", "much", "only", "other", "over", "should", "some", "such",
            "than", "that", "their", "them", "then", "there", "these", "they", "this", "those",
            "very", "were", "what", "when", "where", "which", "while", "with", "would", "yes",
            "sure", "okay", "actually", "believe", "call", "use", "run", "read", "write", "return",
            "update", "create", "delete",
        ]
        .into_iter()
        .collect()
    })
}

fn question_starters() -> &'static HashSet<&'static str> {
    static STARTERS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STARTERS.get_or_init(|| {
        [
            "who", "what", "when", "where", "why", "how", "which", "whose", "is", "are", "was",
            "were", "do", "does", "did", "can", "could", "should", "would",
        ]
        .into_iter()
        .collect()
    })
}

fn number_words() -> &'static [(&'static str, i32)] {
    &[
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
        ("twelve", 12),
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub id: usize,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceWindow {
    pub text: String,
    pub score: f64,
    pub adequacy: f64,
    pub entity_coverage: f64,
    pub number_coverage: f64,
    pub token_coverage: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofStep {
    pub operator: String,
    pub evidence: String,
    pub strength: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Certificate {
    pub claim_id: usize,
    pub claim_text: String,
    pub label: String,
    pub support_strength: f64,
    pub contradiction_strength: f64,
    pub evidence_adequacy: f64,
    pub risk: f64,
    pub proof_path: Vec<ProofStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WitnessResult {
    pub output: String,
    pub certificates: Vec<Certificate>,
    pub summary_score: f64,
    pub n_grounded: usize,
    pub n_unsupported: usize,
    pub n_contradicted: usize,
    pub n_unknown: usize,
    pub latency_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WitnessRewrite {
    pub output: String,
    pub changed: bool,
    pub mode: String,
    pub profile: String,
    pub flagged_count: usize,
    pub suppressed_count: usize,
    pub warned_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WitnessPayload {
    pub witness: WitnessResult,
    pub policy: WitnessRewrite,
    pub output: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuousFeatures {
    pub entity_precision: f64,
    pub number_consistency: f64,
    pub idf_lex_overlap: f64,
    pub quote_support: f64,
    pub forward_entail: f64,
    pub reverse_entail: f64,
    pub negation_polarity: f64,
    pub adequacy: f64,
    pub qa_alignment: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct WitnessConfig {
    pub support_threshold: f64,
    pub contradiction_threshold: f64,
    pub adequacy_threshold: f64,
}

impl Default for WitnessConfig {
    fn default() -> Self {
        Self {
            support_threshold: 0.62,
            contradiction_threshold: 0.70,
            adequacy_threshold: 0.72,
        }
    }
}

pub fn analyze(context: &str, output: &str, config: WitnessConfig) -> WitnessResult {
    let t0 = Instant::now();
    let mut certificates = Vec::new();
    for claim in extract_claims(output) {
        certificates.push(certify_claim(&claim, context, config));
    }

    let mut n_grounded = 0;
    let mut n_unsupported = 0;
    let mut n_contradicted = 0;
    let mut n_unknown = 0;
    for cert in &certificates {
        match cert.label.as_str() {
            "grounded" => n_grounded += 1,
            "unsupported" => n_unsupported += 1,
            "contradicted" => n_contradicted += 1,
            "unknown" => n_unknown += 1,
            _ => {}
        }
    }

    let summary_score = if certificates.is_empty() {
        1.0
    } else {
        let risk_sum: f64 = certificates.iter().map(|c| c.risk).sum();
        (1.0 - risk_sum / certificates.len() as f64).clamp(0.0, 1.0)
    };

    WitnessResult {
        output: output.to_string(),
        certificates,
        summary_score,
        n_grounded,
        n_unsupported,
        n_contradicted,
        n_unknown,
        latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}

pub fn analyze_with_policy(
    context: &str,
    output: &str,
    mode: &str,
    profile: &str,
    config: WitnessConfig,
) -> WitnessPayload {
    let witness = analyze(context, output, config);
    let policy = apply_policy(output, &witness, mode, profile, 6);
    let rewritten_output = policy.output.clone();
    WitnessPayload {
        witness,
        policy,
        output: rewritten_output,
    }
}

pub fn extract_claims(output: &str) -> Vec<Claim> {
    candidate_claim_segments(output)
        .into_iter()
        .filter_map(|(start, segment)| {
            let text = clean_claim_segment(&segment);
            if !is_claim_like(&text) {
                return None;
            }
            let kind = classify_claim(&text);
            Some((start, text, kind))
        })
        .enumerate()
        .map(|(id, (start, text, kind))| Claim {
            id,
            end: start + text.len(),
            start,
            text,
            kind,
        })
        .collect()
}

pub fn select_evidence_windows(
    context: &str,
    claim: &str,
    top_k: usize,
) -> (Vec<EvidenceWindow>, f64) {
    let claim_words = content_words(claim);
    let entities = extract_entities(claim);
    let numbers = extract_numbers(claim);
    let mut windows = Vec::new();

    for text in sentence_windows(context, 900) {
        let lower = text.to_lowercase();
        let words = content_words(&text);
        let entity_cov = coverage_fraction(&entities, &lower);
        let number_cov = coverage_fraction(&numbers, &lower);
        let token_cov = intersection_ratio(&claim_words, &words);
        let adequacy = weighted_adequacy(
            entity_cov,
            number_cov,
            token_cov,
            !entities.is_empty(),
            !numbers.is_empty(),
        );
        let score = 3.0 * entity_cov + 2.0 * number_cov + token_cov + adequacy;
        windows.push(EvidenceWindow {
            text,
            score,
            adequacy,
            entity_coverage: entity_cov,
            number_coverage: number_cov,
            token_coverage: token_cov,
        });
    }

    windows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.adequacy
                    .partial_cmp(&a.adequacy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| b.text.len().cmp(&a.text.len()))
    });
    let top: Vec<EvidenceWindow> = windows.into_iter().take(top_k).collect();
    let combined = top
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let combined_lower = combined.to_lowercase();
    let mut combined_words = HashSet::new();
    for window in &top {
        combined_words.extend(content_words(&window.text));
    }
    let combined_adequacy = weighted_adequacy(
        coverage_fraction(&entities, &combined_lower),
        coverage_fraction(&numbers, &combined_lower),
        intersection_ratio(&claim_words, &combined_words),
        !entities.is_empty(),
        !numbers.is_empty(),
    );
    let best = top
        .iter()
        .map(|w| w.adequacy)
        .fold(combined_adequacy, f64::max);
    (top, best)
}

pub fn apply_policy(
    output: &str,
    result: &WitnessResult,
    mode: &str,
    profile: &str,
    max_items: usize,
) -> WitnessRewrite {
    let normalized_mode = mode.trim().to_lowercase();
    let normalized_profile = normalize_profile(profile, output);
    let flagged: Vec<&Certificate> = result
        .certificates
        .iter()
        .filter(|cert| cert.label != "grounded")
        .collect();

    if normalized_mode == "off" || normalized_mode == "audit" {
        return WitnessRewrite {
            output: output.to_string(),
            changed: false,
            mode: normalized_mode,
            profile: normalized_profile,
            flagged_count: flagged.len(),
            suppressed_count: 0,
            warned_count: 0,
        };
    }

    if flagged.is_empty() {
        return WitnessRewrite {
            output: output.to_string(),
            changed: false,
            mode: normalized_mode,
            profile: normalized_profile,
            flagged_count: 0,
            suppressed_count: 0,
            warned_count: 0,
        };
    }

    if normalized_mode == "annotate" {
        let mut rewritten = output.trim_end().to_string();
        rewritten.push_str("\n\n[Entroly WITNESS] Verification warnings:");
        for cert in flagged.iter().take(max_items) {
            rewritten.push_str(&format!("\n- {}: {}", cert.label, cert.claim_text));
        }
        if flagged.len() > max_items {
            rewritten.push_str(&format!("\n- ... {} more", flagged.len() - max_items));
        }
        return WitnessRewrite {
            output: rewritten,
            changed: true,
            mode: normalized_mode,
            profile: normalized_profile,
            flagged_count: flagged.len(),
            suppressed_count: 0,
            warned_count: flagged.len(),
        };
    }

    if normalized_mode == "strict" {
        let mut rewritten = output.to_string();
        let suppressed: Vec<&Certificate> = flagged
            .iter()
            .copied()
            .filter(|cert| should_suppress(cert, &normalized_profile))
            .collect();
        let warned: Vec<&Certificate> = flagged
            .iter()
            .copied()
            .filter(|cert| !should_suppress(cert, &normalized_profile))
            .collect();
        let suppressed_count = suppressed.len();
        let warned_count = warned.len();
        let mut sorted = suppressed;
        sorted.sort_by_key(|b| std::cmp::Reverse(b.claim_text.len()));
        for cert in sorted {
            rewritten = remove_claim_text(&rewritten, &cert.claim_text);
        }
        rewritten = cleanup_rewritten_output(&rewritten);
        if suppressed_count > 0 && rewritten.trim().is_empty() {
            rewritten =
                "Entroly WITNESS suppressed this answer because its factual claims lacked support."
                    .to_string();
        } else if suppressed_count > 0 && rewritten != output {
            rewritten.push_str(&format!(
                "\n\n[Entroly WITNESS suppressed {} unsupported factual claim(s).]",
                suppressed_count
            ));
        }
        if warned_count > 0 {
            rewritten.push_str(
                "\n\n[Entroly WITNESS warning: some factual claims could not be fully verified:",
            );
            for cert in warned.iter().take(max_items) {
                rewritten.push_str(&format!("\n- {}: {}", cert.label, cert.claim_text));
            }
            if warned_count > max_items {
                rewritten.push_str(&format!("\n- ... {} more", warned_count - max_items));
            }
            rewritten.push(']');
        }
        return WitnessRewrite {
            changed: rewritten != output,
            output: rewritten,
            mode: normalized_mode,
            profile: normalized_profile,
            flagged_count: flagged.len(),
            suppressed_count,
            warned_count,
        };
    }

    WitnessRewrite {
        output: output.to_string(),
        changed: false,
        mode: normalized_mode,
        profile: normalized_profile,
        flagged_count: flagged.len(),
        suppressed_count: 0,
        warned_count: 0,
    }
}

fn certify_claim(claim: &Claim, context: &str, config: WitnessConfig) -> Certificate {
    let (windows, adequacy) = select_evidence_windows(context, &claim.text, 4);
    let mut support_steps = Vec::new();
    let mut contradiction_steps = Vec::new();
    let question = question_from_context(context);

    let quote = quote_support(context, &claim.text);
    if quote > 0.0 && question.is_empty() {
        support_steps.push(ProofStep {
            operator: "quote_support".to_string(),
            evidence: format!("quote_strength={:.2}", quote),
            strength: quote,
        });
    }

    let slot = slot_support(context, &claim.text, adequacy);
    if slot > 0.0 && question.is_empty() {
        support_steps.push(ProofStep {
            operator: "slot_support".to_string(),
            evidence: format!("adequacy={:.2}", adequacy),
            strength: slot,
        });
    }

    let local = local_pav(context, &claim.text, &windows, adequacy);
    if local.0 == "entailment" {
        support_steps.push(ProofStep {
            operator: "local_entailment".to_string(),
            evidence: local.2.chars().take(180).collect(),
            strength: local.1,
        });
    } else if local.0 == "contradiction" {
        contradiction_steps.push(ProofStep {
            operator: "local_contradiction".to_string(),
            evidence: local.2.chars().take(180).collect(),
            strength: local.1,
        });
    }

    let support = support_steps.iter().map(|s| s.strength).fold(0.0, f64::max);
    let contradiction = contradiction_steps
        .iter()
        .map(|s| s.strength)
        .fold(0.0, f64::max);

    let (label, risk, proof_path) = if contradiction >= config.contradiction_threshold {
        let mut proof = contradiction_steps.clone();
        proof.extend(support_steps.iter().take(1).cloned());
        ("contradicted".to_string(), 0.92, proof)
    } else if support >= config.support_threshold {
        (
            "grounded".to_string(),
            (0.35 * (1.0 - support)).max(0.05),
            support_steps,
        )
    } else if adequacy >= config.adequacy_threshold {
        let mut proof = support_steps;
        proof.push(ProofStep {
            operator: "evidence_adequacy".to_string(),
            evidence: top_window_text(&windows),
            strength: adequacy,
        });
        ("unsupported".to_string(), 0.80, proof)
    } else {
        (
            "unknown".to_string(),
            0.55,
            vec![ProofStep {
                operator: "weak_evidence".to_string(),
                evidence: top_window_text(&windows),
                strength: adequacy,
            }],
        )
    };

    Certificate {
        claim_id: claim.id,
        claim_text: claim.text.clone(),
        label,
        support_strength: round4(support),
        contradiction_strength: round4(contradiction),
        evidence_adequacy: round4(adequacy),
        risk: round4(risk),
        proof_path,
    }
}

fn local_pav(
    context: &str,
    claim: &str,
    windows: &[EvidenceWindow],
    adequacy: f64,
) -> (String, f64, String) {
    let evidence = windows
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if let Some((label, confidence)) = comparative_verdict(context, claim) {
        return (label, confidence, evidence);
    }

    if !question_from_context(context).is_empty() {
        return ("neutral".to_string(), 0.35, evidence);
    }

    let quote = quote_support(context, claim);
    if quote >= 0.82 {
        return ("entailment".to_string(), quote.min(0.85), evidence);
    }

    let claim_nums: Vec<String> = extract_numbers(claim)
        .into_iter()
        .filter(|n| !n.is_empty())
        .collect();
    let evidence_nums: Vec<String> = extract_numbers(&evidence)
        .into_iter()
        .filter(|n| !n.is_empty())
        .collect();
    if !claim_nums.is_empty() && adequacy >= 0.72 {
        if claim_nums.iter().all(|n| evidence_nums.contains(n)) {
            return ("entailment".to_string(), 0.68, evidence);
        }
        if !evidence_nums.is_empty() {
            return ("contradiction".to_string(), 0.66, evidence);
        }
    }

    if adequacy >= 0.88 && slot_support(context, claim, adequacy) >= 0.70 {
        return ("entailment".to_string(), 0.66, evidence);
    }
    ("neutral".to_string(), 0.35, evidence)
}

fn quote_support(context: &str, claim: &str) -> f64 {
    let norm_context = normalize_text(context);
    let norm_claim = normalize_text(claim);
    if norm_claim.is_empty() {
        return 0.0;
    }
    if norm_context.contains(&norm_claim) {
        if norm_claim.len() > 80 {
            0.95
        } else {
            0.82
        }
    } else {
        0.0
    }
}

fn slot_support(context: &str, claim: &str, adequacy: f64) -> f64 {
    let words = content_words(claim);
    let entities = extract_entities(claim);
    let numbers = extract_numbers(claim);
    let lower = context.to_lowercase();
    let token_cov = intersection_ratio(&words, &content_words(context));
    let entity_cov = coverage_fraction(&entities, &lower);
    let number_cov = coverage_fraction(&numbers, &lower);
    if !numbers.is_empty() && number_cov < 1.0 {
        return 0.0;
    }
    if !entities.is_empty() && entity_cov < 0.65 {
        return 0.0;
    }
    (0.25 + 0.35 * adequacy + 0.20 * token_cov).min(0.74)
}

fn comparative_verdict(context: &str, claim: &str) -> Option<(String, f64)> {
    let question = question_from_context(context);
    let candidates = candidate_entities(&question);
    if candidates.len() != 2 {
        return None;
    }
    let clean_context = match context.to_lowercase().find("question:") {
        Some(idx) => &context[..idx],
        None => context,
    };
    let values: Vec<Vec<i32>> = candidates
        .iter()
        .map(|candidate| numbers_near(clean_context, candidate, 260))
        .collect();
    if values.iter().any(|v| v.is_empty()) {
        return None;
    }
    let q_lower = question.to_lowercase();
    let expected = if ["first", "earlier", "older", "started", "founded"]
        .iter()
        .any(|term| q_lower.contains(term))
    {
        let i = if values[0].iter().min()? <= values[1].iter().min()? {
            0
        } else {
            1
        };
        Some(candidates[i].clone())
    } else if ["more", "most", "larger", "higher"]
        .iter()
        .any(|term| q_lower.contains(term))
    {
        let i = if values[0].iter().max()? >= values[1].iter().max()? {
            0
        } else {
            1
        };
        Some(candidates[i].clone())
    } else {
        None
    }?;

    let claim_lower = claim.to_lowercase();
    if claim_lower.contains(&expected.to_lowercase()) {
        return Some(("entailment".to_string(), 0.76));
    }
    if candidates
        .iter()
        .any(|c| c != &expected && claim_lower.contains(&c.to_lowercase()))
    {
        return Some(("contradiction".to_string(), 0.78));
    }
    None
}

fn candidate_claim_segments(text: &str) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut seen = HashSet::new();

    for (start, sentence) in split_sentences(text) {
        push_claim_candidate(start, &sentence, &mut out, &mut seen);
        for (part_start, part) in split_compound_claims(start, &sentence) {
            push_claim_candidate(part_start, &part, &mut out, &mut seen);
        }
    }

    let mut line_start = 0usize;
    for raw_line in text.split_inclusive('\n') {
        let line = raw_line.trim_end_matches(['\r', '\n']);
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let leading = line.find(trimmed).unwrap_or(0);
            let cleaned = clean_claim_segment(trimmed);
            let start = line_start + leading;
            if is_list_or_table_row(trimmed) || looks_like_code_claim(&cleaned) {
                push_claim_candidate(start, &cleaned, &mut out, &mut seen);
            }
            for (part_start, part) in split_compound_claims(start, &cleaned) {
                push_claim_candidate(part_start, &part, &mut out, &mut seen);
            }
        }
        line_start += raw_line.len();
    }

    out.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.len().cmp(&a.1.len())));
    out
}

fn push_claim_candidate(
    start: usize,
    text: &str,
    out: &mut Vec<(usize, String)>,
    seen: &mut HashSet<String>,
) {
    let cleaned = clean_claim_segment(text);
    if cleaned.is_empty() {
        return;
    }
    let key = normalize_text(&cleaned);
    if key.is_empty() || !seen.insert(key) {
        return;
    }
    out.push((start, cleaned));
}

fn clean_claim_segment(segment: &str) -> String {
    let mut s = segment.trim().trim_matches('`').trim().to_string();
    for marker in ["- ", "* ", "+ "] {
        if s.starts_with(marker) {
            s = s[marker.len()..].trim().to_string();
        }
    }
    let bytes = s.as_bytes();
    if bytes.len() > 2 && bytes[0].is_ascii_digit() && bytes[1] == b'.' {
        s = s[2..].trim().to_string();
    }
    if s.starts_with('|') && s.ends_with('|') {
        let cells: Vec<&str> = s
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim())
            .filter(|c| {
                !c.is_empty()
                    && !c
                        .chars()
                        .all(|ch| ch == '-' || ch == ':' || ch.is_whitespace())
            })
            .collect();
        if cells.len() >= 2 {
            s = cells.join(" | ");
        }
    }
    s.trim_matches(|c: char| c == '-' || c == '*' || c == ':' || c.is_whitespace())
        .trim()
        .to_string()
}

fn is_list_or_table_row(segment: &str) -> bool {
    let s = segment.trim();
    s.starts_with("- ")
        || s.starts_with("* ")
        || s.starts_with("+ ")
        || s.starts_with("|")
        || s.as_bytes().first().is_some_and(|b| b.is_ascii_digit())
            && s.as_bytes().get(1) == Some(&b'.')
}

fn split_compound_claims(start: usize, segment: &str) -> Vec<(usize, String)> {
    let mut parts = Vec::new();
    for delimiter in ["; ", " and "] {
        let lower = segment.to_lowercase();
        let Some(idx) = lower.find(delimiter) else {
            continue;
        };
        let left = segment[..idx].trim();
        let right = segment[idx + delimiter.len()..].trim();
        if left.len() < 8 || right.len() < 8 {
            continue;
        }
        if has_fact_signal(left) && has_fact_signal(right) {
            parts.push((start, left.to_string()));
            parts.push((start + idx + delimiter.len(), right.to_string()));
        }
    }
    parts
}

fn is_claim_like(sentence: &str) -> bool {
    let s = sentence.trim();
    if s.len() < 2 || is_question_like(s) {
        return false;
    }
    if s.starts_with("```") || s.eq_ignore_ascii_case("json") || s.eq_ignore_ascii_case("python") {
        return false;
    }
    if looks_like_code_claim(s) {
        return true;
    }
    has_fact_signal(s) || contains_claim_verb(s)
}

fn has_fact_signal(text: &str) -> bool {
    !extract_entities(text).is_empty()
        || !extract_numbers(text).is_empty()
        || looks_like_code_claim(text)
}

fn contains_claim_verb(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " had ",
        " uses ",
        " returns ",
        " contains ",
        " supports ",
        " requires ",
        " equals ",
        " means ",
        " runs ",
        " fails ",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn looks_like_code_claim(text: &str) -> bool {
    let lower = text.to_lowercase();
    text.contains("()")
        || text.contains('_')
        || text.contains("::")
        || lower.contains("function")
        || lower.contains("method")
        || lower.contains("class ")
        || lower.contains("module ")
        || lower.contains("import ")
        || lower.contains(".py")
        || lower.contains(".rs")
        || lower.contains(".js")
        || lower.contains(".ts")
}

fn classify_claim(text: &str) -> String {
    if looks_like_code_claim(text) {
        "code_ref".to_string()
    } else if !extract_numbers(text).is_empty() {
        "quantity".to_string()
    } else {
        "proposition".to_string()
    }
}

fn normalize_profile(profile: &str, output: &str) -> String {
    let normalized = profile.trim().to_lowercase().replace('-', "_");
    match normalized.as_str() {
        "code" | "rag" | "qa" | "benchmark_qa" | "summary" | "summarization" | "chat"
        | "dialogue" => {
            if normalized == "summarization" {
                "summary".to_string()
            } else {
                normalized
            }
        }
        _ => infer_profile(output),
    }
}

fn infer_profile(output: &str) -> String {
    let lower = output.to_lowercase();
    if lower.contains("```")
        || lower.contains(".py")
        || lower.contains(".rs")
        || lower.contains("function")
        || lower.contains("class ")
    {
        "code".to_string()
    } else if output.len() > 900
        || output
            .lines()
            .filter(|line| line.trim_start().starts_with("- "))
            .count()
            >= 3
    {
        "summary".to_string()
    } else {
        "rag".to_string()
    }
}

fn should_suppress(cert: &Certificate, profile: &str) -> bool {
    match cert.label.as_str() {
        "grounded" => false,
        "contradicted" => true,
        "unsupported" => !matches!(profile, "chat" | "dialogue"),
        "unknown" => matches!(profile, "code" | "rag" | "qa" | "benchmark_qa") || cert.risk >= 0.75,
        _ => false,
    }
}

fn split_sentences(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut last_boundary = 0usize;
    let mut prev_was_term = false;
    for (idx, ch) in text.char_indices() {
        if ch == '\n' || (prev_was_term && ch.is_whitespace()) {
            let end = if ch == '\n' { idx } else { last_boundary };
            push_sentence(text, start, end, &mut out);
            start = idx + ch.len_utf8();
            prev_was_term = false;
            continue;
        }
        if matches!(ch, '.' | '!' | '?') {
            last_boundary = idx + ch.len_utf8();
            prev_was_term = true;
        } else if !ch.is_whitespace() {
            prev_was_term = false;
        }
    }
    push_sentence(text, start, text.len(), &mut out);
    out
}

fn push_sentence(text: &str, start: usize, end: usize, out: &mut Vec<(usize, String)>) {
    if end <= start || start >= text.len() {
        return;
    }
    let raw = &text[start..end.min(text.len())];
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }
    let leading = raw.find(trimmed).unwrap_or(0);
    out.push((start + leading, trimmed.to_string()));
}

fn is_question_like(sentence: &str) -> bool {
    let s = sentence.trim();
    if s.ends_with('?') {
        return true;
    }
    let first = s
        .split(|c: char| !c.is_ascii_alphabetic())
        .find(|part| !part.is_empty())
        .unwrap_or("")
        .to_lowercase();
    question_starters().contains(first.as_str())
}

fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars().flat_map(|c| c.to_lowercase()) {
        let keep = ch.is_alphanumeric() || matches!(ch, '_' | '%' | '\'' | '-' | '.');
        if keep {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn content_words(text: &str) -> HashSet<String> {
    word_re()
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .filter(|w| !stopwords().contains(w.as_str()))
        .collect()
}

/// Ordered, lowercased word tokens (no stopword filter — order matters
/// for contiguity). Exact analogue of the Python `_tokens` helper: uses
/// `gate_word_re()` (≥2 chars, apostrophe) so the binding/residual gate
/// is computed over the *same* token alphabet as Python, making the
/// data-calibrated 0.34 threshold valid in both runtimes.
fn word_tokens(text: &str) -> Vec<String> {
    gate_word_re()
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

/// Longest contiguous token substring shared by `ans` and `sent` (any
/// offset). Token-level LCS-substring DP, O(|ans|·|sent|); inputs are a
/// short QA answer vs. one sentence. Parity with Python
/// `_longest_contiguous_run`: a genuine extractive answer is a
/// contiguous span of its evidence sentence; a recombination /
/// wrong-option answer is not.
fn longest_contiguous_run(ans: &[String], sent: &[String]) -> usize {
    if ans.is_empty() || sent.is_empty() {
        return 0;
    }
    let mut prev = vec![0usize; sent.len() + 1];
    let mut best = 0usize;
    for a in ans {
        let mut cur = vec![0usize; sent.len() + 1];
        for (j, s) in sent.iter().enumerate() {
            if a == s {
                cur[j + 1] = prev[j] + 1;
                if cur[j + 1] > best {
                    best = cur[j + 1];
                }
            }
        }
        prev = cur;
    }
    best
}

fn extract_numbers(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut nums: Vec<String> = number_re()
        .find_iter(text)
        .map(|m| {
            let mut s = m.as_str().replace([',', '$', '%'], "");
            while s.len() > 1 && s.starts_with('0') {
                s.remove(0);
            }
            s
        })
        .filter(|s| !s.is_empty())
        .collect();
    for (word, value) in number_words() {
        if lower
            .split(|c: char| !c.is_ascii_alphabetic())
            .any(|part| part == *word)
        {
            nums.push(value.to_string());
        }
    }
    nums
}

fn extract_entities(text: &str) -> Vec<String> {
    let mut raw = Vec::new();
    for caps in quoted_re().captures_iter(text) {
        if let Some(m) = caps.get(1) {
            raw.push(m.as_str().trim().to_string());
        }
    }
    for m in caps_re().find_iter(text) {
        raw.push(m.as_str().to_string());
    }
    for m in entity_re().find_iter(text) {
        let value = m.as_str().trim();
        if value.len() >= 3 && !stopwords().contains(value.to_lowercase().as_str()) {
            raw.push(value.to_string());
        }
    }
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for entity in raw {
        let key: String = entity
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if !key.is_empty() && seen.insert(key) {
            unique.push(entity);
        }
    }
    unique
}

fn sentence_windows(context: &str, max_chars: usize) -> Vec<String> {
    let mut windows = Vec::new();
    for (_, sentence) in split_sentences(context) {
        if sentence.len() <= max_chars {
            windows.push(sentence);
        } else {
            let mut start = 0;
            while start < sentence.len() {
                let mut end = (start + max_chars).min(sentence.len());
                while end < sentence.len() && !sentence.is_char_boundary(end) {
                    end += 1;
                }
                windows.push(sentence[start..end].trim().to_string());
                start = end;
            }
        }
    }
    if windows.is_empty() {
        windows.push(context.chars().take(max_chars).collect());
    }
    windows.into_iter().filter(|w| !w.is_empty()).collect()
}

fn coverage_fraction(items: &[String], haystack_lower: &str) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let hits = items
        .iter()
        .filter(|item| !item.is_empty() && haystack_lower.contains(&item.to_lowercase()))
        .count();
    hits as f64 / items.len() as f64
}

pub fn continuous_features(
    claim: &str,
    context: &str,
    adequacy: f64,
    question: Option<&str>,
) -> ContinuousFeatures {
    let entity_precision = continuous_entity_precision(claim, context);
    let number_consistency = continuous_number_consistency(claim, context);
    let idf_lex_overlap = continuous_lex_overlap(claim, context);
    let quote = longest_substring_ratio(claim, context);
    let forward = 0.5 * quote + 0.5 * continuous_reverse_entail(claim, context);
    let reverse = continuous_reverse_entail(claim, context);
    let negation = continuous_negation_polarity(claim, context);
    let qa_alignment = continuous_qa_alignment(claim, context, question.unwrap_or(""));
    ContinuousFeatures {
        entity_precision,
        number_consistency,
        idf_lex_overlap,
        quote_support: quote,
        forward_entail: forward.clamp(0.0, 1.0),
        reverse_entail: reverse,
        negation_polarity: negation,
        adequacy: adequacy.clamp(0.0, 1.0),
        qa_alignment,
    }
}

pub fn continuous_risk(features: &ContinuousFeatures) -> f64 {
    let mut z_soft = 6.0;
    let mut survival_gates = Vec::new();
    for (name, value, weight) in [
        ("entity_precision", features.entity_precision, -1.4),
        ("number_consistency", features.number_consistency, -3.2),
        ("idf_lex_overlap", features.idf_lex_overlap, -1.2),
        ("quote_support", features.quote_support, -2.4),
        ("forward_entail", features.forward_entail, -1.8),
        ("reverse_entail", features.reverse_entail, -2.6),
        ("negation_polarity", features.negation_polarity, -3.0),
        ("adequacy", features.adequacy, -0.8),
        ("qa_alignment", features.qa_alignment, -2.8),
    ] {
        if matches!(
            name,
            "entity_precision" | "number_consistency" | "negation_polarity" | "qa_alignment"
        ) && value < 0.0
        {
            survival_gates.push((-value).clamp(0.0, 1.0));
        } else {
            z_soft += weight * value.max(0.0);
        }
    }
    let rho_soft = sigmoid(z_soft);
    let mut survival = 1.0 - rho_soft;
    for gate in survival_gates {
        survival *= 1.0 - gate;
    }
    (1.0 - survival).clamp(0.0, 1.0)
}

fn continuous_entity_precision(claim: &str, context: &str) -> f64 {
    let entities = extract_entities(claim);
    if entities.is_empty() {
        return 0.0;
    }
    let lower = context.to_lowercase();
    let matched = entities
        .iter()
        .filter(|e| lower.contains(&e.to_lowercase()))
        .count();
    if matched == entities.len() {
        1.0
    } else {
        -((entities.len() - matched) as f64) / entities.len() as f64
    }
}

fn continuous_number_consistency(claim: &str, context: &str) -> f64 {
    let claim_nums = extract_numbers(claim);
    if claim_nums.is_empty() {
        return 0.0;
    }
    let context_nums = extract_numbers(context);
    let missing = claim_nums
        .iter()
        .filter(|n| !context_nums.contains(n))
        .count();
    if missing == 0 {
        1.0
    } else {
        -(missing as f64) / claim_nums.len() as f64
    }
}

fn continuous_lex_overlap(claim: &str, context: &str) -> f64 {
    let claim_words = content_words(claim);
    if claim_words.is_empty() {
        return 1.0;
    }
    let context_words = content_words(context);
    intersection_ratio(&claim_words, &context_words).clamp(0.0, 1.0)
}

fn continuous_reverse_entail(claim: &str, context: &str) -> f64 {
    continuous_lex_overlap(claim, context)
}

fn continuous_negation_polarity(claim: &str, context: &str) -> f64 {
    let shared: HashSet<String> = content_words(claim)
        .intersection(&content_words(context))
        .cloned()
        .collect();
    if shared.is_empty() {
        return 0.0;
    }
    let claim_lower = claim.to_lowercase();
    let context_lower = context.to_lowercase();
    for word in shared {
        let claim_neg = has_negation_before(&claim_lower, &word);
        let context_neg = has_negation_before(&context_lower, &word);
        if claim_neg != context_neg {
            return -1.0;
        }
    }
    1.0
}

fn has_negation_before(text: &str, word: &str) -> bool {
    let Some(idx) = text.find(word) else {
        return false;
    };
    let start = idx.saturating_sub(30);
    let preceding = &text[start..idx];
    [
        "is not", "are not", "was not", "were not", "does not", "do not", "did not", "has not",
        "have not", "had not", "cannot", "can not", "isn't", "aren't", "wasn't", "weren't",
        "doesn't", "don't", "didn't", "hasn't", "haven't", "not ", "never ", " no ",
    ]
    .iter()
    .any(|cue| preceding.contains(cue))
}

fn continuous_qa_alignment(answer: &str, knowledge: &str, question: &str) -> f64 {
    if question.trim().is_empty() || knowledge.trim().is_empty() {
        return 0.0;
    }
    let q_words = content_words(question);
    let a_words = content_words(answer);
    let a_nums = extract_numbers(answer);
    if q_words.is_empty() || (a_words.is_empty() && a_nums.is_empty()) {
        return 0.0;
    }
    let sentences = sentence_windows(&knowledge.to_lowercase(), 800);
    let mut best_score = -1.0;
    let mut best_sentence = String::new();
    for sentence in sentences {
        let sent_words = content_words(&sentence);
        let q_score = q_words.intersection(&sent_words).count() as f64;
        let a_score = a_words.intersection(&sent_words).count() as f64 * 0.1;
        let total = q_score + a_score;
        if total > best_score {
            best_score = total;
            best_sentence = sentence;
        }
    }
    if best_score <= 0.0 {
        return 0.0;
    }
    let best_words = content_words(&best_sentence);
    let mut raw = if a_words.is_empty() {
        1.0
    } else {
        intersection_ratio(&a_words, &best_words)
    };
    if !a_nums.is_empty() {
        let sentence_nums = extract_numbers(&best_sentence);
        let present = a_nums.iter().filter(|n| sentence_nums.contains(n)).count() as f64
            / a_nums.len() as f64;
        raw = if a_words.is_empty() {
            present
        } else {
            0.5 * raw + 0.5 * present
        };
    }
    let answer_norm = normalize_text(answer);
    if answer_norm.len() >= 2 && normalize_text(&best_sentence).contains(&answer_norm) {
        raw = (raw + 0.3).min(1.0);
    }
    let confident_locus = q_words.intersection(&best_words).count() >= 2;
    if confident_locus && raw < 0.2 {
        if !a_nums.is_empty() {
            let sentence_nums = extract_numbers(&best_sentence);
            let all_nums = extract_numbers(knowledge);
            let num_in_best = a_nums.iter().filter(|n| sentence_nums.contains(n)).count();
            let num_in_knowledge = a_nums.iter().filter(|n| all_nums.contains(n)).count();
            if num_in_best == 0 && num_in_knowledge > 0 {
                return -1.0;
            }
        }
        if !a_words.is_empty() {
            return -((0.2 - raw) / 0.2).clamp(0.0, 1.0);
        }
    }

    // ── Extractive-binding + question-residual gates ──────────────
    // Parity with the Python feat_qa_alignment refinement that cut
    // HaluEval-QA exposure 0.639 → 0.340 at flat retention. The block
    // above only fires when answer words are nearly absent from the
    // Q-evidence sentence (raw < 0.2). The dominant miss class is
    // different: the answer REUSES the knowledge's vocabulary (raw ≈
    // 0.4–0.7) but recombines it into a false proposition, picks the
    // wrong option, or parrots the question frame with a wrong filler.
    // Word-presence can't see this; contiguity can. Conservative:
    // needs confident_locus + ≥2 answer tokens, so genuine extractive
    // answers (high bind_q) are never demoted.
    if confident_locus {
        let a_tok = word_tokens(answer);
        if a_tok.len() >= 2 {
            let best_tok = word_tokens(&best_sentence);
            let denom = a_tok.len() as f64;
            let bind_q = longest_contiguous_run(&a_tok, &best_tok) as f64 / denom;
            if bind_q < 0.34 {
                return -(0.6 + (0.34 - bind_q) * 1.2).min(1.0);
            }
            // Question-residual payload: only tokens the answer adds
            // beyond the question's own words carry truth value. If
            // that payload is ENTIRELY absent from the evidence
            // sentence (and the answer is not strongly grounded
            // overall), it is a wrong-slot fill.
            if raw < 0.6 {
                let stop = stopwords();
                let payload: Vec<String> = a_tok
                    .iter()
                    .filter(|t| {
                        t.len() > 2 && !stop.contains(t.as_str()) && !q_words.contains(t.as_str())
                    })
                    .cloned()
                    .collect();
                if !payload.is_empty() {
                    let best_set: HashSet<String> = best_tok.iter().cloned().collect();
                    let present = payload.iter().filter(|t| best_set.contains(*t)).count();
                    if present == 0 {
                        return -0.7;
                    }
                }
            }
        }
    }

    raw.clamp(0.0, 1.0)
}

fn longest_substring_ratio(claim: &str, context: &str) -> f64 {
    let a = claim.to_lowercase();
    let b = context.to_lowercase();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut prev = vec![0usize; b_bytes.len() + 1];
    let mut best = 0usize;
    for i in 1..=a_bytes.len() {
        let mut cur = vec![0usize; b_bytes.len() + 1];
        for j in 1..=b_bytes.len() {
            if a_bytes[i - 1] == b_bytes[j - 1] {
                cur[j] = prev[j - 1] + 1;
                best = best.max(cur[j]);
            }
        }
        prev = cur;
    }
    best as f64 / a_bytes.len().max(1) as f64
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn intersection_ratio(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a.intersection(b).count() as f64 / a.len() as f64
}

fn weighted_adequacy(
    entity_cov: f64,
    number_cov: f64,
    token_cov: f64,
    has_entities: bool,
    has_numbers: bool,
) -> f64 {
    let mut weighted = 0.0;
    let mut denom = 0.0;
    if has_entities {
        weighted += 0.45 * entity_cov;
        denom += 0.45;
    }
    if has_numbers {
        weighted += 0.35 * number_cov;
        denom += 0.35;
    }
    let token_weight = if has_entities || has_numbers {
        0.20
    } else {
        1.0
    };
    weighted += token_weight * token_cov;
    denom += token_weight;
    (weighted / denom.max(1e-9_f64)).clamp(0.0, 1.0)
}

fn question_from_context(context: &str) -> String {
    let lower = context.to_lowercase();
    if let Some(idx) = lower.find("question:") {
        context[idx + "question:".len()..].trim().to_string()
    } else {
        String::new()
    }
}

fn candidate_entities(question: &str) -> Vec<String> {
    if !question.to_lowercase().contains(" or ") {
        return vec![];
    }
    let mut tail = question.trim().trim_end_matches('?').to_string();
    if let Some(idx) = tail.rfind(',') {
        tail = tail[idx + 1..].to_string();
    }
    let lower = tail.to_lowercase();
    for marker in [
        " first ",
        " more ",
        " older ",
        " earlier ",
        " larger ",
        " higher ",
    ] {
        if let Some(idx) = lower.find(marker) {
            tail = tail[idx + marker.len()..].to_string();
            break;
        }
    }
    tail.split(" or ")
        .map(|p| {
            p.trim_matches(|c: char| c.is_whitespace() || ".?\"'".contains(c))
                .to_string()
        })
        .filter(|p| p.len() > 1)
        .collect()
}

fn numbers_near(context: &str, entity: &str, radius: usize) -> Vec<i32> {
    let mut nums = Vec::new();
    let lower = context.to_lowercase();
    let key = entity.to_lowercase();
    let mut start = 0usize;
    while let Some(rel_idx) = lower[start..].find(&key) {
        let idx = start + rel_idx;
        let left = idx.saturating_sub(radius);
        let right = (idx + entity.len() + radius).min(context.len());
        let window = &context[left..right];
        nums.extend(
            number_re()
                .find_iter(window)
                .filter_map(|m| m.as_str().replace([',', '$', '%'], "").parse::<i32>().ok()),
        );
        let window_lower = window.to_lowercase();
        for (word, value) in number_words() {
            if window_lower
                .split(|c: char| !c.is_ascii_alphabetic())
                .any(|part| part == *word)
            {
                nums.push(*value);
            }
        }
        start = idx + key.len();
    }
    nums
}

fn top_window_text(windows: &[EvidenceWindow]) -> String {
    windows
        .first()
        .map(|w| w.text.chars().take(200).collect())
        .unwrap_or_default()
}

fn remove_claim_text(output: &str, claim: &str) -> String {
    if let Some(idx) = output.find(claim) {
        let before = output[..idx].trim_end();
        let after = output[idx + claim.len()..].trim_start();
        return format!("{}\n{}", before, after).trim().to_string();
    }

    let claim_words = content_words(claim);
    if claim_words.is_empty() {
        return output.to_string();
    }
    let mut spans = split_sentences(output);
    spans.sort_by_key(|b| std::cmp::Reverse(b.0));
    let mut rewritten = output.to_string();
    for (start, sentence) in spans {
        let words = content_words(&sentence);
        let overlap = intersection_ratio(&claim_words, &words);
        if overlap >= 0.72 {
            let end = (start + sentence.len()).min(rewritten.len());
            rewritten = format!(
                "{}\n{}",
                rewritten[..start].trim_end(),
                rewritten[end..].trim_start()
            );
            break;
        }
    }
    rewritten
}

fn cleanup_rewritten_output(text: &str) -> String {
    let mut out = Vec::new();
    let mut blank = false;
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            if !blank {
                out.push(String::new());
            }
            blank = true;
        } else if trimmed.trim() != "-" && trimmed.trim() != "*" {
            out.push(trimmed.to_string());
            blank = false;
        }
    }
    out.join("\n").trim().to_string()
}

fn round4(value: f64) -> f64 {
    (value * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grounded_and_strict_suppression() {
        let context = "The Oberoi Group has its head office in Delhi.";
        let output =
            "The Oberoi Group has its head office in Delhi. Paris is the capital of Germany.";
        let payload =
            analyze_with_policy(context, output, "strict", "rag", WitnessConfig::default());
        assert_eq!(payload.witness.n_grounded, 1);
        assert_eq!(payload.witness.n_unknown, 1);
        assert!(!payload.output.contains("Paris is the capital of Germany"));
        assert!(payload.output.contains("Delhi"));
    }

    #[test]
    fn comparative_contradiction() {
        let context = concat!(
            "Arthur's Magazine (1844-1846) was an American literary periodical. ",
            "First for Women is a woman's magazine published by Bauer Media Group, started in 1989.\n\n",
            "Question: Which magazine was started first Arthur's Magazine or First for Women?"
        );
        let result = analyze(
            context,
            "First for Women was started first.",
            WitnessConfig::default(),
        );
        assert_eq!(result.certificates[0].label, "contradicted");
    }

    #[test]
    fn summary_profile_warns_unknown_without_suppressing() {
        let context = "The report says revenue increased in 2024.";
        let output = "Revenue increased in 2024. The report also confirms a new Tokyo office.";
        let payload = analyze_with_policy(
            context,
            output,
            "strict",
            "summary",
            WitnessConfig::default(),
        );
        assert!(payload.output.contains("Tokyo office"));
        assert_eq!(payload.policy.suppressed_count, 0);
        assert!(payload.policy.warned_count >= 1);
    }

    #[test]
    fn extracts_bullets_tables_and_code_refs() {
        let output = "- Alpha launched in 2024.\n| Name | Value |\n| Beta | 42 |\n`load_config()` reads config.yaml.";
        let claims = extract_claims(output);
        assert!(claims.iter().any(|c| c.text.contains("Alpha launched")));
        assert!(claims
            .iter()
            .any(|c| c.text.contains("Beta") && c.text.contains("42")));
        assert!(claims.iter().any(|c| c.kind == "code_ref"));
    }

    #[test]
    fn continuous_risk_full_support_is_low() {
        let features = ContinuousFeatures {
            entity_precision: 1.0,
            number_consistency: 1.0,
            idf_lex_overlap: 1.0,
            quote_support: 1.0,
            forward_entail: 1.0,
            reverse_entail: 1.0,
            negation_polarity: 1.0,
            adequacy: 1.0,
            qa_alignment: 1.0,
        };
        assert!(continuous_risk(&features) < 0.05);
    }

    #[test]
    fn continuous_risk_wrong_number_dominates() {
        let features = continuous_features(
            "Einstein developed relativity in 1962.",
            "Einstein developed relativity in 1915.",
            1.0,
            None,
        );
        assert!(features.number_consistency < 0.0);
        assert!(continuous_risk(&features) > 0.99);
    }

    #[test]
    fn continuous_qa_alignment_catches_misplaced_year() {
        let knowledge = concat!(
            "Albert Einstein developed general relativity in 1915. ",
            "He won the Nobel Prize in 1921 for the photoelectric effect."
        );
        let features = continuous_features(
            "1921",
            knowledge,
            1.0,
            Some("What year did Einstein develop general relativity?"),
        );
        assert!(features.qa_alignment < 0.0);
        assert!(continuous_risk(&features) > 0.99);
    }
}
