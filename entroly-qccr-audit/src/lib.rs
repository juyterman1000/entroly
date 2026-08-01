//! Audited atomic-unit selection over Entroly's shared QCCR file ranker.
//! Exact UTF-8 spans and optimizer residuals are emitted at candidate-unit
//! scope. Semantic preservation requires a separately validated calibration.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const CHARS_PER_TOKEN: usize = 4;
const MAX_FILES: usize = 12;
const EPSILON: f64 = 1e-12;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InFragment {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub fragment_id: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub start_byte: Option<usize>,
    #[serde(default)]
    pub end_byte: Option<usize>,
    #[serde(default = "one")]
    pub feedback_multiplier: f64,
}

impl Default for InFragment {
    fn default() -> Self {
        Self {
            id: String::new(),
            fragment_id: String::new(),
            source: String::new(),
            content: String::new(),
            start_byte: None,
            end_byte: None,
            feedback_multiplier: 1.0,
        }
    }
}

fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceSpan {
    pub fragment_id: String,
    pub source: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_token: usize,
    pub end_token: usize,
    pub token_offset_kind: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutFragment {
    pub id: String,
    pub fragment_id: String,
    pub source: String,
    pub content: String,
    pub token_count: usize,
    pub relevance: f64,
    pub relevance_score: f64,
    pub source_spans: Vec<SourceSpan>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateAudit {
    pub unit_id: String,
    pub source_id: String,
    pub fragment_id: String,
    pub utility: f64,
    pub cost_tokens: usize,
    pub selected: bool,
    pub selection_stage: &'static str,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_token: usize,
    pub end_token: usize,
    pub token_offset_kind: &'static str,
    pub trimmed: bool,
    pub neighbourhood_ids: Vec<String>,
    pub query_anchor_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditMetrics {
    pub captured_mass: f64,
    pub shadow_price: f64,
    pub residual_risk: f64,
    pub cutoff_ambiguity: f64,
    pub query_coverage: f64,
    pub boundary_exposure: f64,
    pub budget_saturation: f64,
    pub source_span_integrity: bool,
    pub excluded_positive_candidates: usize,
    pub verdict: &'static str,
    pub scope: &'static str,
    pub reasons: Vec<String>,
    pub signal_availability: BTreeMap<String, bool>,
    pub calibration_version: Option<String>,
    pub calibration_dataset_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditSelection {
    pub selected: Vec<OutFragment>,
    pub candidates: Vec<CandidateAudit>,
    pub metrics: AuditMetrics,
    pub requested_budget: i64,
    pub raw_tokens: usize,
    pub emitted_tokens: usize,
    pub selection_mode: &'static str,
}

#[derive(Clone, Debug)]
struct Unit {
    id: String,
    source: String,
    fragment_id: String,
    text: String,
    start_byte: usize,
    end_byte: usize,
    start_token: usize,
    end_token: usize,
    cost: usize,
    utility: f64,
    words: BTreeSet<String>,
    anchors: Vec<String>,
    required: Vec<usize>,
    selected: bool,
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(CHARS_PER_TOKEN).max(1)
}

fn is_cjk_character(ch: char) -> bool {
    let code = ch as u32;
    (0x3400..=0x4DBF).contains(&code)
        || (0x4E00..=0x9FFF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
        || (0x3040..=0x30FF).contains(&code)
        || (0x31F0..=0x31FF).contains(&code)
        || (0xAC00..=0xD7AF).contains(&code)
}

fn flush_identifier(buffer: &mut String, terms: &mut BTreeSet<String>) {
    if buffer.is_empty() {
        return;
    }
    let raw = std::mem::take(buffer);
    let folded = raw.to_lowercase();
    if folded.chars().count() >= 2 || folded.chars().all(|ch| ch.is_numeric()) {
        terms.insert(folded.clone());
    }
    for part in raw.split(|ch: char| matches!(ch, '_' | '.' | ':' | '/' | '-' | '\\')) {
        let part = part.to_lowercase();
        if part.chars().count() >= 2 || part.chars().all(|ch| ch.is_numeric()) {
            terms.insert(part);
        }
    }
}

fn flush_cjk(buffer: &mut Vec<char>, terms: &mut BTreeSet<String>) {
    if buffer.is_empty() {
        return;
    }
    let run: String = buffer.iter().collect();
    terms.insert(run);
    for ch in buffer.iter().copied() {
        terms.insert(ch.to_string());
    }
    for pair in buffer.windows(2) {
        terms.insert(pair.iter().copied().collect());
    }
    buffer.clear();
}

fn lexical_words(text: &str) -> BTreeSet<String> {
    let mut terms = BTreeSet::new();
    let mut identifier = String::new();
    let mut cjk = Vec::new();
    for ch in text.chars() {
        if is_cjk_character(ch) {
            flush_identifier(&mut identifier, &mut terms);
            cjk.push(ch);
        } else if ch.is_alphanumeric() || ch == '_' {
            flush_cjk(&mut cjk, &mut terms);
            identifier.push(ch);
        } else {
            flush_identifier(&mut identifier, &mut terms);
            flush_cjk(&mut cjk, &mut terms);
        }
    }
    flush_identifier(&mut identifier, &mut terms);
    flush_cjk(&mut cjk, &mut terms);
    terms
}

fn query_terms(text: &str) -> BTreeSet<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "with",
    ];
    lexical_words(text)
        .into_iter()
        .filter(|word| !STOPWORDS.contains(&word.as_str()))
        .collect()
}

fn stable_fragment_id(fragment: &InFragment, index: usize) -> String {
    if !fragment.fragment_id.is_empty() {
        fragment.fragment_id.clone()
    } else if !fragment.id.is_empty() {
        fragment.id.clone()
    } else {
        format!("input::{index}")
    }
}

fn trimmed_span(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    if start >= end
        || end > text.len()
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        return None;
    }
    let slice = &text[start..end];
    let left = slice.len() - slice.trim_start().len();
    let right = slice.len() - slice.trim_end().len();
    (start + left < end - right).then_some((start + left, end - right))
}

fn atomic_spans(text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let next = chars.peek().map(|(_, candidate)| *candidate);
        let end = index + ch.len_utf8();
        let sentence_end = matches!(ch, '.' | '!' | '?')
            && next.map(char::is_whitespace).unwrap_or(true);
        let structural_end = matches!(ch, ':' | ';' | '}') && next == Some('\n');
        let paragraph_end = ch == '\n' && next == Some('\n');
        if sentence_end || structural_end || paragraph_end {
            if let Some(span) = trimmed_span(text, start, end) {
                result.push(span);
            }
            start = end;
        }
    }
    if let Some(span) = trimmed_span(text, start, text.len()) {
        result.push(span);
    }
    if result.is_empty() {
        trimmed_span(text, 0, text.len()).into_iter().collect()
    } else {
        result
    }
}

fn requires_previous(text: &str) -> bool {
    let trimmed = text.trim_start();
    let first = trimmed
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();
    trimmed
        .chars()
        .next()
        .map(char::is_lowercase)
        .unwrap_or(false)
        || matches!(
            first.as_str(),
            "and"
                | "but"
                | "or"
                | "because"
                | "which"
                | "that"
                | "this"
                | "these"
                | "those"
                | "it"
                | "they"
                | "however"
                | "therefore"
                | "then"
                | "so"
        )
}

fn requires_next(text: &str) -> bool {
    let trimmed = text.trim_end();
    let opens = trimmed
        .chars()
        .filter(|ch| matches!(*ch, '(' | '[' | '{'))
        .count();
    let closes = trimmed
        .chars()
        .filter(|ch| matches!(*ch, ')' | ']' | '}'))
        .count();
    trimmed.ends_with(':')
        || trimmed.ends_with(',')
        || trimmed.ends_with(';')
        || trimmed.ends_with("->")
        || trimmed.ends_with('{')
        || trimmed.ends_with('(')
        || trimmed.ends_with('[')
        || opens > closes
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    }
}

fn median_absolute_deviation(values: &[f64]) -> f64 {
    let centre = median(values);
    median(
        &values
            .iter()
            .map(|value| (value - centre).abs())
            .collect::<Vec<_>>(),
    )
}

fn jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    let numerator = left.intersection(right).count();
    let denominator = left.union(right).count();
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn build_units(input: &[InFragment], allowed: &HashSet<String>, query: &str) -> Vec<Unit> {
    let original_terms = query_terms(query);
    let mut expanded: BTreeSet<String> = entroly_qccr::expand_query_sorted(query)
        .into_iter()
        .collect();
    expanded.extend(original_terms.iter().cloned());
    let mut units = Vec::new();

    for (fragment_index, fragment) in input
        .iter()
        .enumerate()
        .filter(|(_, fragment)| allowed.contains(&fragment.source))
    {
        let fragment_id = stable_fragment_id(fragment, fragment_index);
        let byte_base = fragment.start_byte.unwrap_or(0);
        let group_start = units.len();
        let mut token_offset = 0;

        for (local_index, (start, end)) in atomic_spans(&fragment.content).into_iter().enumerate() {
            let text = fragment.content[start..end].to_string();
            let words = lexical_words(&text);
            let internal_token_count = words.len().max(1);
            let anchors = original_terms.intersection(&words).cloned().collect();
            let unit_index = units.len();
            units.push(Unit {
                id: format!(
                    "{fragment_id}::{}:{}",
                    start + byte_base,
                    end + byte_base
                ),
                source: fragment.source.clone(),
                fragment_id: fragment_id.clone(),
                text,
                start_byte: start + byte_base,
                end_byte: end + byte_base,
                start_token: token_offset,
                end_token: token_offset + internal_token_count,
                cost: estimate_tokens(&fragment.content[start..end]),
                utility: 0.0,
                words,
                anchors,
                required: vec![unit_index],
                selected: false,
            });
            token_offset += internal_token_count;
            if local_index > 0 && requires_previous(&units[unit_index].text) {
                units[unit_index].required.push(unit_index - 1);
            }
        }

        for unit_index in group_start..units.len() {
            if unit_index + 1 < units.len() && requires_next(&units[unit_index].text) {
                units[unit_index].required.push(unit_index + 1);
            }
            units[unit_index].required.sort_unstable();
            units[unit_index].required.dedup();
        }
    }

    let document_count = units.len().max(1) as f64;
    let mut document_frequency = HashMap::<String, usize>::new();
    for unit in &units {
        for word in &unit.words {
            *document_frequency.entry(word.clone()).or_default() += 1;
        }
    }
    for unit in &mut units {
        let mut score = 0.0;
        for query_word in &expanded {
            if unit.words.contains(query_word) {
                let frequency = *document_frequency.get(query_word).unwrap_or(&0) as f64;
                score += (1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)).ln();
            }
        }
        unit.utility = score * (1.0 + 0.2 * unit.anchors.len() as f64);
    }
    units
}

fn transitive_closure(units: &[Unit], anchor: usize) -> BTreeSet<usize> {
    let mut closure = BTreeSet::new();
    let mut pending = vec![anchor];
    while let Some(index) = pending.pop() {
        if !closure.insert(index) {
            continue;
        }
        pending.extend(units[index].required.iter().copied());
    }
    closure
}

fn pack(units: &mut [Unit], budget: usize) {
    let mut chosen: BTreeSet<usize> = BTreeSet::new();
    let mut remaining: BTreeSet<usize> = (0..units.len())
        .filter(|index| units[*index].utility > 0.0)
        .collect();
    let mut used = 0;

    while !remaining.is_empty() && used < budget {
        let best = remaining.iter().copied().max_by(|left, right| {
            let density = |index: usize| {
                let redundancy = chosen
                    .iter()
                    .map(|selected_index| {
                        jaccard(&units[index].words, &units[*selected_index].words)
                    })
                    .fold(0.0, f64::max);
                (0.75 * units[index].utility - 0.25 * redundancy)
                    / units[index].cost.max(1) as f64
            };
            density(*left)
                .total_cmp(&density(*right))
                .then_with(|| units[*right].id.cmp(&units[*left].id))
        });
        let Some(anchor) = best else {
            break;
        };
        remaining.remove(&anchor);

        let closure = transitive_closure(units, anchor);
        let additional_cost: usize = closure
            .iter()
            .filter(|index| !chosen.contains(*index))
            .map(|index| units[*index].cost)
            .sum();
        if additional_cost == 0 || used + additional_cost > budget {
            continue;
        }
        for index in closure {
            if chosen.insert(index) {
                used += units[index].cost;
            }
        }
    }

    for index in chosen {
        units[index].selected = true;
    }
}

fn source_span_integrity(units: &[Unit], input: &[InFragment]) -> bool {
    let fragments: HashMap<String, &InFragment> = input
        .iter()
        .enumerate()
        .map(|(index, fragment)| (stable_fragment_id(fragment, index), fragment))
        .collect();
    units.iter().all(|unit| {
        let Some(fragment) = fragments.get(&unit.fragment_id) else {
            return false;
        };
        let base = fragment.start_byte.unwrap_or(0);
        let start = unit.start_byte.saturating_sub(base);
        let end = unit.end_byte.saturating_sub(base);
        start < end
            && end <= fragment.content.len()
            && fragment.content.is_char_boundary(start)
            && fragment.content.is_char_boundary(end)
            && fragment
                .end_byte
                .map(|declared_end| unit.end_byte <= declared_end)
                .unwrap_or(true)
    })
}

fn audit_metrics(units: &[Unit], budget: usize, spans_valid: bool, query: &str) -> AuditMetrics {
    let total_utility: f64 = units.iter().map(|unit| unit.utility.max(0.0)).sum();
    let retained_utility: f64 = units
        .iter()
        .filter(|unit| unit.selected)
        .map(|unit| unit.utility.max(0.0))
        .sum();
    let captured_mass = if total_utility <= EPSILON {
        0.0
    } else {
        retained_utility / total_utility
    };
    let excluded: Vec<&Unit> = units
        .iter()
        .filter(|unit| !unit.selected && unit.utility > 0.0)
        .collect();
    let shadow_price = excluded
        .iter()
        .map(|unit| unit.utility / unit.cost.max(1) as f64)
        .fold(0.0, f64::max);
    let residual_risk = shadow_price / captured_mass.max(EPSILON);

    let selected_densities: Vec<f64> = units
        .iter()
        .filter(|unit| unit.selected && unit.utility > 0.0)
        .map(|unit| unit.utility / unit.cost.max(1) as f64)
        .collect();
    let excluded_densities: Vec<f64> = excluded
        .iter()
        .map(|unit| unit.utility / unit.cost.max(1) as f64)
        .collect();
    let all_densities: Vec<f64> = selected_densities
        .iter()
        .chain(excluded_densities.iter())
        .copied()
        .collect();
    let cutoff_ambiguity = if selected_densities.is_empty() || excluded_densities.is_empty() {
        0.0
    } else {
        let selected_floor = selected_densities
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let excluded_ceiling = excluded_densities.iter().copied().fold(0.0, f64::max);
        (-(selected_floor - excluded_ceiling).abs()
            / (median_absolute_deviation(&all_densities) + EPSILON))
            .exp()
    };

    let query_words = query_terms(query);
    let retained_words: BTreeSet<String> = units
        .iter()
        .filter(|unit| unit.selected)
        .flat_map(|unit| unit.words.iter().cloned())
        .collect();
    let query_coverage = if query_words.is_empty() {
        1.0
    } else {
        query_words.intersection(&retained_words).count() as f64 / query_words.len() as f64
    };

    let selected_ids: HashSet<&str> = units
        .iter()
        .filter(|unit| unit.selected)
        .map(|unit| unit.id.as_str())
        .collect();
    let anchored: Vec<&Unit> = units
        .iter()
        .filter(|unit| unit.selected && !unit.anchors.is_empty())
        .collect();
    let exposed = anchored
        .iter()
        .filter(|unit| {
            unit.required
                .iter()
                .any(|index| !selected_ids.contains(units[*index].id.as_str()))
        })
        .count();
    let boundary_exposure = if anchored.is_empty() {
        0.0
    } else {
        exposed as f64 / anchored.len() as f64
    };

    let emitted_tokens: usize = units
        .iter()
        .filter(|unit| unit.selected)
        .map(|unit| unit.cost)
        .sum();
    let budget_saturation = if budget == 0 {
        1.0
    } else {
        emitted_tokens as f64 / budget as f64
    };

    let mut reasons = Vec::new();
    let verdict = if !spans_valid {
        reasons.push("source span integrity failed".to_string());
        "degraded"
    } else if boundary_exposure > 0.0 {
        reasons.push("an atomic neighbourhood was severed".to_string());
        "degraded"
    } else if emitted_tokens == 0 {
        reasons.push("no complete positive-utility unit fits the budget".to_string());
        "uncertain"
    } else if query_coverage < 0.5 {
        reasons.push(format!("query coverage is {:.1}%", query_coverage * 100.0));
        "degraded"
    } else if !excluded.is_empty() && budget_saturation >= 0.9 {
        reasons.push(format!(
            "{} positive-utility units remain outside a saturated budget",
            excluded.len()
        ));
        "uncertain"
    } else {
        "sufficient"
    };

    let mut availability = BTreeMap::new();
    for (name, available) in [
        ("candidate_residuals", true),
        ("exact_utf8_byte_offsets", true),
        ("qccr_internal_token_offsets", true),
        ("atomic_neighbourhoods", true),
        ("provider_token_offsets", false),
        ("semantic_calibration", false),
        ("task_oracle", false),
        ("perturbation_stability", false),
    ] {
        availability.insert(name.to_string(), available);
    }

    AuditMetrics {
        captured_mass: rounded(captured_mass),
        shadow_price: rounded(shadow_price),
        residual_risk: rounded(residual_risk),
        cutoff_ambiguity: rounded(cutoff_ambiguity),
        query_coverage: rounded(query_coverage),
        boundary_exposure: rounded(boundary_exposure),
        budget_saturation: rounded(budget_saturation),
        source_span_integrity: spans_valid,
        excluded_positive_candidates: excluded.len(),
        verdict,
        scope: "candidate_units",
        reasons,
        signal_availability: availability,
        calibration_version: None,
        calibration_dataset_fingerprint: None,
    }
}

fn emit_selected(units: &[Unit], file_scores: &HashMap<String, f64>) -> Vec<OutFragment> {
    let mut groups: BTreeMap<String, Vec<&Unit>> = BTreeMap::new();
    for unit in units.iter().filter(|unit| unit.selected) {
        groups.entry(unit.source.clone()).or_default().push(unit);
    }

    let mut output = Vec::new();
    for (source, mut source_units) in groups {
        source_units.sort_by_key(|unit| (unit.fragment_id.clone(), unit.start_byte));
        let content = source_units
            .iter()
            .map(|unit| unit.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let relevance = rounded(*file_scores.get(&source).unwrap_or(&0.0));
        let id = format!("qccr-audit::{source}");
        let source_spans = source_units
            .iter()
            .map(|unit| SourceSpan {
                fragment_id: unit.fragment_id.clone(),
                source: unit.source.clone(),
                start_byte: unit.start_byte,
                end_byte: unit.end_byte,
                start_token: unit.start_token,
                end_token: unit.end_token,
                token_offset_kind: "qccr_internal_v1",
            })
            .collect();
        output.push(OutFragment {
            id: id.clone(),
            fragment_id: id,
            source,
            content,
            token_count: source_units.iter().map(|unit| unit.cost).sum(),
            relevance,
            relevance_score: relevance,
            source_spans,
        });
    }
    output.sort_by(|left, right| {
        right
            .relevance
            .total_cmp(&left.relevance)
            .then_with(|| left.source.cmp(&right.source))
    });
    output
}

pub fn select_with_audit(
    input: &[InFragment],
    budget: i64,
    query: &str,
    overrides: &HashMap<String, f64>,
    preferred: &[String],
) -> AuditSelection {
    let raw_tokens: usize = input
        .iter()
        .map(|fragment| estimate_tokens(&fragment.content))
        .sum();
    let token_budget = budget.max(0) as usize;

    let mut source_order = Vec::new();
    let mut source_text = HashMap::<String, String>::new();
    let mut feedback = HashMap::<String, (f64, usize)>::new();
    for fragment in input {
        if !source_text.contains_key(&fragment.source) {
            source_order.push(fragment.source.clone());
        }
        source_text
            .entry(fragment.source.clone())
            .and_modify(|text| {
                text.push('\n');
                text.push_str(&fragment.content);
            })
            .or_insert_with(|| fragment.content.clone());
        let entry = feedback.entry(fragment.source.clone()).or_default();
        entry.0 += fragment.feedback_multiplier.clamp(0.5, 2.0);
        entry.1 += 1;
    }

    let documents: Vec<String> = source_order
        .iter()
        .map(|source| source_text[source].clone())
        .collect();
    let mut ranked: Vec<(String, f64)> = entroly_qccr::rank_files(
        &source_order,
        &documents,
        query,
        overrides,
    )
    .into_iter()
    .map(|(index, score)| {
        let source = source_order[index].clone();
        let (sum, count) = feedback.get(&source).copied().unwrap_or((1.0, 1));
        (source, score * sum / count.max(1) as f64)
    })
    .collect();

    if !preferred.is_empty() {
        let score_by_source: HashMap<String, f64> = ranked.iter().cloned().collect();
        let preferred_ranked: Vec<(String, f64)> = preferred
            .iter()
            .filter_map(|source| {
                score_by_source
                    .get(source)
                    .map(|score| (source.clone(), *score))
            })
            .collect();
        if !preferred_ranked.is_empty() {
            ranked = preferred_ranked;
        }
    }
    ranked.truncate(MAX_FILES);

    let allowed: HashSet<String> = ranked.iter().map(|item| item.0.clone()).collect();
    let file_scores: HashMap<String, f64> = ranked.into_iter().collect();
    let mut units = build_units(input, &allowed, query);
    if token_budget > 0 && !query.trim().is_empty() {
        pack(&mut units, token_budget);
    }

    let spans_valid = source_span_integrity(&units, input);
    let metrics = audit_metrics(&units, token_budget, spans_valid, query);
    let selected = emit_selected(&units, &file_scores);
    let emitted_tokens = selected.iter().map(|fragment| fragment.token_count).sum();
    let candidates = units
        .iter()
        .map(|unit| CandidateAudit {
            unit_id: unit.id.clone(),
            source_id: unit.source.clone(),
            fragment_id: unit.fragment_id.clone(),
            utility: rounded(unit.utility),
            cost_tokens: unit.cost,
            selected: unit.selected,
            selection_stage: if unit.selected {
                "atomic_pack"
            } else {
                "candidate"
            },
            start_byte: unit.start_byte,
            end_byte: unit.end_byte,
            start_token: unit.start_token,
            end_token: unit.end_token,
            token_offset_kind: "qccr_internal_v1",
            trimmed: false,
            neighbourhood_ids: unit
                .required
                .iter()
                .map(|index| units[*index].id.clone())
                .collect(),
            query_anchor_ids: unit.anchors.clone(),
        })
        .collect();

    AuditSelection {
        selected,
        candidates,
        metrics,
        requested_budget: budget,
        raw_tokens,
        emitted_tokens,
        selection_mode: "atomic_audited",
    }
}

pub fn select_with_audit_json(
    fragments_json: &str,
    budget: i64,
    query: &str,
    overrides_json: &str,
    preferred_json: &str,
) -> String {
    let input: Vec<InFragment> = serde_json::from_str(fragments_json).unwrap_or_default();
    let overrides: HashMap<String, f64> =
        serde_json::from_str(overrides_json).unwrap_or_default();
    let preferred: Vec<String> = serde_json::from_str(preferred_json).unwrap_or_default();
    serde_json::to_string(&select_with_audit(
        &input,
        budget,
        query,
        &overrides,
        &preferred,
    ))
    .unwrap_or_else(|_| {
        "{\"selected\":[],\"candidates\":[],\"metrics\":{\"verdict\":\"uncertain\",\"scope\":\"unavailable\"}}".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragment(text: &str) -> InFragment {
        InFragment {
            fragment_id: "f1".to_string(),
            source: "doc.txt".to_string(),
            content: text.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn unicode_spans_reconstruct() {
        let input = fragment("Résumé. 東京 is the answer.");
        let result = select_with_audit(
            std::slice::from_ref(&input),
            100,
            "東京 answer",
            &HashMap::new(),
            &[],
        );
        for candidate in result.candidates {
            assert!(input.content.is_char_boundary(candidate.start_byte));
            assert!(input.content.is_char_boundary(candidate.end_byte));
        }
        assert!(result.metrics.source_span_integrity);
    }

    #[test]
    fn never_partial_trim() {
        let result = select_with_audit(
            &[fragment("Alpha requested key. Beta requested key.")],
            8,
            "requested key",
            &HashMap::new(),
            &[],
        );
        assert!(result.candidates.iter().all(|candidate| !candidate.trimmed));
        assert!(result.emitted_tokens <= 8);
    }

    #[test]
    fn continuation_closure_keeps_answer() {
        let result = select_with_audit(
            &[fragment("The Dutch name is:\nRhijn. Other note.")],
            30,
            "What is the Dutch name?",
            &HashMap::new(),
            &[],
        );
        let selected = result
            .selected
            .iter()
            .map(|fragment| fragment.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(selected.contains("Rhijn"));
        assert_eq!(result.metrics.boundary_exposure, 0.0);
        assert!(result.metrics.query_coverage >= 1.0 - EPSILON);
    }

    #[test]
    fn residuals_include_excluded_candidates() {
        let result = select_with_audit(
            &[fragment(
                "Authentication rotates. Authentication logs. Authentication recovers.",
            )],
            6,
            "authentication",
            &HashMap::new(),
            &[],
        );
        assert!(result.candidates.iter().any(|candidate| candidate.selected));
        assert!(result.candidates.iter().any(|candidate| !candidate.selected));
    }

    #[test]
    fn cjk_subphrase_contributes_to_relevance_and_coverage() {
        let result = select_with_audit(
            &[fragment("監査ログ。認証失敗を検出した。復旧手順を開始した。")],
            100,
            "認証失敗の原因",
            &HashMap::new(),
            &[],
        );
        let selected = result
            .selected
            .iter()
            .map(|fragment| fragment.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(selected.contains("認証失敗"));
        assert!(result.metrics.query_coverage >= 0.5);
    }

    #[test]
    fn output_is_deterministic() {
        let encoded = serde_json::to_string(&vec![fragment("Alpha. Beta. Gamma.")]).unwrap();
        assert_eq!(
            select_with_audit_json(&encoded, 8, "alpha", "{}", "[]"),
            select_with_audit_json(&encoded, 8, "alpha", "{}", "[]")
        );
    }

    #[test]
    fn no_complete_unit_fit_is_uncertain() {
        let result = select_with_audit(
            &[fragment(
                "One indivisible answer sentence that is far too long.",
            )],
            1,
            "answer",
            &HashMap::new(),
            &[],
        );
        assert!(result.selected.is_empty());
        assert_eq!(result.metrics.verdict, "uncertain");
    }
}
