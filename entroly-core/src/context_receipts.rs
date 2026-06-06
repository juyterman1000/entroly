//! Context Receipts: deterministic multi-document context selection audit trail.
//!
//! The Python CLI is the control plane. This module is the native deterministic
//! engine for ingestion, lexical ranking, dependency scans, budget selection,
//! receipt hashes, and Markdown rendering.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const SCHEMA_VERSION: &str = "context-receipt.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub document_id: String,
    pub source_path: String,
    pub title: String,
    pub fingerprint: String,
    pub token_count: usize,
    pub byte_count: usize,
    pub chunk_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub chunk_id: String,
    pub document_id: String,
    pub source_path: String,
    pub title: String,
    pub section_heading: Option<String>,
    pub page_number: Option<usize>,
    pub chunk_index: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub token_count: usize,
    pub fingerprint: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextIndex {
    pub schema_version: String,
    pub documents: Vec<DocumentRecord>,
    pub chunks: Vec<DocumentChunk>,
    pub chunk_token_limit: usize,
    pub chunk_overlap: usize,
    pub source_fingerprints: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
struct RankedChunk {
    chunk_id: String,
    lexical_score: f64,
    semantic_score: f64,
    rerank_score: f64,
    final_score: f64,
    reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DependencyLink {
    pub source_chunk_id: String,
    pub target_chunk_id: Option<String>,
    pub relation_type: String,
    pub evidence: String,
    pub source_document_id: String,
    pub target_document_id: Option<String>,
    pub resolved: bool,
    pub warning: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectedContextItem {
    pub chunk_id: String,
    pub source_path: String,
    pub section_heading: Option<String>,
    pub page_number: Option<usize>,
    pub byte_start: usize,
    pub byte_end: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub token_count: usize,
    pub score: f64,
    pub reasons: Vec<String>,
    pub dependencies_included: Vec<String>,
    pub dependencies_missing: Vec<String>,
    pub fingerprint: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmittedContextItem {
    pub chunk_id: String,
    pub source_path: String,
    pub section_heading: Option<String>,
    pub page_number: Option<usize>,
    pub token_count: usize,
    pub score: f64,
    pub reasons: Vec<String>,
    pub omission_reason: String,
    pub fingerprint: String,
    pub text_preview: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionRatio {
    pub source_tokens: usize,
    pub selected_tokens: usize,
    pub tokens_saved: usize,
    pub selected_to_source_ratio: f64,
    pub source_to_selected_ratio: f64,
    pub reduction_pct: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextReceipt {
    pub receipt_id: String,
    pub schema_version: String,
    pub query: String,
    pub token_budget: usize,
    pub selected_context: Vec<SelectedContextItem>,
    pub omitted_context: Vec<OmittedContextItem>,
    pub dependency_links: Vec<DependencyLink>,
    pub ranking_reasons: BTreeMap<String, Vec<String>>,
    pub compression_ratio: CompressionRatio,
    pub source_fingerprints: Value,
    pub risk_summary: Value,
    pub warnings: Vec<String>,
    pub reproducibility_hash: String,
    pub outcome_links: Vec<Value>,
}

#[derive(Clone, Debug)]
struct Block {
    text: String,
    start: usize,
    end: usize,
    heading: Option<String>,
    page: Option<usize>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn text_fingerprint(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    format!("sha256:{}", sha256_hex(normalized.as_bytes()))
}

fn stable_hash<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_vec(value).map(|raw| sha256_hex(&raw))
}

fn token_re() -> Regex {
    Regex::new(r"[A-Za-z0-9][A-Za-z0-9_']*").unwrap()
}

fn estimate_tokens(text: &str) -> usize {
    token_re().find_iter(&text.replace('-', " ")).count().max(1)
}

fn tokenize(text: &str) -> Vec<String> {
    token_re()
        .find_iter(&text.replace('-', " "))
        .map(|m| m.as_str().to_ascii_lowercase())
        .filter(|t| t.len() > 1 && !is_stopword(t))
        .collect()
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "the"
            | "and"
            | "for"
            | "with"
            | "that"
            | "this"
            | "does"
            | "have"
            | "has"
            | "from"
            | "into"
            | "which"
            | "what"
            | "where"
            | "when"
            | "shall"
            | "will"
            | "of"
            | "a"
            | "an"
    )
}

fn title_for_path(source_path: &str) -> String {
    let name = source_path
        .rsplit(&['/', '\\'][..])
        .next()
        .unwrap_or(source_path);
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    stem.replace(['_', '-'], " ")
}

fn norm(value: &str) -> String {
    Regex::new(r"[^a-z0-9]+")
        .unwrap()
        .replace_all(&value.to_ascii_lowercase(), " ")
        .trim()
        .to_string()
}

fn clean_heading(line: &str) -> String {
    line.trim().trim_start_matches('#').trim().to_string()
}

fn is_heading(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('#')
        || Regex::new(r"(?i)^(section|article|clause|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+")
            .unwrap()
            .is_match(trimmed)
        || Regex::new(r"^\d+(\.\d+)*\s+\S+").unwrap().is_match(trimmed)
}

fn parse_page(line: &str) -> Option<usize> {
    Regex::new(r"(?i)\bpage\s+(\d+)\b")
        .unwrap()
        .captures(line)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<usize>().ok())
}

fn paragraph_blocks(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut start: Option<usize> = None;
    let mut heading: Option<String> = None;
    let mut page: Option<usize> = None;
    let mut offset = 0usize;

    fn flush(
        blocks: &mut Vec<Block>,
        current: &mut String,
        start: &mut Option<usize>,
        end: usize,
        heading: Option<String>,
        page: Option<usize>,
    ) {
        if let Some(s) = *start {
            let raw = current.trim();
            if !raw.is_empty() {
                blocks.push(Block {
                    text: raw.to_string(),
                    start: s,
                    end,
                    heading,
                    page,
                });
            }
        }
        current.clear();
        *start = None;
    }

    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if let Some(p) = parse_page(trimmed) {
            page = Some(p);
        }
        if line.contains('\u{000C}') {
            page = Some(page.unwrap_or(0) + line.matches('\u{000C}').count());
        }
        if !trimmed.is_empty() && is_heading(trimmed) {
            flush(
                &mut blocks,
                &mut current,
                &mut start,
                offset,
                heading.clone(),
                page,
            );
            heading = Some(clean_heading(trimmed));
        }
        if trimmed.is_empty() {
            flush(
                &mut blocks,
                &mut current,
                &mut start,
                offset,
                heading.clone(),
                page,
            );
        } else {
            if start.is_none() {
                start = Some(offset);
            }
            current.push_str(line);
        }
        offset += line.len();
    }
    flush(
        &mut blocks,
        &mut current,
        &mut start,
        text.len(),
        heading,
        page,
    );
    blocks
}

fn split_large_block(block: &Block, chunk_tokens: usize, overlap_tokens: usize) -> Vec<Block> {
    let replaced = block.text.replace('-', " ");
    let token_matches: Vec<_> = token_re().find_iter(&replaced).collect();
    if token_matches.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let step = chunk_tokens.saturating_sub(overlap_tokens).max(1);
    let mut token_start = 0usize;
    while token_start < token_matches.len() {
        let token_end = (token_start + chunk_tokens).min(token_matches.len());
        let local_start = token_matches[token_start].start();
        let local_end = token_matches[token_end - 1].end();
        out.push(Block {
            text: block.text[local_start..local_end].to_string(),
            start: block.start + local_start,
            end: block.start + local_end,
            heading: block.heading.clone(),
            page: block.page,
        });
        if token_end >= token_matches.len() {
            break;
        }
        token_start += step;
    }
    out
}

fn chunk_document(
    source_path: &str,
    text: &str,
    document_id: &str,
    doc_fingerprint: &str,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> Vec<DocumentChunk> {
    let mut raw = Vec::<Block>::new();
    let mut pending = String::new();
    let mut start: Option<usize> = None;
    let mut end = 0usize;
    let mut heading: Option<String> = None;
    let mut page: Option<usize> = None;
    let mut tokens = 0usize;

    fn flush(
        raw: &mut Vec<Block>,
        pending: &mut String,
        start: &mut Option<usize>,
        end: usize,
        heading: Option<String>,
        page: Option<usize>,
        tokens: &mut usize,
    ) {
        if let Some(s) = *start {
            let text = pending.trim();
            if !text.is_empty() {
                raw.push(Block {
                    text: text.to_string(),
                    start: s,
                    end,
                    heading,
                    page,
                });
            }
        }
        pending.clear();
        *start = None;
        *tokens = 0;
    }

    for block in paragraph_blocks(text) {
        let btokens = estimate_tokens(&block.text);
        if btokens > chunk_tokens {
            flush(
                &mut raw,
                &mut pending,
                &mut start,
                end,
                heading.clone(),
                page,
                &mut tokens,
            );
            raw.extend(split_large_block(&block, chunk_tokens, overlap_tokens));
            continue;
        }
        if !pending.is_empty() && tokens + btokens > chunk_tokens {
            flush(
                &mut raw,
                &mut pending,
                &mut start,
                end,
                heading.clone(),
                page,
                &mut tokens,
            );
        }
        if start.is_none() {
            start = Some(block.start);
            heading = block.heading.clone();
            page = block.page;
        }
        if !pending.is_empty() {
            pending.push_str("\n\n");
        }
        pending.push_str(&block.text);
        end = block.end;
        tokens += btokens;
    }
    flush(
        &mut raw,
        &mut pending,
        &mut start,
        end,
        heading,
        page,
        &mut tokens,
    );

    let mut out = Vec::new();
    let title = title_for_path(source_path);
    let mut running = 0usize;
    for (idx, block) in raw.into_iter().enumerate() {
        let count = estimate_tokens(&block.text);
        let chunk_fp = text_fingerprint(&format!(
            "{}\n{}:{}\n{}",
            doc_fingerprint, block.start, block.end, block.text
        ));
        let chunk_id_hash = stable_hash(&format!(
            "{}:{}:{}:{}",
            document_id, block.start, block.end, chunk_fp
        ))
        .unwrap_or_else(|_| sha256_hex(block.text.as_bytes()));
        out.push(DocumentChunk {
            chunk_id: format!("chk_{}", &chunk_id_hash[..12]),
            document_id: document_id.to_string(),
            source_path: source_path.to_string(),
            title: title.clone(),
            section_heading: block.heading,
            page_number: block.page,
            chunk_index: idx,
            byte_start: block.start,
            byte_end: block.end,
            token_start: running,
            token_end: running + count,
            token_count: count,
            fingerprint: chunk_fp,
            text: block.text,
        });
        running += count;
    }
    out
}

pub fn ingest_documents(
    mut documents: Vec<(String, String)>,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> ContextIndex {
    documents.sort_by(|a, b| a.0.cmp(&b.0));
    let limit = chunk_tokens.max(40);
    let mut records = Vec::new();
    let mut chunks = Vec::new();
    let mut source_fingerprints = BTreeMap::new();

    for (source_path, text) in documents {
        let doc_fp = text_fingerprint(&text);
        let doc_hash = stable_hash(&format!("{}:{}", source_path, doc_fp))
            .unwrap_or_else(|_| sha256_hex(source_path.as_bytes()));
        let doc_id = format!("doc_{}", &doc_hash[..12]);
        let doc_chunks =
            chunk_document(&source_path, &text, &doc_id, &doc_fp, limit, overlap_tokens);
        source_fingerprints.insert(source_path.clone(), doc_fp.clone());
        records.push(DocumentRecord {
            document_id: doc_id,
            source_path: source_path.clone(),
            title: title_for_path(&source_path),
            fingerprint: doc_fp,
            token_count: estimate_tokens(&text),
            byte_count: text.len(),
            chunk_ids: doc_chunks.iter().map(|c| c.chunk_id.clone()).collect(),
        });
        chunks.extend(doc_chunks);
    }

    ContextIndex {
        schema_version: SCHEMA_VERSION.to_string(),
        documents: records,
        chunks,
        chunk_token_limit: limit,
        chunk_overlap: overlap_tokens,
        source_fingerprints,
    }
}

fn rank_chunks(index: &ContextIndex, query: &str) -> Vec<RankedChunk> {
    let query_terms = tokenize(query);
    let mut tokenized: HashMap<String, Vec<String>> = HashMap::new();
    let mut df: HashMap<String, usize> = HashMap::new();
    for chunk in &index.chunks {
        let terms = tokenize(&chunk.text);
        for term in terms.iter().cloned().collect::<HashSet<_>>() {
            *df.entry(term).or_insert(0) += 1;
        }
        tokenized.insert(chunk.chunk_id.clone(), terms);
    }
    let doc_count = index.chunks.len().max(1) as f64;
    let avg_len = tokenized.values().map(|v| v.len()).sum::<usize>() as f64 / doc_count.max(1.0);
    let mut ranked = Vec::new();

    for chunk in &index.chunks {
        let terms = tokenized.get(&chunk.chunk_id).cloned().unwrap_or_default();
        let mut tf: HashMap<String, usize> = HashMap::new();
        for term in &terms {
            *tf.entry(term.clone()).or_insert(0) += 1;
        }
        let mut lexical = 0.0f64;
        let mut matched = BTreeSet::<String>::new();
        let doc_len = terms.len().max(1) as f64;
        let heading = chunk
            .section_heading
            .clone()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let path = chunk.source_path.to_ascii_lowercase();
        for term in &query_terms {
            let freq = *tf.get(term).unwrap_or(&0) as f64;
            let in_heading = heading.contains(term);
            let in_path = path.contains(term);
            if freq > 0.0 || in_heading || in_path {
                matched.insert(term.clone());
            }
            let dfi = *df.get(term).unwrap_or(&0) as f64;
            let idf = ((doc_count - dfi + 0.5) / (dfi + 0.5) + 1.0).ln();
            if freq > 0.0 {
                lexical += idf
                    * ((freq * 2.2)
                        / (freq + 1.2 * (1.0 - 0.75 + 0.75 * doc_len / avg_len.max(1.0))));
            }
            if in_heading {
                lexical += idf * 2.5;
            }
            if in_path {
                lexical += idf * 1.5;
            }
        }
        let coverage = matched.len() as f64 / query_terms.len().max(1) as f64;
        lexical *= 1.0 + coverage;
        let mut reasons = Vec::new();
        if !matched.is_empty() {
            reasons.push(format!(
                "lexical match: {}",
                matched
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let heading_hits: Vec<_> = query_terms
            .iter()
            .filter(|t| heading.contains(*t))
            .cloned()
            .collect();
        if !heading_hits.is_empty() {
            reasons.push(format!(
                "section heading match: {}",
                heading_hits.join(", ")
            ));
        }
        let lower = chunk.text.to_ascii_lowercase();
        if ["as defined in", "subject to", "pursuant to", "see section"]
            .iter()
            .any(|p| lower.contains(p))
        {
            reasons.push("contains explicit dependency/reference language".to_string());
        }
        if reasons.is_empty() {
            reasons.push("low lexical overlap; retained as lower-ranked candidate".to_string());
        }
        ranked.push(RankedChunk {
            chunk_id: chunk.chunk_id.clone(),
            lexical_score: round6(lexical),
            semantic_score: 0.0,
            rerank_score: 0.0,
            final_score: round6(lexical),
            reasons,
        });
    }
    ranked.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    ranked
}

fn defined_terms(chunk: &DocumentChunk) -> Vec<String> {
    let mut terms = Vec::new();
    let quoted =
        Regex::new(r#"(?i)"([^"]{2,80})"\s+(means|shall mean|is defined as|refers to)\b"#).unwrap();
    for cap in quoted.captures_iter(&chunk.text) {
        if let Some(term) = cap.get(1) {
            terms.push(norm(term.as_str()));
        }
    }
    let alias =
        Regex::new(r#"(?i)\b([A-Z][A-Za-z0-9 \-]{2,80})\s+\(the\s+"([^"]{2,80})"\)"#).unwrap();
    for cap in alias.captures_iter(&chunk.text) {
        if let Some(term) = cap.get(2) {
            terms.push(norm(term.as_str()));
        }
    }
    terms.sort();
    terms.dedup();
    terms.into_iter().filter(|t| t.len() >= 2).collect()
}

fn resolve_heading<'a>(
    chunks: &'a [DocumentChunk],
    label: &str,
    source_chunk_id: &str,
) -> Option<&'a DocumentChunk> {
    let normalized = norm(label);
    if normalized.is_empty() {
        return None;
    }
    chunks.iter().find(|chunk| {
        chunk.chunk_id != source_chunk_id
            && (norm(chunk.section_heading.as_deref().unwrap_or("")).contains(&normalized)
                || norm(&chunk.source_path).contains(&normalized)
                || norm(&chunk.text.chars().take(300).collect::<String>()).contains(&normalized))
    })
}

fn detect_dependencies(index: &ContextIndex) -> Vec<DependencyLink> {
    let mut term_defs: HashMap<String, Vec<&DocumentChunk>> = HashMap::new();
    for chunk in &index.chunks {
        for term in defined_terms(chunk) {
            term_defs.entry(term).or_default().push(chunk);
        }
    }
    let mut links = Vec::<DependencyLink>::new();
    let mut seen = HashSet::<String>::new();

    for chunk in &index.chunks {
        let lower = norm(&chunk.text);
        for (term, defs) in &term_defs {
            if lower.contains(term) && defs.iter().all(|d| d.chunk_id != chunk.chunk_id) {
                push_dependency(
                    &mut links,
                    &mut seen,
                    chunk,
                    defs.first().copied(),
                    "defined_term",
                    term,
                );
            }
        }
        for (relation, pattern) in [
            (
                "defined_in",
                r"(?i)\bas defined in\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)",
            ),
            (
                "subject_to",
                r"(?i)\bsubject to\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)",
            ),
            (
                "pursuant_to",
                r"(?i)\bpursuant to\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)",
            ),
            (
                "see_reference",
                r"(?i)\bsee\s+((section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)",
            ),
            (
                "structural_reference",
                r"(?i)\b(section|clause|article|exhibit|schedule|addendum)\s+([A-Za-z0-9.\-]+)",
            ),
        ] {
            let re = Regex::new(pattern).unwrap();
            for cap in re.captures_iter(&chunk.text) {
                let evidence = if relation == "structural_reference" {
                    format!(
                        "{} {}",
                        cap.get(1).unwrap().as_str(),
                        cap.get(2).unwrap().as_str()
                    )
                } else {
                    cap.get(1).unwrap().as_str().to_string()
                };
                if relation == "structural_reference"
                    && norm(chunk.section_heading.as_deref().unwrap_or(""))
                        .contains(&norm(&evidence))
                {
                    continue;
                }
                let target = resolve_heading(&index.chunks, &evidence, &chunk.chunk_id);
                push_dependency(&mut links, &mut seen, chunk, target, relation, &evidence);
            }
        }
    }
    links
}

fn push_dependency(
    links: &mut Vec<DependencyLink>,
    seen: &mut HashSet<String>,
    source: &DocumentChunk,
    target: Option<&DocumentChunk>,
    relation: &str,
    evidence: &str,
) {
    let key = format!(
        "{}:{}:{}:{}",
        source.chunk_id,
        target.map(|t| t.chunk_id.as_str()).unwrap_or(""),
        relation,
        evidence.to_ascii_lowercase()
    );
    if !seen.insert(key) {
        return;
    }
    links.push(DependencyLink {
        source_chunk_id: source.chunk_id.clone(),
        target_chunk_id: target.map(|t| t.chunk_id.clone()),
        relation_type: relation.to_string(),
        evidence: evidence.chars().take(160).collect(),
        source_document_id: source.document_id.clone(),
        target_document_id: target.map(|t| t.document_id.clone()),
        resolved: target.is_some(),
        warning: if target.is_some() {
            None
        } else {
            Some(format!(
                "Unresolved reference: {}",
                evidence.chars().take(80).collect::<String>()
            ))
        },
    });
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.intersection(b).count() as f64 / a.union(b).count().max(1) as f64
}

fn is_redundant(
    candidate: &str,
    selected_ids: &[String],
    token_sets: &HashMap<String, HashSet<String>>,
) -> bool {
    let Some(candidate_set) = token_sets.get(candidate) else {
        return false;
    };
    selected_ids
        .iter()
        .filter_map(|sid| token_sets.get(sid))
        .any(|selected| jaccard(candidate_set, selected) >= 0.82)
}

fn preview(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= 240 {
        compact
    } else {
        format!("{}...", &compact[..237])
    }
}

fn select_context(
    index: &ContextIndex,
    ranked: &[RankedChunk],
    deps: &[DependencyLink],
    token_budget: usize,
) -> (
    Vec<SelectedContextItem>,
    Vec<OmittedContextItem>,
    Vec<String>,
) {
    let chunks: HashMap<String, &DocumentChunk> = index
        .chunks
        .iter()
        .map(|c| (c.chunk_id.clone(), c))
        .collect();
    let ranks: HashMap<String, &RankedChunk> =
        ranked.iter().map(|r| (r.chunk_id.clone(), r)).collect();
    let mut deps_by_source: HashMap<String, Vec<&DependencyLink>> = HashMap::new();
    for dep in deps {
        deps_by_source
            .entry(dep.source_chunk_id.clone())
            .or_default()
            .push(dep);
    }
    let token_sets: HashMap<String, HashSet<String>> = index
        .chunks
        .iter()
        .map(|c| (c.chunk_id.clone(), tokenize(&c.text).into_iter().collect()))
        .collect();

    let mut selected_ids = Vec::<String>::new();
    let mut selected_set = HashSet::<String>::new();
    let mut selected_tokens = 0usize;
    let mut warnings = Vec::<String>::new();

    for rank in ranked {
        if rank.final_score <= 0.0 && !selected_ids.is_empty() {
            continue;
        }
        let Some(chunk) = chunks.get(&rank.chunk_id) else {
            continue;
        };
        if is_redundant(&rank.chunk_id, &selected_ids, &token_sets) {
            continue;
        }
        if !try_add(
            &rank.chunk_id,
            &chunks,
            &mut selected_ids,
            &mut selected_set,
            &mut selected_tokens,
            token_budget,
        ) {
            continue;
        }
        for dep in deps_by_source
            .get(&rank.chunk_id)
            .cloned()
            .unwrap_or_default()
        {
            if let Some(target_id) = &dep.target_chunk_id {
                if !selected_set.contains(target_id)
                    && !try_add(
                        target_id,
                        &chunks,
                        &mut selected_ids,
                        &mut selected_set,
                        &mut selected_tokens,
                        token_budget,
                    )
                {
                    warnings.push(format!(
                        "Dependency not included due to budget: {} -> {} ({})",
                        rank.chunk_id, target_id, dep.relation_type
                    ));
                }
            } else if let Some(warning) = &dep.warning {
                warnings.push(warning.clone());
            }
        }
        let _ = chunk;
    }

    let mut selected = Vec::new();
    for chunk_id in &selected_ids {
        let chunk = chunks.get(chunk_id).unwrap();
        let rank = ranks.get(chunk_id);
        let source_deps = deps_by_source.get(chunk_id).cloned().unwrap_or_default();
        selected.push(SelectedContextItem {
            chunk_id: chunk.chunk_id.clone(),
            source_path: chunk.source_path.clone(),
            section_heading: chunk.section_heading.clone(),
            page_number: chunk.page_number,
            byte_start: chunk.byte_start,
            byte_end: chunk.byte_end,
            token_start: chunk.token_start,
            token_end: chunk.token_end,
            token_count: chunk.token_count,
            score: rank.map(|r| r.final_score).unwrap_or(0.0),
            reasons: rank
                .map(|r| r.reasons.clone())
                .unwrap_or_else(|| vec!["included as dependency".to_string()]),
            dependencies_included: source_deps
                .iter()
                .filter_map(|d| d.target_chunk_id.clone())
                .filter(|id| selected_set.contains(id))
                .collect(),
            dependencies_missing: source_deps
                .iter()
                .filter(|d| {
                    d.target_chunk_id
                        .as_ref()
                        .map(|id| !selected_set.contains(id))
                        .unwrap_or(true)
                })
                .map(|d| {
                    d.target_chunk_id
                        .clone()
                        .unwrap_or_else(|| d.evidence.clone())
                })
                .collect(),
            fingerprint: chunk.fingerprint.clone(),
            text: chunk.text.clone(),
        });
    }

    let mut omitted = Vec::new();
    for rank in ranked {
        if selected_set.contains(&rank.chunk_id) {
            continue;
        }
        let Some(chunk) = chunks.get(&rank.chunk_id) else {
            continue;
        };
        let mut reason = "lower ranked than selected context under token budget".to_string();
        if selected_tokens + chunk.token_count > token_budget {
            reason = "budget_limit".to_string();
        }
        if is_redundant(&rank.chunk_id, &selected_ids, &token_sets) {
            reason = "redundant_with_selected_context".to_string();
        }
        if selected_ids.iter().any(|sid| {
            let selected_chunk = chunks.get(sid).unwrap();
            selected_chunk.document_id == chunk.document_id
                && selected_chunk.chunk_index.abs_diff(chunk.chunk_index) == 1
        }) {
            reason = "nearby_relevant_context_omitted_due_to_budget".to_string();
            warnings.push(format!(
                "Nearby relevant chunk omitted: {} from {}",
                chunk.chunk_id, chunk.source_path
            ));
        }
        if deps.iter().any(|d| {
            d.target_chunk_id.as_deref() == Some(&rank.chunk_id)
                && selected_set.contains(&d.source_chunk_id)
        }) {
            reason = "dependency_not_included_due_to_budget".to_string();
        }
        omitted.push(OmittedContextItem {
            chunk_id: chunk.chunk_id.clone(),
            source_path: chunk.source_path.clone(),
            section_heading: chunk.section_heading.clone(),
            page_number: chunk.page_number,
            token_count: chunk.token_count,
            score: rank.final_score,
            reasons: rank.reasons.clone(),
            omission_reason: reason,
            fingerprint: chunk.fingerprint.clone(),
            text_preview: preview(&chunk.text),
        });
        if omitted.len() >= 20 {
            break;
        }
    }
    if selected.is_empty() {
        warnings.push("No chunks fit inside the token budget.".to_string());
    }
    let unresolved = deps.iter().filter(|d| !d.resolved).count();
    if unresolved > 0 {
        warnings.push(format!(
            "{} dependency reference(s) could not be resolved to an ingested chunk.",
            unresolved
        ));
    }
    warnings.sort();
    warnings.dedup();
    (selected, omitted, warnings)
}

fn try_add(
    chunk_id: &str,
    chunks: &HashMap<String, &DocumentChunk>,
    selected_ids: &mut Vec<String>,
    selected_set: &mut HashSet<String>,
    selected_tokens: &mut usize,
    budget: usize,
) -> bool {
    if selected_set.contains(chunk_id) {
        return false;
    }
    let Some(chunk) = chunks.get(chunk_id) else {
        return false;
    };
    if *selected_tokens + chunk.token_count > budget {
        return false;
    }
    selected_set.insert(chunk_id.to_string());
    selected_ids.push(chunk_id.to_string());
    *selected_tokens += chunk.token_count;
    true
}

fn compression_ratio(source_tokens: usize, selected_tokens: usize) -> CompressionRatio {
    if source_tokens == 0 {
        return CompressionRatio {
            source_tokens: 0,
            selected_tokens,
            tokens_saved: 0,
            selected_to_source_ratio: 1.0,
            source_to_selected_ratio: 1.0,
            reduction_pct: 0.0,
        };
    }
    let selected_ratio = selected_tokens as f64 / source_tokens as f64;
    CompressionRatio {
        source_tokens,
        selected_tokens,
        tokens_saved: source_tokens.saturating_sub(selected_tokens),
        selected_to_source_ratio: round6(selected_ratio),
        source_to_selected_ratio: round6(source_tokens as f64 / selected_tokens.max(1) as f64),
        reduction_pct: round3((1.0 - selected_ratio) * 100.0),
    }
}

fn risk_summary(
    index: &ContextIndex,
    selected_count: usize,
    selected_tokens: usize,
    omitted_relevant: usize,
    deps: &[DependencyLink],
    warnings: &[String],
) -> Value {
    let total_chunks = index.chunks.len();
    let source_tokens = index
        .chunks
        .iter()
        .map(|c| c.token_count)
        .sum::<usize>()
        .max(1);
    let unresolved = deps.iter().filter(|d| !d.resolved).count();
    let missing_dependency_warnings = warnings
        .iter()
        .filter(|w| w.contains("Dependency not included"))
        .count();
    let token_coverage = selected_tokens as f64 / source_tokens as f64;
    let chunk_coverage = selected_count as f64 / total_chunks.max(1) as f64;
    let omission_pressure =
        omitted_relevant as f64 / (selected_count + omitted_relevant).max(1) as f64;
    let dependency_pressure =
        (unresolved + missing_dependency_warnings) as f64 / deps.len().max(1) as f64;
    let coverage_score =
        (0.45 * token_coverage + 0.35 * chunk_coverage + 0.20 * (1.0 - dependency_pressure))
            .clamp(0.0, 1.0);
    let review_level = if coverage_score < 0.45 || dependency_pressure > 0.4 {
        "high"
    } else if coverage_score < 0.70 || omission_pressure > 0.25 {
        "medium"
    } else {
        "low"
    };
    serde_json::json!({
        "coverage_score": round6(coverage_score),
        "review_level": review_level,
        "selected_chunks": selected_count,
        "total_chunks": total_chunks,
        "chunk_coverage": round6(chunk_coverage),
        "token_coverage": round6(token_coverage),
        "omitted_relevant_chunks": omitted_relevant,
        "unresolved_dependency_count": unresolved,
        "missing_dependency_warning_count": missing_dependency_warnings,
        "controls": {
            "dependency_closure": if unresolved == 0 && missing_dependency_warnings == 0 { "complete" } else { "partial" },
            "omitted_evidence_pressure": if omission_pressure > 0.4 { "high" } else if omission_pressure > 0.15 { "medium" } else { "low" },
            "replayable_fingerprints": true,
            "local_no_llm_judgment": true,
        }
    })
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn round3(v: f64) -> f64 {
    (v * 1_000.0).round() / 1_000.0
}

pub fn build_receipt(
    index: &ContextIndex,
    query: &str,
    token_budget: usize,
) -> Result<ContextReceipt, serde_json::Error> {
    let ranked = rank_chunks(index, query);
    let deps = detect_dependencies(index);
    let (selected, omitted, mut warnings) = select_context(index, &ranked, &deps, token_budget);
    let source_tokens = index.chunks.iter().map(|c| c.token_count).sum::<usize>();
    let selected_tokens = selected.iter().map(|c| c.token_count).sum::<usize>();
    let relevant_omitted = omitted.iter().filter(|o| o.score > 0.0).count();
    if relevant_omitted > 0 {
        warnings.push(format!(
            "{} relevant chunk(s) were omitted; inspect omitted_context.",
            relevant_omitted
        ));
    }
    warnings.sort();
    warnings.dedup();

    let mut doc_fps = BTreeMap::new();
    for doc in &index.documents {
        doc_fps.insert(doc.source_path.clone(), doc.fingerprint.clone());
    }
    let mut chunk_fps = BTreeMap::new();
    for chunk in &index.chunks {
        chunk_fps.insert(chunk.chunk_id.clone(), chunk.fingerprint.clone());
    }
    let source_fingerprints = serde_json::json!({
        "documents": doc_fps,
        "chunks": chunk_fps,
    });
    let ranking_reasons: BTreeMap<String, Vec<String>> = ranked
        .iter()
        .map(|r| {
            let mut reasons = r.reasons.clone();
            reasons.push(format!(
                "score breakdown: lexical={:.6}, semantic={:.6}, rerank={:.6}",
                r.lexical_score, r.semantic_score, r.rerank_score
            ));
            (r.chunk_id.clone(), reasons)
        })
        .collect();
    let ratio = compression_ratio(source_tokens, selected_tokens);
    let risk_summary = risk_summary(
        index,
        selected.len(),
        selected_tokens,
        relevant_omitted,
        &deps,
        &warnings,
    );
    let hash_payload = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "token_budget": token_budget,
        "selected_context": &selected,
        "omitted_context": &omitted,
        "dependency_links": &deps,
        "ranking_reasons": &ranking_reasons,
        "compression_ratio": &ratio,
        "source_fingerprints": &source_fingerprints,
        "risk_summary": &risk_summary,
        "warnings": &warnings,
        "outcome_links": [],
    });
    let repro = stable_hash(&hash_payload)?;
    Ok(ContextReceipt {
        receipt_id: format!("cr_{}", &repro[..12]),
        schema_version: SCHEMA_VERSION.to_string(),
        query: query.to_string(),
        token_budget,
        selected_context: selected,
        omitted_context: omitted,
        dependency_links: deps,
        ranking_reasons,
        compression_ratio: ratio,
        source_fingerprints,
        risk_summary,
        warnings,
        reproducibility_hash: repro,
        outcome_links: Vec::new(),
    })
}

pub fn markdown_report(receipt: &ContextReceipt) -> String {
    let ratio = &receipt.compression_ratio;
    let mut lines = vec![
        format!("# Context Receipt {}", receipt.receipt_id),
        String::new(),
        format!("Query: `{}`", receipt.query),
        String::new(),
        "## Token Budget".to_string(),
        String::new(),
        format!("- Budget: {}", receipt.token_budget),
        format!("- Source tokens: {}", ratio.source_tokens),
        format!("- Selected tokens: {}", ratio.selected_tokens),
        format!("- Reduction: {:.1}%", ratio.reduction_pct),
        format!(
            "- Source-to-selected ratio: {:.2}:1",
            ratio.source_to_selected_ratio
        ),
        String::new(),
        "## Coverage And Risk Controls".to_string(),
        String::new(),
        format!(
            "- Coverage score: {:.3}",
            receipt.risk_summary["coverage_score"]
                .as_f64()
                .unwrap_or_default()
        ),
        format!(
            "- Review level: {}",
            receipt.risk_summary["review_level"]
                .as_str()
                .unwrap_or("unknown")
        ),
        format!(
            "- Dependency closure: {}",
            receipt.risk_summary["controls"]["dependency_closure"]
                .as_str()
                .unwrap_or("unknown")
        ),
        format!(
            "- Omitted evidence pressure: {}",
            receipt.risk_summary["controls"]["omitted_evidence_pressure"]
                .as_str()
                .unwrap_or("unknown")
        ),
        format!(
            "- Replayable fingerprints: {}",
            receipt.risk_summary["controls"]["replayable_fingerprints"]
                .as_bool()
                .unwrap_or(false)
        ),
        String::new(),
        "## Included Context".to_string(),
        String::new(),
    ];
    if receipt.selected_context.is_empty() {
        lines.push("No context chunks were selected.".to_string());
        lines.push(String::new());
    } else {
        for item in &receipt.selected_context {
            lines.push(format!("### {}", item.chunk_id));
            lines.push(format!(
                "- Source: `{}`{}",
                item.source_path,
                item.section_heading
                    .as_ref()
                    .map(|h| format!(" - {}", h))
                    .unwrap_or_default()
            ));
            lines.push(format!(
                "- Tokens: {}; score: {:.4}",
                item.token_count, item.score
            ));
            lines.push(format!("- Why: {}", item.reasons.join("; ")));
            lines.push(format!("- Fingerprint: `{}`", item.fingerprint));
            if !item.dependencies_included.is_empty() {
                lines.push(format!(
                    "- Dependencies included: {}",
                    item.dependencies_included.join(", ")
                ));
            }
            if !item.dependencies_missing.is_empty() {
                lines.push(format!(
                    "- Dependencies missing or unresolved: {}",
                    item.dependencies_missing.join(", ")
                ));
            }
            lines.push(String::new());
        }
    }
    lines.push("## Omitted Context".to_string());
    lines.push(String::new());
    if receipt.omitted_context.is_empty() {
        lines.push("No relevant omitted chunks were tracked.".to_string());
        lines.push(String::new());
    } else {
        for item in &receipt.omitted_context {
            lines.push(format!("### {}", item.chunk_id));
            lines.push(format!(
                "- Source: `{}`{}",
                item.source_path,
                item.section_heading
                    .as_ref()
                    .map(|h| format!(" - {}", h))
                    .unwrap_or_default()
            ));
            lines.push(format!(
                "- Tokens: {}; score: {:.4}",
                item.token_count, item.score
            ));
            lines.push(format!("- Why omitted: {}", item.omission_reason));
            lines.push(format!("- Ranking reason: {}", item.reasons.join("; ")));
            lines.push(format!("- Preview: {}", item.text_preview));
            lines.push(String::new());
        }
    }
    lines.push("## Dependency Graph Summary".to_string());
    lines.push(String::new());
    if receipt.dependency_links.is_empty() {
        lines.push("No explicit dependency links were detected.".to_string());
    } else {
        for link in &receipt.dependency_links {
            lines.push(format!(
                "- `{}` -> `{}` ({}, {}): {}",
                link.source_chunk_id,
                link.target_chunk_id.as_deref().unwrap_or("UNRESOLVED"),
                link.relation_type,
                if link.resolved {
                    "resolved"
                } else {
                    "unresolved"
                },
                link.evidence
            ));
        }
    }
    lines.push(String::new());
    lines.push("## Risks And Warnings".to_string());
    lines.push(String::new());
    if receipt.warnings.is_empty() {
        lines.push("- No warnings were emitted by the local heuristics.".to_string());
    } else {
        for warning in &receipt.warnings {
            lines.push(format!("- {}", warning));
        }
    }
    lines.push(String::new());
    lines.push("## Reproducibility".to_string());
    lines.push(String::new());
    lines.push(format!(
        "- Reproducibility hash: `{}`",
        receipt.reproducibility_hash
    ));
    lines.push(format!("- Schema: `{}`", receipt.schema_version));
    lines.join("\n") + "\n"
}

pub fn explain_omitted(receipt: &ContextReceipt, chunk_id: &str) -> String {
    if let Some(item) = receipt
        .omitted_context
        .iter()
        .find(|o| o.chunk_id == chunk_id)
    {
        return format!(
            "{} was omitted from {}: {}. Score={:.4}. Ranking reasons: {}. Preview: {}",
            chunk_id,
            item.source_path,
            item.omission_reason,
            item.score,
            item.reasons.join("; "),
            item.text_preview
        );
    }
    if receipt
        .selected_context
        .iter()
        .any(|s| s.chunk_id == chunk_id)
    {
        return format!(
            "{} was not omitted; it is present in selected_context.",
            chunk_id
        );
    }
    format!(
        "{} is not present in this receipt's selected or omitted context.",
        chunk_id
    )
}

fn to_py_err(err: serde_json::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction(name = "context_receipts_ingest")]
pub fn py_context_receipts_ingest(
    documents: Vec<(String, String)>,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> PyResult<String> {
    serde_json::to_string_pretty(&ingest_documents(documents, chunk_tokens, overlap_tokens))
        .map_err(to_py_err)
}

#[pyfunction(name = "context_receipts_select")]
pub fn py_context_receipts_select(
    index_json: &str,
    query: &str,
    token_budget: usize,
) -> PyResult<String> {
    let index: ContextIndex = serde_json::from_str(index_json).map_err(to_py_err)?;
    let receipt = build_receipt(&index, query, token_budget).map_err(to_py_err)?;
    serde_json::to_string_pretty(&receipt).map_err(to_py_err)
}

#[pyfunction(name = "context_receipts_run")]
pub fn py_context_receipts_run(
    documents: Vec<(String, String)>,
    query: &str,
    token_budget: usize,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> PyResult<String> {
    let index = ingest_documents(documents, chunk_tokens, overlap_tokens);
    let receipt = build_receipt(&index, query, token_budget).map_err(to_py_err)?;
    serde_json::to_string_pretty(&receipt).map_err(to_py_err)
}

#[pyfunction(name = "context_receipts_report")]
pub fn py_context_receipts_report(receipt_json: &str) -> PyResult<String> {
    let receipt: ContextReceipt = serde_json::from_str(receipt_json).map_err(to_py_err)?;
    Ok(markdown_report(&receipt))
}

#[pyfunction(name = "context_receipts_explain_omitted")]
pub fn py_context_receipts_explain_omitted(receipt_json: &str, chunk_id: &str) -> PyResult<String> {
    let receipt: ContextReceipt = serde_json::from_str(receipt_json).map_err(to_py_err)?;
    Ok(explain_omitted(&receipt, chunk_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<(String, String)> {
        vec![
            (
                "master.md".to_string(),
                "# Section 1 Definitions\n\n\"Change of Control\" means a merger or sale.\n".to_string(),
            ),
            (
                "addendum.md".to_string(),
                "# Addendum A\n\nPursuant to Section 1, the Change of Control provision is modified.\n"
                    .to_string(),
            ),
        ]
    }

    #[test]
    fn ingest_is_stable() {
        let a = ingest_documents(docs(), 80, 16);
        let b = ingest_documents(docs(), 80, 16);
        assert_eq!(a.documents[0].fingerprint, b.documents[0].fingerprint);
        assert_eq!(a.chunks[0].chunk_id, b.chunks[0].chunk_id);
    }

    #[test]
    fn receipt_detects_dependencies() {
        let index = ingest_documents(docs(), 80, 16);
        let receipt = build_receipt(
            &index,
            "Does this contract have a change-of-control clause?",
            60,
        )
        .expect("receipt");
        assert!(!receipt.selected_context.is_empty());
        assert!(receipt
            .dependency_links
            .iter()
            .any(|d| d.relation_type == "defined_term" || d.relation_type == "pursuant_to"));
        assert!(
            receipt.compression_ratio.source_tokens >= receipt.compression_ratio.selected_tokens
        );
    }
}
