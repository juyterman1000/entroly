//! QCCR — Query-Conditioned Compressive Retrieval: file-ranking core.
//!
//! Single source of truth for Entroly's retrieval *ranking*. The Python SDK /
//! MCP / proxy call this via PyO3; the WASM build exposes the same code to npm,
//! so every channel ranks identically (no Python/JS drift). The host language
//! keeps only orchestration it owns naturally (sentence extraction, the
//! edit-target reorder, config I/O) — those move here in later increments.
//!
//! Model (Robertson & Zaragoza 2009 + a linear feature layer):
//!   score(file) = w_bm25 · normalize(BM25F(body, path))
//!               + Σ_k w_k · feature_k(file, query)
//!
//! Two-field BM25F (body + path, each with independent length normalization)
//! is the backbone; the additive features are GENERAL code conventions only
//! (test/generated demotion, source/logic-dir alignment, basename hit, and
//! intent-gated structural symbol shapes). No repository-specific identifiers
//! appear here — per-repo specialization is delegated to the learned layer,
//! which supplies weight overrides. Weight DEFAULTS live here (true SSOT);
//! Python passes only overrides loaded from the tuning config.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use pyo3::prelude::*;
use regex::Regex;

// ── BM25 / BM25F constants ─────────────────────────────────────────────────
const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
const BM25F_W_BODY: f64 = 1.0;
const BM25F_W_PATH: f64 = 2.5;
const BM25F_B_PATH: f64 = 0.5;

// ── Stopwords (mirrors entroly/qccr.py _STOPWORDS for tokenizer parity) ─────
fn stopwords() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        "the a an of to in on for with by how does what why is are and or but \
         do between two include shown actual actually that this these those \
         from can could should would has have had will was were been being \
         about into through over under above below after before while when where \
         not no yes all any some most many much more less than then thus therefore \
         you your their his her its our it he she they them we us i me my"
            .split_whitespace()
            .collect()
    })
}

// ── Tokenizer (faithful to qccr.py _IDENT_RE + _split_identifier) ────────────
fn ident_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]{2,}").unwrap())
}

/// Replicates Python's `_CAMEL_RE = [A-Z]?[a-z]+ | [A-Z]+(?=[A-Z]|$) | \d+`,
/// including the acronym lookahead (which the `regex` crate cannot express),
/// e.g. "JSONParser" -> ["JSON", "Parser"], "mapXToY" -> ["map","X","To","Y"].
fn camel_pieces(tok: &str) -> Vec<String> {
    let chars: Vec<char> = tok.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_ascii_digit() {
            let s = i;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            out.push(chars[s..i].iter().collect());
        } else if c.is_ascii_uppercase() {
            if i + 1 < n && chars[i + 1].is_ascii_lowercase() {
                // [A-Z][a-z]+
                let s = i;
                i += 1;
                while i < n && chars[i].is_ascii_lowercase() {
                    i += 1;
                }
                out.push(chars[s..i].iter().collect());
            } else {
                // acronym run: consume uppercase, but stop before an uppercase
                // that begins a [A-Z][a-z] word (the lookahead boundary).
                let s = i;
                while i < n && chars[i].is_ascii_uppercase() {
                    if i > s && i + 1 < n && chars[i + 1].is_ascii_lowercase() {
                        break;
                    }
                    i += 1;
                }
                out.push(chars[s..i].iter().collect());
            }
        } else if c.is_ascii_lowercase() {
            let s = i;
            while i < n && chars[i].is_ascii_lowercase() {
                i += 1;
            }
            out.push(chars[s..i].iter().collect());
        } else {
            i += 1;
        }
    }
    out
}

fn split_identifier(tok: &str) -> Vec<String> {
    let low = tok.to_lowercase();
    let mut parts: HashSet<String> = HashSet::new();
    parts.insert(low.clone());
    for piece in low.split('_') {
        if piece.chars().count() > 2 {
            parts.insert(piece.to_string());
        }
    }
    for piece in camel_pieces(tok) {
        let p = piece.to_lowercase();
        if p.chars().count() > 2 {
            parts.insert(p);
        }
    }
    parts
        .into_iter()
        .filter(|p| !stopwords().contains(p.as_str()))
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for m in ident_re().find_iter(text) {
        out.extend(split_identifier(m.as_str()));
    }
    out
}

fn query_tokens(query: &str) -> HashSet<String> {
    tokenize(query)
        .into_iter()
        .filter(|t| t.chars().count() > 2)
        .collect()
}

fn counter(toks: &[String]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for t in toks {
        *m.entry(t.clone()).or_insert(0usize) += 1;
    }
    m
}

// ── Intent clusters (general programming vocabulary) ────────────────────────
fn intent_clusters() -> &'static HashMap<&'static str, HashSet<&'static str>> {
    static C: OnceLock<HashMap<&'static str, HashSet<&'static str>>> = OnceLock::new();
    C.get_or_init(|| {
        let mut m: HashMap<&str, HashSet<&str>> = HashMap::new();
        m.insert(
            "mapping",
            [
                "map", "mapping", "mapper", "transform", "convert", "converter",
                "serialize", "deserialize", "marshal", "unmarshal", "adapt",
                "translate", "encode", "decode", "parse", "parser",
            ]
            .into_iter()
            .collect(),
        );
        m.insert(
            "schema",
            [
                "schema", "model", "models", "type", "types", "record", "records",
                "entity", "entities", "struct", "dto", "interface",
            ]
            .into_iter()
            .collect(),
        );
        m.insert(
            "ingest",
            [
                "incoming", "request", "payload", "event", "events", "ingest",
                "ingestion", "consume", "consumer", "intake", "receive", "handler",
                "queue", "worker", "processor", "stream",
            ]
            .into_iter()
            .collect(),
        );
        m.insert(
            "persistence",
            [
                "persist", "persistence", "save", "store", "stored", "write",
                "upsert", "insert", "update", "delete", "repository",
                "repositories", "dao", "table", "tables", "database", "migration",
                "query",
            ]
            .into_iter()
            .collect(),
        );
        m
    })
}

fn cluster_links() -> &'static HashMap<&'static str, Vec<&'static str>> {
    static L: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();
    L.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("mapping", vec!["schema"]);
        m.insert("ingest", vec!["mapping", "persistence"]);
        m.insert("persistence", vec!["schema"]);
        m.insert("schema", Vec::new());
        m
    })
}

/// Conservative suffix stripper for intent matching only (persisted->persist,
/// maps->map, scores->score, items->item); never shorter than 3 chars.
fn stem(tok: &str) -> String {
    for suf in ["ings", "ing", "ied", "ies", "ed", "es", "s"] {
        if tok.ends_with(suf) && tok.chars().count().saturating_sub(suf.len()) >= 3 {
            let base = &tok[..tok.len() - suf.len()];
            return if suf == "ied" {
                format!("{base}y")
            } else {
                base.to_string()
            };
        }
    }
    tok.to_string()
}

fn stemmed_clusters() -> &'static HashMap<&'static str, HashSet<String>> {
    static C: OnceLock<HashMap<&'static str, HashSet<String>>> = OnceLock::new();
    C.get_or_init(|| {
        intent_clusters()
            .iter()
            .map(|(k, v)| (*k, v.iter().map(|t| stem(t)).collect()))
            .collect()
    })
}

fn query_intents(base: &HashSet<String>) -> HashSet<&'static str> {
    let stems: HashSet<String> = base.iter().map(|t| stem(t)).collect();
    let mut out = HashSet::new();
    for (name, vocab) in stemmed_clusters() {
        if vocab.iter().any(|v| stems.contains(v)) {
            out.insert(*name);
        }
    }
    out
}

fn expand_query(query: &str) -> HashSet<String> {
    let base = query_tokens(query);
    let mut terms: HashSet<String> = base.clone();
    for name in query_intents(&base) {
        if let Some(vocab) = intent_clusters().get(name) {
            terms.extend(vocab.iter().map(|s| s.to_string()));
        }
        for linked in cluster_links().get(name).cloned().unwrap_or_default() {
            if let Some(vocab) = intent_clusters().get(linked) {
                terms.extend(vocab.iter().map(|s| s.to_string()));
            }
        }
    }
    terms
        .into_iter()
        .filter(|t| !stopwords().contains(t.as_str()) && t.chars().count() > 2)
        .collect()
}

// ── Path + structural feature regexes (general conventions only) ────────────
macro_rules! lazy_re {
    ($name:ident, $pat:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| Regex::new($pat).unwrap())
        }
    };
}
lazy_re!(
    test_path_re,
    r"(?i)(^|/)(__tests__|tests?|specs?|e2e|fixtures?|mocks?|__mocks__)(/|$)|[._-](test|spec|stories)\."
);
lazy_re!(
    generated_path_re,
    r"(?i)/(generated|gen|dist|build|out|coverage|snapshots?|__snapshots__|vendor|migrations?|seed(?:er)?s?)/|[._-](?:generated|min)\.|(?:package-lock|pnpm-lock|yarn\.lock)"
);
lazy_re!(source_dir_re, r"(?i)/(?:src|lib|core|pkg|internal|app)/");
lazy_re!(
    logic_dir_re,
    r"(?i)/(?:services?|repositor(?:y|ies)|models?|domain|handlers?|controllers?|workers?|queues?|jobs?|processors?|usecases?|adapters?|mappers?|stores?|dao|db|database|persistence)/"
);
lazy_re!(ui_dir_re, r"(?i)/(?:components?|pages?|views?|widgets?|ui|styles?)/");
lazy_re!(
    re_mapper,
    r"\b(?:map|convert|transform|to|from)[A-Za-z0-9_]*?(?:To|From|Into)[A-Z][A-Za-z0-9_]*\b"
);
lazy_re!(
    re_transform,
    r"(?i)\b(?:serialize|deserialize|marshal|unmarshal|encode|decode|parse|format|normalize|adapt)[A-Za-z0-9_]*\b"
);
lazy_re!(
    re_persist,
    r"\b(?:insert|upsert|save|persist|write|store|create|update|delete|find|fetch|load|select)[A-Z][A-Za-z0-9_]*\b"
);
lazy_re!(
    re_schema_type,
    r"\b[A-Z][A-Za-z0-9_]*(?:InsertType|UpdateType|RecordType|Record|Schema|Model|Entity|Dto|DTO|Table|Row|Document)\b"
);
lazy_re!(
    re_sql,
    r"(?is)\b(?:INSERT\s+INTO|UPDATE\s+\w+\s+SET|SELECT\b.{0,200}?\bFROM|CREATE\s+TABLE)\b"
);

// ── Log-linear ranking weights — DEFAULTS LIVE HERE (SSOT) ──────────────────
fn default_weights() -> &'static HashMap<&'static str, f64> {
    static W: OnceLock<HashMap<&'static str, f64>> = OnceLock::new();
    W.get_or_init(|| {
        HashMap::from([
            ("bm25f", 1.0),
            ("test_penalty", -0.45),
            ("generated_penalty", -0.55),
            ("source_dir", 0.08),
            ("logic_dir_intent", 0.22),
            ("ui_penalty_for_logic", -0.30),
            ("basename_hit", 0.30),
            ("defines_mapper", 0.40),
            ("defines_transform", 0.16),
            ("defines_persistence", 0.30),
            ("defines_schema_type", 0.22),
        ])
    })
}

fn rank_features(
    source: &str,
    text: &str,
    intents: &HashSet<&str>,
    base: &HashSet<String>,
    q: &HashSet<String>,
    w: &HashMap<String, f64>,
) -> f64 {
    let gw = |k: &str| *w.get(k).unwrap_or(&0.0);
    let s = source.to_lowercase().replace('\\', "/");
    let mut adj = 0.0;
    let wants_tests = base.iter().any(|t| {
        matches!(t.as_str(), "test" | "tests" | "spec" | "assert" | "fixture" | "mock")
    });
    let backend = intents.contains("mapping")
        || intents.contains("ingest")
        || intents.contains("persistence")
        || intents.contains("schema");

    if !wants_tests && test_path_re().is_match(&s) {
        adj += gw("test_penalty");
    }
    if !wants_tests && generated_path_re().is_match(&s) {
        adj += gw("generated_penalty");
    }
    if source_dir_re().is_match(&s) {
        adj += gw("source_dir");
    }
    if backend && logic_dir_re().is_match(&s) {
        adj += gw("logic_dir_intent");
    }
    if backend && ui_dir_re().is_match(&s) {
        adj += gw("ui_penalty_for_logic");
    }

    let last = s.rsplit('/').next().unwrap_or(&s);
    let basename = last.rsplit_once('.').map(|(a, _)| a).unwrap_or(last);
    let btoks: HashSet<String> = tokenize(basename).into_iter().collect();
    if q.iter().any(|t| btoks.contains(t)) {
        adj += gw("basename_hit");
    }

    if (intents.contains("mapping") || intents.contains("schema")) && re_mapper().is_match(text) {
        adj += gw("defines_mapper");
    }
    if (intents.contains("mapping") || intents.contains("ingest")) && re_transform().is_match(text) {
        adj += gw("defines_transform");
    }
    if (intents.contains("persistence") || intents.contains("ingest"))
        && (re_persist().is_match(text) || re_sql().is_match(text))
    {
        adj += gw("defines_persistence");
    }
    if (intents.contains("schema") || intents.contains("mapping") || intents.contains("persistence"))
        && re_schema_type().is_match(text)
    {
        adj += gw("defines_schema_type");
    }
    adj
}

/// Rank files by `w_bm25 · normalize(BM25F) + Σ feature_k`. Returns
/// `(original_index, score)` sorted by score desc (ties by source desc, to
/// match Python's tuple sort). `overrides` are per-repo weight overrides from
/// the learned layer; missing keys fall back to `default_weights()`.
pub fn rank_files(
    sources: &[String],
    texts: &[String],
    query: &str,
    overrides: &HashMap<String, f64>,
) -> Vec<(usize, f64)> {
    let n = sources.len();
    if n == 0 {
        return Vec::new();
    }

    // Effective weights: defaults (SSOT) with per-repo overrides applied.
    let mut w: HashMap<String, f64> = default_weights()
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    for (k, v) in overrides {
        if w.contains_key(k) {
            w.insert(k.clone(), *v);
        }
    }

    let q = expand_query(query);
    let base = query_tokens(query);
    let intents = query_intents(&base);

    let mut body_tf: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    let mut path_tf: Vec<HashMap<String, usize>> = Vec::with_capacity(n);
    let mut body_len = vec![0f64; n];
    let mut path_len = vec![0f64; n];
    let mut df: HashMap<String, usize> = HashMap::new();
    for i in 0..n {
        let bt = counter(&tokenize(&texts[i]));
        let pt = counter(&tokenize(&sources[i].replace('\\', "/")));
        body_len[i] = bt.values().sum::<usize>() as f64;
        path_len[i] = pt.values().sum::<usize>() as f64;
        let mut seen: HashSet<&String> = HashSet::new();
        seen.extend(bt.keys());
        seen.extend(pt.keys());
        for k in seen {
            *df.entry(k.clone()).or_insert(0) += 1;
        }
        body_tf.push(bt);
        path_tf.push(pt);
    }
    let avg_body = (body_len.iter().sum::<f64>() / n as f64).max(1.0);
    let avg_path = (path_len.iter().sum::<f64>() / n as f64).max(1.0);
    let nf = n as f64;

    let mut raw = vec![0f64; n];
    for i in 0..n {
        let mut score = 0.0;
        for term in &q {
            let fb = *body_tf[i].get(term).unwrap_or(&0) as f64;
            let fp = *path_tf[i].get(term).unwrap_or(&0) as f64;
            if fb == 0.0 && fp == 0.0 {
                continue;
            }
            let dfn = *df.get(term).unwrap_or(&0) as f64;
            let idf = (1.0 + (nf - dfn + 0.5) / (dfn + 0.5)).ln();
            let ntf_b = if fb > 0.0 {
                fb / (1.0 - BM25_B + BM25_B * (body_len[i] / avg_body))
            } else {
                0.0
            };
            let ntf_p = if fp > 0.0 {
                fp / (1.0 - BM25F_B_PATH + BM25F_B_PATH * (path_len[i] / avg_path))
            } else {
                0.0
            };
            let wtf = BM25F_W_BODY * ntf_b + BM25F_W_PATH * ntf_p;
            score += idf * (wtf * (BM25_K1 + 1.0)) / (wtf + BM25_K1);
        }
        raw[i] = score;
    }
    let max_b = raw.iter().cloned().fold(0.0f64, f64::max);
    let denom = if max_b > 0.0 { max_b } else { 1.0 };
    let w_bm25 = *w.get("bm25f").unwrap_or(&1.0);

    let mut scored: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let s = w_bm25 * (raw[i] / denom)
                + rank_features(&sources[i], &texts[i], &intents, &base, &q, &w);
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| sources[b.0].cmp(&sources[a.0]))
    });
    scored
}

// ── PyO3 bindings ───────────────────────────────────────────────────────────
/// Rank files; returns `(index, score)` pairs sorted best-first.
#[pyfunction]
#[pyo3(signature = (sources, texts, query, overrides=None))]
pub fn py_qccr_rank_files(
    sources: Vec<String>,
    texts: Vec<String>,
    query: String,
    overrides: Option<HashMap<String, f64>>,
) -> Vec<(usize, f64)> {
    let ov = overrides.unwrap_or_default();
    rank_files(&sources, &texts, &query, &ov)
}

/// Expand a query into its retrieval vocabulary (sorted, deterministic). The
/// host's sentence-level scoring uses this so expansion has one source.
#[pyfunction]
pub fn py_qccr_expand_query(query: String) -> Vec<String> {
    let mut v: Vec<String> = expand_query(&query).into_iter().collect();
    v.sort();
    v
}
