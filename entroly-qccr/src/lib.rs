//! Entroly QCCR — Query-Conditioned Compressive Retrieval (single source of truth).
//!
//! Pure ranking + selection logic with NO language bindings. `entroly-core`
//! wraps it with PyO3 and `entroly-wasm` wraps it with wasm-bindgen, so Python
//! (pip/MCP/SDK), npm, and the CLI all run the exact same retrieval — the code
//! literally cannot drift. The regex engine is feature-selected: full `regex`
//! for the native/PyO3 build, dependency-free `regex-lite` for the lean WASM
//! build.
//!
//! Pipeline (Robertson & Zaragoza 2009 + Carbonell & Goldstein 1998):
//!   1. group fragments by file
//!   2. file ranking: w_bm25·normalize(BM25F(body,path)) + Σ_k w_k·feature_k
//!   3. (optional) caller-supplied reorder (engine_s6 edit-target localizer)
//!   4. budget split ∝ score; per-file sentence BM25 + entity boost + MMR
//!   5. emit excerpts, trim to a hard token budget

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[cfg(feature = "regex-full")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex-full")))]
use regex_lite::Regex;

use serde::{Deserialize, Serialize};

// ── Constants (mirror entroly/qccr.py) ──────────────────────────────────────
const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
const BM25F_W_BODY: f64 = 1.0;
const BM25F_W_PATH: f64 = 2.5;
const BM25F_B_PATH: f64 = 0.5;
const MMR_LAMBDA: f64 = 0.7;
const MIN_SENTENCE_CHARS: usize = 20;
const MAX_FILES_CONSIDERED: usize = 12;
const MAX_MMR_SENTENCE_CANDIDATES: usize = 512;
const CHARS_PER_TOKEN: usize = 4;
const ENTITY_BOOST: f64 = 1.5;

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

/// Replicates Python's `_CAMEL_RE` incl. its acronym lookahead.
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
                let s = i;
                i += 1;
                while i < n && chars[i].is_ascii_lowercase() {
                    i += 1;
                }
                out.push(chars[s..i].iter().collect());
            } else {
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

// ── Intent clusters / expansion ─────────────────────────────────────────────
fn intent_clusters() -> &'static HashMap<&'static str, HashSet<&'static str>> {
    static C: OnceLock<HashMap<&'static str, HashSet<&'static str>>> = OnceLock::new();
    C.get_or_init(|| {
        let mut m: HashMap<&str, HashSet<&str>> = HashMap::new();
        m.insert(
            "mapping",
            [
                "map",
                "mapping",
                "mapper",
                "transform",
                "convert",
                "converter",
                "serialize",
                "deserialize",
                "marshal",
                "unmarshal",
                "adapt",
                "translate",
                "encode",
                "decode",
                "parse",
                "parser",
            ]
            .into_iter()
            .collect(),
        );
        m.insert(
            "schema",
            [
                "schema",
                "model",
                "models",
                "type",
                "types",
                "record",
                "records",
                "entity",
                "entities",
                "struct",
                "dto",
                "interface",
            ]
            .into_iter()
            .collect(),
        );
        m.insert(
            "ingest",
            [
                "incoming",
                "request",
                "payload",
                "event",
                "events",
                "ingest",
                "ingestion",
                "consume",
                "consumer",
                "intake",
                "receive",
                "handler",
                "queue",
                "worker",
                "processor",
                "stream",
            ]
            .into_iter()
            .collect(),
        );
        m.insert(
            "persistence",
            [
                "persist",
                "persistence",
                "save",
                "store",
                "stored",
                "write",
                "upsert",
                "insert",
                "update",
                "delete",
                "repository",
                "repositories",
                "dao",
                "table",
                "tables",
                "database",
                "migration",
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

/// Expand a query into its retrieval vocabulary (general clusters + links).
pub fn expand_query(query: &str) -> HashSet<String> {
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

/// Sorted expansion (deterministic) for the bindings.
pub fn expand_query_sorted(query: &str) -> Vec<String> {
    let mut v: Vec<String> = expand_query(query).into_iter().collect();
    v.sort();
    v
}

// ── Feature regexes (general code conventions) ──────────────────────────────
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
lazy_re!(
    ui_dir_re,
    r"(?i)/(?:components?|pages?|views?|widgets?|ui|styles?)/"
);
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
        matches!(
            t.as_str(),
            "test" | "tests" | "spec" | "assert" | "fixture" | "mock"
        )
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
    if (intents.contains("mapping") || intents.contains("ingest")) && re_transform().is_match(text)
    {
        adj += gw("defines_transform");
    }
    if (intents.contains("persistence") || intents.contains("ingest"))
        && (re_persist().is_match(text) || re_sql().is_match(text))
    {
        adj += gw("defines_persistence");
    }
    if (intents.contains("schema")
        || intents.contains("mapping")
        || intents.contains("persistence"))
        && re_schema_type().is_match(text)
    {
        adj += gw("defines_schema_type");
    }
    adj
}

/// Rank files by `w_bm25 · normalize(BM25F) + Σ feature_k`; `(index, score)`
/// best-first (ties by source desc, matching Python's tuple sort).
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

// ── Sentence-level ──────────────────────────────────────────────────────────
fn entity_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r#"\b[A-Z][a-z]{2,}\b|\b\d+(?:[.,]\d+)*\b|['"]([^'"]{2,40})['"]"#).unwrap()
    })
}

fn query_entities(query: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for caps in entity_re().captures_iter(query) {
        let m = caps
            .get(1)
            .map(|x| x.as_str())
            .unwrap_or_else(|| caps.get(0).unwrap().as_str());
        out.insert(m.to_lowercase());
    }
    out
}

/// Faithful port of qccr.py `_split_sentences` (whose regex uses lookbehind the
/// regex crate can't express): split AFTER `[.!?]`+whitespace (whitespace run
/// consumed), and after `\n\n` / `;\n` / `}\n` (zero-width). Keep chunks whose
/// trimmed length >= MIN_SENTENCE_CHARS.
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if i + 1 < n && matches!(c, '.' | '!' | '?') && chars[i + 1].is_whitespace() {
            cur.push(c);
            chunks.push(std::mem::take(&mut cur));
            i += 1;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            continue;
        }
        cur.push(c);
        i += 1;
        if cur.ends_with("\n\n") || cur.ends_with(";\n") || cur.ends_with("}\n") {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| s.chars().count() >= MIN_SENTENCE_CHARS)
        .collect()
}

fn approx_tokens(s: &str) -> usize {
    (s.chars().count() / CHARS_PER_TOKEN).max(1)
}

type TermFrequencies = HashMap<String, usize>;
type Bm25CorpusStats = (Vec<TermFrequencies>, Vec<f64>, TermFrequencies, f64);

/// Single-field BM25 corpus stats for sentence scoring.
fn bm25_corpus(texts: &[String]) -> Bm25CorpusStats {
    let mut tf_list = Vec::with_capacity(texts.len());
    let mut lens = Vec::with_capacity(texts.len());
    let mut df: HashMap<String, usize> = HashMap::new();
    for t in texts {
        let tf = counter(&tokenize(t));
        lens.push(tf.values().sum::<usize>() as f64);
        for term in tf.keys() {
            *df.entry(term.clone()).or_insert(0) += 1;
        }
        tf_list.push(tf);
    }
    let avgdl = if lens.is_empty() {
        1.0
    } else {
        (lens.iter().sum::<f64>() / lens.len() as f64).max(1.0)
    };
    (tf_list, lens, df, avgdl)
}

fn bm25_score(
    q: &HashSet<String>,
    tf: &HashMap<String, usize>,
    dl: f64,
    df: &HashMap<String, usize>,
    n: usize,
    avgdl: f64,
) -> f64 {
    let nf = n as f64;
    let mut score = 0.0;
    for term in q {
        let f = *tf.get(term).unwrap_or(&0) as f64;
        if f == 0.0 {
            continue;
        }
        let dfn = *df.get(term).unwrap_or(&0) as f64;
        let idf = (1.0 + (nf - dfn + 0.5) / (dfn + 0.5)).ln();
        let norm = 1.0 - BM25_B + BM25_B * (dl / avgdl);
        score += idf * (f * (BM25_K1 + 1.0)) / (f + BM25_K1 * norm);
    }
    score
}

/// First-max argmax over `cands` (matches Python `max` returning the first
/// maximal element, unlike Rust's `max_by` which returns the last).
fn argmax_first(cands: &[usize], key: impl Fn(usize) -> f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for &i in cands {
        let v = key(i);
        match best {
            None => best = Some((i, v)),
            Some((_, bv)) if v > bv => best = Some((i, v)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

/// MMR sentence selection (Carbonell-Goldstein); Jaccard redundancy.
fn mmr_select(
    sentences: &[String],
    tf_list: &[HashMap<String, usize>],
    rel: &[f64],
    budget_tokens: i64,
) -> Vec<usize> {
    let mut n = sentences.len();
    if n == 0 {
        return Vec::new();
    }
    // Candidate cap for huge files.
    let mut index_map: Vec<usize> = (0..n).collect();
    let (sentences, tf_list, rel): (Vec<String>, Vec<HashMap<String, usize>>, Vec<f64>) =
        if n > MAX_MMR_SENTENCE_CANDIDATES {
            let mut ranked: Vec<usize> = (0..n).collect();
            ranked.sort_by(|&a, &b| {
                rel[b]
                    .partial_cmp(&rel[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| tf_list[b].len().cmp(&tf_list[a].len()))
                    .then_with(|| {
                        sentences[b]
                            .chars()
                            .count()
                            .cmp(&sentences[a].chars().count())
                    })
            });
            ranked.truncate(MAX_MMR_SENTENCE_CANDIDATES);
            ranked.sort_unstable();
            index_map = ranked.clone();
            n = ranked.len();
            (
                ranked.iter().map(|&i| sentences[i].clone()).collect(),
                ranked.iter().map(|&i| tf_list[i].clone()).collect(),
                ranked.iter().map(|&i| rel[i]).collect(),
            )
        } else {
            (sentences.to_vec(), tf_list.to_vec(), rel.to_vec())
        };

    let mut remaining: Vec<usize> = (0..n).filter(|&i| rel[i] > 0.0).collect();
    if remaining.is_empty() {
        // Anchor fallback: pack the longest sentences that fit.
        let mut by_len: Vec<usize> = (0..n).collect();
        by_len.sort_by(|&a, &b| {
            sentences[b]
                .chars()
                .count()
                .cmp(&sentences[a].chars().count())
        });
        let mut out = Vec::new();
        let mut used: i64 = 0;
        for i in by_len {
            let cost = approx_tokens(&sentences[i]) as i64;
            if used + cost > budget_tokens && !out.is_empty() {
                break;
            }
            out.push(i);
            used += cost;
            if used >= budget_tokens {
                break;
            }
        }
        out.sort_unstable();
        return out.into_iter().map(|i| index_map[i]).collect();
    }

    let sets: Vec<HashSet<String>> = tf_list
        .iter()
        .map(|tf| tf.keys().cloned().collect())
        .collect();
    let mut max_sim = vec![0.0f64; n];
    let mut selected: Vec<usize> = Vec::new();
    let mut budget_used: i64 = 0;

    while !remaining.is_empty() && budget_used < budget_tokens {
        let best = if selected.is_empty() {
            argmax_first(&remaining, |i| rel[i])
        } else {
            argmax_first(&remaining, |i| {
                MMR_LAMBDA * rel[i] - (1.0 - MMR_LAMBDA) * max_sim[i]
            })
        };
        let best = match best {
            Some(b) => b,
            None => break,
        };
        if rel[best] <= 0.0 {
            break;
        }
        let cost = approx_tokens(&sentences[best]) as i64;
        if budget_used + cost > budget_tokens {
            remaining.retain(|&x| x != best);
            continue;
        }
        selected.push(best);
        remaining.retain(|&x| x != best);
        budget_used += cost;
        let b = &sets[best];
        if !b.is_empty() {
            for &i in &remaining {
                let a = &sets[i];
                if a.is_empty() {
                    continue;
                }
                let inter = a.intersection(b).count();
                let union = a.union(b).count();
                if union == 0 {
                    continue;
                }
                let sim = inter as f64 / union as f64;
                if sim > max_sim[i] {
                    max_sim[i] = sim;
                }
            }
        }
    }
    selected.sort_unstable();
    selected.into_iter().map(|i| index_map[i]).collect()
}

// ── Public types + full select ──────────────────────────────────────────────
#[derive(Deserialize, Default)]
pub struct InFragment {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub content: String,
    #[serde(default = "default_feedback_multiplier")]
    pub feedback_multiplier: f64,
}

fn default_feedback_multiplier() -> f64 {
    1.0
}

#[derive(Serialize)]
pub struct OutFragment {
    pub id: String,
    pub fragment_id: String,
    pub source: String,
    pub content: String,
    pub token_count: usize,
    pub relevance: f64,
    pub relevance_score: f64,
}

/// Full QCCR selection: group → rank → optional caller reorder (`preferred`
/// source order from the engine_s6 localizer) → budget → sentence MMR → emit →
/// hard-trim. The one implementation Python and npm both call.
pub fn select(
    fragments: &[InFragment],
    token_budget: i64,
    query: &str,
    overrides: &HashMap<String, f64>,
    preferred: &[String],
) -> Vec<OutFragment> {
    if fragments.is_empty() || query.is_empty() {
        return Vec::new();
    }
    if expand_query(query).is_empty() {
        return Vec::new();
    }

    // Group by file, first-seen order (mirrors Python dict insertion order).
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    let mut feedback_by_source: HashMap<String, (f64, usize)> = HashMap::new();
    for f in fragments {
        if !groups.contains_key(&f.source) {
            order.push(f.source.clone());
        }
        groups
            .entry(f.source.clone())
            .or_default()
            .push(f.content.clone());
        let feedback = f.feedback_multiplier.clamp(0.5, 2.0);
        let entry = feedback_by_source
            .entry(f.source.clone())
            .or_insert((0.0, 0));
        entry.0 += feedback;
        entry.1 += 1;
    }
    let file_sources: Vec<String> = order.clone();
    let file_texts: Vec<String> = order.iter().map(|s| groups[s].join("\n")).collect();

    let ranked = rank_files(&file_sources, &file_texts, query, overrides);
    let mut file_scores: Vec<(f64, String, String)> = ranked
        .iter()
        .map(|&(i, sc)| {
            let source = &file_sources[i];
            let (sum, count) = feedback_by_source
                .get(source)
                .copied()
                .unwrap_or((1.0, 1));
            (sc * sum / count.max(1) as f64, source.clone(), file_texts[i].clone())
        })
        .collect();
    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Caller-supplied reorder (engine_s6 localizer) — same effect as the Python
    // `localize_files` block: reorder the candidate list, scores preserved.
    if !preferred.is_empty()
        && file_scores.len() > 1
        && file_scores.iter().any(|(s, _, _)| *s > 0.0)
    {
        let by_src: HashMap<String, (f64, String)> = file_scores
            .iter()
            .map(|(sc, src, txt)| (src.clone(), (*sc, txt.clone())))
            .collect();
        let mut reordered = Vec::new();
        for src in preferred {
            if let Some((sc, txt)) = by_src.get(src) {
                reordered.push((*sc, src.clone(), txt.clone()));
            }
        }
        if !reordered.is_empty() {
            file_scores = reordered;
        }
    }

    let mut top_files: Vec<(f64, String, String)> = file_scores
        .iter()
        .take(MAX_FILES_CONSIDERED)
        .filter(|(s, _, _)| *s > 0.0)
        .cloned()
        .collect();
    if top_files.is_empty() {
        top_files = file_scores.into_iter().take(MAX_FILES_CONSIDERED).collect();
        if top_files.is_empty() {
            return Vec::new();
        }
    }

    let total_score: f64 = {
        let s: f64 = top_files.iter().map(|(s, _, _)| *s).sum();
        if s == 0.0 {
            1.0
        } else {
            s
        }
    };
    let mut per_file_budget: HashMap<String, i64> = HashMap::new();
    for (score, src, _) in &top_files {
        let share = (token_budget as f64 * (score / total_score)) as i64;
        per_file_budget.insert(src.clone(), share.max(256));
    }

    let q_terms = expand_query(query);
    let q_ents = query_entities(query);
    let mut output: Vec<OutFragment> = Vec::new();
    let mut budget_left: i64 = token_budget;

    for (score, src, text) in &top_files {
        if budget_left <= 0 {
            break;
        }
        let mut sentences = split_sentences(text);
        if sentences.is_empty() && !text.trim().is_empty() {
            sentences = vec![text.trim().to_string()];
        }
        if sentences.is_empty() {
            continue;
        }
        let (s_tf, s_lens, s_df, s_avg) = bm25_corpus(&sentences);
        let s_n = sentences.len();
        let mut rel: Vec<f64> = (0..s_n)
            .map(|i| bm25_score(&q_terms, &s_tf[i], s_lens[i], &s_df, s_n, s_avg))
            .collect();
        if !q_ents.is_empty() {
            for (i, sent) in sentences.iter().enumerate() {
                let lower = sent.to_lowercase();
                let hits = q_ents.iter().filter(|e| lower.contains(e.as_str())).count();
                if hits > 0 {
                    rel[i] *= 1.0 + ENTITY_BOOST * hits as f64;
                }
            }
        }
        let file_budget = per_file_budget
            .get(src)
            .copied()
            .unwrap_or(256)
            .min(budget_left);
        let chosen = mmr_select(&sentences, &s_tf, &rel, file_budget);
        if chosen.is_empty() {
            continue;
        }
        let excerpt = chosen
            .iter()
            .map(|&i| sentences[i].clone())
            .collect::<Vec<_>>()
            .join("\n");
        let tokens_used = approx_tokens(&excerpt);
        let relevance = (score * 10000.0).round() / 10000.0;
        let fragment_id = format!("qccr::{src}");
        output.push(OutFragment {
            id: fragment_id.clone(),
            fragment_id,
            source: src.clone(),
            content: excerpt,
            token_count: tokens_used,
            relevance,
            relevance_score: relevance,
        });
        budget_left -= tokens_used as i64;
    }

    // Hard budget ceiling: trim trailing excerpts (drop last sentence, then
    // whole excerpts) until the emitted total fits.
    let frag_tokens = |f: &OutFragment| -> usize {
        if f.token_count > 0 {
            f.token_count
        } else {
            approx_tokens(&f.content)
        }
    };
    let mut total: i64 = output.iter().map(|f| frag_tokens(f) as i64).sum();
    while !output.is_empty() && total > token_budget {
        let last = output.last_mut().unwrap();
        let mut lines: Vec<&str> = last.content.split('\n').collect();
        if lines.len() > 1 {
            lines.pop();
            last.content = lines.join("\n");
            last.token_count = approx_tokens(&last.content);
        } else {
            output.pop();
        }
        total = output.iter().map(|f| frag_tokens(f) as i64).sum();
    }

    output
}

// ── JSON-string entry points (uniform across PyO3 + wasm-bindgen) ───────────
/// Run the full selection. `fragments_json` / `overrides_json` / `preferred_json`
/// are JSON; returns a JSON array of selected fragments.
pub fn select_json(
    fragments_json: &str,
    token_budget: i64,
    query: &str,
    overrides_json: &str,
    preferred_json: &str,
) -> String {
    let frags: Vec<InFragment> = serde_json::from_str(fragments_json).unwrap_or_default();
    let overrides: HashMap<String, f64> = serde_json::from_str(overrides_json).unwrap_or_default();
    let preferred: Vec<String> = serde_json::from_str(preferred_json).unwrap_or_default();
    let out = select(&frags, token_budget, query, &overrides, &preferred);
    serde_json::to_string(&out).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_splits_identifiers() {
        let t: HashSet<String> = tokenize("taint_flow CamelCase").into_iter().collect();
        assert!(t.contains("taint") && t.contains("flow"));
        assert!(t.contains("camel") && t.contains("case"));
    }

    #[test]
    fn camel_acronym_boundary_matches_python_lookahead() {
        assert_eq!(camel_pieces("JSONParser"), vec!["JSON", "Parser"]);
        assert_eq!(
            camel_pieces("mapTraceEventsToRecords"),
            vec!["map", "Trace", "Events", "To", "Records"]
        );
    }

    #[test]
    fn stemming_makes_intent_morphology_robust() {
        let base: HashSet<String> = query_tokens("how are scores persisted")
            .into_iter()
            .collect();
        let intents = query_intents(&base);
        assert!(intents.contains("persistence"), "persisted -> persistence");
    }

    #[test]
    fn expansion_reaches_code_vocabulary() {
        let e: HashSet<String> = expand_query("map incoming json to schema")
            .into_iter()
            .collect();
        // mapping + schema + (ingest -> mapping/persistence) clusters reachable
        assert!(e.contains("record") || e.contains("records"));
        assert!(e.contains("transform") || e.contains("mapper"));
    }

    #[test]
    fn split_sentences_breaks_on_boundaries() {
        let s = split_sentences(
            "This is a long enough sentence here. And another sufficiently long sentence follows.",
        );
        assert!(s.len() >= 2, "got {s:?}");
    }

    #[test]
    fn rank_demotes_tests_and_prefers_repository() {
        let sources = vec![
            "file:web/src/components/DatasetTable.tsx".to_string(),
            "file:server/repositories/scores.ts".to_string(),
            "file:server/repositories/scores.test.ts".to_string(),
        ];
        let texts = vec![
            "dataset scores persisted display table ".repeat(20),
            "export const upsertScore = async (s) => { await db.insert(s); };\n\
             export type ScoreRecordInsertType = {};"
                .to_string(),
            "describe('scores', () => { it('upserts', () => expect(true)); });".to_string(),
        ];
        let ranked = rank_files(
            &sources,
            &texts,
            "how are scores persisted",
            &HashMap::new(),
        );
        assert_eq!(ranked[0].0, 1, "repository should rank first");
        // the test file must not be first
        assert_ne!(ranked[0].0, 2);
    }

    #[test]
    fn select_prefers_repository_and_respects_budget() {
        let frags = vec![
            InFragment {
                source: "file:web/components/Table.tsx".into(),
                content: "dataset scores persisted display table. ".repeat(20),
                feedback_multiplier: 1.0,
            },
            InFragment {
                source: "file:server/repositories/scores.ts".into(),
                content: "export const upsertScore = async (s) => { await db.insert(s); };\n\
                          export type ScoreRecordInsertType = {}; the repository upserts scores."
                    .into(),
                feedback_multiplier: 1.0,
            },
        ];
        let out = select(
            &frags,
            300,
            "how are scores persisted",
            &HashMap::new(),
            &[],
        );
        assert!(!out.is_empty());
        assert_eq!(out[0].source, "file:server/repositories/scores.ts");
        let total: usize = out.iter().map(|f| f.token_count).sum();
        assert!(total <= 360, "budget exceeded: {total}");
    }

    #[test]
    fn empty_query_selects_nothing() {
        let frags = vec![InFragment {
            source: "a.py".into(),
            content: "def f(): pass".into(),
            feedback_multiplier: 1.0,
        }];
        assert!(select(&frags, 100, "", &HashMap::new(), &[]).is_empty());
    }

    #[test]
    fn feedback_multiplier_changes_equal_content_ordering() {
        let content = "rate limiter token bucket handles burst traffic";
        let frags = vec![
            InFragment {
                source: "failed.py".into(),
                content: content.into(),
                feedback_multiplier: 0.5,
            },
            InFragment {
                source: "successful.py".into(),
                content: content.into(),
                feedback_multiplier: 2.0,
            },
        ];

        let out = select(
            &frags,
            512,
            "rate limiter burst traffic",
            &HashMap::new(),
            &[],
        );

        assert_eq!(out[0].source, "successful.py");
        assert!(out[0].relevance > out[1].relevance);
    }
}
