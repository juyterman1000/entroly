//! BM25 Retrieval Scoring — replaces SimHash Hamming distance for query-document relevance.
//!
//! # Why BM25?
//! SimHash Hamming distance measures document similarity (near-duplicate detection).
//! It was never designed for query-document relevance — a 15-word query has a completely
//! different term distribution than a 500-line source file, producing random noise (~0.45±0.03).
//!
//! BM25 is the gold standard for term-based retrieval.
//! It answers: "how relevant is this document to this query?" using:
//!   - Term Frequency (TF): saturating — diminishing returns for repeated terms
//!   - Inverse Document Frequency (IDF): rare terms matter more
//!   - Document Length Normalization: long documents don't get unfair advantage
//!
//! # Additional signals (beyond standard BM25):
//!   - **Path Boosting**: query term in file path → strong relevance signal
//!   - **Identifier Matching**: query term matches extracted code identifiers (class, fn names)
//!   - **Camel/Snake Split**: "StateGraph" matches "state" and "graph" individually
//!
//! # Complexity: O(Q × D × L) where Q=query terms, D=documents, L=avg doc length
//! For 500 documents × 10 query terms: ~5000 string scans, <5ms in Rust.

use std::collections::HashMap;

/// BM25 parameters (well-studied defaults)
const K1: f64 = 1.2;    // Term frequency saturation. Higher = more weight to repeated terms.
const B: f64 = 0.75;     // Length normalization. 0 = no normalization, 1 = full normalization.

/// Bonus multipliers for structural signals
const PATH_MATCH_BOOST: f64 = 2.5;       // Query term found in file path
const IDENTIFIER_MATCH_BOOST: f64 = 1.8;  // Query term matches a code identifier
const EXACT_FILENAME_BOOST: f64 = 3.0;    // Query term IS the filename (minus extension)

/// Pre-computed corpus statistics for BM25 IDF calculation.
pub struct BM25Index {
    /// Total number of documents in the corpus.
    pub num_docs: usize,
    /// Average document length (in tokens/terms) across the corpus.
    pub avg_dl: f64,
    /// Document frequency: how many documents contain each term.
    pub df: HashMap<String, usize>,
}

/// Per-document BM25 score breakdown (for explainability).
#[derive(Debug, Clone)]
pub struct BM25Score {
    /// Fraction of query terms that appear in this document [0, 1].
    /// High coverage = document addresses multiple aspects of the query.
    pub query_coverage: f64,
    pub bm25_base: f64,
    pub path_boost: f64,
    pub identifier_boost: f64,
    pub combined: f64,
}

impl BM25Index {
    /// Build a BM25 index from a collection of (document_id, content, source_path) tuples.
    ///
    /// This is O(N × L) where N = documents, L = avg document length.
    /// For 500 documents: ~1-2ms in Rust.
    pub fn build(documents: &[(String, String, String)]) -> Self {
        let num_docs = documents.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_terms = 0usize;

        for (_id, content, source) in documents {
            // Tokenize content + source path
            let tokens = tokenize_code(content);
            let path_tokens = tokenize_path(source);
            total_terms += tokens.len();

            // Count unique terms per document (for DF)
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for t in tokens.iter().chain(path_tokens.iter()) {
                if seen.insert(t.clone()) {
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }

        let avg_dl = if num_docs > 0 {
            total_terms as f64 / num_docs as f64
        } else {
            1.0
        };

        BM25Index { num_docs, avg_dl, df }
    }

    /// Compute IDF for a single term:
    ///   IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    ///
    /// This gives high scores to rare terms and low (but non-negative) scores to common terms.
    #[inline]
    pub fn idf(&self, term: &str) -> f64 {
        let n = self.num_docs as f64;
        let df = self.df.get(term).copied().unwrap_or(0) as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score a single document against a query.
    ///
    /// Returns a BM25Score with the base BM25 score plus structural boosts.
    pub fn score(
        &self,
        query_terms: &[String],
        content: &str,
        source_path: &str,
        identifiers: &[String],
    ) -> BM25Score {
        let doc_tokens = tokenize_code(content);
        let doc_len = doc_tokens.len() as f64;

        // Build term frequency map for this document
        let mut tf_map: HashMap<&str, usize> = HashMap::new();
        for t in &doc_tokens {
            *tf_map.entry(t.as_str()).or_insert(0) += 1;
        }

        // Also tokenize the path for path matching
        let path_lower = source_path.to_lowercase();
        let path_tokens = tokenize_path(source_path);

        // Extract filename without extension for exact match
        let filename = source_path
            .rsplit(&['/', '\\'][..])
            .next()
            .unwrap_or("")
            .rsplit_once('.')
            .map(|(name, _)| name.to_lowercase())
            .unwrap_or_default();

        // Build identifier set (lowercased) for identifier matching
        let id_set: std::collections::HashSet<String> = identifiers
            .iter()
            .flat_map(|id| split_identifier(id))
            .collect();

        let mut bm25_base = 0.0;
        let mut path_boost = 0.0;
        let mut identifier_boost = 0.0;
        let mut terms_matched: usize = 0;
        let total_query_terms = query_terms.len().max(1);

        // Pre-compute query term parts for coverage checking
        let qt_parts_cache: HashMap<String, Vec<String>> = query_terms.iter()
            .map(|qt| (qt.to_lowercase(), split_identifier(qt)))
            .collect();

        for qt in query_terms {
            let qt_lower = qt.to_lowercase();

            // ── Standard BM25 ──
            let raw_tf = tf_map.get(qt_lower.as_str()).copied().unwrap_or(0) as f64;
            let idf = self.idf(&qt_lower);

            // BM25 TF component: tf × (k1 + 1) / (tf + k1 × (1 - b + b × dl/avgdl))
            let tf_component = if raw_tf > 0.0 {
                (raw_tf * (K1 + 1.0)) /
                (raw_tf + K1 * (1.0 - B + B * doc_len / self.avg_dl.max(1.0)))
            } else {
                0.0
            };

            // Track coverage: does this document contain this query term at all?
            // (in content, path, or identifiers)
            let in_content = raw_tf > 0.0;
            let in_path = path_lower.contains(&qt_lower) || path_tokens.contains(&qt_lower);
            let in_ids = qt_parts_cache.get(&qt_lower).map(|parts| {
                parts.iter().any(|p| id_set.contains(p))
            }).unwrap_or(false);
            if in_content || in_path || in_ids {
                terms_matched += 1;
            }

            bm25_base += idf * tf_component;

            // ── Path Boost ──
            // If the query term appears in the file path, it's a strong signal.
            // "StateGraph" query on "graph/state.py" should rank higher.
            if path_lower.contains(&qt_lower) || path_tokens.contains(&qt_lower) {
                path_boost += idf * PATH_MATCH_BOOST;
            }

            // Exact filename match (strongest signal)
            if !filename.is_empty() && filename == qt_lower {
                path_boost += idf * EXACT_FILENAME_BOOST;
            }

            // ── Identifier Boost ──
            // If the query term matches a code identifier (class name, function name),
            // boost relevance. Split camelCase/snake_case for partial matching.
            let qt_parts = split_identifier(&qt_lower);
            for part in &qt_parts {
                if id_set.contains(part) {
                    identifier_boost += idf * IDENTIFIER_MATCH_BOOST;
                    break; // One boost per query term
                }
            }
        }

        // ── Query Coverage Bonus ──
        // Documents matching MORE unique query terms get a superlinear bonus.
        // This prevents a file with high TF for one common term from outranking
        // a file that matches 4/5 query terms with moderate TF each.
        //
        // The coverage function c(k, n) = (k/n)^1.5 rewards breadth of matching.
        // At coverage 1.0 (all terms): bonus = +50% of base BM25
        // At coverage 0.5 (half terms): bonus = +18% of base BM25
        // At coverage 0.2 (one term):   bonus = +4% of base BM25
        let coverage = terms_matched as f64 / total_query_terms as f64;
        let coverage_bonus = coverage.powf(1.5) * 0.5 * bm25_base;

        // Combine: BM25 base + structural signals + coverage bonus
        let combined = bm25_base + path_boost + identifier_boost + coverage_bonus;

        let out = BM25Score {
            query_coverage: coverage,
            bm25_base,
            path_boost,
            identifier_boost,
            combined,
        };

        // Decomposition invariant:
        //   combined = bm25_base + path_boost + identifier_boost + coverage^1.5 * 0.5 * bm25_base
        // Asserting here both guards against scoring-formula drift and ensures every
        // explainability field is a load-bearing part of the final score.
        debug_assert!({
            let expected = out.bm25_base
                + out.path_boost
                + out.identifier_boost
                + out.query_coverage.powf(1.5) * 0.5 * out.bm25_base;
            (out.combined - expected).abs() < 1e-9
        });

        out
    }
}

/// Tokenize code content into lowercase terms.
///
/// Splits on non-alphanumeric boundaries, handles camelCase and snake_case,
/// filters terms < 2 chars and common language keywords.
pub fn tokenize_code(content: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    // First pass: split on whitespace and punctuation
    for word in content.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.len() < 2 {
            continue;
        }

        let lower = word.to_lowercase();

        // Skip single-char tokens and very common programming keywords
        if is_code_stopword(&lower) {
            continue;
        }

        // Split camelCase: "StateGraph" → ["state", "graph"]
        let parts = split_identifier(word);
        if parts.len() > 1 {
            // Add both the full token and its parts
            tokens.push(lower);
            for part in parts {
                if part.len() >= 2 && !is_code_stopword(&part) {
                    tokens.push(part);
                }
            }
        } else {
            tokens.push(lower);
        }
    }

    tokens
}

/// Tokenize a file path into meaningful terms.
///
/// "libs/langgraph/graph/state.py" → ["libs", "langgraph", "graph", "state"]
pub fn tokenize_path(path: &str) -> Vec<String> {
    path.split(&['/', '\\', '.', '-', '_'][..])
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_lowercase())
        .filter(|s| !matches!(s.as_str(), "py" | "rs" | "ts" | "js" | "md" | "txt" |
                                           "json" | "yaml" | "yml" | "toml" | "cfg" |
                                           "src" | "lib" | "file"))
        .collect()
}

/// Split a camelCase or PascalCase identifier into lowercase parts.
///
/// "StateGraph" → ["state", "graph"]
/// "add_conditional_edges" → ["add", "conditional", "edges"]
/// "HTMLParser" → ["html", "parser"]
pub fn split_identifier(id: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();

    // First split on underscores
    for segment in id.split('_') {
        if segment.is_empty() {
            continue;
        }

        // Then split camelCase within each segment
        let chars: Vec<char> = segment.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if i > 0 && ch.is_uppercase() {
                // Check for acronyms: "HTMLParser" → don't split "HTML"
                let prev_upper = chars[i - 1].is_uppercase();
                let next_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());

                if !prev_upper || next_lower {
                    if !current.is_empty() {
                        parts.push(current.to_lowercase());
                    }
                    current = String::new();
                }
            }
            current.push(ch);
        }

        if !current.is_empty() {
            parts.push(current.to_lowercase());
            current = String::new();
        }
    }

    parts
}

/// Common programming keywords that don't help with retrieval.
fn is_code_stopword(word: &str) -> bool {
    matches!(word,
        // Python
        "def" | "class" | "import" | "from" | "return" | "self" | "none" | "true" | "false" |
        "if" | "else" | "elif" | "for" | "while" | "try" | "except" | "with" | "as" |
        "and" | "or" | "not" | "in" | "is" | "pass" | "break" | "continue" | "yield" |
        "lambda" | "raise" | "finally" | "assert" | "del" | "global" | "nonlocal" |
        // Rust
        "fn" | "let" | "mut" | "pub" | "use" | "mod" | "struct" | "enum" | "impl" |
        "trait" | "where" | "match" | "ref" | "move" | "async" | "await" | "unsafe" |
        "crate" | "super" | "type" | "const" | "static" |
        // General
        "var" | "new" | "this" | "null" | "void" | "int" | "str" | "bool" |
        "the" | "are" | "but" | "was" | "all" | "any" | "can" |
        "had" | "her" | "his" | "how" | "its" | "may" | "our" | "out" | "too" | "who" |
        "has" | "have" | "that" | "then" | "than" | "these" | "those" | "does" | "done"
    )
}

/// Normalize BM25 scores to [0.0, 1.0] range for integration with existing PRISM scoring.
///
/// Uses min-max normalization with a floor to prevent zero scores:
///   normalized = (score - min) / (max - min)  clamped to [0.05, 1.0]
///
/// The floor of 0.05 ensures that even irrelevant files get a tiny non-zero score,
/// preventing division-by-zero in downstream scoring.
pub fn normalize_scores(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }

    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);

    let range = max - min;
    if range < 1e-10 {
        // All scores are the same — return uniform 0.5
        return vec![0.5; scores.len()];
    }

    scores.iter()
        .map(|&s| ((s - min) / range * 0.95 + 0.05).clamp(0.05, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_identifier_camel() {
        assert_eq!(split_identifier("StateGraph"), vec!["state", "graph"]);
    }

    #[test]
    fn test_split_identifier_snake() {
        assert_eq!(split_identifier("add_conditional_edges"), vec!["add", "conditional", "edges"]);
    }

    #[test]
    fn test_split_identifier_acronym() {
        let parts = split_identifier("HTMLParser");
        assert_eq!(parts, vec!["html", "parser"]);
    }

    #[test]
    fn test_tokenize_path() {
        let tokens = tokenize_path("libs/langgraph/graph/state.py");
        assert!(tokens.contains(&"langgraph".to_string()));
        assert!(tokens.contains(&"graph".to_string()));
        assert!(tokens.contains(&"state".to_string()));
        assert!(!tokens.contains(&"py".to_string()));
    }

    #[test]
    fn test_bm25_relevance_ordering() {
        let docs = vec![
            ("1".into(), "class StateGraph:\n  def add_node(self):\n    pass\n  def add_edge(self):\n    pass".into(), "graph/state.py".into()),
            ("2".into(), "import os\nDATABASE_URL = 'sqlite:///test.db'".into(), "config.py".into()),
            ("3".into(), "class MemoryStore:\n  def get(self): pass\n  def set(self): pass".into(), "store/memory.py".into()),
        ];

        let index = BM25Index::build(&docs);
        let query_terms: Vec<String> = vec!["state".into(), "graph".into(), "add_node".into(), "add_edge".into()];

        let s1 = index.score(&query_terms, &docs[0].1, &docs[0].2, &["StateGraph".into(), "add_node".into(), "add_edge".into()]);
        let s2 = index.score(&query_terms, &docs[1].1, &docs[1].2, &[]);
        let s3 = index.score(&query_terms, &docs[2].1, &docs[2].2, &["MemoryStore".into()]);

        // state.py MUST score higher than config.py for a "StateGraph" query
        assert!(s1.combined > s2.combined, "state.py ({}) should rank above config.py ({})", s1.combined, s2.combined);
        assert!(s1.combined > s3.combined, "state.py ({}) should rank above memory.py ({})", s1.combined, s3.combined);

        // Score decomposition invariant (coverage-augmented BM25):
        //   combined = bm25_base + path_boost + identifier_boost + coverage^1.5 * 0.5 * bm25_base
        // This guards against silent regressions in the explainability breakdown.
        for s in [&s1, &s2, &s3] {
            let expected = s.bm25_base
                + s.path_boost
                + s.identifier_boost
                + s.query_coverage.powf(1.5) * 0.5 * s.bm25_base;
            assert!(
                (s.combined - expected).abs() < 1e-9,
                "decomposition broken: combined={} vs sum={} (base={}, path={}, id={}, cov={})",
                s.combined, expected, s.bm25_base, s.path_boost, s.identifier_boost, s.query_coverage,
            );
            assert!((0.0..=1.0).contains(&s.query_coverage));
        }
    }

    #[test]
    fn test_normalize_scores() {
        let scores = vec![0.0, 5.0, 10.0];
        let norm = normalize_scores(&scores);
        assert!(norm[0] < 0.1);  // min → ~0.05
        assert!(norm[2] > 0.9);  // max → ~1.0
        assert!(norm[1] > norm[0] && norm[1] < norm[2]); // monotone
    }
}
