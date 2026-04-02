/// Query Analysis Engine
///
/// Provides two capabilities previously implemented in Python (query_refiner.py):
///   1. `analyze_query` – TF-IDF key-term extraction + vagueness scoring
///   2. `refine_heuristic` – reconstruct a grounded query from fragment vocabulary
///
/// Moving to Rust:
///   - TF-IDF scoring is O(N×V) — tight loops benefit from Rust speed
///   - No Python dependencies (regex, Counter, re) needed for core compute
///   - Callable from Python via PyO3 `py_analyze_query` / `py_refine_heuristic`
///
/// What stays in Python (query_refiner.py):
///   - LLM I/O (OpenAI, Anthropic) — external HTTP, must stay Python
///   - `QueryRefiner` class — thin dispatch deciding Rust vs LLM path
use std::collections::HashMap;
use serde::Serialize;

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

/// Common English stop words — filtered out before TF-IDF scoring.
/// IMPORTANT: must be sorted alphabetically for binary_search to work!
static STOP_WORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as",
    "at", "be", "before", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "him", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "may", "me", "might",
    "more", "my", "no", "not", "of", "on", "or", "our", "out", "she",
    "should", "so", "some", "that", "the", "them", "then", "there", "they",
    "this", "to", "up", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
];

/// Generic programming verbs that indicate a vague query.
static GENERIC_VERBS: &[&str] = &[
    "fix", "debug", "add", "help", "check", "look", "find", "show", "get",
    "make", "do", "run", "try", "use", "change", "update", "improve",
    "understand", "explain", "review", "refactor", "write", "work",
    "broken", "issue", "problem", "error", "wrong",
];

/// Technical terms that strongly indicate a specific, well-formed query.
static SPECIFICITY_SIGNALS: &[&str] = &[
    "cwe", "sql", "injection", "xss", "token", "auth", "oauth", "jwt",
    "endpoint", "api", "schema", "migration", "trait", "impl", "struct",
    "async", "await", "cache", "redis", "postgres", "kafka", "docker",
    "hypothesis", "bisect", "panic", "unwrap", "lifetime", "borrow",
    "memory", "unsafe", "overflow", "deadlock", "race", "lock", "mutex",
    "latency", "throughput", "benchmark", "profil", "optim",
];

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct QueryAnalysis {
    /// Vagueness score [0.0, 1.0]. 1.0 = very vague, 0.0 = highly specific.
    pub vagueness_score: f64,
    /// Top key terms extracted (TF-IDF ranked, stop-words removed).
    pub key_terms: Vec<String>,
    /// True if the query should be refined before context selection.
    pub needs_refinement: bool,
    /// Reason for the vagueness classification.
    pub reason: String,
}

// ═══════════════════════════════════════════════════════════════════
// Query Analysis
// ═══════════════════════════════════════════════════════════════════

/// Tokenize text into lowercase words, removing stop words and punctuation.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 2 && !is_stop_word(w))
        .collect()
}

fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.binary_search(&word).is_ok()
}

/// Compute TF (term frequency) for a token list.
fn tf(tokens: &[String]) -> HashMap<String, f64> {
    let n = tokens.len().max(1) as f64;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in tokens {
        *counts.entry(t.clone()).or_insert(0) += 1;
    }
    counts.into_iter()
        .map(|(k, c)| (k, c as f64 / n))
        .collect()
}

/// Compute IDF (inverse document frequency) given a corpus of token lists.
/// Uses log(N / (1 + df)) + 1 (smooth IDF).
fn idf(corpus: &[Vec<String>]) -> HashMap<String, f64> {
    let n = corpus.len() as f64;
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in corpus {
        let unique: std::collections::HashSet<&String> = doc.iter().collect();
        for t in unique {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(k, d)| (k, (n / (1.0 + d as f64)).ln() + 1.0))
        .collect()
}

/// Extract top-N key terms from a query using TF-IDF over the fragment corpus.
///
/// If `fragment_summaries` is empty, falls back to TF-only ranking.
///
/// Returns sorted (descending score) vector of term strings.
pub fn extract_key_terms(
    query: &str,
    fragment_summaries: &[String],
    top_n: usize,
) -> Vec<String> {
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return vec![];
    }

    // Build corpus: query + all fragment summaries
    let mut corpus: Vec<Vec<String>> = vec![query_tokens.clone()];
    for s in fragment_summaries {
        corpus.push(tokenize(s));
    }

    let query_tf = tf(&query_tokens);

    let scored: Vec<(String, f64)> = if corpus.len() > 1 {
        let idf_map = idf(&corpus);
        query_tf.iter()
            .map(|(term, &tf_score)| {
                let idf_score = idf_map.get(term).copied().unwrap_or(1.0);
                (term.clone(), tf_score * idf_score)
            })
            .collect()
    } else {
        // No context: use TF + length bonus (longer words = more specific)
        query_tf.iter()
            .map(|(term, &tf_score)| {
                let len_bonus = (term.len() as f64 / 10.0).min(0.5);
                (term.clone(), tf_score + len_bonus)
            })
            .collect()
    };

    let mut sorted = scored;
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(top_n);
    sorted.into_iter().map(|(t, _)| t).collect()
}

/// Score the vagueness of a query.
///
/// Algorithm:
///   vagueness = generic_verb_ratio × 0.5 + short_penalty × 0.3 − specificity_bonus × 0.2
///
///   - `generic_verb_ratio`: fraction of tokens that are generic verbs ("fix", "help", "add")
///   - `short_penalty`: queries with < 4 meaningful tokens score high vagueness
///   - `specificity_bonus`: presence of technical terms reduces vagueness
///
/// Returns (score [0.0, 1.0], reason string).
pub fn compute_vagueness(query: &str) -> (f64, String) {
    let tokens = tokenize(query);
    let n = tokens.len();

    if n == 0 {
        return (1.0, "Empty query".to_string());
    }

    // Generic verb fraction
    let generic_count = tokens.iter()
        .filter(|t| GENERIC_VERBS.iter().any(|&v| t.contains(v)))
        .count();
    let generic_ratio = generic_count as f64 / n as f64;

    // Short query penalty (stronger — short vague queries need refinement)
    let short_penalty = if n < 3 { 0.7 } else if n < 5 { 0.4 } else if n < 7 { 0.15 } else { 0.0 };

    // Specificity signals
    let specific_count = tokens.iter()
        .filter(|t| SPECIFICITY_SIGNALS.iter().any(|&s| t.contains(s)))
        .count();

    // Also check camelCase/PascalCase identifiers — those are specific
    let has_identifiers = query.split_whitespace()
        .any(|w| w.len() > 4 && w.chars().any(|c| c.is_uppercase()) && !w.chars().all(|c| c.is_uppercase()));
    let specificity_bonus = (specific_count as f64 * 0.15 + if has_identifiers { 0.2 } else { 0.0 }).min(0.4);

    let raw = (generic_ratio * 0.5 + short_penalty * 0.3 - specificity_bonus * 0.7)
        .clamp(0.0, 1.0);

    let reason = if raw >= 0.6 {
        format!(
            "Query is vague: {} of {} tokens are generic verbs ({:.0}%%). \
             Add specific symbol names, error messages, or CWE ids.",
            generic_count, n, generic_ratio * 100.0
        )
    } else if raw >= 0.35 {
        format!(
            "Query is partially specific. Consider adding: file names, function names, \
             or specific error messages. Specificity signals found: {}.",
            specific_count
        )
    } else {
        format!(
            "Query is specific ({} technical signal(s) detected). No refinement needed.",
            specific_count
        )
    };

    ((raw * 1000.0).round() / 1000.0, reason)
}

/// Full query analysis: vagueness + key term extraction.
pub fn analyze_query(query: &str, fragment_summaries: &[String]) -> QueryAnalysis {
    let (vagueness_score, reason) = compute_vagueness(query);
    let key_terms = extract_key_terms(query, fragment_summaries, 12);
    let needs_refinement = vagueness_score >= 0.45;

    QueryAnalysis {
        vagueness_score,
        key_terms,
        needs_refinement,
        reason,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Heuristic Query Refinement
// ═══════════════════════════════════════════════════════════════════

/// Refine a vague query by grounding it in vocabulary from the fragment corpus.
///
/// Algorithm:
///   1. Extract key terms from query
///   2. Find which fragments have highest vocabulary overlap with query terms
///   3. Inject top matching fragments' unique vocabulary into query
///   4. Reconstruct a more specific query string
///
/// This is a deterministic, offline refinement — no LLM call required.
/// Returns the refined query string.
pub fn refine_heuristic(query: &str, fragment_summaries: &[String]) -> String {
    if fragment_summaries.is_empty() {
        return query.to_string();
    }

    let query_terms: std::collections::HashSet<String> = tokenize(query).into_iter().collect();

    // Score each fragment by overlap with query terms
    let mut scored: Vec<(usize, usize)> = fragment_summaries.iter().enumerate()
        .map(|(i, summary)| {
            let frag_tokens: std::collections::HashSet<String> = tokenize(summary).into_iter().collect();
            let overlap = query_terms.intersection(&frag_tokens).count();
            (i, overlap)
        })
        .collect();

    scored.sort_unstable_by_key(|(_, s)| std::cmp::Reverse(*s));

    // Take top 3 most relevant fragments
    let top_n = scored.iter().take(3).filter(|(_, s)| *s > 0);

    // Extract unique terms from top fragments not already in the query
    let mut enrichment_terms: Vec<String> = Vec::new();
    for (idx, _) in top_n {
        let frag_tokens = tokenize(&fragment_summaries[*idx]);
        for token in frag_tokens {
            if !query_terms.contains(&token)
                && token.len() >= 4
                && !enrichment_terms.contains(&token)
                && enrichment_terms.len() < 5
            {
                // Prefer tokens that look like code identifiers
                if token.contains('_') || token.chars().next().is_some_and(|c| c.is_alphabetic()) {
                    enrichment_terms.push(token);
                }
            }
        }
    }

    if enrichment_terms.is_empty() {
        return query.to_string();
    }

    // Reconstruct: original query + grounding context
    format!(
        "{} [context: {}]",
        query,
        enrichment_terms.join(", ")
    )
}



// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vague_query_high_score() {
        let (score, _) = compute_vagueness("fix the bug");
        assert!(score >= 0.35, "Short generic query should be vague, got {}", score);
    }

    #[test]
    fn test_specific_query_low_score() {
        let (score, _) = compute_vagueness("SQL injection in cursor.execute via request.args CWE-89");
        assert!(score < 0.4, "Specific query with CWE and SQL should be low vagueness, got {}", score);
    }

    #[test]
    fn test_key_term_extraction_removes_stopwords() {
        let terms = extract_key_terms("the function is broken in the auth module", &[], 10);
        assert!(!terms.contains(&"the".to_string()));
        assert!(!terms.contains(&"is".to_string()));
        assert!(!terms.contains(&"in".to_string()));
    }

    #[test]
    fn test_key_terms_prefers_specific_tokens() {
        let terms = extract_key_terms("fix the sql injection in cursor.execute", &[], 10);
        assert!(terms.iter().any(|t| t.contains("sql") || t.contains("cursor") || t.contains("injection")));
    }

    #[test]
    fn test_refinement_adds_context_terms() {
        let summaries = vec![
            "cursor.execute query parameterized sql injection".to_string(),
            "request.args user_id validation sanitization".to_string(),
        ];
        let refined = refine_heuristic("fix the sql bug", &summaries);
        // Should add grounding terms
        assert!(refined.len() > "fix the sql bug".len());
        assert!(refined.contains("[context:"));
    }

    #[test]
    fn test_refinement_noop_on_empty_fragments() {
        let refined = refine_heuristic("fix the bug", &[]);
        assert_eq!(refined, "fix the bug");
    }

    #[test]
    fn test_analyze_query_full() {
        let analysis = analyze_query(
            "XSS via dangerouslySetInnerHTML in UserCard component",
            &[],
        );
        assert!(analysis.vagueness_score < 0.5);
        assert!(!analysis.needs_refinement);
        assert!(!analysis.key_terms.is_empty());
    }

    #[test]
    fn test_analyze_query_vague_needs_refinement() {
        let analysis = analyze_query("help me fix this", &[]);
        assert!(analysis.needs_refinement);
    }

    #[test]
    fn test_stopwords_filtered() {
        let tokens = tokenize("the quick brown fox");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
    }

    #[test]
    fn test_idf_penalizes_common_terms() {
        let corpus = vec![
            tokenize("auth token expired"),
            tokenize("auth middleware check"),
            tokenize("auth flow broken"),
            tokenize("token refresh endpoint"),
        ];
        let idf_map = idf(&corpus);
        // "auth" appears in 3/4 docs — lower IDF than "expired"
        let auth_idf = idf_map.get("auth").copied().unwrap_or(0.0);
        let expired_idf = idf_map.get("expired").copied().unwrap_or(0.0);
        assert!(expired_idf >= auth_idf, "Rare term should have higher IDF");
    }
}
