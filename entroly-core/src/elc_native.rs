//! Evidence-Locked Compression native core.
//!
//! This module is designed to be wired into `lib.rs` as the Rust fast path for
//! Python ELC. It avoids regex dependencies and uses simple byte/line scoring so
//! it remains fast, deterministic, and easy to audit.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct ElcNativeOutput {
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub savings_ratio: f64,
    pub changed: bool,
    pub anchor_count: usize,
    pub omitted_spans: Vec<(usize, usize)>,
}

#[inline]
pub fn est_tokens(text: &str) -> usize {
    (text.chars().count() / 4).max(1)
}

pub fn compress_elc_native(text: &str, query: &str, budget_tokens: usize) -> ElcNativeOutput {
    let original_tokens = est_tokens(text);
    if text.trim().is_empty() || budget_tokens == 0 || original_tokens <= budget_tokens {
        return ElcNativeOutput {
            compressed: text.to_string(),
            original_tokens,
            compressed_tokens: original_tokens,
            savings_ratio: 0.0,
            changed: false,
            anchor_count: 0,
            omitted_spans: Vec::new(),
        };
    }

    let lines: Vec<&str> = text.lines().collect();
    let query_terms = clean_query_terms(query);
    let mut keep = BTreeSet::new();
    let mut anchor_count = 0usize;

    for (idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let boundary = idx < 3 || idx + 3 >= lines.len();
        let anchor = has_anchor(&lower);
        let query_hit = !query_terms.is_empty() && query_terms.iter().any(|term| lower.contains(term));
        let pathish = looks_pathish(line);
        if boundary || anchor || query_hit || pathish {
            if anchor {
                anchor_count += 1;
            }
            let start = idx.saturating_sub(2);
            let end = (idx + 2).min(lines.len().saturating_sub(1));
            for i in start..=end {
                keep.insert(i);
            }
        }
    }

    let mut used = keep.iter().map(|&i| est_tokens(lines[i])).sum::<usize>();
    if used < budget_tokens {
        let mut candidates: Vec<(usize, usize)> = lines
            .iter()
            .enumerate()
            .filter(|(i, _)| !keep.contains(i))
            .map(|(i, line)| (i, line_score(line, &query_terms)))
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        for (idx, _) in candidates {
            let cost = est_tokens(lines[idx]);
            if used + cost <= budget_tokens {
                keep.insert(idx);
                used += cost;
            }
        }
    }

    if keep.is_empty() && !lines.is_empty() {
        keep.insert(0);
        keep.insert(lines.len() - 1);
    }

    let (compressed, omitted_spans) = render_kept_lines(&lines, &keep);
    let compressed_tokens = est_tokens(&compressed);
    let savings_ratio = 1.0 - (compressed_tokens as f64 / original_tokens.max(1) as f64);
    ElcNativeOutput {
        compressed,
        original_tokens,
        compressed_tokens,
        savings_ratio,
        changed: compressed_tokens < original_tokens,
        anchor_count,
        omitted_spans,
    }
}

fn clean_query_terms(query: &str) -> Vec<String> {
    let stop = [
        "why", "what", "where", "when", "which", "build", "fail", "failed", "failure",
        "issue", "test", "tests", "log", "logs", "output", "show", "find", "debug",
        "tool", "tools", "proxy", "result", "results",
    ];
    query
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
        .filter_map(|raw| {
            let word = raw.trim().to_ascii_lowercase();
            if word.len() < 4 || stop.contains(&word.as_str()) {
                None
            } else {
                Some(word)
            }
        })
        .collect()
}

fn has_anchor(lower: &str) -> bool {
    [
        "timeout", "panic", "exception", "traceback", "warning", "denied", "refused",
        "invalid", "unexpected", "unresolved", "exit code", "assertion", "segfault",
        "not found", "no such",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn looks_pathish(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    [".py", ".rs", ".ts", ".tsx", ".js", ".java", ".kt", ".go", ".json", ".yaml", ".yml", ".toml"]
        .iter()
        .any(|ext| lower.contains(ext))
}

fn line_score(line: &str, query_terms: &[String]) -> usize {
    let lower = line.to_ascii_lowercase();
    let mut score = lower.len().min(120) / 20;
    if has_anchor(&lower) {
        score += 100;
    }
    if looks_pathish(line) {
        score += 20;
    }
    for term in query_terms {
        if lower.contains(term) {
            score += 30;
        }
    }
    score
}

fn render_kept_lines(lines: &[&str], keep: &BTreeSet<usize>) -> (String, Vec<(usize, usize)>) {
    let mut out = Vec::new();
    let mut omitted = Vec::new();
    let mut prev: Option<usize> = None;
    for &idx in keep {
        if let Some(p) = prev {
            if idx > p + 1 {
                let start = p + 2;
                let end = idx;
                out.push(format!("  ... ({} lines omitted; recoverable span {}-{})", idx - p - 1, start, end));
                omitted.push((start, end));
            }
        } else if idx > 0 {
            out.push(format!("  ... ({} lines omitted; recoverable span 1-{})", idx, idx));
            omitted.push((1, idx));
        }
        out.push(lines[idx].trim().to_string());
        prev = Some(idx);
    }
    if let Some(p) = prev {
        if p + 1 < lines.len() {
            out.push(format!("  ... ({} lines omitted; recoverable span {}-{})", lines.len() - p - 1, p + 2, lines.len()));
            omitted.push((p + 2, lines.len()));
        }
    }
    (out.join("\n"), omitted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_elc_keeps_signal_and_drops_repetition() {
        let mut lines: Vec<String> = (0..1000).map(|i| format!("[build] compiling module_{i} ok")).collect();
        lines.push("src/auth/session.rs:184:9".to_string());
        lines.push("refresh timeout after retry window".to_string());
        lines.push("hint: increase token refresh slack before retry".to_string());
        let text = lines.join("\n");
        let out = compress_elc_native(&text, "why did auth build fail", 300);
        assert!(out.changed);
        assert!(out.savings_ratio > 0.50);
        assert!(out.compressed.contains("src/auth/session.rs:184"));
        assert!(out.compressed.contains("refresh timeout"));
        assert!(out.compressed.contains("increase token refresh slack"));
        assert!(!out.compressed.contains("module_200"));
        assert!(!out.omitted_spans.is_empty());
    }
}
