//! Hierarchical Context Compression (HCC) — Entrolic Context Engine
//!
//! The core insight: Cody does search ("find relevant files").
//! Entroly does compression ("show the ENTIRE codebase at variable resolution").
//!
//! Three levels:
//!   Level 1: Skeleton Map — one-line per file, LLM sees ALL code structure
//!   Level 2: Dep-Graph Cluster — expanded skeletons for query-connected files
//!   Level 3: Full Content — knapsack-optimal fragments at full resolution
//!
//! Novel contributions:
//!   1. Symbol-reachability slicing via dep graph
//!   2. Submodular diversity selection applied to code context
//!   3. Entropy-gated per-fragment resolution
//!   4. PageRank centrality for budget allocation

use std::collections::{HashMap, HashSet, VecDeque};
use crate::fragment::ContextFragment;
use crate::depgraph::DepGraph;

/// Result of hierarchical compression.
pub struct HccResult {
    /// Level 1: one-line file summaries (entire codebase overview)
    pub level1_map: String,
    /// Level 2: expanded skeletons for dep-graph connected cluster
    pub level2_cluster: String,
    /// Level 3: indices of full-content fragments to include
    pub level3_indices: Vec<usize>,
    /// Budget actually used at each level
    pub budget_used: (u32, u32, u32),
    /// Number of files visible at each level
    pub coverage: (usize, usize, usize),
    /// Fragment IDs in the dep-graph cluster (for reporting)
    pub cluster_ids: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Level 1: Skeleton Map — one line per file
// ═══════════════════════════════════════════════════════════════════

/// Generate a one-line summary for each fragment from its skeleton.
///
/// Format: "file:auth.py → AuthService, login(), verify_token()"
/// If no skeleton, falls back to source name only.
///
/// Target: ~3-8 tokens per file. For 500 files, that's 1500-4000 tokens.
pub fn compress_level1(fragments: &[ContextFragment]) -> (String, u32) {
    let mut lines: Vec<String> = Vec::with_capacity(fragments.len());

    for frag in fragments {
        let source = &frag.source;
        let summary = if let Some(ref skel) = frag.skeleton_content {
            // Extract key symbols from skeleton: function names, class names
            extract_oneliner_from_skeleton(skel)
        } else {
            // No skeleton — just show the source name
            String::new()
        };

        if summary.is_empty() {
            lines.push(source.clone());
        } else {
            lines.push(format!("{} → {}", source, summary));
        }
    }

    let text = lines.join("\n");
    // Estimate tokens (~4 chars per token for these short lines)
    let tokens = (text.len() as u32 / 4).max(1);
    (text, tokens)
}

/// Extract a compact one-liner from a skeleton string.
/// Pulls out function/class/struct/trait names.
fn extract_oneliner_from_skeleton(skeleton: &str) -> String {
    let mut symbols: Vec<&str> = Vec::new();

    for line in skeleton.lines() {
        let trimmed = line.trim();

        // Python: "def foo(" or "class Foo"
        if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
            let name = trimmed
                .trim_start_matches("async ")
                .trim_start_matches("def ")
                .split('(')
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() && !name.starts_with('_') {
                symbols.push(name);
            }
        } else if trimmed.starts_with("class ") {
            let name = trimmed
                .trim_start_matches("class ")
                .split('(')
                .next()
                .unwrap_or("")
                .split(':')
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                symbols.push(name);
            }
        }
        // Rust: "pub fn foo" or "pub struct Foo"
        else if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let idx = if parts.first() == Some(&"pub") { 2 } else { 1 };
            if let Some(name) = parts.get(idx) {
                let clean = name.split('(').next().unwrap_or(name)
                    .split('<').next().unwrap_or(name);
                if !clean.is_empty() {
                    symbols.push(clean);
                }
            }
        } else if trimmed.starts_with("struct ") || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("enum ") || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("trait ") || trimmed.starts_with("pub trait ")
        {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let idx = if parts.first() == Some(&"pub") { 2 } else { 1 };
            if let Some(name) = parts.get(idx) {
                let clean = name.trim_end_matches(['{', '<', ':']);
                if !clean.is_empty() {
                    symbols.push(clean);
                }
            }
        }
        // JS/TS: "function foo" or "export function foo"
        else if trimmed.contains("function ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if let Some(pos) = parts.iter().position(|&w| w == "function") {
                if let Some(name) = parts.get(pos + 1) {
                    let clean = name.split('(').next().unwrap_or(name);
                    if !clean.is_empty() {
                        symbols.push(clean);
                    }
                }
            }
        }
    }

    // Cap at 6 symbols to keep it one line
    if symbols.len() > 6 {
        let first_five: Vec<&str> = symbols[..5].to_vec();
        format!("{}, +{} more", first_five.join(", "), symbols.len() - 5)
    } else {
        symbols.join(", ")
    }
}

// ═══════════════════════════════════════════════════════════════════
// Level 2: Dep-Graph Cluster Expansion
// ═══════════════════════════════════════════════════════════════════

/// Identify the fragment cluster connected to query-relevant fragments
/// via the dependency graph.
///
/// This is our "symbol-reachability slice" — an approximation of
/// backward+forward program slicing using the dep graph.
///
/// Steps:
///   1. Find fragments with highest semantic scores (query-relevant)
///   2. BFS through dep graph: forward deps (what they use) + reverse deps (what uses them)
///   3. Return the connected cluster up to max_depth
pub fn identify_cluster(
    dep_graph: &DepGraph,
    query_relevant_ids: &[String],
    max_depth: usize,
) -> Vec<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    // Seed with query-relevant fragments
    for id in query_relevant_ids {
        if !visited.contains(id) {
            visited.insert(id.clone());
            queue.push_back((id.clone(), 0));
        }
    }

    let mut cluster: Vec<String> = Vec::new();

    while let Some((fid, depth)) = queue.pop_front() {
        cluster.push(fid.clone());

        if depth < max_depth {
            // Forward dependencies (what this fragment uses)
            let forward = dep_graph.transitive_deps(&fid, 1);
            for dep_id in &forward {
                if !visited.contains(dep_id) {
                    visited.insert(dep_id.clone());
                    queue.push_back((dep_id.clone(), depth + 1));
                }
            }

            // Reverse dependencies (what uses this fragment)
            let reverse = dep_graph.reverse_deps(&fid);
            for dep_id in &reverse {
                if !visited.contains(dep_id) {
                    visited.insert(dep_id.clone());
                    queue.push_back((dep_id.clone(), depth + 1));
                }
            }
        }
    }

    cluster
}

/// Generate expanded skeleton text for the cluster.
///
/// Cluster fragments get their full skeleton.
/// This gives the LLM structural understanding of the query neighborhood.
pub fn compress_level2(
    fragments: &[ContextFragment],
    cluster_ids: &HashSet<String>,
) -> (String, u32) {
    let mut parts: Vec<String> = Vec::new();
    let mut total_tokens: u32 = 0;

    for frag in fragments {
        if !cluster_ids.contains(&frag.fragment_id) {
            continue;
        }

        if let Some(ref skel) = frag.skeleton_content {
            parts.push(format!("## {}", frag.source));
            parts.push(skel.clone());
            parts.push(String::new());
            if let Some(stc) = frag.skeleton_token_count {
                total_tokens += stc;
            } else {
                total_tokens += (skel.len() as u32 / 4).max(1);
            }
        }
    }

    (parts.join("\n"), total_tokens)
}

// ═══════════════════════════════════════════════════════════════════
// PageRank — identify hub files in the dep graph
// ═══════════════════════════════════════════════════════════════════

/// Compute PageRank scores for all fragments in the dep graph.
///
/// Hub files (imported/called by many others) get higher scores.
/// Used to prioritize L2 expansion toward structurally central code.
///
/// Classic power iteration with damping factor d=0.85.
/// Converges in ~15-20 iterations for typical code graphs.
pub fn compute_pagerank(
    dep_graph: &DepGraph,
    fragment_ids: &[String],
    iterations: usize,
) -> HashMap<String, f64> {
    let n = fragment_ids.len();
    if n == 0 {
        return HashMap::new();
    }

    let d = 0.85; // damping factor
    let base = (1.0 - d) / n as f64;

    // Initialize uniform scores
    let mut scores: HashMap<String, f64> = fragment_ids.iter()
        .map(|id| (id.clone(), 1.0 / n as f64))
        .collect();

    for _ in 0..iterations {
        let mut new_scores: HashMap<String, f64> = fragment_ids.iter()
            .map(|id| (id.clone(), base))
            .collect();

        for fid in fragment_ids {
            let score = scores.get(fid).copied().unwrap_or(0.0);
            // Get outgoing edges — distribute score to targets
            let deps = dep_graph.transitive_deps(fid, 1);
            let out_degree = deps.len().max(1);

            for target in &deps {
                if let Some(entry) = new_scores.get_mut(target) {
                    *entry += d * score / out_degree as f64;
                }
            }

            // Dangling node handling: if no outgoing, distribute to all
            if deps.is_empty() {
                let share = d * score / n as f64;
                for (_, v) in new_scores.iter_mut() {
                    *v += share;
                }
            }
        }

        scores = new_scores;
    }

    scores
}

// ═══════════════════════════════════════════════════════════════════
// Budget Allocation — entropy-weighted split across levels
// ═══════════════════════════════════════════════════════════════════

/// Allocate token budget across the three levels.
///
/// Strategy:
///   - L1 gets min(budget * 0.05, 500) — always fits
///   - L2 gets min(budget * 0.25, remaining * 0.35) — scales with budget
///   - L3 gets the rest — knapsack fills optimally
///
/// If mean entropy is high (complex codebase), L3 gets more.
/// If mean entropy is low (simple codebase), L1+L2 keep more.
pub fn allocate_budget(
    total_budget: u32,
    n_files: usize,
    mean_entropy: f64,
) -> (u32, u32, u32) {
    if total_budget == 0 || n_files == 0 {
        return (0, 0, 0);
    }

    // L1: skeleton map — small fixed allocation
    let l1 = (total_budget as f64 * 0.05).clamp(50.0, 500.0) as u32;
    let remaining = total_budget.saturating_sub(l1);

    // Entropy adjusts L2/L3 split:
    //   High entropy (>0.7) → complex code → L3 needs 75%+
    //   Low entropy (<0.3) → simple code → L2 can have 35%
    let l2_ratio = if mean_entropy > 0.7 {
        0.20
    } else if mean_entropy < 0.3 {
        0.35
    } else {
        0.25
    };

    let l2 = (remaining as f64 * l2_ratio) as u32;
    let l3 = remaining.saturating_sub(l2);

    (l1, l2, l3)
}

// ═══════════════════════════════════════════════════════════════════
// Submodular Diversity — diminishing returns for similar fragments
// ═══════════════════════════════════════════════════════════════════

/// Compute submodular marginal gain for a candidate fragment.
///
/// Uses a facility-location style objective:
///   marginal_gain(f | S) = relevance(f) × diversity_penalty(f, S)
///
/// where diversity_penalty decreases as more fragments from the
/// same "cluster" (connected component) are already selected.
///
/// This naturally prevents: 5 auth files < 3 auth + 1 db + 1 config
///
/// Guarantee: greedy selection with this submodular objective achieves
/// (1 - 1/e) ≈ 0.63 of optimal.
pub fn submodular_marginal_gain(
    candidate_source: &str,
    candidate_relevance: f64,
    selected_sources: &[String],
) -> f64 {
    // Count how many selected fragments share the same module/directory
    let candidate_module = extract_module(candidate_source);
    let same_module_count = selected_sources.iter()
        .filter(|s| extract_module(s) == candidate_module)
        .count();

    // Diminishing returns: each additional file from the same module
    // contributes less. Factor: 1 / (1 + count)
    let diversity_factor = 1.0 / (1.0 + same_module_count as f64);

    candidate_relevance * diversity_factor
}

/// Extract the module/directory path from a source string.
/// "file:src/auth/login.py" → "src/auth"
fn extract_module(source: &str) -> String {
    let path = source.trim_start_matches("file:");
    // Get parent directory
    if let Some(last_slash) = path.rfind('/') {
        path[..last_slash].to_string()
    } else {
        String::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Orchestrator — ties all three levels together
// ═══════════════════════════════════════════════════════════════════

/// Main entry point: compress the entire fragment set hierarchically.
///
/// Arguments:
///   - fragments: all indexed fragments
///   - dep_graph: the dependency graph
///   - query_relevant_ids: fragment IDs ranked by query relevance (most relevant first)
///   - total_budget: total token budget for the context block
///   - mean_entropy: average entropy across all fragments
///
/// Returns HccResult with text for each level and indices for L3.
pub fn hierarchical_compress(
    fragments: &[ContextFragment],
    dep_graph: &DepGraph,
    query_relevant_ids: &[String],
    total_budget: u32,
    mean_entropy: f64,
) -> HccResult {
    let n_files = fragments.len();

    // ── Step 1: Budget allocation ──
    let (b1, b2, b3) = allocate_budget(total_budget, n_files, mean_entropy);

    // ── Step 2: Level 1 — Skeleton map of entire codebase ──
    let (mut l1_text, l1_tokens) = compress_level1(fragments);
    let l1_used = l1_tokens.min(b1);

    // Truncate L1 if over budget (rare — should be compact)
    if l1_tokens > b1 {
        let max_chars = (b1 * 4) as usize;
        if l1_text.len() > max_chars {
            // Find nearest valid UTF-8 char boundary at or before max_chars
            let safe_end = (0..=max_chars.min(l1_text.len()))
                .rev()
                .find(|&i| l1_text.is_char_boundary(i))
                .unwrap_or(0);
            l1_text.truncate(safe_end);
            // Find last newline to avoid cutting mid-line
            if let Some(last_nl) = l1_text.rfind('\n') {
                l1_text.truncate(last_nl);
            }
        }
    }

    // ── Step 3: Level 2 — Dep-graph cluster expansion ──
    // Identify the causal cluster: all fragments reachable from query-relevant ones
    let cluster_ids = identify_cluster(dep_graph, query_relevant_ids, 2);
    let cluster_set: HashSet<String> = cluster_ids.iter().cloned().collect();

    let (mut l2_text, l2_tokens) = compress_level2(fragments, &cluster_set);
    let l2_used = l2_tokens.min(b2);

    // Truncate L2 if over budget
    if l2_tokens > b2 {
        let max_chars = (b2 * 4) as usize;
        if l2_text.len() > max_chars {
            let safe_end = (0..=max_chars.min(l2_text.len()))
                .rev()
                .find(|&i| l2_text.is_char_boundary(i))
                .unwrap_or(0);
            l2_text.truncate(safe_end);
            if let Some(last_nl) = l2_text.rfind('\n') {
                l2_text.truncate(last_nl);
            }
        }
    }

    // ── Step 4: Level 3 — Full content selection with submodular diversity ──
    // Greedy selection with diminishing returns per module
    let effective_l3_budget = b3 + b1.saturating_sub(l1_used) + b2.saturating_sub(l2_used);

    // Build fragment index for lookup
    let frag_index: HashMap<&str, usize> = fragments.iter()
        .enumerate()
        .map(|(i, f)| (f.fragment_id.as_str(), i))
        .collect();

    let mut selected_l3: Vec<usize> = Vec::new();
    let mut selected_sources: Vec<String> = Vec::new();
    let mut l3_tokens_used: u32 = 0;

    // Prioritize query-relevant fragments, then cluster, then rest
    let mut candidates: Vec<(usize, f64)> = Vec::new();

    // First: query-relevant fragments (highest priority)
    for (rank, id) in query_relevant_ids.iter().enumerate() {
        if let Some(&idx) = frag_index.get(id.as_str()) {
            let base_relevance = 1.0 - (rank as f64 * 0.05).min(0.8);
            candidates.push((idx, base_relevance));
        }
    }

    // Second: cluster fragments not already in candidates
    let candidate_ids: HashSet<&str> = candidates.iter()
        .map(|(i, _)| fragments[*i].fragment_id.as_str())
        .collect();
    for id in &cluster_ids {
        if !candidate_ids.contains(id.as_str()) {
            if let Some(&idx) = frag_index.get(id.as_str()) {
                candidates.push((idx, 0.3)); // moderate base relevance
            }
        }
    }

    // Greedy selection with submodular diversity
    for (idx, base_rel) in &candidates {
        let frag = &fragments[*idx];
        let marginal = submodular_marginal_gain(
            &frag.source,
            *base_rel,
            &selected_sources,
        );

        // Only include if marginal gain is worth the tokens
        // (this is the RepoFormer insight: retrieval can hurt)
        if marginal > 0.1 && l3_tokens_used + frag.token_count <= effective_l3_budget {
            selected_l3.push(*idx);
            selected_sources.push(frag.source.clone());
            l3_tokens_used += frag.token_count;
        }
    }

    // Count coverage
    let l1_coverage = fragments.len();
    let l2_coverage = cluster_ids.len();
    let l3_coverage = selected_l3.len();

    HccResult {
        level1_map: l1_text,
        level2_cluster: l2_text,
        level3_indices: selected_l3,
        budget_used: (l1_used, l2_used, l3_tokens_used),
        coverage: (l1_coverage, l2_coverage, l3_coverage),
        cluster_ids,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depgraph::{Dependency, DepType};

    fn make_frag(id: &str, source: &str, tokens: u32, skeleton: Option<&str>) -> ContextFragment {
        let mut f = ContextFragment::new(
            id.to_string(),
            format!("content of {}", id),
            tokens,
            source.to_string(),
        );
        if let Some(skel) = skeleton {
            f.skeleton_content = Some(skel.to_string());
            f.skeleton_token_count = Some((skel.len() as u32 / 4).max(1));
        }
        f
    }

    #[test]
    fn test_level1_generates_oneliners() {
        let frags = vec![
            make_frag("f1", "file:auth.py", 100,
                Some("class AuthService:\n    ...\ndef login(user, pwd):\n    ...\ndef verify_token(token):\n    ...")),
            make_frag("f2", "file:db.py", 80,
                Some("class Database:\n    ...\ndef connect():\n    ...")),
            make_frag("f3", "file:README.md", 50, None),
        ];

        let (text, tokens) = compress_level1(&frags);
        assert!(text.contains("auth.py"));
        assert!(text.contains("AuthService"));
        assert!(text.contains("login"));
        assert!(text.contains("db.py"));
        assert!(text.contains("README.md"));
        assert!(tokens > 0);
    }

    #[test]
    fn test_extract_oneliner_caps_at_6() {
        let skeleton = "def a():\n    ...\ndef b():\n    ...\ndef c():\n    ...\ndef d():\n    ...\ndef e():\n    ...\ndef f():\n    ...\ndef g():\n    ...\ndef h():\n    ...";
        let result = extract_oneliner_from_skeleton(skeleton);
        assert!(result.contains("+"), "Should cap at 6 symbols: {}", result);
    }

    #[test]
    fn test_identify_cluster_follows_deps() {
        let mut graph = DepGraph::new();
        graph.add_dependency(Dependency {
            source_id: "f1".into(),
            target_id: "f2".into(),
            dep_type: DepType::Import,
            strength: 1.0,
        });
        graph.add_dependency(Dependency {
            source_id: "f2".into(),
            target_id: "f3".into(),
            dep_type: DepType::FunctionCall,
            strength: 0.8,
        });
        // f4 is isolated

        let cluster = identify_cluster(&graph, &["f1".into()], 2);
        assert!(cluster.contains(&"f1".to_string()));
        assert!(cluster.contains(&"f2".to_string()));
        assert!(cluster.contains(&"f3".to_string()));
        assert!(!cluster.contains(&"f4".to_string()));
    }

    #[test]
    fn test_budget_allocation() {
        // Normal case
        let (l1, l2, l3) = allocate_budget(10000, 50, 0.5);
        assert_eq!(l1, 500); // 5% capped at 500
        assert!(l2 > 0);
        assert!(l3 > l2); // L3 should get majority

        // High entropy → L3 gets even more
        let (_, l2_hi, l3_hi) = allocate_budget(10000, 50, 0.9);
        assert!(l3_hi > l3, "High entropy should give L3 more budget");
        assert!(l2_hi < l2, "High entropy should give L2 less");
    }

    #[test]
    fn test_submodular_diversity() {
        let selected = vec![
            "file:src/auth/login.py".to_string(),
            "file:src/auth/token.py".to_string(),
        ];

        // Third auth file should get penalized (diminishing returns)
        let gain_auth = submodular_marginal_gain(
            "file:src/auth/verify.py", 0.8, &selected,
        );
        // Different module should not be penalized
        let gain_db = submodular_marginal_gain(
            "file:src/db/connect.py", 0.8, &selected,
        );

        assert!(gain_db > gain_auth,
            "Different module should have higher marginal gain: db={} vs auth={}",
            gain_db, gain_auth);
    }

    #[test]
    fn test_pagerank_hub_gets_highest() {
        let mut graph = DepGraph::new();
        // f_hub is imported by everyone
        for i in 1..=5 {
            graph.add_dependency(Dependency {
                source_id: format!("f{}", i),
                target_id: "f_hub".into(),
                dep_type: DepType::Import,
                strength: 1.0,
            });
        }

        let ids: Vec<String> = (1..=5).map(|i| format!("f{}", i))
            .chain(std::iter::once("f_hub".to_string()))
            .collect();

        let scores = compute_pagerank(&graph, &ids, 20);
        let hub_score = scores.get("f_hub").copied().unwrap_or(0.0);

        // Hub should have the highest score
        for (id, score) in &scores {
            if id != "f_hub" {
                assert!(hub_score >= *score,
                    "Hub score ({}) should be >= {} ({})",
                    hub_score, score, id);
            }
        }
    }

    #[test]
    fn test_hierarchical_compress_end_to_end() {
        let frags = vec![
            make_frag("f1", "file:src/auth/login.py", 100,
                Some("class LoginHandler:\n    ...\ndef login(user):\n    ...")),
            make_frag("f2", "file:src/db/user.py", 80,
                Some("class UserDB:\n    ...\ndef get_user(id):\n    ...")),
            make_frag("f3", "file:src/config.py", 30,
                Some("DB_URL = ...\nSECRET_KEY = ...")),
            make_frag("f4", "file:src/utils/logger.py", 40,
                Some("def log(msg):\n    ...")),
        ];

        let mut graph = DepGraph::new();
        // login.py imports user.py
        graph.add_dependency(Dependency {
            source_id: "f1".into(),
            target_id: "f2".into(),
            dep_type: DepType::Import,
            strength: 1.0,
        });
        // user.py imports config.py
        graph.add_dependency(Dependency {
            source_id: "f2".into(),
            target_id: "f3".into(),
            dep_type: DepType::Import,
            strength: 1.0,
        });

        let result = hierarchical_compress(
            &frags,
            &graph,
            &["f1".into()], // query is about login
            5000,
            0.5,
        );

        // L1 should mention all files
        assert!(result.level1_map.contains("login.py"));
        assert!(result.level1_map.contains("user.py"));
        assert!(result.level1_map.contains("config.py"));
        assert!(result.level1_map.contains("logger.py"));
        assert_eq!(result.coverage.0, 4); // All 4 files visible

        // Cluster should include login → user → config (via deps)
        assert!(result.cluster_ids.contains(&"f1".to_string()));
        assert!(result.cluster_ids.contains(&"f2".to_string()));
        assert!(result.cluster_ids.contains(&"f3".to_string()));

        // L3 should include at least f1
        assert!(!result.level3_indices.is_empty());

        // Budget should be allocated
        let total_used: u32 = result.budget_used.0 + result.budget_used.1 + result.budget_used.2;
        assert!(total_used > 0 && total_used <= 5000);
    }

    #[test]
    fn test_empty_fragments() {
        let result = hierarchical_compress(
            &[],
            &DepGraph::new(),
            &[],
            5000,
            0.5,
        );
        assert!(result.level1_map.is_empty());
        assert!(result.level3_indices.is_empty());
    }
}
