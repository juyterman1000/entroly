//! Falsification harness for scalable Work Graph lease candidate generation.
//!
//! This module is test-only until the indexed candidate generator has proven
//! semantic equivalence to the current all-pairs implementation. The exact
//! overlap functions in `work_graph` remain authoritative in production.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
struct LeaseScope {
    agent: String,
    paths: Vec<String>,
    symbols: Vec<String>,
}

fn normalize_scope_path(path: &str) -> String {
    path.replace('\\', "/").trim_matches('/').to_string()
}

/// Return only prefixes that can themselves be normalized lease paths under
/// the production `paths_overlap` rule. Repeated separators are preserved:
/// `src//auth` has the ancestor `src` and the full path `src//auth`; it must not
/// silently become `src/auth` in the index.
fn path_prefixes(path: &str) -> Vec<String> {
    let normalized = normalize_scope_path(path);
    if normalized.is_empty() {
        return Vec::new();
    }
    let mut prefixes = Vec::new();
    for (offset, ch) in normalized.char_indices() {
        if ch != '/' || offset == 0 {
            continue;
        }
        let prefix = &normalized[..offset];
        if !prefix.ends_with('/') {
            prefixes.push(prefix.to_string());
        }
    }
    prefixes.push(normalized);
    prefixes
}

fn paths_overlap(left: &str, right: &str) -> bool {
    let left = normalize_scope_path(left);
    let right = normalize_scope_path(right);
    if left.is_empty() || right.is_empty() {
        return false;
    }
    left == right
        || left.starts_with(&(right.clone() + "/"))
        || right.starts_with(&(left.clone() + "/"))
}

fn scopes_overlap(left: &LeaseScope, right: &LeaseScope) -> bool {
    left.paths
        .iter()
        .any(|a| right.paths.iter().any(|b| paths_overlap(a, b)))
        || left
            .symbols
            .iter()
            .any(|symbol| right.symbols.iter().any(|other| symbol == other))
}

fn naive_pairs(leases: &[LeaseScope]) -> BTreeSet<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    for left in 0..leases.len() {
        for right in (left + 1)..leases.len() {
            if leases[left].agent != leases[right].agent
                && scopes_overlap(&leases[left], &leases[right])
            {
                pairs.insert((left, right));
            }
        }
    }
    pairs
}

/// Generate only lease pairs that *may* overlap.
///
/// For paths, two scopes overlap exactly when one normalized path is an equal
/// or segment-boundary prefix of the other. `exact_paths` finds previously
/// inserted ancestors; `descendants` maps every valid literal prefix to paths
/// below it and finds previously inserted descendants. Symbols use an exact
/// inverted index. The production overlap functions still perform the final
/// decision after this candidate filter is promoted.
fn indexed_pairs(leases: &[LeaseScope]) -> BTreeSet<(usize, usize)> {
    let mut exact_paths: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let mut descendants: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let mut symbols: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let mut pairs = BTreeSet::new();

    for (index, lease) in leases.iter().enumerate() {
        let mut candidates = BTreeSet::new();

        for path in &lease.paths {
            let normalized = normalize_scope_path(path);
            if normalized.is_empty() {
                continue;
            }
            for prefix in path_prefixes(&normalized) {
                if let Some(indices) = exact_paths.get(&prefix) {
                    candidates.extend(indices.iter().copied());
                }
            }
            if let Some(indices) = descendants.get(&normalized) {
                candidates.extend(indices.iter().copied());
            }
        }

        for symbol in &lease.symbols {
            if symbol.is_empty() {
                continue;
            }
            if let Some(indices) = symbols.get(symbol) {
                candidates.extend(indices.iter().copied());
            }
        }

        for previous in candidates {
            if leases[previous].agent != lease.agent {
                pairs.insert((previous, index));
            }
        }

        for path in &lease.paths {
            let normalized = normalize_scope_path(path);
            if normalized.is_empty() {
                continue;
            }
            exact_paths
                .entry(normalized.clone())
                .or_default()
                .insert(index);
            for prefix in path_prefixes(&normalized) {
                descendants.entry(prefix).or_default().insert(index);
            }
        }
        for symbol in &lease.symbols {
            if !symbol.is_empty() {
                symbols.entry(symbol.clone()).or_default().insert(index);
            }
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn pick<'a>(&mut self, values: &'a [&'a str]) -> &'a str {
            values[(self.next() as usize) % values.len()]
        }

        fn bounded(&mut self, limit: usize) -> usize {
            (self.next() as usize) % limit
        }
    }

    fn random_scope(rng: &mut Lcg, index: usize) -> LeaseScope {
        const ROOTS: &[&str] = &["src", "tests", "docs", "crates", "packages"];
        const AREAS: &[&str] = &["auth", "cache", "graph", "mcp", "router", "memory"];
        const FILES: &[&str] = &["mod.rs", "api.rs", "state.py", "index.js", "types.ts"];
        const SYMBOLS: &[&str] = &[
            "Auth.refresh",
            "Cache.lookup",
            "Graph.resume",
            "Mcp.serve",
            "Router.route",
            "Memory.recall",
        ];

        let path_count = rng.bounded(4);
        let mut paths = BTreeSet::new();
        for _ in 0..path_count {
            let root = rng.pick(ROOTS);
            let area = rng.pick(AREAS);
            let depth = 1 + rng.bounded(3);
            let path = match depth {
                1 => root.to_string(),
                2 => format!("{root}/{area}"),
                _ => format!("{root}/{area}/{}", rng.pick(FILES)),
            };
            paths.insert(path);
        }

        let symbol_count = rng.bounded(3);
        let mut symbols = BTreeSet::new();
        for _ in 0..symbol_count {
            symbols.insert(rng.pick(SYMBOLS).to_string());
        }

        LeaseScope {
            // Reuse some identities deliberately: same-agent overlaps must not
            // become conflicts in either implementation.
            agent: format!("agent-{}", (index + rng.bounded(5)) % 7),
            paths: paths.into_iter().collect(),
            symbols: symbols.into_iter().collect(),
        }
    }

    #[test]
    fn indexed_candidates_match_naive_oracle_on_randomized_scopes() {
        for seed in 0..2_000_u64 {
            let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
            let count = 1 + rng.bounded(48);
            let leases: Vec<_> = (0..count)
                .map(|index| random_scope(&mut rng, index))
                .collect();
            assert_eq!(
                indexed_pairs(&leases),
                naive_pairs(&leases),
                "candidate mismatch for deterministic seed {seed}"
            );
        }
    }

    #[test]
    fn indexed_candidates_handle_ancestor_descendant_and_exact_symbol_cases() {
        let leases = vec![
            LeaseScope {
                agent: "claude".into(),
                paths: vec!["src/auth".into()],
                symbols: vec![],
            },
            LeaseScope {
                agent: "codex".into(),
                paths: vec!["src/auth/token.rs".into()],
                symbols: vec![],
            },
            LeaseScope {
                agent: "kimi".into(),
                paths: vec!["docs/auth".into()],
                symbols: vec!["Auth.refresh".into()],
            },
            LeaseScope {
                agent: "copilot".into(),
                paths: vec!["packages/web".into()],
                symbols: vec!["Auth.refresh".into()],
            },
        ];
        let expected = BTreeSet::from([(0, 1), (2, 3)]);
        assert_eq!(indexed_pairs(&leases), expected);
        assert_eq!(indexed_pairs(&leases), naive_pairs(&leases));
    }

    #[test]
    fn indexed_candidates_preserve_literal_path_normalization_edge_cases() {
        let leases = vec![
            LeaseScope {
                agent: "claude".into(),
                paths: vec!["//src\\auth//".into()],
                symbols: vec![],
            },
            LeaseScope {
                agent: "codex".into(),
                paths: vec!["src/auth/token.rs".into()],
                symbols: vec![],
            },
            LeaseScope {
                agent: "kimi".into(),
                paths: vec!["src//graph".into()],
                symbols: vec![],
            },
            LeaseScope {
                agent: "copilot".into(),
                paths: vec!["src//graph/api.rs".into()],
                symbols: vec![],
            },
            LeaseScope {
                agent: "deepseek".into(),
                paths: vec!["src/graph".into()],
                symbols: vec![],
            },
        ];
        // `src//graph` and `src/graph` are intentionally distinct under the
        // existing production rule even though both share the textual root.
        let expected = BTreeSet::from([(0, 1), (2, 3)]);
        assert_eq!(indexed_pairs(&leases), expected);
        assert_eq!(indexed_pairs(&leases), naive_pairs(&leases));
    }

    #[test]
    fn thousand_disjoint_agents_produce_zero_candidates() {
        let leases: Vec<_> = (0..1_000)
            .map(|index| LeaseScope {
                agent: format!("agent-{index}"),
                paths: vec![format!("workspace-{index}/file.rs")],
                symbols: vec![format!("Symbol{index}")],
            })
            .collect();
        assert!(indexed_pairs(&leases).is_empty());
        assert!(naive_pairs(&leases).is_empty());
    }
}
