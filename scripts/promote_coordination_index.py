from pathlib import Path

module = Path("entroly-engine/src/coordination_index.rs")
text = module.read_text(encoding="utf-8")

old_header = '''#[derive(Debug, Clone)]
struct LeaseScope {
    agent: String,
    paths: Vec<String>,
    symbols: Vec<String>,
}
'''
new_header = '''#[derive(Debug, Clone, Copy)]
pub(crate) struct CoordinationScope<'a> {
    pub(crate) agent: &'a str,
    pub(crate) paths: &'a [String],
    pub(crate) symbols: &'a [String],
}
'''
if text.count(old_header) != 1:
    raise SystemExit("coordination scope anchor changed")
text = text.replace(old_header, new_header, 1)

old_fn = "fn indexed_pairs(leases: &[LeaseScope]) -> BTreeSet<(usize, usize)> {"
new_fn = "pub(crate) fn candidate_pairs(leases: &[CoordinationScope<'_>]) -> BTreeSet<(usize, usize)> {"
if text.count(old_fn) != 1:
    raise SystemExit("indexed_pairs anchor changed")
text = text.replace(old_fn, new_fn, 1)

old_scopes = "fn scopes_overlap(left: &LeaseScope, right: &LeaseScope) -> bool {"
old_naive = "fn naive_pairs(leases: &[LeaseScope]) -> BTreeSet<(usize, usize)> {"
if text.count(old_scopes) != 1 or text.count(old_naive) != 1:
    raise SystemExit("naive-oracle anchors changed")
text = text.replace(
    old_scopes,
    "#[cfg(test)]\nfn scopes_overlap(left: &TestLeaseScope, right: &TestLeaseScope) -> bool {",
    1,
)
text = text.replace(
    old_naive,
    "#[cfg(test)]\nfn naive_pairs(leases: &[TestLeaseScope]) -> BTreeSet<(usize, usize)> {",
    1,
)

test_anchor = "#[cfg(test)]\nmod tests {"
if text.count(test_anchor) != 1:
    raise SystemExit("coordination test anchor changed")
test_support = '''#[cfg(test)]
#[derive(Debug, Clone)]
struct TestLeaseScope {
    agent: String,
    paths: Vec<String>,
    symbols: Vec<String>,
}

#[cfg(test)]
fn borrowed_scopes(leases: &[TestLeaseScope]) -> Vec<CoordinationScope<'_>> {
    leases
        .iter()
        .map(|lease| CoordinationScope {
            agent: &lease.agent,
            paths: &lease.paths,
            symbols: &lease.symbols,
        })
        .collect()
}

'''
text = text.replace(test_anchor, test_support + test_anchor, 1)
text = text.replace("-> LeaseScope {", "-> TestLeaseScope {")
text = text.replace("        LeaseScope {", "        TestLeaseScope {")
text = text.replace("            LeaseScope {", "            TestLeaseScope {")
text = text.replace("                LeaseScope {", "                TestLeaseScope {")
text = text.replace(
    "indexed_pairs(&leases)",
    "candidate_pairs(&borrowed_scopes(&leases))",
)
module.write_text(text, encoding="utf-8")

lib = Path("entroly-engine/src/lib.rs")
lib_text = lib.read_text(encoding="utf-8")
old_lib = '''pub mod work_graph;

#[cfg(test)]
mod coordination_index;
'''
new_lib = '''pub mod work_graph;

mod coordination_index;
'''
if lib_text.count(old_lib) != 1:
    raise SystemExit("lib coordination module anchor changed")
lib.write_text(lib_text.replace(old_lib, new_lib, 1), encoding="utf-8")

graph = Path("entroly-engine/src/work_graph.rs")
graph_text = graph.read_text(encoding="utf-8")
use_anchor = "use std::fmt;\n"
if graph_text.count(use_anchor) != 1:
    raise SystemExit("work_graph import anchor changed")
graph_text = graph_text.replace(
    use_anchor,
    use_anchor + "use crate::coordination_index::{candidate_pairs, CoordinationScope};\n",
    1,
)

old_loop = '''        let mut conflicts = Vec::new();
        for i in 0..leases.len() {
            for j in (i + 1)..leases.len() {
                let a = &leases[i];
                let b = &leases[j];
                if a.agent == b.agent || a.expires <= now_ms || b.expires <= now_ms {
                    continue;
                }
                let overlapping_paths = overlap_paths(&a.paths, &b.paths);
                let overlapping_symbols = overlap_exact(&a.symbols, &b.symbols);
                if overlapping_paths.is_empty() && overlapping_symbols.is_empty() {
                    continue;
                }
                conflicts.push(CoordinationConflict {
                    lease_a: a.id.clone(),
                    lease_b: b.id.clone(),
                    agent_a: a.agent.clone(),
                    agent_b: b.agent.clone(),
                    task_a: a.task.clone(),
                    task_b: b.task.clone(),
                    overlapping_paths,
                    overlapping_symbols,
                    reason: "active advisory work scopes overlap".to_string(),
                });
            }
        }
'''
new_loop = '''        let scopes: Vec<CoordinationScope<'_>> = leases
            .iter()
            .map(|lease| CoordinationScope {
                agent: &lease.agent,
                paths: &lease.paths,
                symbols: &lease.symbols,
            })
            .collect();
        let mut conflicts = Vec::new();
        for (i, j) in candidate_pairs(&scopes) {
            let a = &leases[i];
            let b = &leases[j];
            // Candidate generation is only a performance filter. Keep the
            // pre-existing exact overlap functions authoritative so conflict
            // semantics cannot drift with the index.
            if a.agent == b.agent || a.expires <= now_ms || b.expires <= now_ms {
                continue;
            }
            let overlapping_paths = overlap_paths(&a.paths, &b.paths);
            let overlapping_symbols = overlap_exact(&a.symbols, &b.symbols);
            if overlapping_paths.is_empty() && overlapping_symbols.is_empty() {
                continue;
            }
            conflicts.push(CoordinationConflict {
                lease_a: a.id.clone(),
                lease_b: b.id.clone(),
                agent_a: a.agent.clone(),
                agent_b: b.agent.clone(),
                task_a: a.task.clone(),
                task_b: b.task.clone(),
                overlapping_paths,
                overlapping_symbols,
                reason: "active advisory work scopes overlap".to_string(),
            });
        }
'''
if graph_text.count(old_loop) != 1:
    raise SystemExit("coordination production loop anchor changed")
graph.write_text(graph_text.replace(old_loop, new_loop, 1), encoding="utf-8")

print("coordination index promotion patch applied")
