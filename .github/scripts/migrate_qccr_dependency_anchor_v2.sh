#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('entroly-qccr/src/lib.rs')
text = path.read_text(encoding='utf-8')

if 'protected_anchors_survive, query_dependency_anchors' in text:
    print('QCCR dependency integration is already v2')
    raise SystemExit(0)
if 'mod dependency_anchor;' not in text:
    raise SystemExit('expected persisted v1 dependency integration is absent')

old_import = '''use dependency_anchor::{
    query_dependency_anchors, render_dependency_anchored_excerpt, DIRECT_DEPENDENCY_BOOST,
};
'''
new_import = '''use dependency_anchor::{
    protected_anchors_survive, query_dependency_anchors, render_dependency_anchored_excerpt,
};
'''

old_boost = '''    if !dependency_order.is_empty() {
        for (score, source, _) in &mut file_scores {
            if dependency_by_source.contains_key(source) {
                *score = (*score + DIRECT_DEPENDENCY_BOOST).max(DIRECT_DEPENDENCY_BOOST);
            }
        }
    }
    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Caller-supplied reorder (engine_s6 localizer) — same effect as the Python
'''
new_boost = '''    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let ranked_by_src: HashMap<String, (f64, String)> = file_scores
        .iter()
        .map(|(sc, src, txt)| (src.clone(), (*sc, txt.clone())))
        .collect();

    // Caller-supplied reorder (engine_s6 localizer) — same effect as the Python
'''

old_reorder = '''        let by_src: HashMap<String, (f64, String)> = file_scores
            .iter()
            .map(|(sc, src, txt)| (src.clone(), (*sc, txt.clone())))
            .collect();
        let mut reordered = Vec::new();
        let mut reordered_seen = HashSet::new();
        for src in &dependency_order {
            if let Some((sc, txt)) = by_src.get(src) {
                reordered.push((*sc, src.clone(), txt.clone()));
                reordered_seen.insert(src.clone());
            }
        }
        for src in preferred {
            if reordered_seen.insert(src.clone()) {
                if let Some((sc, txt)) = by_src.get(src) {
                    reordered.push((*sc, src.clone(), txt.clone()));
                }
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
'''
new_reorder = '''        let mut reordered = Vec::new();
        for src in preferred {
            if let Some((sc, txt)) = ranked_by_src.get(src) {
                reordered.push((*sc, src.clone(), txt.clone()));
            }
        }
        if !reordered.is_empty() {
            file_scores = reordered;
        }
    }

    // Proven direct dependencies of a callable explicitly named by the query
    // are structural evidence. Admit them before lexical/localizer truncation,
    // preserving the original score only for proportional budget allocation.
    if !dependency_order.is_empty() {
        let mut reordered = Vec::new();
        let mut seen = HashSet::new();
        for src in &dependency_order {
            if let Some((sc, txt)) = ranked_by_src.get(src) {
                reordered.push((*sc, src.clone(), txt.clone()));
                seen.insert(src.clone());
            }
        }
        for (sc, src, txt) in file_scores {
            if seen.insert(src.clone()) {
                reordered.push((sc, src, txt));
            }
        }
        file_scores = reordered;
    }

    let dependency_sources: HashSet<String> = dependency_order.iter().cloned().collect();
    let mut top_files: Vec<(f64, String, String)> = file_scores
        .iter()
        .filter(|(s, src, _)| *s > 0.0 || dependency_sources.contains(src))
        .take(MAX_FILES_CONSIDERED)
        .cloned()
        .collect();
'''

old_anchor_cost = '''        let anchor_cost = anchors
            .map(|items| {
                items
                    .iter()
                    .map(|item| approx_tokens(item) as i64)
                    .sum::<i64>()
            })
            .unwrap_or(0)
            .min(file_budget);
'''
new_anchor_cost = '''        let anchor_cost = anchors
            .map(|items| approx_tokens(&items.join("\\n")) as i64)
            .unwrap_or(0);
'''

old_trim = '''    // Hard budget ceiling: trim trailing excerpts (drop last sentence, then
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
        let mut lines: Vec<&str> = last.content.split('\\n').collect();
        if lines.len() > 1 {
            lines.pop();
            last.content = lines.join("\\n");
            last.token_count = approx_tokens(&last.content);
        } else {
            output.pop();
        }
        total = output.iter().map(|f| frag_tokens(f) as i64).sum();
    }

    output
'''
new_trim = '''    // Hard budget ceiling. Query-rooted dependency signatures are protected
    // evidence atoms: reconciliation may trim only while every protected atom
    // remains byte-for-byte intact. If the fixed budget cannot satisfy that
    // invariant, fail closed instead of emitting a deceptively partial header.
    let frag_tokens = |f: &OutFragment| -> usize {
        if f.token_count > 0 {
            f.token_count
        } else {
            approx_tokens(&f.content)
        }
    };
    let mut total: i64 = output.iter().map(|f| frag_tokens(f) as i64).sum();
    while !output.is_empty() && total > token_budget {
        let mut changed = false;
        for index in (0..output.len()).rev() {
            let protected = dependency_by_source.get(&output[index].source);
            let lines: Vec<&str> = output[index].content.split('\\n').collect();
            if lines.len() > 1 {
                let candidate = lines[..lines.len() - 1].join("\\n");
                let protected_ok = protected
                    .map(|anchors| protected_anchors_survive(&candidate, anchors))
                    .unwrap_or(true);
                if protected_ok && !candidate.is_empty() {
                    output[index].content = candidate;
                    output[index].token_count = approx_tokens(&output[index].content);
                    changed = true;
                    break;
                }
            }
            if protected.is_none() {
                output.remove(index);
                changed = true;
                break;
            }
        }
        if !changed {
            return Vec::new();
        }
        total = output.iter().map(|f| frag_tokens(f) as i64).sum();
    }

    output
'''

test_marker = '''    #[test]
    fn split_sentences_breaks_on_boundaries() {
'''
top_k_test = r'''    #[test]
    fn select_admits_structural_dependency_ahead_of_lexical_top_k() {
        let mut fragments = vec![
            InFragment {
                source: "file:pkg/api.py".to_string(),
                content: "from .dep import target\n\ndef caller():\n    return target(1, 2)\n".to_string(),
                feedback_multiplier: 1.0,
            },
            InFragment {
                source: "file:pkg/dep.py".to_string(),
                content: "def target(\n    alpha: int,\n    beta: int,\n) -> int:\n    return alpha + beta\n".to_string(),
                feedback_multiplier: 1.0,
            },
        ];
        for index in 0..16 {
            fragments.push(InFragment {
                source: format!("file:noise_{index}.py"),
                content: "caller explain behavior deterministic repository index ".repeat(20),
                feedback_multiplier: 1.0,
            });
        }
        let out = select(
            &fragments,
            800,
            "caller explain behavior deterministic repository index",
            &HashMap::new(),
            &[],
        );
        let target = out
            .iter()
            .find(|fragment| fragment.source == "file:pkg/dep.py")
            .expect("structural dependency must survive lexical top-k pressure");
        assert!(target.content.contains("def target("), "{}", target.content);
        assert!(target.content.contains("beta: int"), "{}", target.content);
    }

'''

replacements = [
    ('import', old_import, new_import),
    ('boost', old_boost, new_boost),
    ('reorder', old_reorder, new_reorder),
    ('anchor_cost', old_anchor_cost, new_anchor_cost),
    ('trim', old_trim, new_trim),
]
for label, old, new in replacements:
    if old not in text:
        raise SystemExit(f'{label} v1 marker not found')
    text = text.replace(old, new, 1)

if 'fn select_admits_structural_dependency_ahead_of_lexical_top_k()' not in text:
    if test_marker not in text:
        raise SystemExit('test insertion marker not found')
    text = text.replace(test_marker, top_k_test + test_marker, 1)

path.write_text(text, encoding='utf-8')
PY

git diff --check
