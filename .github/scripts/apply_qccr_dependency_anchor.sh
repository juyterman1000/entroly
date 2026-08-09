#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

path = Path('entroly-qccr/src/lib.rs')
text = path.read_text(encoding='utf-8')
if 'mod dependency_anchor;' in text:
    raise SystemExit('dependency-anchor integration already applied')

old_import = '''use serde::{Deserialize, Serialize};
'''
new_import = '''use serde::{Deserialize, Serialize};

mod dependency_anchor;
use dependency_anchor::{
    query_dependency_anchors, render_dependency_anchored_excerpt, DIRECT_DEPENDENCY_BOOST,
};
'''

old_file_texts = '''    let file_sources: Vec<String> = order.clone();
    let file_texts: Vec<String> = order.iter().map(|s| groups[s].join("\\n")).collect();

    let ranked = rank_files(&file_sources, &file_texts, query, overrides);
'''
new_file_texts = '''    let file_sources: Vec<String> = order.clone();
    let file_texts: Vec<String> = order.iter().map(|s| groups[s].join("\\n")).collect();
    let dependency_anchors = query_dependency_anchors(&file_sources, &file_texts, query);
    let mut dependency_by_source: HashMap<String, Vec<String>> = HashMap::new();
    let mut dependency_order: Vec<String> = Vec::new();
    for anchor in &dependency_anchors {
        if !dependency_order.contains(&anchor.source) {
            dependency_order.push(anchor.source.clone());
        }
        dependency_by_source
            .entry(anchor.source.clone())
            .or_default()
            .push(anchor.signature.clone());
    }

    let ranked = rank_files(&file_sources, &file_texts, query, overrides);
'''

old_sort = '''    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Caller-supplied reorder (engine_s6 localizer) — same effect as the Python
'''
new_sort = '''    if !dependency_order.is_empty() {
        for (score, source, _) in &mut file_scores {
            if dependency_by_source.contains_key(source) {
                *score = (*score + DIRECT_DEPENDENCY_BOOST).max(DIRECT_DEPENDENCY_BOOST);
            }
        }
    }
    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Caller-supplied reorder (engine_s6 localizer) — same effect as the Python
'''

old_reorder = '''        let mut reordered = Vec::new();
        for src in preferred {
            if let Some((sc, txt)) = by_src.get(src) {
                reordered.push((*sc, src.clone(), txt.clone()));
            }
        }
'''
new_reorder = '''        let mut reordered = Vec::new();
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
'''

old_choose = '''        let chosen = mmr_select(&sentences, &s_tf, &rel, file_budget);
        if chosen.is_empty() {
            continue;
        }
        let excerpt = chosen
            .iter()
            .map(|&i| sentences[i].clone())
            .collect::<Vec<_>>()
            .join("\\n");
'''
new_choose = '''        let anchors = dependency_by_source.get(src);
        let anchor_cost = anchors
            .map(|items| items.iter().map(|item| approx_tokens(item) as i64).sum::<i64>())
            .unwrap_or(0)
            .min(file_budget);
        let sentence_budget = (file_budget - anchor_cost).max(0);
        let chosen = if sentence_budget > 0 {
            mmr_select(&sentences, &s_tf, &rel, sentence_budget)
        } else {
            Vec::new()
        };
        if chosen.is_empty() && anchors.is_none() {
            continue;
        }
        let excerpt = if let Some(anchors) = anchors {
            render_dependency_anchored_excerpt(anchors, &sentences, &chosen, file_budget)
        } else {
            chosen
                .iter()
                .map(|&i| sentences[i].clone())
                .collect::<Vec<_>>()
                .join("\\n")
        };
        if excerpt.is_empty() {
            continue;
        }
'''

test_marker = '''    #[test]
    fn split_sentences_breaks_on_boundaries() {
'''
test = r'''    #[test]
    fn select_keeps_direct_dependency_signature_inside_budget() {
        let fragments = vec![
            InFragment {
                source: "file:pkg/api.py".to_string(),
                content: "from .dep import target\n\ndef caller():\n    return target(1, 2)\n".to_string(),
                feedback_multiplier: 1.0,
            },
            InFragment {
                source: "file:pkg/dep.py".to_string(),
                content: "def target(\n    alpha: int,\n    beta: int,\n) -> int:\n    return alpha + beta\n\ndef noise():\n    return 42\n".to_string(),
                feedback_multiplier: 1.0,
            },
        ];
        let out = select(
            &fragments,
            160,
            "caller explain behavior",
            &HashMap::new(),
            &[],
        );
        let target = out
            .iter()
            .find(|fragment| fragment.source == "file:pkg/dep.py")
            .expect("dependency file must be selected");
        assert!(target.content.contains("def target("), "{}", target.content);
        assert!(target.content.contains("alpha: int"), "{}", target.content);
        assert!(target.content.contains("beta: int"), "{}", target.content);
        let total: usize = out.iter().map(|fragment| fragment.token_count).sum();
        assert!(total <= 160, "total={total}");
    }

'''

replacements = [
    ('import', old_import, new_import),
    ('file_texts', old_file_texts, new_file_texts),
    ('sort', old_sort, new_sort),
    ('reorder', old_reorder, new_reorder),
    ('choose', old_choose, new_choose),
    ('test', test_marker, test + test_marker),
]
for label, old, new in replacements:
    if old not in text:
        raise SystemExit(f'{label} marker not found')
    text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
PY

cargo fmt --manifest-path entroly-qccr/Cargo.toml
cargo fmt --manifest-path entroly-qccr/Cargo.toml -- --check
git diff --check
