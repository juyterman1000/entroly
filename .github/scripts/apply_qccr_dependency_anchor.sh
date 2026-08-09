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
    protected_anchors_survive, query_dependency_anchors,
    render_dependency_anchored_excerpt,
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

old_reorder_block = '''    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));

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
'''
new_reorder_block = '''    file_scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let ranked_by_src: HashMap<String, (f64, String)> = file_scores
        .iter()
        .map(|(sc, src, txt)| (src.clone(), (*sc, txt.clone())))
        .collect();

    // Caller-supplied reorder (engine_s6 localizer) — same effect as the Python
    // `localize_files` block: reorder the candidate list, scores preserved.
    if !preferred.is_empty()
        && file_scores.len() > 1
        && file_scores.iter().any(|(s, _, _)| *s > 0.0)
    {
        let mut reordered = Vec::new();
        for src in preferred {
            if let Some((sc, txt)) = ranked_by_src.get(src) {
                reordered.push((*sc, src.clone(), txt.clone()));
            }
        }
        if !reordered.is_empty() {
            file_scores = reordered;
        }
    }

    // A proven direct dependency of a callable explicitly named by the query is
    // structural evidence, not another lexical hint. Admit those files before
    // lexical/localizer truncation while preserving their original utility
    // scores for budget allocation. The remaining slots keep the existing order.
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
            .map(|items| approx_tokens(&items.join("\\n")) as i64)
            .unwrap_or(0);
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
new_trim = '''    // Hard budget ceiling. Protected dependency signatures are atomic: the
    // reconciler may trim only content that leaves every protected signature
    // byte-for-byte intact. If no such reduction is possible, fail closed
    // instead of emitting a deceptively partial declaration.
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

    #[test]
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
    ('file_texts', old_file_texts, new_file_texts),
    ('reorder', old_reorder_block, new_reorder_block),
    ('choose', old_choose, new_choose),
    ('trim', old_trim, new_trim),
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