from __future__ import annotations

from pathlib import Path

PATH = Path("entroly-qccr/src/lib.rs")
text = PATH.read_text(encoding="utf-8")

if "render_python_selection_with_complete_signatures" in text:
    raise SystemExit("signature completion already applied")

marker = '''fn approx_tokens(s: &str) -> usize {
    (s.chars().count() / CHARS_PER_TOKEN).max(1)
}
'''
helper = r'''fn sentence_offsets(text: &str, sentences: &[String]) -> Vec<Option<(usize, usize)>> {
    let mut cursor = 0usize;
    sentences
        .iter()
        .map(|sentence| {
            let found = text[cursor..]
                .find(sentence)
                .map(|relative| cursor + relative)
                .or_else(|| text.find(sentence));
            if let Some(start) = found {
                let end = start.saturating_add(sentence.len()).min(text.len());
                cursor = end;
                Some((start, end))
            } else {
                None
            }
        })
        .collect()
}

fn python_signature_spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut lines = Vec::new();
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let without_newline = line.strip_suffix('\n').unwrap_or(line);
        lines.push((offset, without_newline));
        offset = offset.saturating_add(line.len());
    }
    if offset < text.len() {
        lines.push((offset, &text[offset..]));
    }

    let mut spans = Vec::new();
    for start_index in 0..lines.len() {
        let (start_offset, raw) = lines[start_index];
        let trimmed = raw.trim_start();
        if !(trimmed.starts_with("def ")
            || trimmed.starts_with("async def ")
            || trimmed.starts_with("class "))
        {
            continue;
        }

        let mut balance = 0i32;
        let mut rendered = Vec::new();
        for end_index in start_index..lines.len().min(start_index.saturating_add(32)) {
            let (line_offset, line) = lines[end_index];
            rendered.push(line.trim_end().to_string());
            for ch in line.chars() {
                match ch {
                    '(' | '[' | '{' => balance += 1,
                    ')' | ']' | '}' => balance -= 1,
                    _ => {}
                }
            }
            if balance <= 0 && line.trim_end().ends_with(':') {
                let end_offset = line_offset.saturating_add(line.len());
                spans.push((start_offset, end_offset, rendered.join("\n")));
                break;
            }
        }
    }
    spans
}

fn render_python_selection_with_complete_signatures(
    source: &str,
    text: &str,
    sentences: &[String],
    chosen: &[usize],
    file_budget: i64,
) -> String {
    let plain = chosen
        .iter()
        .map(|&i| sentences[i].clone())
        .collect::<Vec<_>>()
        .join("\n");
    let source_lower = source.to_ascii_lowercase();
    if !(source_lower.ends_with(".py")
        || source_lower.ends_with(".pyi")
        || source_lower.ends_with(".pyw"))
    {
        return plain;
    }

    let spans = python_signature_spans(text);
    if spans.is_empty() {
        return plain;
    }
    let offsets = sentence_offsets(text, sentences);
    let mut items: Vec<(String, bool)> = Vec::new();
    let mut seen = HashSet::new();

    for &index in chosen {
        let sentence = sentences[index].clone();
        let mut replacement = None;
        if let Some(Some((sentence_start, sentence_end))) = offsets.get(index) {
            for (signature_start, signature_end, signature) in &spans {
                let overlaps = *sentence_start < *signature_end && *sentence_end > *signature_start;
                if overlaps {
                    replacement = Some(signature.clone());
                    break;
                }
            }
        }
        let (rendered, protected) = replacement
            .map(|signature| (signature, true))
            .unwrap_or((sentence, false));
        if seen.insert(rendered.clone()) {
            items.push((rendered, protected));
        }
    }

    let protected_only = items
        .iter()
        .filter(|(_, protected)| *protected)
        .map(|(item, _)| item.clone())
        .collect::<Vec<_>>()
        .join("\n");
    if !protected_only.is_empty() && approx_tokens(&protected_only) as i64 > file_budget {
        return plain;
    }

    loop {
        let rendered = items
            .iter()
            .map(|(item, _)| item.clone())
            .collect::<Vec<_>>()
            .join("\n");
        if approx_tokens(&rendered) as i64 <= file_budget {
            return rendered;
        }
        if let Some(position) = items.iter().rposition(|(_, protected)| !*protected) {
            items.remove(position);
            continue;
        }
        return protected_only;
    }
}

'''

old_excerpt = '''        let excerpt = chosen
            .iter()
            .map(|&i| sentences[i].clone())
            .collect::<Vec<_>>()
            .join("\\n");
'''
new_excerpt = '''        let excerpt = render_python_selection_with_complete_signatures(
            src,
            text,
            &sentences,
            &chosen,
            file_budget,
        );
'''

test_marker = '''    #[test]
    fn split_sentences_breaks_on_boundaries() {
'''
test = r'''    #[test]
    fn selected_python_signature_fragment_is_completed_atomically() {
        let source = "def target(\n    alpha: str,\n    beta: int,\n    force: bool = False,\n) -> dict:\n    return {\"ok\": force}\n";
        let sentences = split_sentences(source);
        let chosen = vec![sentences
            .iter()
            .position(|sentence| sentence.contains("force: bool"))
            .expect("signature fragment")];
        let rendered = render_python_selection_with_complete_signatures(
            "file:service.py",
            source,
            &sentences,
            &chosen,
            96,
        );
        assert!(rendered.contains("def target("), "{rendered}");
        assert!(rendered.contains("alpha: str,"), "{rendered}");
        assert!(rendered.contains("force: bool = False,"), "{rendered}");
        assert!(rendered.contains(") -> dict:"), "{rendered}");
        assert!(approx_tokens(&rendered) <= 96, "{rendered}");
    }

'''

for label, needle in (("approx_tokens", marker), ("excerpt", old_excerpt), ("test", test_marker)):
    if needle not in text:
        raise SystemExit(f"{label} marker not found")

text = text.replace(marker, helper + marker, 1)
text = text.replace(old_excerpt, new_excerpt, 1)
text = text.replace(test_marker, test + test_marker, 1)
PATH.write_text(text, encoding="utf-8")
