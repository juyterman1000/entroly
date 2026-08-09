use std::collections::{HashMap, HashSet};

use super::{approx_tokens, ident_re};

pub(crate) const DIRECT_DEPENDENCY_BOOST: f64 = 0.85;
const MAX_DIRECT_DEPENDENCY_ANCHORS: usize = 8;

#[derive(Clone, Debug)]
pub(crate) struct DependencyAnchor {
    pub(crate) source: String,
    pub(crate) signature: String,
}

fn normalized_source_path(source: &str) -> String {
    source
        .strip_prefix("file:")
        .unwrap_or(source)
        .replace('\\', "/")
}

fn is_python_source(source: &str) -> bool {
    let path = normalized_source_path(source).to_ascii_lowercase();
    path.ends_with(".py") || path.ends_with(".pyi") || path.ends_with(".pyw")
}

fn python_indent(line: &str) -> usize {
    line.chars()
        .take_while(|c| matches!(c, ' ' | '\t'))
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

fn python_header_end(lines: &[&str], start: usize) -> Option<usize> {
    let mut balance = 0i32;
    let mut saw_group = false;
    for (index, line) in lines.iter().enumerate().skip(start).take(32) {
        for ch in line.chars() {
            match ch {
                '(' | '[' | '{' => {
                    balance += 1;
                    saw_group = true;
                }
                ')' | ']' | '}' => balance -= 1,
                _ => {}
            }
        }
        if line.trim_end().ends_with(':') && (!saw_group || balance <= 0) {
            return Some(index);
        }
    }
    None
}

fn python_signature_from_lines(lines: &[&str], start: usize) -> Option<String> {
    let end = python_header_end(lines, start)?;
    Some(
        lines[start..=end]
            .iter()
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn python_signature_for_symbol(text: &str, symbol: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let def_prefix = format!("def {symbol}(");
    let async_prefix = format!("async def {symbol}(");
    let class_plain = format!("class {symbol}:");
    let class_args = format!("class {symbol}(");

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&def_prefix) || trimmed.starts_with(&async_prefix) {
            return python_signature_from_lines(&lines, index);
        }
        if trimmed.starts_with(&class_plain) || trimmed.starts_with(&class_args) {
            let class_indent = python_indent(line);
            let class_header_end = python_header_end(&lines, index)?;
            for inner in class_header_end + 1..lines.len() {
                let inner_line = lines[inner];
                let inner_trimmed = inner_line.trim_start();
                if inner_trimmed.is_empty() || inner_trimmed.starts_with('#') {
                    continue;
                }
                let indent = python_indent(inner_line);
                if indent <= class_indent {
                    break;
                }
                if inner_trimmed.starts_with("def __init__(")
                    || inner_trimmed.starts_with("async def __init__(")
                {
                    return python_signature_from_lines(&lines, inner);
                }
            }
        }
    }
    None
}

fn python_callable_body(text: &str, symbol: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let def_prefix = format!("def {symbol}(");
    let async_prefix = format!("async def {symbol}(");
    let class_plain = format!("class {symbol}:");
    let class_args = format!("class {symbol}(");

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if !(trimmed.starts_with(&def_prefix)
            || trimmed.starts_with(&async_prefix)
            || trimmed.starts_with(&class_plain)
            || trimmed.starts_with(&class_args))
        {
            continue;
        }
        let base_indent = python_indent(line);
        let header_end = python_header_end(&lines, index)?;
        let mut end = lines.len();
        for (inner, inner_line) in lines.iter().enumerate().skip(header_end + 1) {
            let inner_trimmed = inner_line.trim();
            if inner_trimmed.is_empty() || inner_trimmed.starts_with('#') {
                continue;
            }
            if python_indent(inner_line) <= base_indent {
                end = inner;
                break;
            }
        }
        return Some(lines[header_end + 1..end].join("\n"));
    }
    None
}

fn resolve_python_module_source(source: &str, module: &str) -> Option<String> {
    let path = normalized_source_path(source);
    let mut parent: Vec<String> = path.split('/').map(str::to_string).collect();
    parent.pop()?;

    let dot_count = module.chars().take_while(|c| *c == '.').count();
    let remainder = &module[dot_count..];
    if dot_count == 0 {
        parent.clear();
    } else {
        for _ in 1..dot_count {
            parent.pop()?;
        }
    }
    if !remainder.is_empty() {
        parent.extend(
            remainder
                .split('.')
                .filter(|part| !part.is_empty())
                .map(str::to_string),
        );
    }
    if parent.is_empty() {
        return None;
    }
    Some(format!("{}.py", parent.join("/")))
}

fn python_import_bindings(source: &str, text: &str) -> Vec<(String, String, String)> {
    let mut statements = Vec::new();
    let mut pending = String::new();
    let mut depth = 0i32;

    for raw in text.lines() {
        let trimmed = raw.trim();
        if pending.is_empty() && !trimmed.starts_with("from ") {
            continue;
        }
        if !pending.is_empty() {
            pending.push(' ');
        }
        pending.push_str(trimmed.trim_end_matches('\\'));
        for ch in trimmed.chars() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
        }
        if depth > 0 || trimmed.ends_with('\\') {
            continue;
        }
        statements.push(std::mem::take(&mut pending));
        depth = 0;
    }

    let mut out = Vec::new();
    for statement in statements {
        let Some(rest) = statement.strip_prefix("from ") else {
            continue;
        };
        let Some((module, names)) = rest.split_once(" import ") else {
            continue;
        };
        let Some(target_source) = resolve_python_module_source(source, module.trim()) else {
            continue;
        };
        let cleaned = names.trim().trim_start_matches('(').trim_end_matches(')');
        for item in cleaned.split(',') {
            let item = item.trim();
            if item.is_empty() || item == "*" {
                continue;
            }
            let words: Vec<&str> = item.split_whitespace().collect();
            let original = words[0].trim();
            if original.is_empty() {
                continue;
            }
            let local = if words.len() >= 3 && words[1] == "as" {
                words[2]
            } else {
                original
            };
            out.push((
                local.to_string(),
                original.to_string(),
                target_source.clone(),
            ));
        }
    }
    out
}

fn python_call_position(body: &str, name: &str) -> Option<usize> {
    for (index, _) in body.match_indices(name) {
        let bytes = body.as_bytes();
        let before_ok = index == 0
            || !matches!(
                bytes[index - 1],
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'
            );
        if !before_ok {
            continue;
        }
        let mut cursor = index + name.len();
        if cursor < bytes.len()
            && matches!(
                bytes[cursor],
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_'
            )
        {
            continue;
        }
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'(' {
            return Some(index);
        }
    }
    None
}

pub(crate) fn query_dependency_anchors(
    sources: &[String],
    texts: &[String],
    query: &str,
) -> Vec<DependencyAnchor> {
    let mut query_symbols = Vec::new();
    let mut seen_query = HashSet::new();
    for matched in ident_re().find_iter(query) {
        let symbol = matched.as_str().to_string();
        if symbol.len() >= 3 && seen_query.insert(symbol.clone()) {
            query_symbols.push(symbol);
        }
    }

    let path_to_index: HashMap<String, usize> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (normalized_source_path(source), index))
        .collect();
    let mut candidates: Vec<(usize, String, String, String)> = Vec::new();

    for (root_index, source) in sources.iter().enumerate() {
        if !is_python_source(source) {
            continue;
        }
        let bindings = python_import_bindings(source, &texts[root_index]);
        if bindings.is_empty() {
            continue;
        }
        for query_symbol in &query_symbols {
            let Some(body) = python_callable_body(&texts[root_index], query_symbol) else {
                continue;
            };
            for (local, original, target_path) in &bindings {
                let Some(call_pos) = python_call_position(&body, local) else {
                    continue;
                };
                let target_index = path_to_index.get(target_path).copied().or_else(|| {
                    let package_path =
                        format!("{}/__init__.py", target_path.trim_end_matches(".py"));
                    path_to_index.get(&package_path).copied()
                });
                let Some(target_index) = target_index else {
                    continue;
                };
                let Some(signature) = python_signature_for_symbol(&texts[target_index], original)
                else {
                    continue;
                };
                candidates.push((
                    call_pos,
                    sources[target_index].clone(),
                    original.clone(),
                    signature,
                ));
            }
        }
    }

    candidates.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (_, source, symbol, signature) in candidates {
        if !seen.insert((source.clone(), symbol.clone())) {
            continue;
        }
        out.push(DependencyAnchor { source, signature });
        if out.len() >= MAX_DIRECT_DEPENDENCY_ANCHORS {
            break;
        }
    }
    out
}

pub(crate) fn render_dependency_anchored_excerpt(
    anchors: &[String],
    sentences: &[String],
    chosen: &[usize],
    file_budget: i64,
) -> String {
    let mut pieces: Vec<String> = Vec::new();
    let mut used = 0i64;
    for anchor in anchors {
        let cost = approx_tokens(anchor) as i64;
        if cost > file_budget || used + cost > file_budget {
            continue;
        }
        if !pieces.contains(anchor) {
            pieces.push(anchor.clone());
            used += cost;
        }
    }
    for &index in chosen {
        let sentence = sentences[index].clone();
        if pieces
            .iter()
            .any(|piece| piece.contains(&sentence) || sentence.contains(piece))
        {
            continue;
        }
        let cost = approx_tokens(&sentence) as i64;
        if used + cost > file_budget {
            continue;
        }
        pieces.push(sentence);
        used += cost;
    }
    pieces.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_import_and_complete_signature() {
        let sources = vec!["file:pkg/api.py".to_string(), "file:pkg/dep.py".to_string()];
        let texts = vec![
            "from .dep import target\n\ndef caller():\n    return target(1, 2)\n".to_string(),
            "def target(\n    alpha: int,\n    beta: int,\n) -> int:\n    return alpha + beta\n"
                .to_string(),
        ];
        let anchors = query_dependency_anchors(&sources, &texts, "caller explain behavior");
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!(anchors[0].source, "file:pkg/dep.py");
        assert!(anchors[0].signature.contains("alpha: int"));
        assert!(anchors[0].signature.contains("beta: int"));
    }

    #[test]
    fn resolves_dependency_from_multiline_caller_signature() {
        let sources = vec!["file:pkg/api.py".to_string(), "file:pkg/dep.py".to_string()];
        let texts = vec![
            "from .dep import target\n\ndef caller(\n    value: int,\n) -> int:\n    return target(value)\n"
                .to_string(),
            "def target(\n    alpha: int,\n    beta: int = 0,\n) -> int:\n    return alpha + beta\n"
                .to_string(),
        ];
        let anchors = query_dependency_anchors(&sources, &texts, "caller explain behavior");
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!(anchors[0].source, "file:pkg/dep.py");
        assert!(anchors[0].signature.contains("beta: int = 0"));
    }

    #[test]
    fn uses_constructor_signature_for_called_class() {
        let sources = vec![
            "file:pkg/api.py".to_string(),
            "file:pkg/cache.py".to_string(),
        ];
        let texts = vec![
            "from .cache import Cache\n\ndef caller():\n    return Cache()\n".to_string(),
            "class Cache:\n    def __init__(\n        self,\n        threshold: float = 0.9,\n        max_clients: int = 100,\n    ):\n        pass\n"
                .to_string(),
        ];
        let anchors = query_dependency_anchors(&sources, &texts, "caller cache setup");
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert!(anchors[0].signature.contains("def __init__("));
        assert!(anchors[0].signature.contains("max_clients: int = 100"));
    }
}
