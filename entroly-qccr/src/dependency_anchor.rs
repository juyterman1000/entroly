use std::collections::{HashMap, HashSet};

use super::{approx_tokens, ident_re};

const MAX_QUERY_ROOTS: usize = 4;
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
    let mut quote: Option<char> = None;
    let mut escaped = false;

    for (index, line) in lines.iter().enumerate().skip(start).take(32) {
        for ch in line.chars() {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' && quote.is_some() {
                escaped = true;
                continue;
            }
            if let Some(active) = quote {
                if ch == active {
                    quote = None;
                }
                continue;
            }
            if matches!(ch, '\'' | '"') {
                quote = Some(ch);
                continue;
            }
            if ch == '#' {
                break;
            }
            match ch {
                '(' | '[' | '{' => {
                    balance += 1;
                    saw_group = true;
                }
                ')' | ']' | '}' => balance -= 1,
                _ => {}
            }
        }
        if quote.is_none() && line.trim_end().ends_with(':') && (!saw_group || balance <= 0) {
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
        let module = module.trim();
        let Some(base_target) = resolve_python_module_source(source, module) else {
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
            let target_source = if module.ends_with('.') {
                format!("{}/{}.py", base_target.trim_end_matches(".py"), original)
            } else {
                base_target.clone()
            };
            out.push((local.to_string(), original.to_string(), target_source));
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

fn query_roots(
    sources: &[String],
    texts: &[String],
    query: &str,
) -> Vec<(usize, usize, String, String)> {
    let mut identifiers = Vec::new();
    let mut seen = HashSet::new();
    for matched in ident_re().find_iter(query) {
        let symbol = matched.as_str().to_string();
        if symbol.len() >= 3 && seen.insert(symbol.clone()) {
            identifiers.push((matched.start(), symbol));
        }
    }

    let mut roots = Vec::new();
    for (query_position, symbol) in identifiers {
        for (source_index, source) in sources.iter().enumerate() {
            if !is_python_source(source) {
                continue;
            }
            if let Some(body) = python_callable_body(&texts[source_index], &symbol) {
                roots.push((query_position, source_index, symbol.clone(), body));
            }
        }
    }
    roots.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| sources[a.1].cmp(&sources[b.1]))
    });
    roots.truncate(MAX_QUERY_ROOTS);
    roots
}

pub(crate) fn query_dependency_anchors(
    sources: &[String],
    texts: &[String],
    query: &str,
) -> Vec<DependencyAnchor> {
    let path_to_index: HashMap<String, usize> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| (normalized_source_path(source), index))
        .collect();
    let roots = query_roots(sources, texts, query);
    let mut candidates: Vec<(usize, usize, String, String, String)> = Vec::new();

    for (root_rank, (_, root_index, _root_symbol, body)) in roots.iter().enumerate() {
        let bindings = python_import_bindings(&sources[*root_index], &texts[*root_index]);
        for (local, original, target_path) in bindings {
            let Some(call_pos) = python_call_position(body, &local) else {
                continue;
            };
            let target_index = path_to_index.get(&target_path).copied().or_else(|| {
                let package_path = format!("{}/__init__.py", target_path.trim_end_matches(".py"));
                path_to_index.get(&package_path).copied()
            });
            let Some(target_index) = target_index else {
                continue;
            };
            let Some(signature) = python_signature_for_symbol(&texts[target_index], &original)
            else {
                continue;
            };
            candidates.push((
                root_rank,
                call_pos,
                sources[target_index].clone(),
                original,
                signature,
            ));
        }
    }

    candidates.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
    });
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (_, _, source, symbol, signature) in candidates {
        if !seen.insert((source.clone(), symbol)) {
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
    if file_budget <= 0 {
        return String::new();
    }
    let mut pieces: Vec<String> = Vec::new();
    for anchor in anchors {
        if pieces.contains(anchor) {
            continue;
        }
        let mut candidate = pieces.clone();
        candidate.push(anchor.clone());
        if approx_tokens(&candidate.join("\n")) as i64 <= file_budget {
            pieces = candidate;
        }
    }
    if anchors.iter().any(|anchor| !pieces.contains(anchor)) {
        return String::new();
    }
    for &index in chosen {
        let sentence = sentences[index].clone();
        if pieces
            .iter()
            .any(|piece| piece.contains(&sentence) || sentence.contains(piece))
        {
            continue;
        }
        let mut candidate = pieces.clone();
        candidate.push(sentence);
        if approx_tokens(&candidate.join("\n")) as i64 <= file_budget {
            pieces = candidate;
        }
    }
    pieces.join("\n")
}

pub(crate) fn protected_anchors_survive(content: &str, anchors: &[String]) -> bool {
    anchors.iter().all(|anchor| content.contains(anchor))
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
    fn resolves_function_local_class_import() {
        let sources = vec![
            "file:entroly/sdk.py".to_string(),
            "file:entroly/cache_aligner.py".to_string(),
        ];
        let texts = vec![
            "def _cache_align_older(\n    client_key: str, older_msgs: list[dict[str, object]]\n) -> list[dict[str, object]]:\n    try:\n        if True:\n            from .cache_aligner import CacheAligner\n            aligner = CacheAligner()\n        return older_msgs\n    except Exception:\n        return older_msgs\n"
                .to_string(),
            "class CacheAligner:\n    def __init__(\n        self,\n        similarity_threshold: float = 0.90,\n        max_clients: int = 100,\n    ):\n        pass\n"
                .to_string(),
        ];
        let anchors = query_dependency_anchors(
            &sources,
            &texts,
            "_cache_align_older Reuse the previous compressed older-context",
        );
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!(anchors[0].source, "file:entroly/cache_aligner.py");
        assert!(anchors[0].signature.contains("def __init__("));
        assert!(anchors[0].signature.contains("max_clients: int = 100"));
    }

    #[test]
    fn resolves_multi_name_relative_import() {
        let sources = vec![
            "file:entroly/repository_intelligence/__init__.py".to_string(),
            "file:entroly/repository_intelligence/graph.py".to_string(),
        ];
        let texts = vec![
            "from .graph import analyze_change_impact, localize_tests, resolve_calls, resolve_imports\n\ndef build_repository_index(\n    root: str,\n    *,\n    limits: object | None = None,\n) -> object:\n    parsed = {}\n    symbols = {}\n    policy = limits\n    calls, unresolved_calls = resolve_calls(parsed, symbols, policy)\n    return calls, unresolved_calls\n"
                .to_string(),
            "def resolve_calls(\n    parsed: object,\n    symbols: object,\n    limits: object,\n) -> tuple:\n    return (), ()\n"
                .to_string(),
        ];
        let anchors = query_dependency_anchors(
            &sources,
            &texts,
            "build_repository_index Build a deterministic, resource-bounded repository index.",
        );
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!(
            anchors[0].source,
            "file:entroly/repository_intelligence/graph.py"
        );
        assert!(anchors[0].signature.contains("parsed: object"));
        assert!(anchors[0].signature.contains("limits: object"));
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

    #[test]
    fn anchored_renderer_never_splits_protected_signature() {
        let anchor = "def target(\n    alpha: int,\n    beta: int,\n) -> int:".to_string();
        let sentences = vec![
            "This relevant sentence is intentionally long enough for selection.".to_string(),
            "Another selected sentence would exceed a deliberately small budget.".to_string(),
        ];
        let budget = approx_tokens(&anchor) as i64 + 2;
        let rendered =
            render_dependency_anchored_excerpt(&[anchor.clone()], &sentences, &[0, 1], budget);
        assert!(protected_anchors_survive(&rendered, &[anchor.clone()]));
        assert!(approx_tokens(&rendered) as i64 <= budget, "{rendered}");
        assert!(rendered.contains("beta: int"), "{rendered}");
    }
}
