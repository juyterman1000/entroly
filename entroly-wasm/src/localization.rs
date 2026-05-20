//! Tier-0 file localizer — Rust port of `entroly/localization.py`.
//!
//! Mirrors the Python `Tier0Localizer.rerank_edit_target` byte-faithfully
//! so npm/WASM users get the SAME engine_s6 behaviour as pip / Python-MCP.
//!
//! The rerank is a deterministic structural prior over an existing base
//! ranking (typically the BM25 recall):
//!
//!   1. Window-only top-`EDIT_WINDOW` (20) — tail preserved (recall floor).
//!   2. Explicit cues frozen at top: S1 path/traceback hits + unique
//!      symbol definers, in extraction order. Never re-ordered later.
//!   3. Class re-prio inside the window: source > test > non-source.
//!   4. Intent guards: `_DOC_INTENT_WORDS` in the issue collapses
//!      non-source class → source; `_TEST_INTENT_WORDS` collapses
//!      test class → source. Both prevent legitimate doc/test issues
//!      from being demoted.
//!   5. Test → source mirror: each in-window test promotes its
//!      basename-mirror source file (if any) immediately after it.
//!
//! Validation: SWE-bench Lite n=36 seed=7 paired McNemar over the
//! engine_s5 baseline — b=0 / c=5 at hit@10 (one-sided p=0.031, strict
//! Pareto improvement). See `bench/_engine_s6_paired.py`.
//!
//! No `regex` dependency: the Python regexes are byte-level patterns we
//! re-implement with manual scans. Same outputs on the same inputs.

use std::collections::{BTreeSet, HashMap, HashSet};

// ── Constants (mirror localization.py) ──────────────────────────────────

pub const EDIT_WINDOW: usize = 20;

const NS_EXTS: &[&str] = &[
    ".rst", ".md", ".txt", ".yml", ".yaml", ".cfg", ".toml", ".ini",
];
const NS_PATH_HINTS: &[&str] = &[
    "docs/",
    "/doc/",
    ".github/",
    "/examples/",
    "/tutorial/",
    "/tutorials/",
];
const NS_NAME_HINTS: &[&str] = &["history", "changelog", "changes", "authors", "contributors"];
const TEST_PATH_HINTS: &[&str] = &["/tests/", "tests/", "/test/"];

const DOC_INTENT_WORDS: &[&str] = &[
    "documentation",
    "docstring",
    "tutorial",
    "readme",
    "rst",
    "the docs",
    "configuration file",
    " config file",
    "changelog",
    "yaml",
    " toml ",
    " ini ",
    "rst_prolog",
];
const TEST_INTENT_WORDS: &[&str] = &[
    "unit test",
    "unit-test",
    "failing test",
    "test failure",
    "test case",
    "pytest fixture",
    "conftest",
    "broken test",
    "test that ",
    "regression test",
];

// ── Classifiers (mirror Tier0Localizer._is_*) ───────────────────────────

fn lower(s: &str) -> String {
    s.to_ascii_lowercase()
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn basename_noext(path: &str) -> String {
    let base = basename(path);
    match base.rfind('.') {
        Some(i) => base[..i].to_string(),
        None => base.to_string(),
    }
}

fn is_test_path(path: &str) -> bool {
    let p = lower(path);
    if TEST_PATH_HINTS.iter().any(|h| p.contains(h)) {
        return true;
    }
    let b = basename(&p);
    b.starts_with("test_") || b.ends_with("_test.py") || b == "conftest.py"
}

fn is_non_source(path: &str) -> bool {
    let p = lower(path);
    if NS_EXTS.iter().any(|e| p.ends_with(e)) {
        return true;
    }
    if NS_PATH_HINTS.iter().any(|h| p.contains(h)) {
        return true;
    }
    let base = basename_noext(&p);
    NS_NAME_HINTS.iter().any(|h| *h == base)
}

/// Returns the corpus path of a test file's basename-mirror source,
/// or None. e.g. `tests/test_auth.py` → `pkg/auth.py` if present and
/// not itself a test or non-source. Deterministic: first matching
/// corpus entry in iteration order.
fn test_mirror<'a>(test_path: &str, paths: &'a [String]) -> Option<&'a String> {
    let mut base = basename_noext(test_path).to_ascii_lowercase();
    if let Some(rest) = base.strip_prefix("test_") {
        base = rest.to_string();
    } else if let Some(rest) = base.strip_suffix("_test") {
        base = rest.to_string();
    }
    if base.is_empty() {
        return None;
    }
    let target_suffix = format!("/{}.py", base);
    let target_root = format!("{}.py", base);
    for p in paths {
        let pl = lower(p);
        if (pl.ends_with(&target_suffix) || pl == target_root)
            && !is_test_path(&pl)
            && !is_non_source(&pl)
        {
            return Some(p);
        }
    }
    None
}

// ── S1 explicit-cue extraction (manual byte scans, no regex crate) ──────

/// Find substrings of the issue that look like `*.py` paths. Mirrors
/// Python's `_TRACEBACK_FILE_RE` + `_PATHLIKE_RE` outputs combined,
/// in occurrence order, with traceback paths preferred.
fn find_paths(issue: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = issue.as_bytes();

    // Pass 1: `File "....py"` traceback frames (Python: _TRACEBACK_FILE_RE).
    let mut i = 0usize;
    while i + 6 < bytes.len() {
        if &bytes[i..i + 6] == b"File \"" {
            let start = i + 6;
            if let Some(end_rel) = bytes[start..].iter().position(|&b| b == b'"') {
                let path = &issue[start..start + end_rel];
                if path.ends_with(".py") {
                    out.push(path.to_string());
                }
                i = start + end_rel + 1;
                continue;
            }
        }
        i += 1;
    }

    // Pass 2: bare `\b[\w/]+\.py\b` (Python: _PATHLIKE_RE).
    fn is_pathword(c: u8) -> bool {
        c.is_ascii_alphanumeric() || c == b'_' || c == b'/'
    }
    let mut j = 0usize;
    while j < bytes.len() {
        if is_pathword(bytes[j]) && (j == 0 || !is_pathword(bytes[j - 1])) {
            let start = j;
            while j < bytes.len() && is_pathword(bytes[j]) {
                j += 1;
            }
            // optional `.py`
            if j + 3 <= bytes.len() && &bytes[j..j + 3] == b".py" {
                let end = j + 3;
                let after_ok = end == bytes.len() || !is_pathword(bytes[end]);
                if after_ok {
                    let cand = &issue[start..end];
                    if !out.iter().any(|p| p == cand) {
                        out.push(cand.to_string());
                    }
                    j = end;
                    continue;
                }
            }
            continue;
        }
        j += 1;
    }
    out
}

/// Dotted-module references `a.b.c` → candidate paths `a/b/c.py` and
/// `a/b/c/__init__.py`. Mirrors Python's `_DOTTED_RE` + `_module_to_paths`.
fn dotted_module_paths(issue: &str) -> Vec<String> {
    let bytes = issue.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let is_id_start = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let is_id_cont = |c: u8| c.is_ascii_alphanumeric() || c == b'_';

    let mut i = 0usize;
    while i < bytes.len() {
        if !is_id_start(bytes[i]) || (i > 0 && (is_id_cont(bytes[i - 1]) || bytes[i - 1] == b'.')) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_id_cont(bytes[i]) {
            i += 1;
        }
        // collect dot-separated continuation
        let mut segs = 0;
        while i < bytes.len()
            && bytes[i] == b'.'
            && i + 1 < bytes.len()
            && is_id_start(bytes[i + 1])
        {
            i += 1;
            let s = i;
            while i < bytes.len() && is_id_cont(bytes[i]) {
                i += 1;
            }
            if i > s {
                segs += 1;
                if segs >= 6 {
                    break;
                }
            }
        }
        if segs >= 1 {
            let dotted = &issue[start..i];
            let mut joined = String::with_capacity(dotted.len());
            for (k, part) in dotted.split('.').enumerate() {
                if k > 0 {
                    joined.push('/');
                }
                joined.push_str(part);
            }
            out.push(format!("{}.py", joined));
            out.push(format!("{}/__init__.py", joined));
        }
    }
    out
}

// ── Symbol-definer index (mirror Tier0Localizer.sym_def) ────────────────

/// Extract `def Name` / `class Name` from a file's content. Mirrors
/// Python's `_DEF_RE` / `_CLASS_RE` (multiline anchored).
fn extract_defined_symbols(content: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        for kw in &["async def ", "def ", "class "] {
            if let Some(rest) = trimmed.strip_prefix(kw) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty()
                    && name
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_alphabetic() || c == '_')
                        .unwrap_or(false)
                {
                    out.push(name);
                }
                break;
            }
        }
    }
    out
}

fn build_symbol_definers(files: &[(String, String)]) -> HashMap<String, Vec<String>> {
    let mut sym: HashMap<String, BTreeSet<String>> = HashMap::new();
    for (path, content) in files {
        for s in extract_defined_symbols(content) {
            sym.entry(s).or_default().insert(path.clone());
        }
    }
    sym.into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect()
}

/// Identifiers in backticks (Python `_BACKTICK_RE` + `_IDENT_RE`)
/// plus bare identifiers ≥4 chars.
fn extract_issue_identifiers(issue: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = issue.as_bytes();
    let is_id_start = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let is_id_cont = |c: u8| c.is_ascii_alphanumeric() || c == b'_';

    // pass 1: inside backticks
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i + 1;
            if let Some(end_rel) = bytes[start..].iter().position(|&b| b == b'`') {
                let inner = &issue[start..start + end_rel];
                let mut j = 0usize;
                let ib = inner.as_bytes();
                while j < ib.len() {
                    if is_id_start(ib[j]) {
                        let s = j;
                        while j < ib.len() && is_id_cont(ib[j]) {
                            j += 1;
                        }
                        if j - s >= 4 {
                            out.push(inner[s..j].to_string());
                        }
                    } else {
                        j += 1;
                    }
                }
                i = start + end_rel + 1;
                continue;
            }
        }
        i += 1;
    }

    // pass 2: bare identifiers length>=4
    let mut k = 0usize;
    while k < bytes.len() {
        if is_id_start(bytes[k]) && (k == 0 || !is_id_cont(bytes[k - 1])) {
            let s = k;
            while k < bytes.len() && is_id_cont(bytes[k]) {
                k += 1;
            }
            if k - s >= 4 {
                out.push(issue[s..k].to_string());
            }
        } else {
            k += 1;
        }
    }
    out
}

// ── Public API: rerank_edit_target ──────────────────────────────────────

/// Deterministic edit-target rerank on top of `base_ranked`. See the
/// module docstring for the spec; mirrors `Tier0Localizer.rerank_edit_target`.
///
/// `files` is the `{path: content}` corpus the base ranking was drawn
/// from. `base_ranked` is the input order; the output is at most `k`
/// entries (cues first, then reranked window, then preserved tail,
/// then test-mirror promotions). Pure function — no I/O, no shared state.
pub fn rerank_edit_target(
    files: &[(String, String)],
    base_ranked: &[String],
    issue: &str,
    k: usize,
) -> Vec<String> {
    if base_ranked.is_empty() {
        return Vec::new();
    }
    let paths: Vec<String> = files.iter().map(|(p, _)| p.clone()).collect();
    let path_set: HashSet<&str> = paths.iter().map(|s| s.as_str()).collect();

    let win = EDIT_WINDOW.min(base_ranked.len());
    let window: &[String] = &base_ranked[..win];
    let tail: &[String] = &base_ranked[win..];

    let ilow = lower(issue);
    let doc_intent = DOC_INTENT_WORDS.iter().any(|w| ilow.contains(w));
    let test_intent = TEST_INTENT_WORDS.iter().any(|w| ilow.contains(w));

    // ── Frozen explicit cues — S1 path/traceback + unique-symbol definers
    let mut cue_order: Vec<String> = Vec::new();
    let mut cue_set: HashSet<String> = HashSet::new();
    let add_cue = |p: String, order: &mut Vec<String>, set: &mut HashSet<String>| {
        if path_set.contains(p.as_str()) && !set.contains(&p) {
            set.insert(p.clone());
            order.push(p);
        }
    };

    // direct hits, traceback frames first (handled in find_paths order)
    for raw in find_paths(issue) {
        let stripped = raw.strip_prefix("./").unwrap_or(&raw).to_string();
        if path_set.contains(stripped.as_str()) {
            add_cue(stripped, &mut cue_order, &mut cue_set);
        } else {
            // suffix-match: `path.endswith("/" + raw) or path.endswith(raw)`
            let needle_slash = format!("/{}", stripped);
            for p in paths.iter() {
                if p.ends_with(&needle_slash) || p.ends_with(&stripped) {
                    add_cue(p.clone(), &mut cue_order, &mut cue_set);
                    break;
                }
            }
        }
    }

    // dotted modules
    for cand in dotted_module_paths(issue) {
        if path_set.contains(cand.as_str()) {
            add_cue(cand, &mut cue_order, &mut cue_set);
        }
    }

    // unique-symbol definers
    let sym_def = build_symbol_definers(files);
    let mut seen_syms: HashSet<String> = HashSet::new();
    for s in extract_issue_identifiers(issue) {
        if !seen_syms.insert(s.clone()) {
            continue;
        }
        if let Some(definers) = sym_def.get(&s) {
            if definers.len() == 1 {
                let f = definers[0].clone();
                add_cue(f, &mut cue_order, &mut cue_set);
            }
        }
    }

    // ── Re-classify the remaining window ──
    // 0 = cue (handled above); 1 = source, 2 = test, 3 = non-source.
    // Intent guards collapse classes 2/3 to 1.
    fn classify(p: &str, doc_intent: bool, test_intent: bool) -> u8 {
        if is_non_source(p) {
            if doc_intent {
                1
            } else {
                3
            }
        } else if is_test_path(p) {
            if test_intent {
                1
            } else {
                2
            }
        } else {
            1
        }
    }

    let rest: Vec<&String> = window.iter().filter(|p| !cue_set.contains(*p)).collect();
    let mut indexed: Vec<(u8, usize, &String)> = rest
        .iter()
        .enumerate()
        .map(|(i, p)| (classify(p, doc_intent, test_intent), i, *p))
        .collect();
    // Stable sort by (class asc, orig_pos asc) — matches Python `sorted(...)`.
    indexed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // ── Build output: cues → reranked window (with test-mirror inserts)
    //                  → tail (recall floor)
    let mut built: Vec<String> = cue_order.clone();
    let mut seen: HashSet<String> = cue_set.clone();
    for (_, _, p) in indexed {
        if !seen.contains(p) {
            built.push(p.clone());
            seen.insert(p.clone());
        }
        if is_test_path(p) {
            if let Some(mirror) = test_mirror(p, &paths) {
                if !seen.contains(mirror) {
                    built.push(mirror.clone());
                    seen.insert(mirror.clone());
                }
            }
        }
    }
    for p in tail {
        if !seen.contains(p) {
            built.push(p.clone());
            seen.insert(p.clone());
        }
    }
    built.truncate(k);
    built
}

// ── Rust unit tests (mirror the Python deterministic-guard tests) ───────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_repo() -> Vec<(String, String)> {
        vec![
            ("pkg/auth.py", "def login():\n    return True\n"),
            ("pkg/cache.py", "def lookup():\n    pass\n"),
            ("pkg/router.py", "def route():\n    pass\n"),
            ("pkg/util.py", "def helper():\n    pass\n"),
            ("tests/test_auth.py", "def test_login():\n    assert True\n"),
            (
                "tests/test_cache.py",
                "def test_lookup():\n    assert True\n",
            ),
            ("docs/index.rst", "Welcome to the docs.\n"),
            ("docs/intro.md", "# Intro\n"),
            ("HISTORY.rst", "0.1.0 first release\n"),
            ("CHANGELOG.md", "## 0.1.0\n"),
            (".github/ISSUE_TEMPLATE/bug_report.yml", "name: Bug\n"),
        ]
        .into_iter()
        .map(|(p, c)| (p.to_string(), c.to_string()))
        .collect()
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn empty_input_returns_empty() {
        let files = make_repo();
        let out = rerank_edit_target(&files, &[], "anything", 20);
        assert!(out.is_empty());
    }

    #[test]
    fn source_outranks_non_source_in_window() {
        let files = make_repo();
        let base = s(&[
            "docs/index.rst",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            "HISTORY.rst",
            "pkg/auth.py",
            "pkg/util.py",
        ]);
        let out = rerank_edit_target(&files, &base, "the login flow is broken", 20);
        let pos = |needle: &str| out.iter().position(|x| x == needle).unwrap();
        let src_max = pos("pkg/auth.py").max(pos("pkg/util.py"));
        let ns_min = pos("docs/index.rst")
            .min(pos("HISTORY.rst"))
            .min(pos(".github/ISSUE_TEMPLATE/bug_report.yml"));
        assert!(src_max < ns_min);
    }

    #[test]
    fn doc_intent_guard_keeps_docs() {
        let files = make_repo();
        let base = s(&["docs/index.rst", "pkg/util.py"]);
        let out = rerank_edit_target(
            &files,
            &base,
            "the documentation is misleading and needs updating",
            20,
        );
        let a = out.iter().position(|x| x == "docs/index.rst").unwrap();
        let b = out.iter().position(|x| x == "pkg/util.py").unwrap();
        assert!(a < b);
    }

    #[test]
    fn test_intent_guard_keeps_tests() {
        let files = make_repo();
        let base = s(&["tests/test_auth.py", "pkg/util.py"]);
        let out = rerank_edit_target(
            &files,
            &base,
            "the failing test does not catch the regression",
            20,
        );
        let a = out.iter().position(|x| x == "tests/test_auth.py").unwrap();
        let b = out.iter().position(|x| x == "pkg/util.py").unwrap();
        assert!(a < b);
    }

    #[test]
    fn explicit_path_cue_frozen_first() {
        let files = make_repo();
        let base = s(&["docs/index.rst", "pkg/auth.py", "pkg/router.py"]);
        let out = rerank_edit_target(
            &files,
            &base,
            "see pkg/router.py - route() never returns",
            20,
        );
        assert_eq!(out[0], "pkg/router.py");
    }

    #[test]
    fn cue_order_follows_extraction_order() {
        let files = make_repo();
        let base = s(&["pkg/util.py", "pkg/auth.py", "pkg/router.py"]);
        let out = rerank_edit_target(
            &files,
            &base,
            "first pkg/router.py then pkg/auth.py fail",
            20,
        );
        assert_eq!(
            &out[..2],
            &["pkg/router.py".to_string(), "pkg/auth.py".to_string()]
        );
    }

    #[test]
    fn test_to_source_mirror_promotion() {
        let files = make_repo();
        let base = s(&["tests/test_auth.py", "pkg/util.py"]);
        let out = rerank_edit_target(&files, &base, "auth logic is wrong", 20);
        let pu = out.iter().position(|x| x == "pkg/util.py").unwrap();
        let pt = out.iter().position(|x| x == "tests/test_auth.py").unwrap();
        let pa = out.iter().position(|x| x == "pkg/auth.py").unwrap();
        // util(src) < test < mirror
        assert!(pu < pt);
        assert_eq!(pa, pt + 1);
    }

    #[test]
    fn window_guard_tail_untouched() {
        let files = make_repo();
        let mut base: Vec<String> = (0..22).map(|i| format!("pkg/_pad_{}.py", i)).collect();
        // re-inject some real paths in the window
        base[0] = "docs/index.rst".to_string();
        base[1] = "pkg/auth.py".to_string();
        let out = rerank_edit_target(&files, &base, "router crash", 50);
        // tail two preserve relative order
        let t0 = out.iter().position(|x| x == &base[20]).unwrap();
        let t1 = out.iter().position(|x| x == &base[21]).unwrap();
        assert!(t0 < t1);
    }

    #[test]
    fn pathological_does_not_crash() {
        let files = make_repo();
        let base = s(&["pkg/auth.py", "docs/index.rst", "tests/test_auth.py"]);
        for bad in &[
            "",
            "```\n```",
            "::::....",
            "0 AND OR ((:::***",
            "\u{0}\u{1}\u{2}",
        ] {
            let out = rerank_edit_target(&files, &base, bad, 20);
            // recall-safe
            for b in &base {
                assert!(out.contains(b), "missing {} for input {:?}", b, bad);
            }
        }
    }

    #[test]
    fn deterministic() {
        let files = make_repo();
        let base = s(&[
            "docs/index.rst",
            "pkg/auth.py",
            "tests/test_auth.py",
            "pkg/util.py",
            "HISTORY.rst",
        ]);
        let q = "login broken when cache empty";
        let a = rerank_edit_target(&files, &base, q, 20);
        let b = rerank_edit_target(&files, &base, q, 20);
        assert_eq!(a, b);
    }
}
