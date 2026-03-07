/// Skeleton Extraction — structural outline of code fragments.
///
/// Extracts function signatures, class/struct definitions, imports, and
/// top-level declarations while stripping function bodies.  This produces
/// a "skeleton" that carries ~90% of the structural information at ~10-30%
/// of the token cost.
///
/// Used by hierarchical fragmentation: the knapsack optimizer can choose
/// the full fragment when budget allows, or fall back to the skeleton
/// when the budget is tight.
///
/// Design choices:
///   - Pattern-based (no regex, no tree-sitter) — matches depgraph.rs style
///   - Language detected from source file extension
///   - Returns None for unknown languages or when skeleton is >70% of original

/// Detect language from source file extension.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Lang {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Unknown,
}

fn detect_lang(source: &str) -> Lang {
    let lower = source.to_lowercase();
    if lower.ends_with(".py") || lower.ends_with(".pyw") {
        Lang::Python
    } else if lower.ends_with(".rs") {
        Lang::Rust
    } else if lower.ends_with(".js") || lower.ends_with(".jsx") || lower.ends_with(".mjs") {
        Lang::JavaScript
    } else if lower.ends_with(".ts") || lower.ends_with(".tsx") {
        Lang::TypeScript
    } else {
        Lang::Unknown
    }
}

/// Extract a skeleton (structural outline) from a code fragment.
///
/// Returns `None` if:
///   - Language is not recognized
///   - Content is too short to benefit from skeletonization
///   - Skeleton would be >70% of original (not worth the overhead)
pub fn extract_skeleton(content: &str, source: &str) -> Option<String> {
    let lang = detect_lang(source);
    if lang == Lang::Unknown {
        return None;
    }

    // Skip very short fragments (< 5 lines)
    let line_count = content.lines().count();
    if line_count < 5 {
        return None;
    }

    let skeleton = match lang {
        Lang::Python => extract_python_skeleton(content),
        Lang::Rust => extract_rust_skeleton(content),
        Lang::JavaScript | Lang::TypeScript => extract_js_skeleton(content),
        Lang::Unknown => return None,
    };

    // Skip if skeleton is >70% of original (not worth it)
    if skeleton.len() as f64 > content.len() as f64 * 0.70 {
        return None;
    }

    // Skip if skeleton is empty or trivially small
    if skeleton.trim().is_empty() || skeleton.len() < 10 {
        return None;
    }

    Some(skeleton)
}

/// Python skeleton: keep imports, decorators, class/def signatures, top-level assignments.
/// Replace function/method bodies with `...`
fn extract_python_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let indent = leading_spaces(line);

        // Module docstring at top (keep first triple-quote block)
        if i == 0 && (trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''")) {
            let quote = &trimmed[..3];
            out.push(line.to_string());
            if !trimmed[3..].contains(quote) {
                // Multi-line docstring — find closing
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains(quote) {
                        break;
                    }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Blank lines — keep between top-level items for readability
        if trimmed.is_empty() {
            if !out.is_empty() {
                out.push(String::new());
            }
            i += 1;
            continue;
        }

        // Comments at top level — keep
        if trimmed.starts_with('#') && indent == 0 {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Imports — always keep
        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            out.push(line.to_string());
            // Handle continuation lines
            i += 1;
            while i < lines.len() && lines[i].trim().starts_with(',') {
                out.push(lines[i].to_string());
                i += 1;
            }
            continue;
        }

        // Decorators — keep (they're part of the signature)
        if trimmed.starts_with('@') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Class definition — keep the signature line
        if trimmed.starts_with("class ") && trimmed.contains(':') {
            out.push(line.to_string());
            i += 1;
            // Keep the docstring if present
            if i < lines.len() {
                let next = lines[i].trim();
                if next.starts_with("\"\"\"") || next.starts_with("'''") {
                    let quote = &next[..3];
                    out.push(lines[i].to_string());
                    if !next[3..].contains(quote) {
                        i += 1;
                        while i < lines.len() {
                            out.push(lines[i].to_string());
                            if lines[i].contains(quote) {
                                break;
                            }
                            i += 1;
                        }
                    }
                    i += 1;
                }
            }
            continue;
        }

        // Function/method definition — keep signature, skip body
        if (trimmed.starts_with("def ") || trimmed.starts_with("async def "))
            && trimmed.contains(':')
        {
            out.push(line.to_string());
            let def_indent = indent;
            i += 1;

            // Keep the docstring if present
            if i < lines.len() {
                let next = lines[i].trim();
                if next.starts_with("\"\"\"") || next.starts_with("'''") {
                    let quote = &next[..3];
                    out.push(lines[i].to_string());
                    if !next[3..].contains(quote) {
                        i += 1;
                        while i < lines.len() {
                            out.push(lines[i].to_string());
                            if lines[i].contains(quote) {
                                break;
                            }
                            i += 1;
                        }
                    }
                    i += 1;
                }
            }

            // Add placeholder and skip the body
            let body_indent = " ".repeat(def_indent + 4);
            out.push(format!("{}...", body_indent));

            // Skip body lines (any line indented deeper than the def)
            while i < lines.len() {
                let next_trimmed = lines[i].trim();
                let next_indent = leading_spaces(lines[i]);
                if next_trimmed.is_empty() {
                    i += 1;
                    continue;
                }
                if next_indent <= def_indent {
                    break; // Back to same or outer level
                }
                i += 1;
            }
            continue;
        }

        // Top-level assignments and constants — keep
        if indent == 0 && (trimmed.contains('=') || trimmed.starts_with("__")) {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Type annotations at top level
        if indent == 0 && trimmed.contains(':') && !trimmed.starts_with('#') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Everything else — skip
        i += 1;
    }

    // Remove trailing blank lines
    while out.last().map_or(false, |l| l.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

/// Rust skeleton: keep use/mod, pub struct/enum/trait/fn signatures, impl blocks.
/// Replace `{ ... }` function bodies with `{ ... }`
fn extract_rust_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Blank lines
        if trimmed.is_empty() {
            if !out.is_empty() {
                out.push(String::new());
            }
            i += 1;
            continue;
        }

        // Doc comments (/// and //!)
        if trimmed.starts_with("///") || trimmed.starts_with("//!") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Module-level attributes (#[...])
        if trimmed.starts_with("#[") || trimmed.starts_with("#![") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // use/mod/extern statements
        if trimmed.starts_with("use ") || trimmed.starts_with("mod ")
            || trimmed.starts_with("extern ")
        {
            out.push(line.to_string());
            // Handle multi-line use with braces
            if trimmed.contains('{') && !trimmed.contains('}') {
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains('}') {
                        break;
                    }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // const/static/type aliases
        if trimmed.starts_with("const ") || trimmed.starts_with("static ")
            || trimmed.starts_with("pub const ") || trimmed.starts_with("pub static ")
            || trimmed.starts_with("type ") || trimmed.starts_with("pub type ")
        {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Struct/enum/trait definitions — keep the signature + fields for structs
        if is_rust_type_def(trimmed) {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut brace_depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if brace_depth > 0 {
                    // Keep struct/enum fields (they're part of the type signature)
                    i += 1;
                    while i < lines.len() && brace_depth > 0 {
                        let l = lines[i];
                        brace_depth += count_char(l, '{') as i32 - count_char(l, '}') as i32;
                        out.push(l.to_string());
                        i += 1;
                    }
                    continue;
                }
            } else if trimmed.ends_with(';') {
                // Unit struct or type alias
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }

        // impl blocks — keep the impl header + fn signatures
        if trimmed.starts_with("impl ") || trimmed.starts_with("impl<") {
            out.push(line.to_string());
            i += 1;
            let mut brace_depth = if trimmed.contains('{') {
                count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32
            } else {
                0
            };

            while i < lines.len() {
                let l = lines[i];
                let t = l.trim();

                if brace_depth <= 0 && t.contains('{') {
                    brace_depth += count_char(t, '{') as i32 - count_char(t, '}') as i32;
                    if !out.last().map_or(false, |last| last.contains('{')) {
                        out.push(l.to_string());
                    }
                    i += 1;
                    continue;
                }

                brace_depth += count_char(t, '{') as i32 - count_char(t, '}') as i32;

                // fn signature inside impl
                if (t.starts_with("pub fn ") || t.starts_with("fn ")
                    || t.starts_with("pub async fn ") || t.starts_with("async fn ")
                    || t.starts_with("pub(crate) fn "))
                {
                    // Output the signature
                    out.push(l.to_string());
                    // Skip the body
                    if t.contains('{') {
                        let mut fn_depth = count_char(t, '{') as i32 - count_char(t, '}') as i32;
                        if fn_depth > 0 {
                            out.push(format!("{}    ...", &l[..leading_spaces(l)]));
                            i += 1;
                            while i < lines.len() && fn_depth > 0 {
                                fn_depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                                if fn_depth <= 0 {
                                    out.push(format!("{}}}", &" ".repeat(leading_spaces(l))));
                                }
                                i += 1;
                            }
                            continue;
                        }
                    }
                } else if t == "}" && brace_depth <= 0 {
                    out.push(l.to_string());
                    i += 1;
                    break;
                } else if t.starts_with("///") || t.starts_with("#[") {
                    out.push(l.to_string());
                }

                i += 1;
            }
            continue;
        }

        // Free-standing fn definitions
        if (trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub async fn ") || trimmed.starts_with("async fn ")
            || trimmed.starts_with("pub(crate) fn "))
        {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut fn_depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if fn_depth > 0 {
                    let indent_str = " ".repeat(leading_spaces(line));
                    out.push(format!("{}    ...", indent_str));
                    i += 1;
                    while i < lines.len() && fn_depth > 0 {
                        fn_depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                        i += 1;
                    }
                    out.push(format!("{}}}", indent_str));
                    continue;
                }
            }
            i += 1;
            continue;
        }

        // Skip everything else
        i += 1;
    }

    while out.last().map_or(false, |l| l.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

/// JS/TS skeleton: keep imports, exports, class/function signatures, top-level const/let.
fn extract_js_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let indent = leading_spaces(line);

        // Blank lines
        if trimmed.is_empty() {
            if !out.is_empty() {
                out.push(String::new());
            }
            i += 1;
            continue;
        }

        // JSDoc comments (/** ... */)
        if trimmed.starts_with("/**") {
            out.push(line.to_string());
            if !trimmed.contains("*/") {
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains("*/") {
                        break;
                    }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Import statements (not export class/function/const/etc.)
        if trimmed.starts_with("import ") {
            let mut stmt = line.to_string();
            i += 1;
            while !stmt.contains(';') && !stmt.contains(" from ") && i < lines.len() {
                stmt.push('\n');
                stmt.push_str(lines[i]);
                i += 1;
            }
            out.push(stmt);
            continue;
        }

        // Top-level function declarations
        if indent == 0 && (trimmed.starts_with("function ") || trimmed.starts_with("async function ")
            || trimmed.starts_with("export function ") || trimmed.starts_with("export async function ")
            || trimmed.starts_with("export default function "))
        {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut fn_depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if fn_depth > 0 {
                    out.push("    ...".to_string());
                    i += 1;
                    while i < lines.len() && fn_depth > 0 {
                        fn_depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                        i += 1;
                    }
                    out.push("}".to_string());
                    continue;
                }
            }
            i += 1;
            continue;
        }

        // Top-level class declarations
        if indent == 0 && (trimmed.starts_with("class ") || trimmed.starts_with("export class ")
            || trimmed.starts_with("export default class "))
        {
            out.push(line.to_string());
            i += 1;
            let mut brace_depth = if trimmed.contains('{') {
                count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32
            } else {
                0
            };

            // Keep method signatures inside class.
            // IMPORTANT: brace_depth tracks the class-level nesting.
            // When we skip a method body, we must update brace_depth for
            // each line consumed so the class-close `}` is detected correctly.
            while i < lines.len() {
                let l = lines[i];
                let t = l.trim();

                // Update class-level brace depth for this line
                let line_delta = count_char(t, '{') as i32 - count_char(t, '}') as i32;
                brace_depth += line_delta;

                if brace_depth <= 0 {
                    // This is the class-closing `}`
                    out.push(l.to_string());
                    i += 1;
                    break;
                }

                // Method / constructor signature — keep sig, strip body
                if is_js_method_sig(t) && t.contains('{') {
                    out.push(l.to_string());
                    // method_depth tracks depth *inside* this method.
                    // brace_depth already accounts for this line's `{`.
                    // We need to consume lines until those braces close.
                    let mut method_depth = line_delta; // net open braces on sig line
                    if method_depth > 0 {
                        let indent_str = " ".repeat(leading_spaces(l));
                        out.push(format!("{}    ...", indent_str));
                        i += 1;
                        while i < lines.len() && method_depth > 0 {
                            let body_delta = count_char(lines[i], '{') as i32
                                - count_char(lines[i], '}') as i32;
                            method_depth += body_delta;
                            // These lines are also class-body lines — keep brace_depth in sync
                            brace_depth += body_delta;
                            i += 1;
                        }
                        out.push(format!("{}}}", indent_str));
                        continue;
                    }
                }

                i += 1;
            }
            continue;
        }

        // Top-level const/let/var (type definitions, config, etc.)
        if indent == 0 && (trimmed.starts_with("const ") || trimmed.starts_with("let ")
            || trimmed.starts_with("var ") || trimmed.starts_with("export const ")
            || trimmed.starts_with("export let "))
        {
            // Keep just the declaration line (not multi-line object literals)
            out.push(line.to_string());
            if trimmed.contains('{') && !trimmed.contains('}') {
                // Skip the object body
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                i += 1;
                while i < lines.len() && depth > 0 {
                    depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                    i += 1;
                }
                out.push("};".to_string());
                continue;
            }
            i += 1;
            continue;
        }

        // Interface/type declarations (TypeScript)
        if indent == 0 && (trimmed.starts_with("interface ") || trimmed.starts_with("export interface ")
            || trimmed.starts_with("type ") || trimmed.starts_with("export type "))
        {
            out.push(line.to_string());
            if trimmed.contains('{') && !trimmed.contains('}') {
                // Keep full interface body (it's all type info)
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                i += 1;
                while i < lines.len() && depth > 0 {
                    depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                    out.push(lines[i].to_string());
                    i += 1;
                }
                continue;
            }
            i += 1;
            continue;
        }

        // Skip everything else
        i += 1;
    }

    while out.last().map_or(false, |l| l.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn leading_spaces(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

fn is_rust_type_def(trimmed: &str) -> bool {
    trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub enum ") || trimmed.starts_with("enum ")
        || trimmed.starts_with("pub trait ") || trimmed.starts_with("trait ")
        || trimmed.starts_with("pub(crate) struct ") || trimmed.starts_with("pub(crate) enum ")
}

fn is_js_method_sig(trimmed: &str) -> bool {
    // Matches: async foo(, foo(, get foo(, set foo(, static foo(, constructor(
    if trimmed.starts_with("constructor(") || trimmed.starts_with("async ") {
        return true;
    }
    // Static/get/set prefixes
    let t = if trimmed.starts_with("static ") || trimmed.starts_with("get ")
        || trimmed.starts_with("set ")
    {
        trimmed.splitn(2, ' ').nth(1).unwrap_or("")
    } else {
        trimmed
    };
    // Method name followed by (
    let first_word_end = t.find('(');
    if let Some(pos) = first_word_end {
        let name = &t[..pos];
        !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_')
    } else {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_skeleton() {
        let code = r#"
import os
from pathlib import Path

class Engine:
    """Main engine class."""

    def __init__(self, config):
        """Initialize engine."""
        self.config = config
        self.data = {}
        self._setup()

    def process(self, input_data: str) -> dict:
        """Process input data and return results."""
        result = {}
        for item in input_data.split(','):
            key, val = item.split('=')
            result[key.strip()] = val.strip()
        return result

    async def fetch(self, url: str) -> bytes:
        response = await self.client.get(url)
        return response.content

MAX_RETRIES = 3
"#.trim();

        let skel = extract_skeleton(code, "engine.py").unwrap();
        assert!(skel.contains("import os"), "Should keep imports");
        assert!(skel.contains("from pathlib import Path"), "Should keep from imports");
        assert!(skel.contains("class Engine:"), "Should keep class def");
        assert!(skel.contains("def __init__(self, config):"), "Should keep method signatures");
        assert!(skel.contains("def process(self, input_data: str) -> dict:"), "Should keep typed signatures");
        assert!(skel.contains("async def fetch(self, url: str) -> bytes:"), "Should keep async def");
        assert!(skel.contains("MAX_RETRIES = 3"), "Should keep top-level constants");
        assert!(!skel.contains("self._setup()"), "Should strip method body");
        assert!(!skel.contains("for item in"), "Should strip loop body");
        assert!(skel.contains("..."), "Should have placeholder");
        assert!(skel.len() < code.len(), "Skeleton should be shorter: {} vs {}", skel.len(), code.len());
    }

    #[test]
    fn test_rust_skeleton() {
        let code = r#"
use std::collections::HashMap;

/// A context fragment.
pub struct Fragment {
    pub id: String,
    pub content: String,
    pub score: f64,
}

impl Fragment {
    pub fn new(id: String) -> Self {
        Fragment {
            id,
            content: String::new(),
            score: 0.0,
        }
    }

    pub fn score(&self) -> f64 {
        let base = self.content.len() as f64;
        base * 0.5 + self.score
    }
}

fn helper(x: u32) -> u32 {
    x * 2 + 1
}
"#.trim();

        let skel = extract_skeleton(code, "fragment.rs").unwrap();
        assert!(skel.contains("use std::collections::HashMap;"), "Should keep use");
        assert!(skel.contains("pub struct Fragment"), "Should keep struct");
        assert!(skel.contains("pub id: String"), "Should keep struct fields");
        assert!(skel.contains("impl Fragment"), "Should keep impl");
        assert!(skel.contains("pub fn new(id: String) -> Self"), "Should keep fn signatures");
        assert!(skel.contains("pub fn score(&self) -> f64"), "Should keep fn signatures");
        assert!(skel.contains("fn helper(x: u32) -> u32"), "Should keep free fn");
        assert!(!skel.contains("x * 2 + 1"), "Should strip fn body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_js_skeleton() {
        let code = r#"
import { useState } from 'react';
import axios from 'axios';

export class UserService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.cache = new Map();
    }

    async fetchUser(id) {
        const response = await axios.get(`${this.baseUrl}/users/${id}`);
        return response.data;
    }
}

export function formatDate(date) {
    const d = new Date(date);
    return d.toISOString().split('T')[0];
}

const API_URL = 'https://api.example.com';
"#.trim();

        let skel = extract_skeleton(code, "service.js").unwrap();
        assert!(skel.contains("import { useState } from 'react';"), "Should keep imports");
        assert!(skel.contains("export class UserService"), "Should keep class");
        assert!(skel.contains("constructor(baseUrl)"), "Should keep constructor sig");
        assert!(skel.contains("async fetchUser(id)"), "Should keep method sig");
        assert!(skel.contains("export function formatDate(date)"), "Should keep function sig");
        assert!(skel.contains("const API_URL"), "Should keep top-level const");
        assert!(!skel.contains("new Map()"), "Should strip constructor body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_ts_interface_kept() {
        // Enough body content so skeleton < 70% of original
        let code = r#"
export interface Config {
    debug: boolean;
    logLevel: string;
    maxRetries: number;
    timeout: number;
    retryDelay: number;
}

export type Result<T> = { ok: true; data: T } | { ok: false; error: string };

export function process(config: Config): Result<string> {
    const result = validate(config);
    if (!result.ok) return result;
    const trimmed = result.data.trim();
    if (trimmed.length === 0) {
        return { ok: false, error: 'empty result' };
    }
    return { ok: true, data: trimmed };
}

export function validate(config: Config): Result<Config> {
    if (config.maxRetries < 0) {
        return { ok: false, error: 'maxRetries must be >= 0' };
    }
    if (config.timeout <= 0) {
        return { ok: false, error: 'timeout must be > 0' };
    }
    return { ok: true, data: config };
}
"#.trim();

        let skel = extract_skeleton(code, "types.ts").unwrap();
        assert!(skel.contains("export interface Config"), "Should keep interface");
        assert!(skel.contains("debug: boolean"), "Should keep interface fields");
        assert!(skel.contains("export type Result<T>"), "Should keep type alias");
        assert!(skel.contains("export function process(config: Config): Result<string>"), "Should keep fn sig");
        assert!(skel.contains("export function validate(config: Config): Result<Config>"), "Should keep second fn sig");
        assert!(!skel.contains("trimmed.length"), "Should strip fn body");
        assert!(!skel.contains("maxRetries must be"), "Should strip fn body strings");
        assert!(skel.len() < code.len(), "Skeleton should be shorter than original");
    }

    #[test]
    fn test_unknown_lang_returns_none() {
        assert!(extract_skeleton("some content here\nwith multiple\nlines of text\nfor testing\npurposes", "readme.md").is_none());
        assert!(extract_skeleton("key: value", "config.yaml").is_none());
    }

    #[test]
    fn test_short_content_returns_none() {
        assert!(extract_skeleton("x = 1", "short.py").is_none());
        assert!(extract_skeleton("fn f() {}", "short.rs").is_none());
    }

    #[test]
    fn test_skeleton_too_similar_returns_none() {
        // All imports, no bodies — skeleton ≈ original
        let code = "import os\nimport sys\nimport json\nimport time\nimport math\n";
        assert!(extract_skeleton(code, "imports.py").is_none());
    }
}
