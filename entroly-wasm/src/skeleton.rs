//! Skeleton Extraction — structural outline of code fragments.
//!
//! Extracts function signatures, class/struct definitions, imports, and
//! top-level declarations while stripping function bodies.  This produces
//! a "skeleton" that carries ~90% of the structural information at ~10-30%
//! of the token cost.
//!
//! Used by hierarchical fragmentation: the knapsack optimizer can choose
//! the full fragment when budget allows, or fall back to the skeleton
//! when the budget is tight.
//!
//! Design choices:
//!   - Pattern-based (no regex, no tree-sitter) — matches depgraph.rs style
//!   - Language detected from source file extension
//!   - Returns None for unknown languages or when skeleton is >70% of original

/// Detect language from source file extension.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Lang {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Go,
    Java,
    CSharp,
    Swift,
    Cpp,
    Shell,
    Ruby,
    Php,
    Vue,
    Svelte,
    Html,
    Css,
    Unknown,
}

fn detect_lang(source: &str) -> Lang {
    let lower = source.to_lowercase();
    if lower.ends_with(".py") || lower.ends_with(".pyw") {
        Lang::Python
    } else if lower.ends_with(".rs") {
        Lang::Rust
    } else if lower.ends_with(".js") || lower.ends_with(".jsx") || lower.ends_with(".mjs") || lower.ends_with(".cjs") {
        Lang::JavaScript
    } else if lower.ends_with(".ts") || lower.ends_with(".tsx") || lower.ends_with(".mts") || lower.ends_with(".cts") {
        Lang::TypeScript
    } else if lower.ends_with(".go") {
        Lang::Go
    } else if lower.ends_with(".java") || lower.ends_with(".kt") {
        Lang::Java
    } else if lower.ends_with(".cs") || lower.ends_with(".csx") {
        Lang::CSharp
    } else if lower.ends_with(".swift") {
        Lang::Swift
    } else if lower.ends_with(".c") || lower.ends_with(".cpp") || lower.ends_with(".cc")
        || lower.ends_with(".h") || lower.ends_with(".hpp") || lower.ends_with(".hxx") {
        Lang::Cpp
    } else if lower.ends_with(".sh") || lower.ends_with(".bash") || lower.ends_with(".zsh") {
        Lang::Shell
    } else if lower.ends_with(".rb") {
        Lang::Ruby
    } else if lower.ends_with(".php") {
        Lang::Php
    } else if lower.ends_with(".vue") {
        Lang::Vue
    } else if lower.ends_with(".svelte") {
        Lang::Svelte
    } else if lower.ends_with(".html") || lower.ends_with(".htm") {
        Lang::Html
    } else if lower.ends_with(".css") || lower.ends_with(".scss") || lower.ends_with(".less") {
        Lang::Css
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
        Lang::Go => extract_go_skeleton(content),
        Lang::Java | Lang::CSharp => extract_java_skeleton(content),
        Lang::Swift => extract_swift_skeleton(content),
        Lang::Cpp => extract_cpp_skeleton(content),
        Lang::Shell => extract_shell_skeleton(content),
        Lang::Ruby => extract_ruby_skeleton(content),
        Lang::Php => extract_php_skeleton(content),
        Lang::Vue | Lang::Svelte => extract_sfc_skeleton(content),
        Lang::Html => extract_html_skeleton(content),
        Lang::Css => extract_css_skeleton(content),
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

/// Count leading spaces in a line.
fn leading_spaces(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Count occurrences of a character in a string.
fn count_char(s: &str, ch: char) -> usize {
    s.chars().filter(|&c| c == ch).count()
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
    while out.last().is_some_and(|l: &String| l.trim().is_empty()) {
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
                    if !out.last().is_some_and(|last: &String| last.contains('{')) {
                        out.push(l.to_string());
                    }
                    i += 1;
                    continue;
                }

                brace_depth += count_char(t, '{') as i32 - count_char(t, '}') as i32;

                // fn signature inside impl
                if t.starts_with("pub fn ") || t.starts_with("fn ")
                    || t.starts_with("pub async fn ") || t.starts_with("async fn ")
                    || t.starts_with("pub(crate) fn ")
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
        if trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub async fn ") || trimmed.starts_with("async fn ")
            || trimmed.starts_with("pub(crate) fn ")
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

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) {
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

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) {
        out.pop();
    }

    out.join("\n")
}

/// Go skeleton: keep package, imports, func signatures, type definitions, const blocks.
/// Replace function bodies with `// ...`
fn extract_go_skeleton(content: &str) -> String {
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

        // Comments (// and doc comments)
        if trimmed.starts_with("//") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Package declaration
        if trimmed.starts_with("package ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Import block: import ( ... )
        if trimmed.starts_with("import (") {
            out.push(line.to_string());
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with(')') {
                out.push(lines[i].to_string());
                i += 1;
            }
            if i < lines.len() {
                out.push(lines[i].to_string());
            }
            i += 1;
            continue;
        }

        // Single import
        if trimmed.starts_with("import ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Const/var blocks
        if trimmed.starts_with("const (") || trimmed.starts_with("var (") {
            out.push(line.to_string());
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with(')') {
                out.push(lines[i].to_string());
                i += 1;
            }
            if i < lines.len() {
                out.push(lines[i].to_string());
            }
            i += 1;
            continue;
        }
        if trimmed.starts_with("const ") || trimmed.starts_with("var ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Type definitions: type Foo struct/interface { ... }
        if trimmed.starts_with("type ") {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if depth > 0 {
                    i += 1;
                    while i < lines.len() && depth > 0 {
                        let l = lines[i];
                        depth += count_char(l, '{') as i32 - count_char(l, '}') as i32;
                        // Keep interface method sigs and struct fields
                        let t = l.trim();
                        if !t.is_empty() {
                            out.push(l.to_string());
                        }
                        i += 1;
                    }
                    continue;
                }
            }
            i += 1;
            continue;
        }

        // Function/method definitions — keep signature, skip body
        if trimmed.starts_with("func ") {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if depth > 0 {
                    out.push("\t// ...".to_string());
                    i += 1;
                    while i < lines.len() && depth > 0 {
                        depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                        i += 1;
                    }
                    out.push("}".to_string());
                    continue;
                }
            }
            i += 1;
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Java/Kotlin skeleton: keep package, imports, class/interface signatures,
/// method signatures, annotations. Strip method bodies.
fn extract_java_skeleton(content: &str) -> String {
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

        // Javadoc / block comments
        if trimmed.starts_with("/**") || trimmed.starts_with("/*") {
            out.push(line.to_string());
            if !trimmed.contains("*/") {
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains("*/") { break; }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Line comments
        if trimmed.starts_with("//") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Package / import
        if trimmed.starts_with("package ") || trimmed.starts_with("import ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Annotations (@Override, @Service, etc.)
        if trimmed.starts_with('@') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Class / interface / enum declarations
        if is_java_type_def(trimmed) {
            out.push(line.to_string());
            i += 1;
            // Enter class body — keep method signatures, skip bodies
            if trimmed.contains('{') {
                let mut class_depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                while i < lines.len() && class_depth > 0 {
                    let l = lines[i];
                    let t = l.trim();
                    let delta = count_char(t, '{') as i32 - count_char(t, '}') as i32;
                    class_depth += delta;

                    if class_depth <= 0 {
                        out.push(l.to_string());
                        i += 1;
                        break;
                    }

                    // Annotations
                    if t.starts_with('@') {
                        out.push(l.to_string());
                        i += 1;
                        continue;
                    }

                    // Nested class/interface
                    if is_java_type_def(t) {
                        out.push(l.to_string());
                        i += 1;
                        continue;
                    }

                    // Method signature (has parens, visibility modifier)
                    if t.contains('(') && is_java_method_line(t) {
                        out.push(l.to_string());
                        if t.contains('{') && delta > 0 {
                            let indent = " ".repeat(leading_spaces(l));
                            out.push(format!("{}    // ...", indent));
                            i += 1;
                            let mut method_depth = delta;
                            while i < lines.len() && method_depth > 0 {
                                let md = count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                                method_depth += md;
                                class_depth += md;
                                i += 1;
                            }
                            out.push(format!("{}}}", indent));
                            continue;
                        }
                        i += 1;
                        continue;
                    }

                    // Field declarations (no parens, has semicolon)
                    if t.ends_with(';') && !t.contains('(') {
                        out.push(l.to_string());
                    }

                    i += 1;
                }
            }
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// C/C++ skeleton: keep #include, namespace, class/struct definitions,
/// function signatures. Strip function bodies.
fn extract_cpp_skeleton(content: &str) -> String {
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

        // Comments
        if trimmed.starts_with("//") {
            out.push(line.to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with("/*") {
            out.push(line.to_string());
            if !trimmed.contains("*/") {
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains("*/") { break; }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Preprocessor directives (#include, #define, #pragma, #ifdef, etc.)
        if trimmed.starts_with('#') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // using / typedef
        if trimmed.starts_with("using ") || trimmed.starts_with("typedef ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // namespace
        if trimmed.starts_with("namespace ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // class / struct definition — keep members, skip method bodies
        if (trimmed.starts_with("class ") || trimmed.starts_with("struct ")
            || trimmed.starts_with("template"))
            && (trimmed.contains('{') || !trimmed.ends_with(';'))
        {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                i += 1;
                while i < lines.len() && depth > 0 {
                    let l = lines[i];
                    let t = l.trim();
                    let delta = count_char(t, '{') as i32 - count_char(t, '}') as i32;
                    depth += delta;

                    if depth <= 0 {
                        out.push(l.to_string());
                        i += 1;
                        break;
                    }

                    // Inline method with body — keep sig, skip body
                    if t.contains('(') && t.contains('{') && delta > 0 {
                        out.push(l.to_string());
                        let indent = " ".repeat(leading_spaces(l));
                        out.push(format!("{}    // ...", indent));
                        i += 1;
                        let mut fn_depth = delta;
                        while i < lines.len() && fn_depth > 0 {
                            let fd = count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                            fn_depth += fd;
                            depth += fd;
                            i += 1;
                        }
                        out.push(format!("{}}}", indent));
                        continue;
                    }

                    // Keep declarations and access specifiers
                    if !t.is_empty() {
                        out.push(l.to_string());
                    }
                    i += 1;
                }
                continue;
            }
            i += 1;
            continue;
        }

        // Free function definitions — keep signature, skip body
        if is_cpp_function_def(trimmed) {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if depth > 0 {
                    let indent = " ".repeat(leading_spaces(line));
                    out.push(format!("{}    // ...", indent));
                    i += 1;
                    while i < lines.len() && depth > 0 {
                        depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                        i += 1;
                    }
                    out.push(format!("{}}}", indent));
                    continue;
                }
            }
            i += 1;
            continue;
        }

        // Enum definitions
        if trimmed.starts_with("enum ") {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                i += 1;
                while i < lines.len() && depth > 0 {
                    let l = lines[i];
                    depth += count_char(l, '{') as i32 - count_char(l, '}') as i32;
                    out.push(l.to_string());
                    i += 1;
                }
                continue;
            }
            i += 1;
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Swift skeleton: keep import, class/struct/protocol/enum definitions,
/// func signatures, property declarations. Strip function bodies.
fn extract_swift_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let indent = leading_spaces(line);

        // Blank lines
        if trimmed.is_empty() {
            if !out.is_empty() { out.push(String::new()); }
            i += 1;
            continue;
        }

        // Comments
        if trimmed.starts_with("//") {
            out.push(line.to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with("/*") {
            out.push(line.to_string());
            if !trimmed.contains("*/") {
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains("*/") { break; }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // Import
        if trimmed.starts_with("import ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Attributes (@objc, @available, etc.)
        if trimmed.starts_with('@') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Class / struct / protocol / enum / extension
        if indent == 0 && is_swift_type_def(trimmed) {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                i += 1;
                while i < lines.len() && depth > 0 {
                    let l = lines[i];
                    let t = l.trim();
                    let delta = count_char(t, '{') as i32 - count_char(t, '}') as i32;
                    depth += delta;

                    if depth <= 0 {
                        out.push(l.to_string());
                        i += 1;
                        break;
                    }

                    // func signature — keep sig, skip body
                    if t.starts_with("func ") || t.starts_with("static func ")
                        || t.starts_with("class func ") || t.starts_with("override func ")
                        || t.starts_with("private func ") || t.starts_with("public func ")
                    {
                        out.push(l.to_string());
                        if t.contains('{') && delta > 0 {
                            let ind = " ".repeat(leading_spaces(l));
                            out.push(format!("{}    // ...", ind));
                            i += 1;
                            let mut fn_d = delta;
                            while i < lines.len() && fn_d > 0 {
                                let fd = count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                                fn_d += fd;
                                depth += fd;
                                i += 1;
                            }
                            out.push(format!("{}}}", ind));
                            continue;
                        }
                        i += 1;
                        continue;
                    }

                    // Property declarations (var/let)
                    if (t.starts_with("var ") || t.starts_with("let ")
                        || t.starts_with("private ") || t.starts_with("public "))
                        && !t.contains('{')
                    {
                        out.push(l.to_string());
                    }

                    i += 1;
                }
            } else {
                i += 1;
            }
            continue;
        }

        // Top-level func
        if indent == 0 && (trimmed.starts_with("func ") || trimmed.starts_with("public func ")
            || trimmed.starts_with("private func "))
        {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if depth > 0 {
                    out.push("    // ...".to_string());
                    i += 1;
                    while i < lines.len() && depth > 0 {
                        depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                        i += 1;
                    }
                    out.push("}".to_string());
                    continue;
                }
            }
            i += 1;
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
}

/// Shell skeleton: keep shebang, function definitions (signature only),
/// export/source statements, global variable assignments.
fn extract_shell_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Blank lines
        if trimmed.is_empty() {
            if !out.is_empty() { out.push(String::new()); }
            i += 1;
            continue;
        }

        // Shebang
        if trimmed.starts_with("#!") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Comments
        if trimmed.starts_with('#') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Source / dot-source
        if trimmed.starts_with("source ") || trimmed.starts_with(". ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Export statements
        if trimmed.starts_with("export ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Function definitions: function_name() { or function function_name {
        let is_func = (trimmed.contains("()") && trimmed.contains('{'))
            || trimmed.starts_with("function ");
        if is_func && trimmed.contains('{') {
            out.push(line.to_string());
            let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
            if depth > 0 {
                out.push("    # ...".to_string());
                i += 1;
                while i < lines.len() && depth > 0 {
                    depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                    i += 1;
                }
                out.push("}".to_string());
                continue;
            }
            i += 1;
            continue;
        }

        // Top-level variable assignments (no indentation)
        if leading_spaces(line) == 0 && trimmed.contains('=') && !trimmed.starts_with("if ")
            && !trimmed.starts_with("while ") && !trimmed.starts_with("for ")
        {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

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
        trimmed.split_once(' ').map_or("", |(_, rest)| rest)
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

fn is_java_type_def(trimmed: &str) -> bool {
    let keywords = ["class ", "interface ", "enum "];
    let modifiers = ["public ", "private ", "protected ", "abstract ", "final ",
                     "static ", "sealed ", "open ", "data class "];
    for kw in &keywords {
        if trimmed.starts_with(kw) { return true; }
    }
    for m in &modifiers {
        if trimmed.starts_with(m) {
            for kw in &keywords {
                if trimmed.contains(kw) { return true; }
            }
        }
    }
    false
}

fn is_java_method_line(trimmed: &str) -> bool {
    let modifiers = ["public ", "private ", "protected ", "static ", "final ",
                     "abstract ", "synchronized ", "native ", "override ",
                     "suspend ", "default "];
    modifiers.iter().any(|m| trimmed.starts_with(m))
}

fn is_cpp_function_def(trimmed: &str) -> bool {
    // C++ free function: return_type name(args) { or return_type name(args) const {
    // Must have '(' and '{' or just '(' at end, not be a control statement
    if !trimmed.contains('(') { return false; }
    if trimmed.starts_with("if ") || trimmed.starts_with("for ")
        || trimmed.starts_with("while ") || trimmed.starts_with("switch ")
        || trimmed.starts_with("return ") { return false; }
    // Has a return type and function name before (
    let before_paren = trimmed.split('(').next().unwrap_or("");
    let words: Vec<&str> = before_paren.split_whitespace().collect();
    // At least two words: return_type function_name
    words.len() >= 2 && trimmed.contains('{')
}

fn is_swift_type_def(trimmed: &str) -> bool {
    let keywords = ["class ", "struct ", "enum ", "protocol ", "extension ", "actor "];
    let modifiers = ["public ", "private ", "internal ", "open ", "final "];
    for kw in &keywords {
        if trimmed.starts_with(kw) { return true; }
    }
    for m in &modifiers {
        if trimmed.starts_with(m) {
            for kw in &keywords {
                if trimmed.contains(kw) { return true; }
            }
        }
    }
    false
}

/// Ruby skeleton: keep require, class/module definitions, def signatures,
/// attr_accessor/reader/writer, comments. Strip method bodies.
fn extract_ruby_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Blank lines
        if trimmed.is_empty() {
            if !out.is_empty() { out.push(String::new()); }
            i += 1;
            continue;
        }

        // Comments
        if trimmed.starts_with('#') {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // require / require_relative / include / extend
        if trimmed.starts_with("require ") || trimmed.starts_with("require_relative ")
            || trimmed.starts_with("include ") || trimmed.starts_with("extend ")
            || trimmed.starts_with("gem ")
        {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // attr_accessor / attr_reader / attr_writer
        if trimmed.starts_with("attr_") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // class / module definition
        if trimmed.starts_with("class ") || trimmed.starts_with("module ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // def — keep signature, skip body until matching end
        if trimmed.starts_with("def ") {
            out.push(line.to_string());
            let indent = leading_spaces(line);
            let ind = " ".repeat(indent);
            out.push(format!("{}  # ...", ind));
            i += 1;
            // Skip until matching 'end' at same or lower indent
            let mut depth = 1i32;
            while i < lines.len() && depth > 0 {
                let t = lines[i].trim();
                if t == "end" || (t.starts_with("end ") && leading_spaces(lines[i]) <= indent) {
                    depth -= 1;
                } else if t.starts_with("def ") || t.starts_with("class ")
                    || t.starts_with("module ") || t.starts_with("do")
                    || t.ends_with(" do") || t.ends_with(" do |")
                    || (t.starts_with("if ") && !t.contains("then"))
                    || t.starts_with("unless ") || t.starts_with("while ")
                    || t.starts_with("begin")
                {
                    depth += 1;
                }
                if depth <= 0 { break; }
                i += 1;
            }
            out.push(format!("{}end", ind));
            i += 1;
            continue;
        }

        // end (for class/module)
        if trimmed == "end" {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Constants
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase()) && trimmed.contains(" = ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
}

/// PHP skeleton: keep namespace, use, class/interface/trait definitions,
/// function/method signatures, property declarations. Strip function bodies.
fn extract_php_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Blank lines
        if trimmed.is_empty() {
            if !out.is_empty() { out.push(String::new()); }
            i += 1;
            continue;
        }

        // PHP tag
        if trimmed.starts_with("<?") || trimmed.starts_with("?>") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // Comments
        if trimmed.starts_with("//") || trimmed.starts_with('#') {
            out.push(line.to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with("/*") || trimmed.starts_with("/**") {
            out.push(line.to_string());
            if !trimmed.contains("*/") {
                i += 1;
                while i < lines.len() {
                    out.push(lines[i].to_string());
                    if lines[i].contains("*/") { break; }
                    i += 1;
                }
            }
            i += 1;
            continue;
        }

        // namespace / use
        if trimmed.starts_with("namespace ") || trimmed.starts_with("use ") {
            out.push(line.to_string());
            i += 1;
            continue;
        }

        // class / interface / trait / abstract class
        if is_php_type_def(trimmed) {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                i += 1;
                while i < lines.len() && depth > 0 {
                    let l = lines[i];
                    let t = l.trim();
                    let delta = count_char(t, '{') as i32 - count_char(t, '}') as i32;
                    depth += delta;

                    if depth <= 0 {
                        out.push(l.to_string());
                        i += 1;
                        break;
                    }

                    // Function/method — keep sig, skip body
                    if is_php_function_line(t) {
                        out.push(l.to_string());
                        if t.contains('{') && delta > 0 {
                            let ind = " ".repeat(leading_spaces(l));
                            out.push(format!("{}    // ...", ind));
                            i += 1;
                            let mut fn_d = delta;
                            while i < lines.len() && fn_d > 0 {
                                let fd = count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                                fn_d += fd;
                                depth += fd;
                                i += 1;
                            }
                            out.push(format!("{}}}", ind));
                            continue;
                        }
                        i += 1;
                        continue;
                    }

                    // Property declarations, constants
                    if (t.starts_with("public ") || t.starts_with("private ")
                        || t.starts_with("protected ") || t.starts_with("const "))
                        && (t.contains('$') || t.contains(" const "))
                        && !t.contains('{')
                    {
                        out.push(l.to_string());
                    }

                    i += 1;
                }
            } else {
                i += 1;
            }
            continue;
        }

        // Top-level function
        if trimmed.starts_with("function ") {
            out.push(line.to_string());
            if trimmed.contains('{') {
                let mut depth = count_char(trimmed, '{') as i32 - count_char(trimmed, '}') as i32;
                if depth > 0 {
                    out.push("    // ...".to_string());
                    i += 1;
                    while i < lines.len() && depth > 0 {
                        depth += count_char(lines[i], '{') as i32 - count_char(lines[i], '}') as i32;
                        i += 1;
                    }
                    out.push("}".to_string());
                    continue;
                }
            }
            i += 1;
            continue;
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
}

fn is_php_type_def(trimmed: &str) -> bool {
    let keywords = ["class ", "interface ", "trait ", "enum "];
    let modifiers = ["abstract ", "final ", "readonly "];
    for kw in &keywords {
        if trimmed.starts_with(kw) { return true; }
    }
    for m in &modifiers {
        if trimmed.starts_with(m) {
            for kw in &keywords {
                if trimmed.contains(kw) { return true; }
            }
        }
    }
    false
}

fn is_php_function_line(trimmed: &str) -> bool {
    let modifiers = ["public ", "private ", "protected ", "static ",
                     "abstract ", "final "];
    if trimmed.starts_with("function ") { return true; }
    modifiers.iter().any(|m| trimmed.starts_with(m) && trimmed.contains("function "))
}

/// Vue/Svelte Single File Component skeleton.
///
/// SFC structure: `<template>`, `<script>`, `<style>` blocks.
/// Strategy:
///   - `<template>`: keep structural elements (div, section, header, nav,
///     component tags), strip text content and most attributes except
///     key structural ones (v-if, v-for, :key, class, id, slot, #)
///   - `<script>`: extract using JS/TS skeleton (reuse extract_js_skeleton)
///   - `<style>`: extract using CSS skeleton (reuse extract_css_skeleton)
fn extract_sfc_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    #[derive(PartialEq)]
    enum Block { None, Template, Script, Style }
    let mut block = Block::None;
    let mut script_lines = Vec::new();
    let mut style_lines = Vec::new();

    while i < lines.len() {
        let trimmed = lines[i].trim();

        // Detect block boundaries
        if trimmed.starts_with("<template") {
            block = Block::Template;
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }
        if trimmed == "</template>" {
            block = Block::None;
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with("<script") {
            out.push(lines[i].to_string());
            block = Block::Script;
            i += 1;
            continue;
        }
        if trimmed == "</script>" {
            // Process accumulated script through JS skeleton
            let script_content = script_lines.join("\n");
            let skel = extract_js_skeleton(&script_content);
            if !skel.trim().is_empty() {
                out.push(skel);
            }
            script_lines.clear();
            out.push(lines[i].to_string());
            block = Block::None;
            i += 1;
            continue;
        }
        if trimmed.starts_with("<style") {
            out.push(lines[i].to_string());
            block = Block::Style;
            i += 1;
            continue;
        }
        if trimmed == "</style>" {
            let style_content = style_lines.join("\n");
            let skel = extract_css_skeleton(&style_content);
            if !skel.trim().is_empty() {
                out.push(skel);
            }
            style_lines.clear();
            out.push(lines[i].to_string());
            block = Block::None;
            i += 1;
            continue;
        }

        match block {
            Block::Script => {
                script_lines.push(lines[i].to_string());
            }
            Block::Style => {
                style_lines.push(lines[i].to_string());
            }
            Block::Template => {
                // Keep structural HTML: elements with tags, strip text-only lines
                if trimmed.is_empty() {
                    if !out.is_empty() { out.push(String::new()); }
                } else if trimmed.starts_with("<!--") {
                    // Keep comments (may contain directives like eslint-disable)
                    out.push(lines[i].to_string());
                } else if trimmed.starts_with('<') || trimmed.starts_with("{{") || trimmed.ends_with('>') {
                    // Keep lines with HTML tags or template interpolation
                    out.push(lines[i].to_string());
                }
                // Strip pure text content lines
            }
            Block::None => {
                // Top-level (outside blocks): keep everything
                out.push(lines[i].to_string());
            }
        }

        i += 1;
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
}

/// HTML skeleton: keep document structure, strip text content.
///
/// Keeps: doctype, structural tags, component references, directives,
///        script/link/meta tags, comments
/// Strips: raw text content between tags, most inline styles
fn extract_html_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut in_script = false;
    let mut in_style = false;
    let mut script_lines = Vec::new();
    let mut style_lines = Vec::new();

    for line in &lines {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if !out.is_empty() { out.push(String::new()); }
            continue;
        }

        let trimmed_lower = trimmed.to_lowercase();

        // Track <script> blocks — delegate to JS skeleton
        if trimmed_lower.starts_with("<script") {
            out.push(line.to_string());
            // Self-closing or inline: <script src="..."></script>
            if trimmed_lower.contains("</script") {
                // Complete tag on one line — no script block to extract
                continue;
            }
            in_script = true;
            continue;
        }
        if trimmed_lower.starts_with("</script") {
            let js = script_lines.join("\n");
            let skel = extract_js_skeleton(&js);
            if !skel.trim().is_empty() {
                out.push(skel);
            }
            script_lines.clear();
            in_script = false;
            out.push(line.to_string());
            continue;
        }
        if in_script {
            script_lines.push(line.to_string());
            continue;
        }

        // Track <style> blocks — delegate to CSS skeleton
        if trimmed_lower.starts_with("<style") {
            out.push(line.to_string());
            if trimmed_lower.contains("</style") {
                continue;
            }
            in_style = true;
            continue;
        }
        if trimmed_lower.starts_with("</style") {
            let css = style_lines.join("\n");
            let skel = extract_css_skeleton(&css);
            if !skel.trim().is_empty() {
                out.push(skel);
            }
            style_lines.clear();
            in_style = false;
            out.push(line.to_string());
            continue;
        }
        if in_style {
            style_lines.push(line.to_string());
            continue;
        }

        // Keep structural lines (tags, comments, doctype)
        if trimmed.starts_with("<!") || trimmed.starts_with('<') || trimmed.ends_with('>')
            || trimmed.starts_with("<!--")
        {
            out.push(line.to_string());
        }
        // Strip pure text lines (between tags)
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
}

/// CSS/SCSS/LESS skeleton: keep selectors, @rules, custom properties.
/// Strip property-value declarations (the bulk of CSS).
///
/// Keeps: selectors (lines ending with `{`), closing braces,
///        @import/@media/@keyframes/@font-face, CSS custom properties (--),
///        comments
/// Strips: property: value; lines
fn extract_css_skeleton(content: &str) -> String {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for line in &lines {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if !out.is_empty() { out.push(String::new()); }
            continue;
        }

        // Comments
        if trimmed.starts_with("/*") || trimmed.starts_with("*") || trimmed.starts_with("//") {
            out.push(line.to_string());
            continue;
        }

        // @rules: @import, @media, @keyframes, @font-face, @charset, @mixin, @include
        if trimmed.starts_with('@') {
            out.push(line.to_string());
            continue;
        }

        // Selectors (lines containing { or just a selector)
        if trimmed.ends_with('{') || trimmed.ends_with(',') {
            out.push(line.to_string());
            continue;
        }

        // Closing braces
        if trimmed.starts_with('}') || trimmed == "}" {
            out.push(line.to_string());
            continue;
        }

        // CSS custom properties (variables)
        if trimmed.starts_with("--") {
            out.push(line.to_string());
            continue;
        }

        // SCSS/LESS variables
        if trimmed.starts_with('$') || trimmed.starts_with('@') {
            out.push(line.to_string());
            continue;
        }

        // SCSS: nesting selectors (lines that contain { in the middle)
        if trimmed.contains('{') {
            out.push(line.to_string());
            continue;
        }

        // Strip regular property: value; declarations
    }

    while out.last().is_some_and(|l: &String| l.trim().is_empty()) { out.pop(); }
    out.join("\n")
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

    #[test]
    fn test_go_skeleton() {
        let code = r#"
package handlers

import (
    "fmt"
    "net/http"
)

// HandleRequest processes incoming HTTP requests.
type Config struct {
    Port    int
    Host    string
    Debug   bool
}

func HandleRequest(w http.ResponseWriter, r *http.Request) {
    body := readBody(r)
    if body == nil {
        http.Error(w, "bad request", 400)
        return
    }
    result := processBody(body)
    fmt.Fprintf(w, "%s", result)
}

func (s *Server) Start() error {
    listener, err := net.Listen("tcp", s.addr)
    if err != nil {
        return err
    }
    return s.serve(listener)
}
"#.trim();

        let skel = extract_skeleton(code, "handler.go").unwrap();
        assert!(skel.contains("package handlers"), "Should keep package");
        assert!(skel.contains("\"net/http\""), "Should keep imports");
        assert!(skel.contains("type Config struct"), "Should keep type def");
        assert!(skel.contains("Port    int"), "Should keep struct fields");
        assert!(skel.contains("func HandleRequest("), "Should keep func sig");
        assert!(skel.contains("func (s *Server) Start()"), "Should keep method sig");
        assert!(!skel.contains("readBody(r)"), "Should strip func body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_java_skeleton() {
        let code = r#"
package com.example.service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * User service for managing users.
 */
@Service
public class UserService {
    private final UserRepository repo;
    private final CacheManager cache;

    @Autowired
    public UserService(UserRepository repo, CacheManager cache) {
        this.repo = repo;
        this.cache = cache;
        this.init();
        logger.info("UserService initialized");
    }

    public Optional<User> findById(Long id) {
        User cached = cache.get("user:" + id);
        if (cached != null) {
            logger.debug("Cache hit for user {}", id);
            return Optional.of(cached);
        }
        Optional<User> user = repo.findById(id);
        user.ifPresent(u -> cache.put("user:" + id, u));
        return user.map(this::enrichUser);
    }

    public List<User> findAll() {
        List<User> users = repo.findAll();
        return users.stream()
            .filter(u -> u.isActive())
            .map(this::enrichUser)
            .sorted((a, b) -> a.getName().compareTo(b.getName()))
            .collect(Collectors.toList());
    }

    private User enrichUser(User user) {
        user.setFullName(user.getFirst() + " " + user.getLast());
        user.setDisplayName(user.getFullName().toLowerCase());
        user.setLastAccessed(Instant.now());
        return user;
    }
}
"#.trim();

        let skel = extract_skeleton(code, "UserService.java").unwrap();
        assert!(skel.contains("package com.example.service;"), "Should keep package");
        assert!(skel.contains("import java.util.List;"), "Should keep imports");
        assert!(skel.contains("@Service"), "Should keep annotations");
        assert!(skel.contains("public class UserService"), "Should keep class");
        assert!(skel.contains("private final UserRepository repo;"), "Should keep fields");
        assert!(skel.contains("public Optional<User> findById(Long id)"), "Should keep method sig");
        assert!(!skel.contains("Cache hit"), "Should strip method body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_cpp_skeleton() {
        let code = r#"
#include <iostream>
#include <vector>
#include <string>

namespace app {

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }

    double multiply(double a, double b) {
        double result = a * b;
        if (result > 1000) {
            result = 1000;
        }
        return result;
    }

private:
    std::vector<int> history;
};

int main(int argc, char* argv[]) {
    Calculator calc;
    int result = calc.add(1, 2);
    std::cout << result << std::endl;
    return 0;
}

} // namespace app
"#.trim();

        let skel = extract_skeleton(code, "calc.cpp").unwrap();
        assert!(skel.contains("#include <iostream>"), "Should keep includes");
        assert!(skel.contains("namespace app"), "Should keep namespace");
        assert!(skel.contains("class Calculator"), "Should keep class");
        assert!(skel.contains("int add(int a, int b)"), "Should keep method sig");
        assert!(skel.contains("private:"), "Should keep access specifiers");
        assert!(!skel.contains("a + b"), "Should strip method body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_ruby_skeleton() {
        let code = r#"
require 'json'
require_relative 'helpers'

module PaymentGateway
  class Processor
    attr_accessor :api_key, :timeout

    MAX_RETRIES = 3

    def initialize(api_key)
      @api_key = api_key
      @timeout = 30
      @retries = 0
    end

    def process_payment(amount, currency)
      validate_amount(amount)
      response = make_request(amount, currency)
      if response.success?
        log_success(response)
        response.transaction_id
      else
        handle_failure(response)
      end
    end

    private

    def validate_amount(amount)
      raise ArgumentError, "Amount must be positive" unless amount > 0
    end

    def make_request(amount, currency)
      HTTP.post("/charge", body: { amount: amount, currency: currency })
    end
  end
end
"#.trim();

        let skel = extract_skeleton(code, "processor.rb").unwrap();
        assert!(skel.contains("require 'json'"), "Should keep require");
        assert!(skel.contains("require_relative 'helpers'"), "Should keep require_relative");
        assert!(skel.contains("module PaymentGateway"), "Should keep module");
        assert!(skel.contains("class Processor"), "Should keep class");
        assert!(skel.contains("attr_accessor"), "Should keep attr_accessor");
        assert!(skel.contains("MAX_RETRIES = 3"), "Should keep constants");
        assert!(skel.contains("def initialize"), "Should keep def signatures");
        assert!(skel.contains("def process_payment"), "Should keep def signatures");
        // The skeleton keeps "def validate_amount" as a signature, but should NOT
        // contain the CALL to validate_amount inside process_payment's body.
        // Check that body-only content like log_success and HTTP.post are stripped.
        assert!(!skel.contains("log_success"), "Should strip method body calls");
        assert!(!skel.contains("HTTP.post"), "Should strip method body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_php_skeleton() {
        let code = r#"
<?php

namespace App\Services;

use App\Models\User;
use Illuminate\Support\Facades\Log;

/**
 * Handles user authentication.
 */
class AuthService {
    private string $secretKey;
    protected int $maxAttempts = 5;

    public function __construct(string $secretKey) {
        $this->secretKey = $secretKey;
        $this->attempts = 0;
    }

    public function authenticate(string $email, string $password): ?User {
        $user = User::findByEmail($email);
        if (!$user || !password_verify($password, $user->password_hash)) {
            Log::warning("Failed login for: {$email}");
            return null;
        }
        return $user;
    }

    private function generateToken(User $user): string {
        return hash_hmac('sha256', $user->id . time(), $this->secretKey);
    }
}
"#.trim();

        let skel = extract_skeleton(code, "auth.php").unwrap();
        assert!(skel.contains("<?php"), "Should keep PHP tag");
        assert!(skel.contains("namespace App\\Services"), "Should keep namespace");
        assert!(skel.contains("use App\\Models\\User"), "Should keep use");
        assert!(skel.contains("class AuthService"), "Should keep class");
        assert!(skel.contains("private string $secretKey"), "Should keep properties");
        assert!(skel.contains("public function __construct"), "Should keep method sigs");
        assert!(skel.contains("public function authenticate"), "Should keep method sigs");
        assert!(!skel.contains("findByEmail"), "Should strip method body");
        assert!(!skel.contains("hash_hmac"), "Should strip method body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    // ── Frontend: Vue/Svelte SFC skeleton ────────────────────────────

    #[test]
    fn test_vue_sfc_skeleton() {
        let code = r#"
<template>
  <div class="app">
    <h1>{{ title }}</h1>
    <p>Some descriptive text about the application</p>
    <UserCard v-for="user in users" :key="user.id" :user="user" />
    <button @click="handleSubmit">Submit</button>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import UserCard from './components/UserCard.vue'

const title = ref('Dashboard')
const users = ref([])

function handleSubmit() {
  const data = collectFormData()
  const validated = validateFormData(data)
  if (!validated) {
    showError('Invalid form data')
    return
  }
  api.post('/submit', data)
  users.value = []
  showSuccess('Form submitted successfully')
  resetForm()
  trackAnalytics('form_submit', { count: users.value.length })
}

const activeUsers = computed(() => {
  return users.value.filter(u => u.active)
})

function resetForm() {
  document.querySelectorAll('input').forEach(input => {
    input.value = ''
    input.classList.remove('error')
  })
  title.value = 'Dashboard'
}
</script>

<style scoped>
.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

h1 {
  color: #333;
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 1rem;
  line-height: 1.2;
}

.user-card {
  padding: 1rem;
  border: 1px solid #eee;
  border-radius: 8px;
  background: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.2s ease;
}
</style>
"#.trim();

        let skel = extract_skeleton(code, "Dashboard.vue").unwrap();
        assert!(skel.contains("<template>"), "Should keep template tag");
        assert!(skel.contains("</template>"), "Should keep closing template");
        assert!(skel.contains("<div class=\"app\">"), "Should keep structural elements");
        assert!(skel.contains("UserCard"), "Should keep component references");
        assert!(skel.contains("<script"), "Should keep script tag");
        assert!(skel.contains("import"), "Should keep imports in script");
        assert!(skel.contains("<style"), "Should keep style tag");
        assert!(skel.contains(".app {"), "Should keep CSS selectors");
        assert!(!skel.contains("collectFormData"), "Should strip JS function body");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_html_skeleton() {
        let code = r#"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My App</title>
  <link rel="stylesheet" href="styles.css">
  <script src="app.js"></script>
</head>
<body>
  <header>
    <nav class="main-nav">
      <a href="/">Home</a>
      <a href="/about">About</a>
    </nav>
  </header>
  <main id="content">
    Welcome to the application. This is some paragraph text
    that describes what the application does.
  </main>
  <script>
    function initApp() {
      const data = fetchData()
      renderDashboard(data)
      return data
    }
    initApp()
  </script>
</body>
</html>
"#.trim();

        let skel = extract_skeleton(code, "index.html").unwrap();
        assert!(skel.contains("<!DOCTYPE html>"), "Should keep doctype");
        assert!(skel.contains("<html"), "Should keep html tag");
        assert!(skel.contains("<head>"), "Should keep head");
        assert!(skel.contains("<script src=\"app.js\">"), "Should keep script tags");
        assert!(skel.contains("<link"), "Should keep link tags");
        assert!(skel.contains("<nav class=\"main-nav\">"), "Should keep structural elements");
        assert!(!skel.contains("Welcome to the application"), "Should strip text content");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }

    #[test]
    fn test_css_skeleton() {
        let code = r#"
/* Base styles */
@import url('https://fonts.googleapis.com/css2?family=Inter');

:root {
  --primary-color: #3498db;
  --secondary-color: #2ecc71;
  --spacing-unit: 8px;
}

@media (max-width: 768px) {
  .container {
    padding: 1rem;
    margin: 0;
    display: flex;
    flex-direction: column;
  }
}

.header {
  background: var(--primary-color);
  color: white;
  padding: calc(var(--spacing-unit) * 3);
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.nav-link {
  text-decoration: none;
  font-weight: 600;
  transition: color 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
"#.trim();

        let skel = extract_skeleton(code, "styles.css").unwrap();
        assert!(skel.contains("/* Base styles */"), "Should keep comments");
        assert!(skel.contains("@import"), "Should keep @import");
        assert!(skel.contains(":root {"), "Should keep selectors");
        assert!(skel.contains("--primary-color"), "Should keep custom properties");
        assert!(skel.contains("@media"), "Should keep media queries");
        assert!(skel.contains(".header {"), "Should keep class selectors");
        assert!(skel.contains("@keyframes"), "Should keep keyframes");
        assert!(!skel.contains("text-decoration"), "Should strip property declarations");
        assert!(!skel.contains("font-weight"), "Should strip property declarations");
        assert!(skel.len() < code.len(), "Skeleton should be shorter");
    }
}
