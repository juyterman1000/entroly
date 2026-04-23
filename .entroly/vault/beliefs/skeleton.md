---
claim_id: e62d3f78-be9a-4417-a281-7c244f6e05f6
entity: skeleton
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\skeleton.rs:1981
  - entroly-wasm\src\skeleton.rs:39
  - entroly-wasm\src\skeleton.rs:85
  - entroly-wasm\src\skeleton.rs:128
  - entroly-wasm\src\skeleton.rs:133
  - entroly-wasm\src\skeleton.rs:139
  - entroly-wasm\src\skeleton.rs:305
  - entroly-wasm\src\skeleton.rs:489
  - entroly-wasm\src\skeleton.rs:676
  - entroly-wasm\src\skeleton.rs:805
last_checked: 2026-04-23T03:07:07.911029+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: skeleton

**Language:** rust
**Lines of code:** 2574

## Types
- `pub struct Fragment` — A context fragment.

## Functions
- `fn detect_lang(source: &str) -> Lang`
- `fn extract_skeleton(content: &str, source: &str) -> Option<String>` — Extract a skeleton (structural outline) from a code fragment.  Returns `None` if: - Language is not recognized - Content is too short to benefit from skeletonization - Skeleton would be >70% of origin
- `fn leading_spaces(line: &str) -> usize` — Count leading spaces in a line.
- `fn count_char(s: &str, ch: char) -> usize` — Count occurrences of a character in a string.
- `fn extract_python_skeleton(content: &str) -> String` — Python skeleton: keep imports, decorators, class/def signatures, top-level assignments. Replace function/method bodies with `...`
- `fn extract_rust_skeleton(content: &str) -> String` — Rust skeleton: keep use/mod, pub struct/enum/trait/fn signatures, impl blocks. Replace `{ ... }` function bodies with `{ ... }`
- `fn extract_js_skeleton(content: &str) -> String` — JS/TS skeleton: keep imports, exports, class/function signatures, top-level const/let.
- `fn extract_go_skeleton(content: &str) -> String` — Go skeleton: keep package, imports, func signatures, type definitions, const blocks. Replace function bodies with `// ...`
- `fn extract_java_skeleton(content: &str) -> String` — Java/Kotlin skeleton: keep package, imports, class/interface signatures, method signatures, annotations. Strip method bodies.
- `fn extract_cpp_skeleton(content: &str) -> String` — C/C++ skeleton: keep #include, namespace, class/struct definitions, function signatures. Strip function bodies.
- `fn extract_swift_skeleton(content: &str) -> String` — Swift skeleton: keep import, class/struct/protocol/enum definitions, func signatures, property declarations. Strip function bodies.
- `fn extract_shell_skeleton(content: &str) -> String` — Shell skeleton: keep shebang, function definitions (signature only), export/source statements, global variable assignments.
- `fn is_rust_type_def(trimmed: &str) -> bool`
- `fn is_js_method_sig(trimmed: &str) -> bool`
- `fn is_java_type_def(trimmed: &str) -> bool`
- `fn is_java_method_line(trimmed: &str) -> bool`
- `fn is_cpp_function_def(trimmed: &str) -> bool`
- `fn is_swift_type_def(trimmed: &str) -> bool`
- `fn extract_ruby_skeleton(content: &str) -> String` — Ruby skeleton: keep require, class/module definitions, def signatures, attr_accessor/reader/writer, comments. Strip method bodies.
- `fn extract_php_skeleton(content: &str) -> String` — PHP skeleton: keep namespace, use, class/interface/trait definitions, function/method signatures, property declarations. Strip function bodies.
- `fn is_php_type_def(trimmed: &str) -> bool`
- `fn is_php_function_line(trimmed: &str) -> bool`
- `fn extract_sfc_skeleton(content: &str) -> String` — Vue/Svelte Single File Component skeleton.  SFC structure: `<template>`, `<script>`, `<style>` blocks. Strategy: - `<template>`: keep structural elements (div, section, header, nav, component tags), s
- `fn extract_html_skeleton(content: &str) -> String` — HTML skeleton: keep document structure, strip text content.  Keeps: doctype, structural tags, component references, directives, script/link/meta tags, comments Strips: raw text content between tags, m
- `fn extract_css_skeleton(content: &str) -> String` — CSS/SCSS/LESS skeleton: keep selectors, @rules, custom properties. Strip property-value declarations (the bulk of CSS).  Keeps: selectors (lines ending with `{`), closing braces, @import/@media/@keyfr
- `fn helper(x: u32) -> u32`

## Dependencies
- `App`
- `Illuminate`
- `std::collections::HashMap`
