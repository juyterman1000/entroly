//! Dependency Graph Engine
//!
//! Addresses the core weakness: context is NOT additive.
//!
//!   value(A + B) ≠ value(A) + value(B)
//!
//! Code has dependencies. A function call without its definition
//! is useless. A type reference without its schema is confusing.
//! Removing one fragment can destroy the value of others.
//!
//! This module implements:
//!   1. **Dependency extraction** — parse imports, calls, types from source
//!   2. **Graph construction** — directed graph of fragment → fragment deps
//!   3. **Graph-aware selection** — if A is selected and A depends on B,
//!      B's value is boosted (or B is force-included)
//!   4. **Connected component analysis** — fragments in the same component
//!      should be selected or dropped together
//!
//! This transforms the problem from flat knapsack to:
//!   Graph-constrained knapsack (NP-hard in general, but tractable
//!   for typical code dependency graphs with ~500 nodes)
//!
//! References:
//!   - Sourcegraph's Code Intelligence — graph-based code navigation
//!   - Code Property Graphs (Yamaguchi et al., 2014)

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

/// A directed dependency between two fragments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// The fragment that USES/REQUIRES the target
    pub source_id: String,
    /// The fragment that PROVIDES the dependency
    pub target_id: String,
    /// Type of dependency
    pub dep_type: DepType,
    /// Strength of the dependency [0, 1]
    pub strength: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DepType {
    /// Source imports Target
    Import,
    /// Source calls a function defined in Target
    FunctionCall,
    /// Source uses a type defined in Target
    TypeReference,
    /// Source and Target are in the same module/file
    SameModule,
    /// Source's tests test Target
    TestOf,
    /// Cross-language FFI boundary (PyO3, JNI, CGo, WASM, N-API, C FFI)
    CrossLanguageFFI,
}

/// The dependency graph.
#[derive(Serialize, Deserialize)]
pub struct DepGraph {
    /// fragment_id → list of dependencies WHERE this fragment is the source
    outgoing: HashMap<String, Vec<Dependency>>,
    /// fragment_id → list of dependencies WHERE this fragment is the target
    incoming: HashMap<String, Vec<Dependency>>,
    /// All known symbols: "function_name" → fragment_id that defines it
    symbol_table: HashMap<String, String>,
    /// Cross-language FFI exports: symbol → (fragment_id, bridge_type)
    /// e.g. ("process", ("frag_rust_lib", "pyo3"))
    cross_lang_exports: HashMap<String, (String, String)>,
}

impl DepGraph {
    pub fn new() -> Self {
        DepGraph {
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            symbol_table: HashMap::new(),
            cross_lang_exports: HashMap::new(),
        }
    }

    /// Register a symbol definition (e.g., function, class, type).
    pub fn register_symbol(&mut self, symbol: &str, fragment_id: &str) {
        self.symbol_table.insert(symbol.to_string(), fragment_id.to_string());
    }

    /// Add a dependency edge.
    pub fn add_dependency(&mut self, dep: Dependency) {
        let source = dep.source_id.clone();
        let target = dep.target_id.clone();

        self.outgoing.entry(source).or_default().push(dep.clone());
        self.incoming.entry(target).or_default().push(dep);
    }

    /// Auto-link: given a fragment's content, extract symbols it references
    /// and create dependency edges to the fragments that define those symbols.
    pub fn auto_link(&mut self, fragment_id: &str, content: &str) {
        let lower = content.to_lowercase();

        // Detect explicit imports first (strongest dependency type)
        let import_targets: HashSet<String> = Self::extract_import_targets(content);

        // Extract identifiers from content
        let identifiers = extract_identifiers(content);

        for ident in &identifiers {
            if let Some(defining_frag) = self.symbol_table.get(ident) {
                if defining_frag != fragment_id {
                    // Classify the dependency type based on how the symbol appears
                    let (dep_type, strength) = if import_targets.contains(ident) {
                        // Explicitly imported → strongest link
                        (DepType::Import, 1.0)
                    } else if Self::is_type_reference(&lower, ident) {
                        // Used as a type annotation → strong link
                        (DepType::TypeReference, 0.9)
                    } else {
                        // General identifier usage (function call, variable access)
                        (DepType::FunctionCall, 0.7)
                    };

                    self.add_dependency(Dependency {
                        source_id: fragment_id.to_string(),
                        target_id: defining_frag.clone(),
                        dep_type,
                        strength,
                    });
                }
            }
        }

        // Extract symbols this fragment DEFINES
        let definitions = extract_definitions(content);
        for def in &definitions {
            self.register_symbol(def, fragment_id);
        }

        // Cross-language FFI: extract exported symbols and link across boundaries
        let ffi_exports = Self::extract_ffi_exports(content);
        for (symbol, bridge_type) in &ffi_exports {
            self.cross_lang_exports.insert(
                symbol.clone(),
                (fragment_id.to_string(), bridge_type.clone()),
            );
            // Also register in the normal symbol table so imports resolve
            self.register_symbol(symbol, fragment_id);
        }

        // Check if this fragment's imports match any cross-language export
        for ident in &identifiers {
            if let Some((export_frag, _bridge)) = self.cross_lang_exports.get(ident) {
                if export_frag != fragment_id {
                    self.add_dependency(Dependency {
                        source_id: fragment_id.to_string(),
                        target_id: export_frag.clone(),
                        dep_type: DepType::CrossLanguageFFI,
                        strength: 0.95,  // Very strong: cross-lang deps are critical context
                    });
                }
            }
        }
    }

    /// Extract symbols that are explicitly imported in source code.
    /// Parses Python, Rust, JS/TS, Go, Java, C/C++, and Ruby imports.
    fn extract_import_targets(content: &str) -> HashSet<String> {
        let mut targets = HashSet::new();
        for line in content.lines() {
            let trimmed = line.trim();

            // Python: from module import foo, bar
            if trimmed.starts_with("from ") && trimmed.contains(" import ") {
                if let Some(import_part) = trimmed.split(" import ").nth(1) {
                    for name in import_part.split(',') {
                        let clean = name.trim().split(" as ").next().unwrap_or("").trim();
                        if !clean.is_empty() && clean != "*" {
                            targets.insert(clean.to_string());
                        }
                    }
                }
            }
            // Python: import module (the module name itself)
            else if trimmed.starts_with("import ") && !trimmed.contains("from") {
                for name in trimmed.trim_start_matches("import ").split(',') {
                    let clean = name.trim().split(" as ").next().unwrap_or("").trim();
                    // Use the last component of dotted imports
                    let last = clean.rsplit('.').next().unwrap_or(clean);
                    if !last.is_empty() {
                        targets.insert(last.to_string());
                    }
                }
            }
            // Rust: use crate::module::Symbol;
            else if trimmed.starts_with("use ") {
                let path = trimmed.trim_start_matches("use ").trim_end_matches(';');
                // Handle `use foo::{A, B}` brace imports
                if let Some(brace_start) = path.find('{') {
                    if let Some(brace_end) = path.find('}') {
                        if brace_start + 1 < brace_end {
                            let inner = &path[brace_start+1..brace_end];
                            for name in inner.split(',') {
                                let clean = name.trim().split(" as ").next().unwrap_or("").trim();
                                if !clean.is_empty() {
                                    targets.insert(clean.to_string());
                                }
                            }
                        }
                    }
                    // No closing brace (multi-line use) — skip safely
                } else {
                    // use foo::Bar; → import "Bar"
                    let last = path.rsplit("::").next().unwrap_or(path).trim();
                    if !last.is_empty() {
                        targets.insert(last.to_string());
                    }
                }
            }
            // JS/TS: import { foo, bar } from '...'  /  import React from '...'
            else if trimmed.starts_with("import ") {
                if let Some(brace_start) = trimmed.find('{') {
                    if let Some(brace_end) = trimmed.find('}') {
                        let inner = &trimmed[brace_start+1..brace_end];
                        for name in inner.split(',') {
                            let clean = name.trim().split(" as ").next().unwrap_or("").trim();
                            if !clean.is_empty() {
                                targets.insert(clean.to_string());
                            }
                        }
                    }
                } else {
                    // Default import: import React from 'react'
                    let after_import = trimmed.trim_start_matches("import ");
                    if let Some(name) = after_import.split_whitespace().next() {
                        let clean = name.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                        if !clean.is_empty() && clean != "type" && clean != "*" {
                            targets.insert(clean.to_string());
                        }
                    }
                }
            }
            // JS/TS: const foo = require('module')
            else if trimmed.contains("require(") {
                if let Some(var) = extract_require_lhs(trimmed) {
                    targets.insert(var);
                }
            }
            // Go: import "fmt" / import alias "pkg/path"
            // Also handles import block:  import ( "fmt"\n "net/http" )
            else if trimmed.starts_with("import ") && trimmed.contains('"') {
                // Single-line: import "fmt" or import http "net/http"
                for segment in trimmed.split('"') {
                    let seg = segment.trim();
                    if !seg.is_empty() && !seg.starts_with("import") && !seg.starts_with("(") {
                        // Use the last path component: "net/http" → "http"
                        let last = seg.rsplit('/').next().unwrap_or(seg);
                        if !last.is_empty() {
                            targets.insert(last.to_string());
                        }
                    }
                }
            }
            // Go import block lines: "fmt" or alias "pkg/path"
            else if trimmed.starts_with('"') && trimmed.ends_with('"') {
                let pkg = trimmed.trim_matches('"');
                let last = pkg.rsplit('/').next().unwrap_or(pkg);
                if !last.is_empty() {
                    targets.insert(last.to_string());
                }
            }
            // Java/Kotlin: import com.example.ClassName;
            else if trimmed.starts_with("import ") && trimmed.contains('.') && trimmed.ends_with(';') {
                let path = trimmed.trim_start_matches("import ")
                    .trim_start_matches("static ")
                    .trim_end_matches(';')
                    .trim();
                if path.ends_with(".*") {
                    // Wildcard import — use the package name
                    let pkg = path.trim_end_matches(".*").rsplit('.').next().unwrap_or("");
                    if !pkg.is_empty() {
                        targets.insert(pkg.to_string());
                    }
                } else {
                    let class = path.rsplit('.').next().unwrap_or(path);
                    if !class.is_empty() {
                        targets.insert(class.to_string());
                    }
                }
            }
            // C/C++: #include <header> or #include "header"
            else if trimmed.starts_with("#include") {
                let after = trimmed.trim_start_matches("#include").trim();
                let header = after.trim_matches(|c: char| c == '<' || c == '>' || c == '"');
                // "stdio.h" → "stdio", "vector" → "vector", "boost/asio.hpp" → "asio"
                let last = header.rsplit('/').next().unwrap_or(header);
                let name = last.split('.').next().unwrap_or(last);
                if !name.is_empty() {
                    targets.insert(name.to_string());
                }
            }
            // Ruby: require 'module' / require_relative 'path'
            else if trimmed.starts_with("require ") || trimmed.starts_with("require_relative ") {
                let after = if trimmed.starts_with("require_relative") {
                    trimmed.trim_start_matches("require_relative").trim()
                } else {
                    trimmed.trim_start_matches("require").trim()
                };
                let module = after.trim_matches(|c: char| c == '\'' || c == '"');
                let last = module.rsplit('/').next().unwrap_or(module);
                if !last.is_empty() {
                    targets.insert(last.to_string());
                }
            }
            // C#: using Namespace.Class;
            else if trimmed.starts_with("using ") && trimmed.ends_with(';') {
                let path = trimmed.trim_start_matches("using ")
                    .trim_start_matches("static ")
                    .trim_end_matches(';')
                    .trim();
                if !path.contains(' ') {  // skip "using var x = ..."
                    let last = path.rsplit('.').next().unwrap_or(path);
                    if !last.is_empty() {
                        targets.insert(last.to_string());
                    }
                }
            }
            // Swift: import Module
            else if trimmed.starts_with("import ") && !trimmed.contains('{')
                && !trimmed.contains('"') && !trimmed.contains('\'')
            {
                let module = trimmed.trim_start_matches("import ").trim();
                // "import Foundation" → "Foundation"
                // "import class UIKit.UIView" → "UIView"
                let last = module.rsplit('.').next()
                    .unwrap_or(module)
                    .rsplit_once(' ')
                    .map(|(_, r)| r)
                    .unwrap_or(module);
                if !last.is_empty() {
                    targets.insert(last.to_string());
                }
            }
            // PHP: use App\Models\User;
            else if trimmed.starts_with("use ") && trimmed.contains('\\') && trimmed.ends_with(';') {
                let path = trimmed.trim_start_matches("use ")
                    .trim_end_matches(';')
                    .trim();
                // Handle aliased: use App\Models\User as AppUser;
                let effective = path.split(" as ").next().unwrap_or(path);
                let last = effective.rsplit('\\').next().unwrap_or(effective);
                if !last.is_empty() {
                    targets.insert(last.to_string());
                }
            }

            // ── HTML/Vue/Svelte: component and resource references ──
            // <script src="..."> or <link href="...">
            if (trimmed.contains("<script") || trimmed.contains("<link"))
                && (trimmed.contains("src=") || trimmed.contains("href="))
            {
                // Extract path from src="..." or href="..."
                for attr in &["src=\"", "href=\"", "src='", "href='"] {
                    if let Some(start) = trimmed.find(attr) {
                        let after = &trimmed[start + attr.len()..];
                        let quote = if attr.ends_with('"') { '"' } else { '\'' };
                        if let Some(end) = after.find(quote) {
                            let path = &after[..end];
                            // Extract filename without extension as symbol
                            let name = path.rsplit('/').next().unwrap_or(path);
                            let name = name.split('.').next().unwrap_or(name);
                            if !name.is_empty() && name != "index" {
                                targets.insert(name.to_string());
                            }
                        }
                    }
                }
            }
            // Vue/Svelte: component imports are standard JS (handled above)
            // Angular: @Component({templateUrl: '...'})
            if trimmed.contains("templateurl") || trimmed.contains("styleurls") {
                for attr in &["templateurl:", "styleurls:"] {
                    if let Some(start) = trimmed.to_lowercase().find(attr) {
                        let after = &trimmed[start + attr.len()..];
                        // Extract path from quotes
                        if let Some(q1) = after.find('\'').or_else(|| after.find('"')) {
                            let rest = &after[q1+1..];
                            if let Some(q2) = rest.find('\'').or_else(|| rest.find('"')) {
                                let path = &rest[..q2];
                                let name = path.rsplit('/').next().unwrap_or(path).split('.').next().unwrap_or("");
                                if !name.is_empty() {
                                    targets.insert(name.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // ── Cross-language FFI boundary detection ──────────────
            // PyO3: Rust → Python bridge
            if trimmed.contains("#[pyfunction]") || trimmed.contains("#[pyclass]")
                || trimmed.contains("#[pymethods]") || trimmed.contains("pyo3::prelude")
            {
                targets.insert("__pyo3_bridge__".to_string());
            }
            // JNI: Java → C/C++ bridge
            if trimmed.starts_with("native ") || trimmed.contains(" native ")
                || trimmed.contains("System.loadLibrary")
                || trimmed.contains("System.load(")
            {
                targets.insert("__jni_bridge__".to_string());
            }
            // JNI C-side: JNIEXPORT
            if trimmed.contains("JNIEXPORT") || trimmed.contains("JNIEnv") {
                targets.insert("__jni_bridge__".to_string());
            }
            // C FFI: extern "C" / ctypes / cffi / dlopen
            if trimmed.contains("extern \"C\"") || trimmed.contains("extern \"c\"") {
                targets.insert("__c_ffi__".to_string());
            }
            if trimmed.contains("ctypes.") || trimmed.contains("from ctypes")
                || trimmed.contains("cffi") || trimmed.contains("dlopen")
            {
                targets.insert("__c_ffi__".to_string());
            }
            // CGo: import "C"
            if trimmed == "import \"C\"" || trimmed.contains("/*\n#include") {
                targets.insert("__cgo_bridge__".to_string());
            }
            // WASM: wasm_bindgen
            if trimmed.contains("wasm_bindgen") || trimmed.contains("wasm-bindgen") {
                targets.insert("__wasm_bridge__".to_string());
            }
            // Node.js N-API / neon
            if trimmed.contains("napi::") || trimmed.contains("#[napi]")
                || trimmed.contains("neon::prelude")
            {
                targets.insert("__node_native__".to_string());
            }
        }
        targets
    }

    /// Extract symbols exported across FFI boundaries.
    ///
    /// Detects:
    ///   - **PyO3**: `#[pyfunction] fn process()` → exports "process"
    ///   - **PyO3**: `#[pyclass] struct Engine` → exports "Engine"
    ///   - **JNI**: `JNIEXPORT ... Java_com_example_Class_method` → exports "method"
    ///   - **JNI (Java-side)**: `native void process()` → exports "process"
    ///   - **CGo**: `//export ProcessData` → exports "ProcessData"
    ///   - **WASM**: `#[wasm_bindgen] pub fn greet()` → exports "greet"
    ///   - **N-API**: `#[napi] fn compute()` → exports "compute"
    ///   - **C FFI**: `extern "C" fn handler()` → exports "handler"
    ///   - **ctypes/cffi (Python)**: symbol loaded via ctypes/cffi
    fn extract_ffi_exports(content: &str) -> Vec<(String, String)> {
        let mut exports = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // ── PyO3: #[pyfunction] / #[pyclass] ──────────────────────
            // Pattern: #[pyfunction] on one line, fn name( on next
            if trimmed.contains("#[pyfunction]") || trimmed.contains("#[pyo3(name") {
                // Check same line: #[pyfunction] fn foo(
                if let Some(name) = extract_fn_name_from_line(trimmed) {
                    exports.push((name, "pyo3".to_string()));
                }
                // Check next line
                else if i + 1 < lines.len() {
                    if let Some(name) = extract_fn_name_from_line(lines[i + 1].trim()) {
                        exports.push((name, "pyo3".to_string()));
                    }
                }
            }
            if trimmed.contains("#[pyclass]") || trimmed.contains("#[pyclass(") {
                if let Some(name) = extract_struct_name_from_line(trimmed) {
                    exports.push((name, "pyo3".to_string()));
                } else if i + 1 < lines.len() {
                    if let Some(name) = extract_struct_name_from_line(lines[i + 1].trim()) {
                        exports.push((name, "pyo3".to_string()));
                    }
                }
            }
            // #[pymethods] impl Foo { fn bar() → export "bar"
            if trimmed.contains("#[pymethods]") {
                if let Some(impl_line) = lines.get(i + 1) {
                    let t = impl_line.trim();
                    if t.starts_with("impl ") {
                        // Scan methods inside this impl block
                        let mut j = i + 2;
                        let mut depth = 0i32;
                        while j < lines.len() {
                            let lt = lines[j].trim();
                            depth += lt.matches('{').count() as i32 - lt.matches('}').count() as i32;
                            if depth < 0 { break; }
                            if (lt.starts_with("fn ") || lt.starts_with("pub fn ")) && lt.contains('(') {
                                if let Some(name) = extract_fn_name_from_line(lt) {
                                    exports.push((name, "pyo3".to_string()));
                                }
                            }
                            j += 1;
                        }
                    }
                }
            }

            // ── JNI: JNIEXPORT ... Java_pkg_Class_method ──────────────
            if trimmed.contains("JNIEXPORT") && trimmed.contains("Java_") {
                // Extract: Java_com_example_ClassName_methodName
                if let Some(java_pos) = trimmed.find("Java_") {
                    let rest = &trimmed[java_pos..];
                    let jni_name: String = rest.chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    // Last component after final underscore is the method name
                    if let Some(method) = jni_name.rsplit('_').next() {
                        if !method.is_empty() {
                            exports.push((method.to_string(), "jni".to_string()));
                        }
                    }
                }
            }
            // JNI Java-side: public native void process(
            if (trimmed.contains("native ") || trimmed.starts_with("native "))
                && trimmed.contains('(')
            {
                let before_paren = trimmed.split('(').next().unwrap_or("");
                let words: Vec<&str> = before_paren.split_whitespace().collect();
                if let Some(name) = words.last() {
                    let clean = name.trim();
                    if !clean.is_empty() && clean != "native" {
                        exports.push((clean.to_string(), "jni".to_string()));
                    }
                }
            }

            // ── CGo: //export FunctionName ─────────────────────────────
            if trimmed.starts_with("//export ") {
                let name = trimmed.trim_start_matches("//export ").trim();
                if !name.is_empty() {
                    exports.push((name.to_string(), "cgo".to_string()));
                }
            }

            // ── WASM: #[wasm_bindgen] ─────────────────────────────────
            if trimmed.contains("#[wasm_bindgen]") || trimmed.contains("#[wasm_bindgen(") {
                if let Some(name) = extract_fn_name_from_line(trimmed) {
                    exports.push((name, "wasm".to_string()));
                } else if i + 1 < lines.len() {
                    if let Some(name) = extract_fn_name_from_line(lines[i + 1].trim()) {
                        exports.push((name, "wasm".to_string()));
                    }
                }
            }

            // ── N-API: #[napi] ────────────────────────────────────────
            if trimmed.contains("#[napi]") || trimmed.contains("#[napi(") {
                if let Some(name) = extract_fn_name_from_line(trimmed) {
                    exports.push((name, "napi".to_string()));
                } else if i + 1 < lines.len() {
                    if let Some(name) = extract_fn_name_from_line(lines[i + 1].trim()) {
                        exports.push((name, "napi".to_string()));
                    }
                }
            }

            // ── C FFI: extern "C" fn name( ────────────────────────────
            if trimmed.contains("extern \"C\"") && trimmed.contains("fn ") && trimmed.contains('(') {
                if let Some(name) = extract_fn_name_from_line(trimmed) {
                    exports.push((name, "c_ffi".to_string()));
                }
            }
            // C header-style: void __attribute__((visibility("default"))) func_name(
            // or simply exported C functions
            if trimmed.contains("__attribute__") && trimmed.contains("visibility")
                && trimmed.contains('(')
            {
                let before_paren = trimmed.split('(').next().unwrap_or("");
                let words: Vec<&str> = before_paren.split_whitespace().collect();
                if let Some(name) = words.last() {
                    let clean = name.trim();
                    if !clean.is_empty() {
                        exports.push((clean.to_string(), "c_ffi".to_string()));
                    }
                }
            }
        }

        exports
    }

    /// Check if a symbol appears to be used as a type reference
    /// (e.g., `: TypeName`, `-> TypeName`, `<TypeName>`).
    fn is_type_reference(lower_content: &str, ident: &str) -> bool {
        let lower_ident = ident.to_lowercase();
        // Check for type annotation patterns: ": Type", "-> Type", "<Type>"
        let patterns = [
            format!(": {}", lower_ident),
            format!("-> {}", lower_ident),
            format!("<{}", lower_ident),
            format!("isinstance(_, {}", lower_ident),
        ];
        patterns.iter().any(|p| lower_content.contains(p.as_str()))
    }

    /// Get all fragments that this fragment depends on (transitively).
    pub fn transitive_deps(&self, fragment_id: &str, max_depth: usize) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((fragment_id.to_string(), 0usize));
        visited.insert(fragment_id.to_string());

        let mut result = Vec::new();

        while let Some((fid, depth)) = queue.pop_front() {
            if depth > 0 {
                result.push(fid.clone());
            }
            if depth < max_depth {
                if let Some(deps) = self.outgoing.get(&fid) {
                    for dep in deps {
                        if !visited.contains(&dep.target_id) {
                            visited.insert(dep.target_id.clone());
                            queue.push_back((dep.target_id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        result
    }

    /// Get all fragments that DEPEND ON this fragment (reverse deps).
    pub fn reverse_deps(&self, fragment_id: &str) -> Vec<String> {
        let mut result = Vec::new();
        if let Some(deps) = self.incoming.get(fragment_id) {
            for dep in deps {
                result.push(dep.source_id.clone());
            }
        }
        result
    }

    /// Compute dependency boost for each fragment.
    ///
    /// If fragment A is selected and A depends on B, then B gets
    /// a boost proportional to the dependency strength.
    ///
    /// This solves the "context is not additive" problem:
    /// instead of value(A+B) = value(A) + value(B),
    /// we get value(A+B) = value(A) + value(B) + dep_bonus(A,B).
    pub fn compute_dep_boosts(
        &self,
        selected_ids: &HashSet<String>,
    ) -> HashMap<String, f64> {
        let mut boosts: HashMap<String, f64> = HashMap::new();

        for selected_id in selected_ids {
            if let Some(deps) = self.outgoing.get(selected_id) {
                for dep in deps {
                    // If a selected fragment depends on target,
                    // boost the target's relevance
                    let entry = boosts.entry(dep.target_id.clone()).or_insert(0.0);
                    *entry += dep.strength;
                }
            }
        }

        // Normalize to [0, 1]
        if let Some(&max_boost) = boosts.values().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            if max_boost > 0.0 {
                for v in boosts.values_mut() {
                    *v = (*v / max_boost).min(1.0);
                }
            }
        }

        boosts
    }

    /// Find connected components — fragments that should be
    /// selected or dropped together.
#[allow(dead_code)]
    pub fn connected_components(&self, fragment_ids: &[String]) -> Vec<Vec<String>> {
        let id_set: HashSet<&str> = fragment_ids.iter().map(|s| s.as_str()).collect();
        let mut visited: HashSet<String> = HashSet::new();
        let mut components: Vec<Vec<String>> = Vec::new();

        for fid in fragment_ids {
            if visited.contains(fid) {
                continue;
            }

            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(fid.clone());
            visited.insert(fid.clone());

            while let Some(current) = queue.pop_front() {
                component.push(current.clone());

                // Check outgoing
                if let Some(deps) = self.outgoing.get(&current) {
                    for dep in deps {
                        if id_set.contains(dep.target_id.as_str()) && !visited.contains(&dep.target_id) {
                            visited.insert(dep.target_id.clone());
                            queue.push_back(dep.target_id.clone());
                        }
                    }
                }

                // Check incoming (undirected connectivity)
                if let Some(deps) = self.incoming.get(&current) {
                    for dep in deps {
                        if id_set.contains(dep.source_id.as_str()) && !visited.contains(&dep.source_id) {
                            visited.insert(dep.source_id.clone());
                            queue.push_back(dep.source_id.clone());
                        }
                    }
                }
            }

            if !component.is_empty() {
                components.push(component);
            }
        }

        components
    }

    /// Get all symbol → fragment_id mappings.
    pub fn symbol_definitions(&self) -> &HashMap<String, String> {
        &self.symbol_table
    }

    /// Check if a symbol is defined anywhere.
    pub fn has_symbol(&self, symbol: &str) -> bool {
        self.symbol_table.contains_key(symbol)
    }

    pub fn edge_count(&self) -> usize {
        self.outgoing.values().map(|v| v.len()).sum()
    }

    pub fn node_count(&self) -> usize {
        let mut nodes: HashSet<&str> = HashSet::new();
        for (k, deps) in &self.outgoing {
            nodes.insert(k);
            for dep in deps {
                nodes.insert(&dep.target_id);
            }
        }
        nodes.len()
    }
}

/// Extract identifiers (function names, variable names) from source code.
/// Fast, regex-free extraction for supported languages.
pub fn extract_identifiers(content: &str) -> Vec<String> {
    let mut identifiers = Vec::new();
    let mut chars = content.chars().peekable();
    let mut current = String::new();

    while let Some(&ch) = chars.peek() {
        if ch.is_alphanumeric() || ch == '_' { 
            current.push(ch);
            chars.next();
        } else {
            if !current.is_empty() && current.len() > 1 {
                // Filter out common keywords
                if !is_keyword(&current) {
                    identifiers.push(current.clone());
                }
                current.clear();
            } else {
                current.clear();
            }
            chars.next();
        }
    }

    if !current.is_empty() && current.len() > 1 && !is_keyword(&current) {
        identifiers.push(current);
    }

    identifiers
}

/// Extract symbol definitions (def, class, fn, struct, etc.)
fn extract_definitions(content: &str) -> Vec<String> {
    let mut defs = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Python: def foo(, class Foo(
        if trimmed.starts_with("def ") || trimmed.starts_with("class ") {
            if let Some(name) = trimmed.split_whitespace().nth(1) {
                // Split on '(' to get the name before params
                let clean = name.split('(').next().unwrap_or(name);
                let clean = clean.trim_end_matches(':');
                if !clean.is_empty() {
                    defs.push(clean.to_string());
                }
            }
        }
        // Rust: fn foo(, struct Foo, enum Foo, trait Foo
        else if trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("trait ")
        {
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            // Skip visibility modifier
        let name_idx = if words.first() == Some(&"pub") { 2 } else { 1 };
            if let Some(name) = words.get(name_idx) {
                let clean = name.split('(').next().unwrap_or(name);
                let clean = clean.trim_end_matches(['{', '<', ':']);
                if !clean.is_empty() {
                    defs.push(clean.to_string());
                }
            }
        }
        // JS/TS: function foo(, export function foo(
        else if trimmed.starts_with("function ") || trimmed.starts_with("export function ")
            || trimmed.starts_with("export default function ")
            || trimmed.starts_with("async function ")
            || trimmed.starts_with("export async function ")
        {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let fn_idx = parts.iter().position(|&w| w == "function").map(|i| i + 1);
            if let Some(idx) = fn_idx {
                if let Some(name) = parts.get(idx) {
                    let clean = name.split('(').next().unwrap_or(name);
                    let clean = clean.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');
                    if !clean.is_empty() {
                        defs.push(clean.to_string());
                    }
                }
            }
        }
        // JS/TS: const App = () => {, const handler = async () => {, export const foo = (
        // Also catches: const schema = z.object({, const router = express.Router()
        else if (trimmed.starts_with("const ") || trimmed.starts_with("let ")
            || trimmed.starts_with("export const ") || trimmed.starts_with("export let ")
            || trimmed.starts_with("export default "))
            && (trimmed.contains("=>") || trimmed.contains("= function")
                || trimmed.contains("= async function") || trimmed.contains("= ("))
        {
            // Extract: const NAME = ...
            let after_kw = trimmed
                .trim_start_matches("export ")
                .trim_start_matches("default ")
                .trim_start_matches("const ")
                .trim_start_matches("let ");
            if let Some(name) = after_kw.split([' ', ':', '=']).next() {
                let clean = name.trim();
                if !clean.is_empty() && clean.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') {
                    defs.push(clean.to_string());
                }
            }
        }
        // TS: interface Foo {, export interface Foo {
        else if trimmed.starts_with("interface ") || trimmed.starts_with("export interface ") {
            let after_kw = trimmed
                .trim_start_matches("export ")
                .trim_start_matches("interface ");
            if let Some(name) = after_kw.split([' ', '{', '<']).next() {
                let clean = name.trim();
                if !clean.is_empty() {
                    defs.push(clean.to_string());
                }
            }
        }
        // TS: type Foo = ..., export type Foo = ...
        else if (trimmed.starts_with("type ") || trimmed.starts_with("export type "))
            && trimmed.contains('=')
        {
            let after_kw = trimmed
                .trim_start_matches("export ")
                .trim_start_matches("type ");
            if let Some(name) = after_kw.split([' ', '=', '<']).next() {
                let clean = name.trim();
                if !clean.is_empty() {
                    defs.push(clean.to_string());
                }
            }
        }
        // TS/JS: enum Status {, export enum Status {
        else if trimmed.starts_with("enum ") || trimmed.starts_with("export enum ")
            || trimmed.starts_with("const enum ") || trimmed.starts_with("export const enum ")
        {
            let after_kw = trimmed
                .trim_start_matches("export ")
                .trim_start_matches("const ")
                .trim_start_matches("enum ");
            if let Some(name) = after_kw.split([' ', '{']).next() {
                let clean = name.trim();
                if !clean.is_empty() {
                    defs.push(clean.to_string());
                }
            }
        }
        // Go: func HandleRequest(, func (s *Server) Start(, type Config struct
        else if trimmed.starts_with("func ") {
            // Method: func (recv) Name(  or  Function: func Name(
            if trimmed.starts_with("func (") {
                // Method receiver: func (s *Server) Name(
                if let Some(after_paren) = trimmed.find(") ") {
                    let rest = &trimmed[after_paren + 2..];
                    if let Some(name) = rest.split('(').next() {
                        let clean = name.trim();
                        if !clean.is_empty() {
                            defs.push(clean.to_string());
                        }
                    }
                }
            } else {
                // Regular function: func Name(
                let rest = trimmed.trim_start_matches("func ");
                if let Some(name) = rest.split('(').next() {
                    let clean = name.trim();
                    if !clean.is_empty() {
                        defs.push(clean.to_string());
                    }
                }
            }
        }
        // Go: type Config struct {, type Handler interface {, type ID = string
        else if trimmed.starts_with("type ") && !trimmed.contains('=') {
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            if words.len() >= 3 {
                let name = words[1];
                if !name.is_empty() {
                    defs.push(name.to_string());
                }
            }
        }
        // Java/Kotlin: public class Foo {, abstract class Bar, interface Baz
        else if (trimmed.contains("class ") || trimmed.contains("interface "))
            && (trimmed.starts_with("public ") || trimmed.starts_with("private ")
                || trimmed.starts_with("protected ") || trimmed.starts_with("abstract ")
                || trimmed.starts_with("final ") || trimmed.starts_with("class ")
                || trimmed.starts_with("interface ") || trimmed.starts_with("data class ")
                || trimmed.starts_with("sealed ") || trimmed.starts_with("open "))
        {
            // Find "class Name" or "interface Name"
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            for (i, &w) in words.iter().enumerate() {
                if (w == "class" || w == "interface") && i + 1 < words.len() {
                    let name = words[i + 1].split(['{', '<', '(', ':']).next().unwrap_or("");
                    if !name.is_empty() {
                        defs.push(name.to_string());
                    }
                    break;
                }
            }
        }
        // Java/Kotlin: public void handleRequest(, public static String process(
        else if (trimmed.starts_with("public ") || trimmed.starts_with("private ")
            || trimmed.starts_with("protected ") || trimmed.starts_with("static ")
            || trimmed.starts_with("override ") || trimmed.starts_with("suspend "))
            && trimmed.contains('(') && !trimmed.contains("class ") && !trimmed.contains("interface ")
        {
            let before_paren = trimmed.split('(').next().unwrap_or("");
            let words: Vec<&str> = before_paren.split_whitespace().collect();
            // Last word before '(' is the method name
            if let Some(name) = words.last() {
                let clean = name.trim();
                if !clean.is_empty() && !is_keyword(clean) {
                    defs.push(clean.to_string());
                }
            }
        }
        // C/C++: class Foo {, struct Bar {, namespace Baz {
        else if (trimmed.starts_with("class ") || trimmed.starts_with("struct ")
            || trimmed.starts_with("namespace "))
            && !trimmed.starts_with("class ") // avoid conflicting with Java above (Java caught above)
            || (trimmed.starts_with("typedef ") && trimmed.contains("struct"))
        {
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            if words.len() >= 2 {
                let name = words[1].split(['{', ';', ':']).next().unwrap_or("");
                if !name.is_empty() && !is_keyword(name) {
                    defs.push(name.to_string());
                }
            }
        }
        // Ruby: class Foo, module Bar, def method_name
        else if trimmed.starts_with("module ") && !trimmed.starts_with("module.") {
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            if words.len() >= 2 {
                let name = words[1].split([';', '<', ':']).next().unwrap_or("");
                if !name.is_empty() {
                    defs.push(name.to_string());
                }
            }
        }
    }

    defs
}

/// Extract the LHS variable name from `const foo = require('...')`.
fn extract_require_lhs(line: &str) -> Option<String> {
    let trimmed = line.trim();
    // Match: const/let/var NAME = require(
    for kw in &["const ", "let ", "var "] {
        if let Some(rest) = trimmed.strip_prefix(kw) {
            if let Some(eq_pos) = rest.find('=') {
                let before_eq = rest[..eq_pos].trim();
                let after_eq = rest[eq_pos + 1..].trim();
                if after_eq.starts_with("require(") {
                    // Handle destructuring: const { foo, bar } = require(...)
                    if before_eq.starts_with('{') && before_eq.contains('}') {
                        let inner = &before_eq[1..before_eq.find('}')?];
                        for name in inner.split(',') {
                            let clean = name.trim().split(':').next()?.trim();
                            if !clean.is_empty() {
                                return Some(clean.to_string());
                            }
                        }
                    } else {
                        let clean = before_eq.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                        if !clean.is_empty() {
                            return Some(clean.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract function name from a line like "fn process(" or "pub fn process(".
fn extract_fn_name_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let fn_pos = trimmed.find("fn ")?;
    let after_fn = &trimmed[fn_pos + 3..];
    let name: String = after_fn.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

/// Extract struct/class name from a line like "struct Engine" or "pub struct Engine {".
fn extract_struct_name_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    for kw in &["struct ", "class ", "enum "] {
        if let Some(pos) = trimmed.find(kw) {
            let after = &trimmed[pos + kw.len()..];
            let name: String = after.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() { return Some(name); }
        }
    }
    None
}

/// Check if an identifier is a language keyword (ignore these).
fn is_keyword(word: &str) -> bool {
    matches!(word,
        // Python
        "def" | "class" | "import" | "from" | "return" | "if" | "else" | "elif"
        | "for" | "while" | "try" | "except" | "finally" | "with" | "as" | "and"
        | "or" | "not" | "in" | "is" | "None" | "True" | "False" | "self" | "pass"
        | "break" | "continue" | "raise" | "yield" | "lambda" | "async" | "await"
        // Rust
        | "fn" | "let" | "mut" | "pub" | "use" | "mod" | "impl" | "struct"
        | "enum" | "trait" | "where" | "match" | "loop" | "move" | "ref"
        | "static" | "const" | "type" | "unsafe" | "extern" | "crate" | "super"
        // JS/TS
        | "function" | "var" | "this" | "new" | "typeof" | "instanceof"
        | "void" | "delete" | "throw" | "catch" | "switch" | "case" | "default"
        | "export" | "require" | "module" | "extends" | "constructor"
        | "interface" | "implements" | "abstract" | "declare" | "namespace"
        | "readonly" | "keyof" | "infer" | "never" | "any" | "unknown"
        | "satisfies" | "override" | "private" | "protected" | "public"
        // Go
        | "func" | "package" | "chan" | "defer" | "go" | "select" | "range"
        | "map" | "fallthrough" | "goto"
        // Java / C#
        | "final" | "synchronized" | "throws" | "volatile" | "transient"
        | "native" | "strictfp" | "assert"
        | "sealed" | "record" | "permits"
        | "using" | "virtual" | "internal" | "partial" | "event" | "delegate"
        | "foreach" | "checked" | "unchecked" | "fixed" | "lock" | "params"
        // Swift
        | "guard" | "protocol" | "extension" | "typealias"
        | "associatedtype" | "inout" | "lazy" | "weak" | "unowned"
        | "convenience" | "required" | "mutating" | "nonmutating" | "indirect"
        // C/C++
        | "auto" | "register" | "inline" | "sizeof" | "template" | "typename"
        | "explicit" | "friend" | "mutable" | "operator" | "dynamic_cast"
        | "static_cast" | "reinterpret_cast" | "const_cast" | "noexcept"
        | "constexpr" | "decltype" | "nullptr" | "alignas" | "alignof"
        // Ruby
        | "begin" | "end" | "rescue" | "ensure" | "defined" | "do" | "then"
        | "elsif" | "unless" | "until" | "when" | "redo" | "retry"
        // PHP
        | "echo" | "print" | "isset" | "unset" | "empty" | "die" | "exit"
        | "include" | "include_once" | "require_once" | "list" | "global"
        | "endfor" | "endforeach" | "endif" | "endwhile" | "endswitch"
        // Common
        | "true" | "false" | "null" | "undefined" | "int" | "str" | "float"
        | "bool" | "string" | "number" | "object" | "array" | "symbol" | "bigint"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_definitions_python() {
        let code = "def process_payment(amount):\n    return amount * rate\n\nclass PaymentProcessor:\n    pass";
        let defs = extract_definitions(code);
        assert!(defs.contains(&"process_payment".to_string()));
        assert!(defs.contains(&"PaymentProcessor".to_string()));
    }

    #[test]
    fn test_extract_definitions_rust() {
        let code = "pub fn calculate_tax(income: f64) -> f64 {\n    income * 0.3\n}\n\nstruct TaxResult {\n    amount: f64,\n}";
        let defs = extract_definitions(code);
        assert!(defs.contains(&"calculate_tax".to_string()));
        assert!(defs.contains(&"TaxResult".to_string()));
    }

    #[test]
    fn test_dependency_boost() {
        let mut graph = DepGraph::new();
        graph.register_symbol("process_payment", "frag_payments");
        graph.register_symbol("get_rate", "frag_rates");

        graph.add_dependency(Dependency {
            source_id: "frag_payments".into(),
            target_id: "frag_rates".into(),
            dep_type: DepType::FunctionCall,
            strength: 0.8,
        });

        let mut selected = HashSet::new();
        selected.insert("frag_payments".to_string());

        let boosts = graph.compute_dep_boosts(&selected);
        assert!(boosts.get("frag_rates").unwrap_or(&0.0) > &0.0,
            "frag_rates should get boost when frag_payments is selected");
    }

    #[test]
    fn test_connected_components() {
        let mut graph = DepGraph::new();
        graph.add_dependency(Dependency {
            source_id: "a".into(),
            target_id: "b".into(),
            dep_type: DepType::Import,
            strength: 1.0,
        });
        // "c" is isolated

        let components = graph.connected_components(&[
            "a".into(), "b".into(), "c".into(),
        ]);

        // Should be 2 components: {a, b} and {c}
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_transitive_deps() {
        let mut graph = DepGraph::new();
        graph.add_dependency(Dependency {
            source_id: "a".into(), target_id: "b".into(),
            dep_type: DepType::FunctionCall, strength: 1.0,
        });
        graph.add_dependency(Dependency {
            source_id: "b".into(), target_id: "c".into(),
            dep_type: DepType::Import, strength: 1.0,
        });

        let deps = graph.transitive_deps("a", 3);
        assert!(deps.contains(&"b".to_string()));
        assert!(deps.contains(&"c".to_string()));
    }

    #[test]
    fn test_extract_definitions_js_function() {
        // This was the dead-symbol false positive: "function App()" was
        // registered as "App()" instead of "App", so the reference "App"
        // (from <App />) didn't match and App was flagged as dead code.
        let code = "function App() {\n  return <div>Hello</div>;\n}\n\nexport function handleClick(event) {\n  console.log(event);\n}";
        let defs = extract_definitions(code);
        assert!(defs.contains(&"App".to_string()),
            "function App() should extract 'App', got: {:?}", defs);
        assert!(!defs.iter().any(|d| d.contains('(')),
            "No definition should contain parentheses, got: {:?}", defs);
        assert!(defs.contains(&"handleClick".to_string()),
            "export function handleClick() should extract 'handleClick'");
    }

    #[test]
    fn test_extract_definitions_arrow_functions() {
        let code = "const App = () => {\n  return <div/>;\n}\n\nexport const handler = async () => {\n  await fetch();\n}";
        let defs = extract_definitions(code);
        assert!(defs.contains(&"App".to_string()), "arrow fn: {:?}", defs);
        assert!(defs.contains(&"handler".to_string()), "export arrow fn: {:?}", defs);
    }

    #[test]
    fn test_extract_definitions_ts_interface_type_enum() {
        let code = "interface User {\n  name: string;\n}\n\nexport type Status = 'active' | 'inactive';\n\nenum Color {\n  Red,\n  Blue,\n}";
        let defs = extract_definitions(code);
        assert!(defs.contains(&"User".to_string()), "interface: {:?}", defs);
        assert!(defs.contains(&"Status".to_string()), "type alias: {:?}", defs);
        assert!(defs.contains(&"Color".to_string()), "enum: {:?}", defs);
    }

    #[test]
    fn test_extract_import_targets_default_import() {
        let code = "import React from 'react';\nimport { useState } from 'react';";
        let targets = DepGraph::extract_import_targets(code);
        assert!(targets.contains("React"), "default import: {:?}", targets);
        assert!(targets.contains("useState"), "named import: {:?}", targets);
    }

    #[test]
    fn test_extract_import_targets_require() {
        let code = "const express = require('express');\nconst { Router } = require('express');";
        let targets = DepGraph::extract_import_targets(code);
        assert!(targets.contains("express"), "require: {:?}", targets);
    }

    // ── P2: Cross-language FFI dep tracking ─────────────────────────

    #[test]
    fn test_pyo3_rust_to_python_linking() {
        let mut graph = DepGraph::new();

        // Rust side: #[pyfunction] fn process_data(...)
        let rust_code = r#"
use pyo3::prelude::*;

#[pyfunction]
fn process_data(input: &str) -> PyResult<String> {
    Ok(input.to_uppercase())
}

#[pyclass]
struct Engine {
    config: String,
}
"#;
        graph.auto_link("frag_rust_lib", rust_code);

        // Python side: from mymodule import process_data
        let python_code = r#"
from mymodule import process_data, Engine

result = process_data("hello")
engine = Engine()
"#;
        graph.auto_link("frag_python_app", python_code);

        // Verify cross-language edges were created
        let deps = graph.outgoing.get("frag_python_app").expect("Should have deps");
        let cross_lang_deps: Vec<_> = deps.iter()
            .filter(|d| d.dep_type == DepType::CrossLanguageFFI)
            .collect();
        assert!(!cross_lang_deps.is_empty(),
            "Python→Rust cross-language edge should exist, deps: {:?}",
            deps.iter().map(|d| (&d.target_id, &d.dep_type)).collect::<Vec<_>>());
        assert!(cross_lang_deps.iter().any(|d| d.target_id == "frag_rust_lib"),
            "Should link to the Rust fragment");
    }

    #[test]
    fn test_jni_java_to_c_linking() {
        let mut graph = DepGraph::new();

        // C side: JNIEXPORT void JNICALL Java_com_example_App_processData
        let c_code = r#"
#include <jni.h>
JNIEXPORT void JNICALL Java_com_example_App_processData(JNIEnv *env, jobject obj) {
    // native impl
}
"#;
        graph.auto_link("frag_c_native", c_code);

        // Java side: native void processData()
        let java_code = r#"
public class App {
    public native void processData();
    static { System.loadLibrary("mylib"); }
}
"#;
        graph.auto_link("frag_java_app", java_code);

        // Both sides should export "processData" via JNI
        assert!(graph.cross_lang_exports.contains_key("processData"),
            "processData should be in cross_lang_exports: {:?}",
            graph.cross_lang_exports.keys().collect::<Vec<_>>());
    }

    #[test]
    fn test_cgo_export_linking() {
        let mut graph = DepGraph::new();

        let go_code = r#"
package main

import "C"

//export ProcessData
func ProcessData(input *C.char) *C.char {
    return C.CString("processed")
}
"#;
        graph.auto_link("frag_go_lib", go_code);

        assert!(graph.cross_lang_exports.contains_key("ProcessData"),
            "CGo export should register: {:?}",
            graph.cross_lang_exports.keys().collect::<Vec<_>>());
    }

    #[test]
    fn test_wasm_bindgen_linking() {
        let mut graph = DepGraph::new();

        let rust_code = r#"
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;
        graph.auto_link("frag_wasm_lib", rust_code);

        // JS side imports greet
        let js_code = "import { greet } from './pkg/my_wasm';\nconst msg = greet('world');";
        graph.auto_link("frag_js_app", js_code);

        assert!(graph.cross_lang_exports.contains_key("greet"),
            "wasm_bindgen export should register");
        let deps = graph.outgoing.get("frag_js_app").expect("Should have deps");
        assert!(deps.iter().any(|d| d.dep_type == DepType::CrossLanguageFFI),
            "JS→WASM cross-language edge should exist");
    }

    #[test]
    fn test_napi_linking() {
        let mut graph = DepGraph::new();

        let rust_code = "#[napi]\nfn compute(a: i32, b: i32) -> i32 { a + b }";
        graph.auto_link("frag_napi_lib", rust_code);

        assert!(graph.cross_lang_exports.contains_key("compute"),
            "N-API export should register");
    }

    #[test]
    fn test_c_ffi_extern_c_linking() {
        let mut graph = DepGraph::new();

        let rust_code = r#"
#[no_mangle]
pub extern "C" fn init_engine(config: *const c_char) -> *mut Engine {
    Box::into_raw(Box::new(Engine::new()))
}
"#;
        graph.auto_link("frag_rust_ffi", rust_code);

        // Python ctypes consumer
        let python_code = "import ctypes\nlib = ctypes.CDLL('./libengine.so')\nresult = lib.init_engine(config)";
        graph.auto_link("frag_python_ctypes", python_code);

        assert!(graph.cross_lang_exports.contains_key("init_engine"),
            "extern C export should register: {:?}",
            graph.cross_lang_exports.keys().collect::<Vec<_>>());
    }

    #[test]
    fn test_cross_lang_dep_boost() {
        // Cross-language deps should participate in dependency boost
        let mut graph = DepGraph::new();

        let rust_code = "#[pyfunction]\nfn analyze(data: &str) -> PyResult<String> { Ok(data.to_string()) }";
        graph.auto_link("frag_rust", rust_code);

        let python_code = "from mymod import analyze\nresult = analyze(raw_data)";
        graph.auto_link("frag_python", python_code);

        let mut selected = HashSet::new();
        selected.insert("frag_python".to_string());

        let boosts = graph.compute_dep_boosts(&selected);
        assert!(boosts.get("frag_rust").unwrap_or(&0.0) > &0.0,
            "Rust fragment should get boost when Python consumer is selected");
    }

    #[test]
    fn test_pymethods_exports_methods() {
        let mut graph = DepGraph::new();

        let rust_code = r#"
#[pymethods]
impl Engine {
    fn run(&self, input: &str) -> String {
        input.to_uppercase()
    }
    pub fn stop(&self) {
        // cleanup
    }
}
"#;
        graph.auto_link("frag_engine", rust_code);

        assert!(graph.cross_lang_exports.contains_key("run"),
            "#[pymethods] should export method 'run': {:?}",
            graph.cross_lang_exports.keys().collect::<Vec<_>>());
        assert!(graph.cross_lang_exports.contains_key("stop"),
            "#[pymethods] should export method 'stop'");
    }
}
