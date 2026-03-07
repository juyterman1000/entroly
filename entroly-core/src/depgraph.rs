/// Dependency Graph Engine
///
/// Addresses the core weakness: context is NOT additive.
///
///   value(A + B) ≠ value(A) + value(B)
///
/// Code has dependencies. A function call without its definition
/// is useless. A type reference without its schema is confusing.
/// Removing one fragment can destroy the value of others.
///
/// This module implements:
///   1. **Dependency extraction** — parse imports, calls, types from source
///   2. **Graph construction** — directed graph of fragment → fragment deps
///   3. **Graph-aware selection** — if A is selected and A depends on B,
///      B's value is boosted (or B is force-included)
///   4. **Connected component analysis** — fragments in the same component
///      should be selected or dropped together
///
/// This transforms the problem from flat knapsack to:
///   Graph-constrained knapsack (NP-hard in general, but tractable
///   for typical code dependency graphs with ~500 nodes)
///
/// References:
///   - Sourcegraph's Code Intelligence — graph-based code navigation
///   - Code Property Graphs (Yamaguchi et al., 2014)

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
}

impl DepGraph {
    pub fn new() -> Self {
        DepGraph {
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            symbol_table: HashMap::new(),
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

        self.outgoing.entry(source).or_insert_with(Vec::new).push(dep.clone());
        self.incoming.entry(target).or_insert_with(Vec::new).push(dep);
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
    }

    /// Extract symbols that are explicitly imported in source code.
    /// Parses Python `from X import Y`, `import X`, Rust `use x::Y`,
    /// and JS `import { Y } from`, `require('X')`.
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
                    let inner = &path[brace_start+1..path.len()-path.chars().rev().position(|c| c == '}').unwrap_or(0)-1];
                    for name in inner.split(',') {
                        let clean = name.trim().split(" as ").next().unwrap_or("").trim();
                        if !clean.is_empty() {
                            targets.insert(clean.to_string());
                        }
                    }
                } else {
                    // use foo::Bar; → import "Bar"
                    let last = path.rsplit("::").next().unwrap_or(path).trim();
                    if !last.is_empty() {
                        targets.insert(last.to_string());
                    }
                }
            }
            // JS/TS: import { foo, bar } from '...'
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
                }
            }
        }
        targets
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
                let clean = clean.trim_end_matches(|c: char| c == '{' || c == '<' || c == ':');
                if !clean.is_empty() {
                    defs.push(clean.to_string());
                }
            }
        }
        // JS/TS: function foo(, const foo =, export function
        else if trimmed.starts_with("function ") || trimmed.starts_with("export function ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let fn_idx = parts.iter().position(|&w| w == "function").map(|i| i + 1);
            if let Some(idx) = fn_idx {
                if let Some(name) = parts.get(idx) {
                    let clean = name.trim_end_matches('(');
                    if !clean.is_empty() {
                        defs.push(clean.to_string());
                    }
                }
            }
        }
    }

    defs
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
        | "function" | "var" | "const" | "this" | "new" | "typeof" | "instanceof"
        | "void" | "delete" | "throw" | "catch" | "switch" | "case" | "default"
        | "export" | "require" | "module" | "extends" | "constructor"
        // Common
        | "true" | "false" | "null" | "undefined" | "int" | "str" | "float"
        | "bool" | "string" | "number" | "object" | "array"
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
}
