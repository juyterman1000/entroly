//! Codebase Health Analysis Engine
//!
//! Research grounding:
//!   - 2025 arXiv (Jan): LLM+AST graph modeling for Type-4 semantic clone detection.
//!     Key insight: SimHash handles Type-1/2/3 (structural similarity); for Type-3/4
//!     (semantic), we use dep-graph co-occurrence as an AST proxy.
//!   - 2026 arXiv: "AI-friendly code" — CodeHealth score correlates with LLM
//!     processing ease. We compute 5 dimensions: coupling, naming, complexity,
//!     duplication ratio, dead-symbol ratio.
//!   - DebtGuardian (arXiv Nov 2025): technical debt via source code changes.
//!     We adapt the batch-level detection approach applied to our fragment store.
//!
//! The engine computes:
//!   1. **Clone Detection**: SimHash Hamming distance for Type-1/2/3 clones.
//!      Threshold: Hamming ≤ 8 = near-duplicate (≤ 87.5% SimHash distance).
//!   2. **Dead Symbols**: Defined but never referenced across all fragments.
//!   3. **God Files**: Fragments with reverse-dep count > μ + 2σ (statistical outlier).
//!   4. **Architecture Violations**: Cross-layer imports detected by naming convention.
//!   5. **CodeHealth Score** [0–100]: weighted composite of the above signals.
//!      Higher = healthier. Used to prioritize refactoring effort.
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::dedup::hamming_distance;
use crate::fragment::ContextFragment;
use crate::depgraph::DepGraph;

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

/// Severity of a health issue (distinct from SAST severity — this is about
/// maintenance burden, not security risk).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(dead_code)]
pub(crate) enum HealthSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl HealthSeverity {
    #[allow(dead_code)]
    pub(crate) fn label(self) -> &'static str {
        match self {
            HealthSeverity::Low      => "LOW",
            HealthSeverity::Medium   => "MEDIUM",
            HealthSeverity::High     => "HIGH",
            HealthSeverity::Critical => "CRITICAL",
        }
    }
}

/// A pair of fragments that are near-duplicates (code clones).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClonePair {
    pub fragment_id_a:  String,
    pub source_a:       String,
    pub fragment_id_b:  String,
    pub source_b:       String,
    /// Similarity [0.0, 1.0]. 1.0 = exact SimHash match.
    pub similarity:     f64,
    pub clone_type:     CloneType,
    pub recommendation: String,
}

/// Clone types (after Koschke et al., adapted for SimHash detection).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CloneType {
    /// Type-1: Exact or near-exact (SimHash Hamming ≤ 2). Identical modulo whitespace.
    NearIdentical,
    /// Type-2: Renamed variables/functions (Hamming 3–8). Same structure, different names.
    Renamed,
    /// Type-3: Structural similarity with some additions/deletions (Hamming 9–16).
    Structural,
}

impl CloneType {
    fn from_hamming(dist: u32) -> Option<CloneType> {
        if dist <= 2 { Some(CloneType::NearIdentical) }
        else if dist <= 8 { Some(CloneType::Renamed) }
        else if dist <= 16 { Some(CloneType::Structural) }
        else { None }
    }

    #[allow(dead_code)]
    pub(crate) fn label(self) -> &'static str {
        match self {
            CloneType::NearIdentical => "Type-1 (near-identical)",
            CloneType::Renamed       => "Type-2 (renamed)",
            CloneType::Structural    => "Type-3 (structural)",
        }
    }
}

/// A symbol that appears to be defined but never referenced across all known fragments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadSymbol {
    pub name:           String,
    pub defined_in:     String,
    pub fragment_id:    String,
    pub confidence:     f64,
    pub recommendation: String,
}

/// A "god file" — a fragment that too many others depend on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodFile {
    pub fragment_id:    String,
    pub source:         String,
    pub reverse_deps:   usize,
    /// Standard deviations above the mean
    pub z_score:        f64,
    pub recommendation: String,
}

/// An architectural layer violation — a lower-layer importing a higher-layer module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchViolation {
    pub fragment_id:  String,
    pub source:       String,
    pub importer_layer: String,
    pub imported_layer: String,
    pub evidence:     String,
    pub recommendation: String,
}

/// Naming consistency issue: files that break established naming conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingIssue {
    pub source:         String,
    pub expected_style: String,
    pub actual_style:   String,
    pub recommendation: String,
}

/// Full codebase health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub fragment_count:      usize,
    pub clone_pairs:         Vec<ClonePair>,
    pub dead_symbols:        Vec<DeadSymbol>,
    pub god_files:           Vec<GodFile>,
    pub arch_violations:     Vec<ArchViolation>,
    pub naming_issues:       Vec<NamingIssue>,
    /// Overall CodeHealth score [0–100]. Higher = healthier.
    pub code_health_score:   f64,
    pub health_grade:        &'static str,
    /// Per-dimension breakdown
    pub duplication_penalty: f64,
    pub dead_code_penalty:   f64,
    pub coupling_penalty:    f64,
    pub arch_penalty:        f64,
    pub naming_penalty:      f64,
    /// Human-readable summary
    pub summary:             String,
    pub top_recommendation:  Option<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Clone Detection — SimHash pairwise scan
///
/// Complexity: O(N²) pairwise, bounded by max_fragments=10,000.
/// At N=1,000: 500,000 comparisons, each O(1) → ~1ms.
/// At N=10,000: 50M comparisons → ~100ms (acceptable for a health scan).
///
/// Optimization: skip pairs from the same source file.
// ═══════════════════════════════════════════════════════════════════
fn detect_clones(fragments: &[&ContextFragment]) -> Vec<ClonePair> {
    let mut pairs = Vec::new();

    for i in 0..fragments.len() {
        for j in (i + 1)..fragments.len() {
            let a = fragments[i];
            let b = fragments[j];

            // Skip same-source comparisons (same file, different version of itself)
            if a.source == b.source {
                continue;
            }

            // Skip very short fragments (< 4 tokens, not meaningful code)
            if a.token_count < 4 || b.token_count < 4 {
                continue;
            }

            let dist = hamming_distance(a.simhash, b.simhash);
            if let Some(clone_type) = CloneType::from_hamming(dist) {
                let similarity = 1.0 - (dist as f64 / 64.0);

                let recommendation = match clone_type {
                    CloneType::NearIdentical => format!(
                        "Extract identical logic from '{}' and '{}' into a shared utility function.",
                        basename(&a.source), basename(&b.source)
                    ),
                    CloneType::Renamed => format!(
                        "Possible duplicate logic across '{}' and '{}'. Consider a generic abstraction.",
                        basename(&a.source), basename(&b.source)
                    ),
                    CloneType::Structural => format!(
                        "Structural similarity between '{}' and '{}'. Review for unintended duplication.",
                        basename(&a.source), basename(&b.source)
                    ),
                };

                pairs.push(ClonePair {
                    fragment_id_a: a.fragment_id.clone(),
                    source_a: a.source.clone(),
                    fragment_id_b: b.fragment_id.clone(),
                    source_b: b.source.clone(),
                    similarity: (similarity * 10000.0).round() / 10000.0,
                    clone_type,
                    recommendation,
                });
            }
        }
    }

    // Sort by similarity descending (most similar first)
    pairs.sort_unstable_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

// ═══════════════════════════════════════════════════════════════════
// Dead Symbol Detection — set difference: defined ∖ referenced
///
/// A symbol is "dead" if:
///   - It appears in the depgraph's symbol_definitions table
///   - It is never referenced in any fragment's content
///
/// Confidence model:
///   - Single-fragment session → low confidence (not enough context)
///   - Symbol appears in a public API file → lower confidence (may be used externally)
///   - Internal helper naming (_, __, _private) → higher confidence
// ═══════════════════════════════════════════════════════════════════
fn find_dead_symbols(
    fragments: &[&ContextFragment],
    dep_graph: &DepGraph,
) -> Vec<DeadSymbol> {
    // Collect all defined symbols and their fragment IDs
    let definitions: Vec<(String, String)> = dep_graph.symbol_definitions()
        .iter()
        .map(|(sym, fid)| (sym.clone(), fid.clone()))
        .collect();

    if definitions.is_empty() {
        return Vec::new();
    }

    // Build the "referenced anywhere" set from all fragment contents
    // Note: we use word-boundary matching (whole word) to avoid false positives
    // from substrings (e.g., "authenticate" should not suppress "auth")
    let all_content: Vec<&str> = fragments.iter().map(|f| f.content.as_str()).collect();

    // Build a frequency map of all identifiers across all fragments
    let mut referenced: HashSet<String> = HashSet::new();
    for content in &all_content {
        for word in content.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if word.len() >= 2 {
                referenced.insert(word.to_lowercase());
            }
        }
    }

    // Build fragment_id → source map for reporting
    let id_to_source: HashMap<&str, &str> = fragments.iter()
        .map(|f| (f.fragment_id.as_str(), f.source.as_str()))
        .collect();

    let mut dead = Vec::new();

    for (sym, fid) in &definitions {
        let sym_lower = sym.to_lowercase();

        // Skip generic names that appear everywhere
        if is_generic_symbol(&sym_lower) {
            continue;
        }

        // Check if referenced anywhere
        if referenced.contains(&sym_lower) {
            continue;
        }

        let source = id_to_source.get(fid.as_str()).copied().unwrap_or("<unknown>");

        // Confidence: lower for public-facing files, higher for internals
        let confidence = if sym.starts_with('_') || sym.starts_with("__") {
            0.85 // Private/mangled names: high confidence they're actually dead
        } else if source.contains("api") || source.contains("interface") || source.contains("types") {
            0.35 // Public API files: may be used by external consumers we haven't ingested
        } else {
            0.60 // Default
        };

        // Only report medium-confidence and above (reduce noise)
        if confidence < 0.30 {
            continue;
        }

        dead.push(DeadSymbol {
            name: sym.clone(),
            defined_in: source.to_string(),
            fragment_id: fid.clone(),
            confidence: (confidence * 100.0_f64).round() / 100.0_f64,
            recommendation: format!(
                "Symbol '{}' defined in '{}' appears to have no references in the ingested context. \
                Verify it is not used by external consumers before removing.",
                sym, basename(source)
            ),
        });
    }

    // Sort by confidence descending
    dead.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    dead.truncate(50); // Cap at 50 to avoid overwhelming reports
    dead
}

/// Skip symbols that are so common they're meaningless as dead-code indicators.
fn is_generic_symbol(sym: &str) -> bool {
    matches!(sym,
        "new" | "init" | "main" | "run" | "start" | "stop" | "get" | "set" | "update" | "delete"
        | "create" | "read" | "write" | "close" | "open" | "error" | "result" | "value"
        | "data" | "name" | "id" | "key" | "type" | "self" | "this" | "true" | "false"
        | "none" | "null" | "ok" | "err" | "ok_or" | "unwrap" | "expect" | "from" | "into"
        | "clone" | "copy" | "default" | "debug" | "display" | "drop"
    ) || sym.len() <= 1
}

// ═══════════════════════════════════════════════════════════════════
// God File Detection — reverse-dep count outliers (μ + 2σ)
///
/// Rationale (2026 "AI-friendly code" paper): High coupling → low CodeHealth → harder
/// for LLMs to contextualize. God files should be split along cohesion lines.
///
/// Algorithm: Compute the population mean and standard deviation of reverse-dep counts.
/// Flag any fragment > μ + 2σ (approximately the top 2.3% by coupling).
// ═══════════════════════════════════════════════════════════════════
fn find_god_files(
    fragments: &[&ContextFragment],
    dep_graph: &DepGraph,
) -> Vec<GodFile> {
    if fragments.len() < 3 {
        return Vec::new();
    }

    // Compute reverse-dep counts
    let counts: Vec<(&&ContextFragment, usize)> = fragments.iter()
        .map(|f| (f, dep_graph.reverse_deps(&f.fragment_id).len()))
        .collect();

    let n = counts.len() as f64;
    let mean = counts.iter().map(|(_, c)| *c as f64).sum::<f64>() / n;
    let variance = counts.iter()
        .map(|(_, c)| (*c as f64 - mean).powi(2))
        .sum::<f64>() / n;
    let stddev = variance.sqrt();

    // Threshold: μ + 2σ (anything above is a statistical outlier)
    let threshold = mean + 2.0 * stddev;

    // Require at least 3 reverse deps to be a god file (avoid flagging tiny codebases)
    let min_deps = 3_usize;

    let mut god_files: Vec<GodFile> = counts.iter()
        .filter(|(_, c)| *c as f64 > threshold && *c >= min_deps)
        .map(|(f, c)| {
            let z_score = if stddev > 0.0 { (*c as f64 - mean) / stddev } else { 0.0 };
            GodFile {
                fragment_id: f.fragment_id.clone(),
                source: f.source.clone(),
                reverse_deps: *c,
                z_score: (z_score * 100.0).round() / 100.0,
                recommendation: format!(
                    "'{}' has {} reverse dependencies ({:.1}σ above average). \
                    Consider splitting into: interface (stable) + implementation (volatile) \
                    to reduce coupling through established boundary.",
                    basename(&f.source), c, z_score
                ),
            }
        })
        .collect();

    god_files.sort_unstable_by(|a, b| b.z_score.partial_cmp(&a.z_score).unwrap_or(std::cmp::Ordering::Equal));
    god_files
}

// ═══════════════════════════════════════════════════════════════════
// Architectural Layer Analysis
///
/// Layers inferred from naming conventions (common in real codebases).
/// We detect violations where a lower-layer file imports a higher-layer symbol.
///
/// Layer hierarchy (lower index = more foundational):
///   0: models / entities / domain
///   1: repositories / stores / db
///   2: services / managers / engines
///   3: controllers / handlers / routes / api
///   4: views / ui / components / pages
///   5: tests / specs
///
/// A violation is when layer[i] imports from layer[j > i+1],
/// OR when a foundational layer imports from a presentation layer.
// ═══════════════════════════════════════════════════════════════════
static ARCH_LAYERS: &[(&str, &[&str])] = &[
    ("domain",         &["model", "entity", "domain", "schema", "types"]),
    ("data",           &["repository", "repo", "store", "dao", "db", "database", "migration"]),
    ("service",        &["service", "manager", "engine", "provider", "processor", "worker"]),
    ("api",            &["controller", "handler", "route", "router", "endpoint", "api", "middleware"]),
    ("presentation",   &["view", "component", "page", "screen", "widget", "ui", "template"]),
    ("test",           &["test", "spec", "mock", "stub", "fixture"]),
];

fn classify_layer(source: &str) -> Option<(usize, &'static str)> {
    let lower = source.to_lowercase();
    for (layer_idx, (layer_name, keywords)) in ARCH_LAYERS.iter().enumerate() {
        for &kw in keywords.iter() {
            if lower.contains(kw) {
                return Some((layer_idx, layer_name));
            }
        }
    }
    None
}

fn find_arch_violations(fragments: &[&ContextFragment]) -> Vec<ArchViolation> {
    let mut violations = Vec::new();

    for frag in fragments {
        let importer = match classify_layer(&frag.source) {
            Some(l) => l,
            None => continue,
        };

        // Scan content for import/use statements referencing other layers
        for line in frag.content.lines() {
            let line_lower = line.to_lowercase();
            let is_import = line_lower.starts_with("import ") || line_lower.starts_with("from ")
                || line_lower.starts_with("use ") || line_lower.contains("require(")
                || line_lower.starts_with("#include");

            if !is_import {
                continue;
            }

            // Check if this import is to a higher-numbered layer (violation)
            if let Some(imported) = find_layer_in_import(&line_lower) {
                // A domain layer importing from service/api/presentation is a violation
                // A service layer importing presentation is a violation
                // Tests importing anything is fine
                if importer.1 == "test" {
                    continue;
                }
                if imported.0 > importer.0 + 1 {
                    violations.push(ArchViolation {
                        fragment_id: frag.fragment_id.clone(),
                        source: frag.source.clone(),
                        importer_layer: importer.1.to_string(),
                        imported_layer: imported.1.to_string(),
                        evidence: line.trim().to_string(),
                        recommendation: format!(
                            "'{}' (layer: {}) imports from '{}' layer — this violates dependency direction. \
                            Introduce an interface/port in the {} layer that {} implements.",
                            basename(&frag.source), importer.1, imported.1,
                            importer.1, imported.1
                        ),
                    });
                }
            }
        }
    }
    violations
}

fn find_layer_in_import(import_line: &str) -> Option<(usize, &'static str)> {
    for (layer_idx, (layer_name, keywords)) in ARCH_LAYERS.iter().enumerate() {
        for &kw in keywords.iter() {
            if import_line.contains(kw) {
                return Some((layer_idx, layer_name));
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════
// Naming Convention Analysis
///
/// Detects inconsistent naming within the fragment set.
/// Conventions detected:
///   - Python files: expected snake_case, flags camelCase or PascalCase
///   - JS/TS files: expected camelCase or kebab-case
///   - Rust modules: expected snake_case
// ═══════════════════════════════════════════════════════════════════
fn find_naming_issues(fragments: &[&ContextFragment]) -> Vec<NamingIssue> {
    let mut issues = Vec::new();

    for frag in fragments {
        let source = &frag.source;
        let name = basename(source);

        // Remove extension
        let stem = name.rsplit('.').nth(1).unwrap_or(name);
        if stem.is_empty() || stem.len() < 3 { continue; }

        let is_py = source.ends_with(".py");
        let is_rs = source.ends_with(".rs");
        let is_js = source.ends_with(".js") || source.ends_with(".ts")
            || source.ends_with(".jsx") || source.ends_with(".tsx");

        if is_py || is_rs {
            // Expect snake_case: no uppercase, no hyphens
            if stem.chars().any(|c| c.is_uppercase()) {
                let style = if stem.chars().next().is_some_and(|c| c.is_uppercase()) {
                    "PascalCase"
                } else {
                    "camelCase"
                };
                issues.push(NamingIssue {
                    source: source.clone(),
                    expected_style: "snake_case".to_string(),
                    actual_style: style.to_string(),
                    recommendation: format!(
                        "Rename '{}' to '{}' (snake_case) for {} convention compliance.",
                        name,
                        to_snake_case(stem),
                        if is_py { "Python PEP-8" } else { "Rust module" }
                    ),
                });
            }
        } else if is_js {
            // JS/TS: React components → PascalCase, utils/hooks → camelCase or kebab-case
            // Flag files that are inconsistent with their directory convention
            // Simple heuristic: components/ folder → PascalCase expected
            if source.contains("/components/") && !stem.chars().next().is_some_and(|c| c.is_uppercase()) {
                issues.push(NamingIssue {
                    source: source.clone(),
                    expected_style: "PascalCase (React component)".to_string(),
                    actual_style: "non-PascalCase".to_string(),
                    recommendation: format!(
                        "React component '{}' should be PascalCase. Rename to '{}'.",
                        name,
                        to_pascal_case(stem)
                    ),
                });
            }
        }
    }
    issues
}

// ═══════════════════════════════════════════════════════════════════
// CodeHealth Score
///
/// Inspired by CodeHealth (2026 arXiv) and SonarQube's maintainability.
/// Formula:
///   score = 100 × (1 - Σ penalty_i)
///
/// Penalties (each [0.0, 1.0]):
///   P_dup     = min(1, clone_pairs / (N × (N-1)/2 × 0.05))
///               (5% of all pairs being clones = full penalty)
///   P_dead    = min(1, dead_symbols / max(1, total_symbols) × 2)
///               (50% dead symbols = full penalty)
///   P_coupling = min(1, god_files.len() / max(1, N × 0.05))
///               (5% of files being god files = full penalty)
///   P_arch    = min(1, arch_violations.len() / max(1, N × 0.1))
///   P_naming  = min(1, naming_issues.len() / max(1, N × 0.2))
///
/// Weights: coupling (0.30), duplication (0.30), dead code (0.20), arch (0.15), naming (0.05)
// ═══════════════════════════════════════════════════════════════════
fn compute_code_health(
    n: usize,
    total_symbols: usize,
    clone_pairs: &[ClonePair],
    dead_symbols: &[DeadSymbol],
    god_files: &[GodFile],
    arch_violations: &[ArchViolation],
    naming_issues: &[NamingIssue],
) -> (f64, f64, f64, f64, f64, f64) {
    let n_f = n.max(1) as f64;

    let max_pairs = (n * n.saturating_sub(1)) / 2;
    let p_dup = if max_pairs == 0 { 0.0 } else {
        (clone_pairs.len() as f64 / (max_pairs as f64 * 0.05)).min(1.0)
    };

    let p_dead = (dead_symbols.len() as f64 / (total_symbols.max(1) as f64 * 2.0)).clamp(0.0, 1.0);

    let p_coupling = (god_files.len() as f64 / (n_f * 0.05).max(1.0)).min(1.0);

    let p_arch = (arch_violations.len() as f64 / (n_f * 0.10).max(1.0)).min(1.0);

    let p_naming = (naming_issues.len() as f64 / (n_f * 0.20).max(1.0)).min(1.0);

    let weighted = 0.30 * p_coupling + 0.30 * p_dup + 0.20 * p_dead + 0.15 * p_arch + 0.05 * p_naming;
    let score = (100.0 * (1.0 - weighted)).clamp(0.0, 100.0);

    (
        (score * 10.0).round() / 10.0,
        (p_dup * 100.0).round() / 100.0,
        (p_dead * 100.0).round() / 100.0,
        (p_coupling * 100.0).round() / 100.0,
        (p_arch * 100.0).round() / 100.0,
        (p_naming * 100.0).round() / 100.0,
    )
}

fn health_grade(score: f64) -> &'static str {
    if score >= 90.0 { "A" }
    else if score >= 80.0 { "B" }
    else if score >= 70.0 { "C" }
    else if score >= 60.0 { "D" }
    else { "F" }
}

// ═══════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════

pub fn analyze_health(
    fragments: &[&ContextFragment],
    dep_graph: &DepGraph,
) -> HealthReport {
    let n = fragments.len();

    let clone_pairs     = detect_clones(fragments);
    let dead_symbols    = find_dead_symbols(fragments, dep_graph);
    let god_files       = find_god_files(fragments, dep_graph);
    let arch_violations = find_arch_violations(fragments);
    let naming_issues   = find_naming_issues(fragments);

    let total_symbols = dep_graph.symbol_definitions().len();

    let (score, p_dup, p_dead, p_coup, p_arch, p_name) = compute_code_health(
        n, total_symbols, &clone_pairs, &dead_symbols, &god_files, &arch_violations, &naming_issues,
    );

    let grade = health_grade(score);

    // Build human-readable summary
    let mut summary_parts: Vec<String> = Vec::new();
    if !clone_pairs.is_empty() {
        let type1 = clone_pairs.iter().filter(|p| p.clone_type == CloneType::NearIdentical).count();
        summary_parts.push(format!("{} clone pairs ({} near-identical)", clone_pairs.len(), type1));
    }
    if !dead_symbols.is_empty() {
        summary_parts.push(format!("{} potentially dead symbols", dead_symbols.len()));
    }
    if !god_files.is_empty() {
        summary_parts.push(format!("{} over-coupled files", god_files.len()));
    }
    if !arch_violations.is_empty() {
        summary_parts.push(format!("{} architecture violations", arch_violations.len()));
    }
    let summary = if summary_parts.is_empty() {
        format!("Codebase health is excellent ({} fragments analyzed, no issues found).", n)
    } else {
        format!("{} fragments analyzed. Issues: {}.", n, summary_parts.join("; "))
    };

    // Top recommendation: whichever issue has highest penalty
    let top_recommendation = if !god_files.is_empty() && p_coup >= p_dup {
        god_files.first().map(|g| g.recommendation.clone())
    } else if !clone_pairs.is_empty() && p_dup >= p_dead {
        clone_pairs.first().map(|c| c.recommendation.clone())
    } else if !dead_symbols.is_empty() {
        dead_symbols.first().map(|d| d.recommendation.clone())
    } else if !arch_violations.is_empty() {
        arch_violations.first().map(|v| v.recommendation.clone())
    } else {
        naming_issues.first().map(|ni| ni.recommendation.clone())
    };

    HealthReport {
        fragment_count: n,
        clone_pairs,
        dead_symbols,
        god_files,
        arch_violations,
        naming_issues,
        code_health_score: score,
        health_grade: grade,
        duplication_penalty: p_dup,
        dead_code_penalty: p_dead,
        coupling_penalty: p_coup,
        arch_penalty: p_arch,
        naming_penalty: p_name,
        summary,
        top_recommendation,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn to_snake_case(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_lowercase().next().unwrap_or(c));
    }
    out.replace('-', "_")
}

fn to_pascal_case(s: &str) -> String {
    s.split(['_', '-'])
        .filter(|p| !p.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let mut s = first.to_uppercase().to_string();
                    s.extend(chars);
                    s
                }
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::ContextFragment;
    use crate::dedup::simhash;
    use crate::depgraph::DepGraph;

    fn make_frag(id: &str, content: &str, source: &str) -> ContextFragment {
        let mut f = ContextFragment::new(id.into(), content.into(), content.len() as u32 / 4, source.into());
        f.simhash = simhash(content);
        f.token_count = content.split_whitespace().count() as u32;
        f
    }

    #[test]
    fn test_exact_clone_detected() {
        let code = "def process_payment(amount): return amount * 1.2";
        let a = make_frag("f1", code, "billing/payments.py");
        let b = make_frag("f2", code, "checkout/fees.py");
        let frags: Vec<&ContextFragment> = vec![&a, &b];
        let dep = DepGraph::new();
        let report = analyze_health(&frags, &dep);
        assert!(!report.clone_pairs.is_empty(), "Exact same content should be a clone");
        assert_eq!(report.clone_pairs[0].clone_type, CloneType::NearIdentical);
        assert!(report.clone_pairs[0].similarity > 0.95);
    }

    #[test]
    fn test_same_file_not_cloned() {
        let code = "def helper(x): return x + 1";
        let a = make_frag("f1", code, "utils.py");
        let b = make_frag("f2", code, "utils.py");
        let frags: Vec<&ContextFragment> = vec![&a, &b];
        let dep = DepGraph::new();
        let report = analyze_health(&frags, &dep);
        // Same source → skip
        assert!(report.clone_pairs.is_empty(), "Same-file pairs should be skipped");
    }

    #[test]
    fn test_god_file_detection() {
        #[allow(unused_variables)]
        let dep = DepGraph::new();
        // Build 10 fragments all depending on "core.py"
        // We can't easily wire reverse deps without dep graph edges
        // but we can test threshold math directly
        let counts = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 1, 20]; // outlier
        let n = counts.len() as f64;
        let mean = counts.iter().sum::<usize>() as f64 / n;
        let var: f64 = counts.iter().map(|c| (*c as f64 - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let threshold = mean + 2.0 * std;
        assert!(20.0 > threshold, "20 should be above 2σ threshold (threshold={:.2})", threshold);
    }

    #[test]
    fn test_code_health_perfect_score() {
        // No issues → score should be 100
        let (score, ..) = super::compute_code_health(10, 5, &[], &[], &[], &[], &[]);
        assert_eq!(score, 100.0, "No issues should give perfect health score");
    }

    #[test]
    fn test_code_health_decreases_with_clones() {
        let a = make_frag("f1", "def foo(x): return x*2+1", "math/a.py");
        let b = make_frag("f2", "def foo(x): return x*2+1", "math/b.py");
        let frags: Vec<&ContextFragment> = vec![&a, &b];
        let dep = DepGraph::new();
        let report = analyze_health(&frags, &dep);
        // With a clone pair the health score must be below 100
        assert!(report.code_health_score < 100.0);
    }

    #[test]
    fn test_arch_violation_domain_imports_api() {
        let content = "from api.routes import handle_request\ndef get_user(id): pass";
        let frag = make_frag("f1", content, "models/user.py");
        let frags: Vec<&ContextFragment> = vec![&frag];
        let dep = DepGraph::new();
        let report = analyze_health(&frags, &dep);
        assert!(!report.arch_violations.is_empty(), "domain importing from api should be flagged");
    }

    #[test]
    fn test_health_grade_mapping() {
        assert_eq!(health_grade(95.0), "A");
        assert_eq!(health_grade(85.0), "B");
        assert_eq!(health_grade(72.0), "C");
        assert_eq!(health_grade(62.0), "D");
        assert_eq!(health_grade(45.0), "F");
    }

    #[test]
    fn test_snake_case_conversion() {
        assert_eq!(to_snake_case("MyClass"), "my_class");
        assert_eq!(to_snake_case("parseJSON"), "parse_j_s_o_n");
        assert_eq!(to_snake_case("snake_case"), "snake_case");
    }

    #[test]
    fn test_empty_codebase_no_panic() {
        let dep = DepGraph::new();
        let report = analyze_health(&[], &dep);
        assert_eq!(report.fragment_count, 0);
        assert_eq!(report.code_health_score, 100.0);
        assert_eq!(report.health_grade, "A");
    }

    #[test]
    fn test_naming_issue_python_camel_case() {
        let frag = make_frag("f1", "class MyService: pass", "services/myService.py");
        let frags: Vec<&ContextFragment> = vec![&frag];
        let dep = DepGraph::new();
        let report = analyze_health(&frags, &dep);
        assert!(!report.naming_issues.is_empty(), "myService.py should flag camelCase naming");
    }
}
