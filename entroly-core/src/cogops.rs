//! CogOps — Epistemic Engine (Rust Core)
//!
//! All CogOps logic lives here in Rust. Python is a thin MCP wrapper.
//!
//! Sub-engines:
//!   1. BeliefCompiler:      Truth → Belief (entity extraction, vault writing)
//!   2. VerificationEngine:  Contradiction detection, staleness, blast radius
//!   3. ChangePipeline:      Diff analysis, PR briefs, code review
//!   4. FlowOrchestrator:    Chains the 5 canonical flows
//!   5. SkillEngine:         Skill synthesis, benchmarking, promotion
//!   6. EpistemicRouter:     Intent classification, routing matrix

#![allow(
    dead_code,
    unused_assignments,
    unused_variables,
    clippy::manual_strip,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicIntent {
    Architecture,
    PrBrief,
    CodeGeneration,
    Report,
    Research,
    Incident,
    Audit,
    Reuse,
    Onboarding,
    TestGap,
    Release,
    Repair,
    General,
}

impl EpistemicIntent {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Architecture => "architecture",
            Self::PrBrief => "pr_brief",
            Self::CodeGeneration => "code_generation",
            Self::Report => "report",
            Self::Research => "research",
            Self::Incident => "incident",
            Self::Audit => "audit",
            Self::Reuse => "reuse",
            Self::Onboarding => "onboarding",
            Self::TestGap => "test_gap",
            Self::Release => "release",
            Self::Repair => "repair",
            Self::General => "general",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicFlow {
    FastAnswer,
    VerifyBefore,
    CompileOnDemand,
    ChangeDriven,
    SelfImprovement,
}

impl EpistemicFlow {
    fn as_str(&self) -> &'static str {
        match self {
            Self::FastAnswer => "fast_answer",
            Self::VerifyBefore => "verify_before_answer",
            Self::CompileOnDemand => "compile_on_demand",
            Self::ChangeDriven => "change_driven",
            Self::SelfImprovement => "self_improvement",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl RiskLevel {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Data Structures
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntity {
    pub name: String,
    pub kind: String, // function, class, struct, trait, enum, const
    pub file_path: String,
    pub line: usize,
    pub docstring: String,
    pub signature: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefArtifact {
    pub claim_id: String,
    pub entity: String,
    pub title: String,
    pub status: String,
    pub confidence: f64,
    pub sources: Vec<String>,
    pub derived_from: Vec<String>,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub flow: String,
    pub intent: String,
    pub risk: String,
    pub reasoning: String,
    pub belief_exists: bool,
    pub belief_fresh: bool,
    pub belief_verified: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub entity: String,
    pub conflict_type: String,
    pub description: String,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSet {
    pub files_added: Vec<String>,
    pub files_modified: Vec<String>,
    pub files_deleted: Vec<String>,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub intent: String,
    pub functions_changed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewFinding {
    pub severity: String,
    pub category: String,
    pub file: String,
    pub line: usize,
    pub message: String,
    pub suggestion: String,
}

// ═══════════════════════════════════════════════════════════════════
// Intent Classification (Rust — zero-allocation keyword scan)
// ═══════════════════════════════════════════════════════════════════

/// Keyword groups for intent classification, ordered by priority.
static INTENT_PATTERNS: &[(&[&str], EpistemicIntent)] = &[
    (
        &[
            "security",
            "vulnerability",
            "cve",
            "pii",
            "compliance",
            "audit",
            "gdpr",
            "hipaa",
            "sox",
            "penetration",
        ],
        EpistemicIntent::Audit,
    ),
    (
        &[
            "incident", "outage", "downtime", "crash", "spike", "latency", "alert", "pager",
        ],
        EpistemicIntent::Incident,
    ),
    (
        &[
            "fix",
            "repair",
            "broken",
            "failing",
            "error",
            "bug",
            "root cause",
            "regression",
            "debug",
        ],
        EpistemicIntent::Repair,
    ),
    (
        &[
            "test",
            "coverage",
            "uncovered",
            "missing test",
            "test gap",
            "untested",
        ],
        EpistemicIntent::TestGap,
    ),
    (
        &[
            "release",
            "deploy",
            "rollout",
            "ready to ship",
            "changelog",
            "version",
        ],
        EpistemicIntent::Release,
    ),
    (
        &["pr", "pull request", "diff", "review", "merge", "commit"],
        EpistemicIntent::PrBrief,
    ),
    (
        &[
            "generate",
            "write code",
            "implement",
            "scaffold",
            "create function",
            "migration",
        ],
        EpistemicIntent::CodeGeneration,
    ),
    (
        &[
            "report",
            "slide",
            "presentation",
            "diagram",
            "chart",
            "visuali",
        ],
        EpistemicIntent::Report,
    ),
    (
        &[
            "research",
            "benchmark",
            "compare",
            "evaluate",
            "study",
            "analyze options",
        ],
        EpistemicIntent::Research,
    ),
    (
        &[
            "architecture",
            "design",
            "how does",
            "system",
            "module",
            "component",
            "dependency",
            "flow",
            "pipeline",
        ],
        EpistemicIntent::Architecture,
    ),
    (
        &[
            "reuse",
            "existing",
            "already have",
            "duplicate",
            "shared",
            "utility",
            "helper",
        ],
        EpistemicIntent::Reuse,
    ),
    (
        &[
            "onboard",
            "explain",
            "new engineer",
            "getting started",
            "walkthrough",
            "tutorial",
        ],
        EpistemicIntent::Onboarding,
    ),
];

pub fn classify_intent(query: &str) -> EpistemicIntent {
    let lower = query.to_lowercase();
    for (keywords, intent) in INTENT_PATTERNS {
        for kw in *keywords {
            if lower.contains(kw) {
                return *intent;
            }
        }
    }
    EpistemicIntent::General
}

/// Assess risk level from query content.
pub fn assess_risk(query: &str) -> RiskLevel {
    let lower = query.to_lowercase();
    let high_risk = [
        "security",
        "vulnerability",
        "pii",
        "compliance",
        "audit",
        "gdpr",
        "hipaa",
        "credential",
        "secret",
        "encrypt",
        "auth",
    ];
    let medium_risk = [
        "deploy",
        "migration",
        "database",
        "schema",
        "release",
        "production",
        "breaking",
        "api change",
    ];
    for kw in &high_risk {
        if lower.contains(kw) {
            return RiskLevel::High;
        }
    }
    for kw in &medium_risk {
        if lower.contains(kw) {
            return RiskLevel::Medium;
        }
    }
    RiskLevel::Low
}

// ═══════════════════════════════════════════════════════════════════
// Entity Extraction (Rust — pattern-based, no regex)
// ═══════════════════════════════════════════════════════════════════

pub fn extract_entities(content: &str, file_path: &str) -> Vec<CodeEntity> {
    let ext = file_path.rsplit('.').next().unwrap_or("");
    match ext {
        "py" | "pyw" => extract_python_entities(content, file_path),
        "rs" => extract_rust_entities(content, file_path),
        "js" | "jsx" | "ts" | "tsx" => extract_js_entities(content, file_path),
        _ => Vec::new(),
    }
}

fn extract_python_entities(content: &str, file_path: &str) -> Vec<CodeEntity> {
    let mut entities = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Classes
        if trimmed.starts_with("class ") && trimmed.contains(':') {
            let name = trimmed
                .strip_prefix("class ")
                .unwrap_or("")
                .split(&['(', ':'][..])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                let doc = get_next_docstring(&lines, i + 1);
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "class".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: doc,
                    signature: trimmed.to_string(),
                    dependencies: Vec::new(),
                });
            }
        }
        // Functions
        if (trimmed.starts_with("def ") || trimmed.starts_with("async def "))
            && trimmed.contains(':')
        {
            let after = if trimmed.starts_with("async def ") {
                &trimmed[10..]
            } else {
                &trimmed[4..]
            };
            let name = after.split('(').next().unwrap_or("").trim();
            if !name.is_empty() && !name.starts_with('_') || name == "__init__" {
                let doc = get_next_docstring(&lines, i + 1);
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "function".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: doc,
                    signature: trimmed.trim_end_matches(':').to_string(),
                    dependencies: Vec::new(),
                });
            }
        }
    }
    entities
}

fn extract_rust_entities(content: &str, file_path: &str) -> Vec<CodeEntity> {
    let mut entities = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Structs
        if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
            let after = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            let name = after
                .strip_prefix("struct ")
                .unwrap_or("")
                .split(&[' ', '<', '{', '(', ';'][..])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                let doc = get_rust_doc(&lines, i);
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "struct".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: doc,
                    signature: format!("pub struct {}", name),
                    dependencies: Vec::new(),
                });
            }
        }
        // Enums
        if trimmed.starts_with("pub enum ")
            || (trimmed.starts_with("enum ") && !trimmed.contains("enumeration"))
        {
            let after = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            let name = after
                .strip_prefix("enum ")
                .unwrap_or("")
                .split(&[' ', '<', '{'][..])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "enum".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: get_rust_doc(&lines, i),
                    signature: format!("pub enum {}", name),
                    dependencies: Vec::new(),
                });
            }
        }
        // Traits
        if trimmed.starts_with("pub trait ") || trimmed.starts_with("trait ") {
            let after = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            let name = after
                .strip_prefix("trait ")
                .unwrap_or("")
                .split(&[' ', '<', '{', ':'][..])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "trait".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: get_rust_doc(&lines, i),
                    signature: format!("pub trait {}", name),
                    dependencies: Vec::new(),
                });
            }
        }
        // Functions
        if (trimmed.starts_with("pub fn ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub async fn ")
            || trimmed.starts_with("async fn ")
            || trimmed.starts_with("pub(crate) fn "))
            && trimmed.contains('(')
        {
            let fn_start = trimmed.find("fn ").unwrap_or(0) + 3;
            let name = trimmed[fn_start..]
                .split(&['(', '<'][..])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() && !name.starts_with('_') {
                let sig_end = trimmed.find('{').unwrap_or(trimmed.len());
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "function".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: get_rust_doc(&lines, i),
                    signature: trimmed[..sig_end].trim().to_string(),
                    dependencies: Vec::new(),
                });
            }
        }
    }
    entities
}

fn extract_js_entities(content: &str, file_path: &str) -> Vec<CodeEntity> {
    let mut entities = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.contains("class ")
            && (trimmed.starts_with("class ") || trimmed.starts_with("export class "))
        {
            let after = if let Some(a) = trimmed.strip_prefix("export ") {
                a
            } else {
                trimmed
            };
            let name = after
                .strip_prefix("class ")
                .unwrap_or("")
                .split(&[' ', '{', '<'][..])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "class".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: String::new(),
                    signature: trimmed.to_string(),
                    dependencies: Vec::new(),
                });
            }
        }
        if trimmed.contains("function ")
            && (trimmed.starts_with("function ")
                || trimmed.starts_with("export function ")
                || trimmed.starts_with("async function ")
                || trimmed.starts_with("export async function "))
        {
            let fn_pos = trimmed.find("function ").unwrap_or(0) + 9;
            let name = trimmed[fn_pos..].split('(').next().unwrap_or("").trim();
            if !name.is_empty() {
                entities.push(CodeEntity {
                    name: name.to_string(),
                    kind: "function".into(),
                    file_path: file_path.into(),
                    line: i + 1,
                    docstring: String::new(),
                    signature: trimmed.to_string(),
                    dependencies: Vec::new(),
                });
            }
        }
    }
    entities
}

fn get_next_docstring(lines: &[&str], after: usize) -> String {
    if after >= lines.len() {
        return String::new();
    }
    let trimmed = lines[after].trim();
    if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
        let quote = &trimmed[..3];
        if trimmed.len() > 6 && trimmed[3..].contains(quote) {
            return trimmed[3..trimmed.len() - 3].trim().to_string();
        }
        let mut doc = trimmed[3..].to_string();
        for j in (after + 1)..lines.len().min(after + 10) {
            if lines[j].contains(quote) {
                break;
            }
            doc.push(' ');
            doc.push_str(lines[j].trim());
        }
        return doc.chars().take(200).collect();
    }
    String::new()
}

fn get_rust_doc(lines: &[&str], before: usize) -> String {
    let mut docs = Vec::new();
    let start = before.saturating_sub(10);
    for i in (start..before).rev() {
        let trimmed = lines[i].trim();
        if trimmed.starts_with("///") {
            docs.push(trimmed[3..].trim());
        } else if trimmed.starts_with("#[") || trimmed.is_empty() {
            continue;
        } else {
            break;
        }
    }
    docs.reverse();
    docs.join(" ").chars().take(200).collect()
}

// ═══════════════════════════════════════════════════════════════════
// Vault I/O (Rust — filesystem operations)
// ═══════════════════════════════════════════════════════════════════

fn ensure_vault(vault_path: &Path) {
    for dir in &[
        "beliefs",
        "verification",
        "actions",
        "evolution/skills",
        "media",
    ] {
        let _ = fs::create_dir_all(vault_path.join(dir));
    }
    let registry = vault_path.join("evolution/registry.md");
    if !registry.exists() {
        let _ = fs::write(
            &registry,
            "# Skill Registry\n\n| ID | Status | Created | Description |\n|---|---|---|---|\n",
        );
    }
}

fn generate_claim_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:016x}{:08x}", ts.as_nanos(), ts.subsec_nanos())
}

fn write_belief_artifact(vault_path: &Path, artifact: &BeliefArtifact) -> Result<String, String> {
    ensure_vault(vault_path);
    let now = chrono_iso();
    let safe_entity = artifact.entity.replace("::", "_").replace('/', "_");
    let filename = format!(
        "{}_{}.md",
        safe_entity,
        &artifact.claim_id[..8.min(artifact.claim_id.len())]
    );
    let path = vault_path.join("beliefs").join(&filename);

    let sources_yaml: String = artifact
        .sources
        .iter()
        .map(|s| format!("  - {}", s))
        .collect::<Vec<_>>()
        .join("\n");
    let derived_yaml: String = artifact
        .derived_from
        .iter()
        .map(|s| format!("  - {}", s))
        .collect::<Vec<_>>()
        .join("\n");

    let content = format!(
        "---\nclaim_id: {}\nentity: {}\nstatus: {}\nconfidence: {:.2}\nsources:\n{}\nlast_checked: {}\nderived_from:\n{}\n---\n\n# {}\n\n{}",
        artifact.claim_id, artifact.entity, artifact.status, artifact.confidence,
        sources_yaml, now, derived_yaml, artifact.title, artifact.body
    );
    fs::write(&path, &content).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

fn read_all_beliefs(vault_path: &Path) -> Vec<BeliefArtifact> {
    let beliefs_dir = vault_path.join("beliefs");
    let mut results = Vec::new();
    let mut files = Vec::new();
    collect_markdown_files(&beliefs_dir, &mut files);
    for path in files {
        if let Ok(content) = fs::read_to_string(path) {
            if let Some(artifact) = parse_belief_frontmatter(&content) {
                results.push(artifact);
            }
        }
    }
    results
}

fn parse_belief_frontmatter(content: &str) -> Option<BeliefArtifact> {
    let parts: Vec<&str> = content.splitn(3, "---").collect();
    if parts.len() < 3 {
        return None;
    }
    let fm = parts[1];
    let body = parts[2].trim().to_string();
    let mut claim_id = String::new();
    let mut entity = String::new();
    let mut status = String::from("inferred");
    let mut confidence: f64 = 0.5;
    let mut sources = Vec::new();
    let mut derived_from = Vec::new();
    let mut in_sources = false;
    let mut in_derived = false;

    for line in fm.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("claim_id:") {
            claim_id = trimmed[9..].trim().to_string();
            in_sources = false;
            in_derived = false;
        } else if trimmed.starts_with("entity:") {
            entity = trimmed[7..].trim().to_string();
            in_sources = false;
            in_derived = false;
        } else if trimmed.starts_with("status:") {
            status = trimmed[7..].trim().to_string();
            in_sources = false;
            in_derived = false;
        } else if trimmed.starts_with("confidence:") {
            confidence = trimmed[11..].trim().parse().unwrap_or(0.5);
            in_sources = false;
            in_derived = false;
        } else if trimmed.starts_with("sources:") {
            in_sources = true;
            in_derived = false;
        } else if trimmed.starts_with("derived_from:") {
            in_derived = true;
            in_sources = false;
        } else if trimmed.starts_with("- ") {
            let val = trimmed[2..].trim().to_string();
            if in_sources {
                sources.push(val);
            } else if in_derived {
                derived_from.push(val);
            }
        } else if !trimmed.is_empty() && !trimmed.starts_with("last_checked:") {
            in_sources = false;
            in_derived = false;
        }
    }

    if claim_id.is_empty() {
        return None;
    }
    let title = body
        .lines()
        .find(|l| l.starts_with("# "))
        .map(|l| l[2..].trim().to_string())
        .unwrap_or_default();

    Some(BeliefArtifact {
        claim_id,
        entity,
        title,
        status,
        confidence,
        sources,
        derived_from,
        body,
    })
}

fn collect_markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_markdown_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
                out.push(path);
            }
        }
    }
}

fn normalize_rel_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

fn source_pointer_path(source: &str) -> String {
    normalize_rel_path(source.split(':').next().unwrap_or(source))
}

fn belief_matches_changed_file(belief: &BeliefArtifact, changed_file: &str) -> bool {
    let changed_norm = normalize_rel_path(changed_file);
    let changed_stem = Path::new(&changed_norm)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if !changed_stem.is_empty() && belief.entity.to_lowercase().contains(&changed_stem) {
        return true;
    }

    for source in &belief.sources {
        let source_norm = source_pointer_path(source);
        if source_norm == changed_norm
            || source_norm.ends_with(&changed_norm)
            || changed_norm.ends_with(&source_norm)
        {
            return true;
        }

        if !changed_stem.is_empty() {
            let source_stem = Path::new(&source_norm)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if source_stem == changed_stem || source_norm.to_lowercase().contains(&changed_stem) {
                return true;
            }
        }
    }

    false
}

fn mark_belief_stale_content(content: &str) -> Option<String> {
    let mut inside_frontmatter = false;
    let mut fence_count = 0usize;
    let mut changed = false;
    let mut lines = Vec::new();

    for line in content.lines() {
        if line.trim() == "---" {
            fence_count += 1;
            inside_frontmatter = fence_count == 1;
            lines.push(line.to_string());
            continue;
        }

        if inside_frontmatter && line.trim_start().starts_with("status:") {
            let current = line.trim_start()["status:".len()..].trim();
            if current != "stale" {
                lines.push("status: stale".to_string());
                changed = true;
            } else {
                lines.push(line.to_string());
            }
            continue;
        }

        lines.push(line.to_string());
    }

    if changed {
        Some(lines.join("\n"))
    } else {
        None
    }
}

fn mark_beliefs_stale(vault_path: &Path, changed_files: &[String]) -> Vec<String> {
    let beliefs_dir = vault_path.join("beliefs");
    let mut belief_files = Vec::new();
    let mut refreshed = Vec::new();
    collect_markdown_files(&beliefs_dir, &mut belief_files);

    for path in belief_files {
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let Some(artifact) = parse_belief_frontmatter(&content) else {
            continue;
        };
        if !changed_files
            .iter()
            .any(|cf| belief_matches_changed_file(&artifact, cf))
        {
            continue;
        }

        if let Some(updated) = mark_belief_stale_content(&content) {
            if fs::write(&path, updated).is_ok() {
                refreshed.push(
                    path.strip_prefix(vault_path)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string(),
                );
            }
        }
    }

    refreshed
}

fn build_belief_artifact(
    root: &Path,
    fpath: &Path,
    content: &str,
) -> Option<(BeliefArtifact, usize)> {
    let rel = normalize_rel_path(&fpath.strip_prefix(root).unwrap_or(fpath).to_string_lossy());
    let entities = extract_entities(content, &rel);
    if entities.is_empty() {
        return None;
    }

    let module_name = fpath
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let lang = fpath.extension().and_then(|e| e.to_str()).unwrap_or("");
    let loc = content.lines().count();
    let mut body = format!("**Language:** {}\n**Lines of code:** {}\n\n", lang, loc);

    let classes: Vec<&CodeEntity> = entities
        .iter()
        .filter(|e| matches!(e.kind.as_str(), "class" | "struct" | "enum" | "trait"))
        .collect();
    let funcs: Vec<&CodeEntity> = entities.iter().filter(|e| e.kind == "function").collect();

    if !classes.is_empty() {
        body.push_str("## Types\n");
        for c in &classes {
            let doc = if c.docstring.is_empty() {
                String::new()
            } else {
                format!(" - {}", c.docstring)
            };
            body.push_str(&format!("- `{}`{}\n", c.signature, doc));
        }
    }

    if !funcs.is_empty() {
        body.push_str("\n## Functions\n");
        for f in &funcs {
            let doc = if f.docstring.is_empty() {
                String::new()
            } else {
                format!(" - {}", f.docstring)
            };
            body.push_str(&format!("- `{}`{}\n", f.signature, doc));
        }
    }

    let sources: Vec<String> = entities
        .iter()
        .take(10)
        .map(|e| format!("{}:{}", rel, e.line))
        .collect();

    let artifact = BeliefArtifact {
        claim_id: generate_claim_id(),
        entity: module_name.to_string(),
        title: format!("Module: {}", module_name),
        status: "inferred".into(),
        confidence: 0.75,
        sources,
        derived_from: vec!["cogops_compiler".into(), "sast".into()],
        body,
    };

    Some((artifact, entities.len()))
}

fn compile_source_paths(
    vault_path: &Path,
    root: &Path,
    source_paths: &[PathBuf],
) -> (u32, u32, u32, Vec<String>) {
    let mut seen = HashSet::new();
    let mut files_processed = 0u32;
    let mut beliefs_written = 0u32;
    let mut entities_extracted = 0u32;
    let mut errors = Vec::new();

    for fpath in source_paths {
        let path_key = normalize_rel_path(&fpath.to_string_lossy());
        if !seen.insert(path_key) || !fpath.is_file() {
            continue;
        }

        files_processed += 1;
        match fs::read_to_string(fpath) {
            Ok(content) => {
                if let Some((artifact, entity_count)) = build_belief_artifact(root, fpath, &content)
                {
                    entities_extracted += entity_count as u32;
                    if let Err(err) = write_belief_artifact(vault_path, &artifact) {
                        let rel = normalize_rel_path(
                            &fpath.strip_prefix(root).unwrap_or(fpath).to_string_lossy(),
                        );
                        errors.push(format!("{}: {}", rel, err));
                    } else {
                        beliefs_written += 1;
                    }
                }
            }
            Err(err) => {
                let rel = normalize_rel_path(
                    &fpath.strip_prefix(root).unwrap_or(fpath).to_string_lossy(),
                );
                errors.push(format!("{}: {}", rel, err));
            }
        }
    }

    (files_processed, beliefs_written, entities_extracted, errors)
}

// ═══════════════════════════════════════════════════════════════════
// Verification Logic (Rust)
// ═══════════════════════════════════════════════════════════════════

pub fn detect_contradictions(beliefs: &[BeliefArtifact]) -> Vec<Contradiction> {
    let mut contras = Vec::new();
    let mut by_entity: HashMap<&str, Vec<&BeliefArtifact>> = HashMap::new();
    for b in beliefs {
        by_entity.entry(b.entity.as_str()).or_default().push(b);
    }
    for (entity, group) in &by_entity {
        if group.len() < 2 {
            continue;
        }
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = group[i];
                let b = group[j];
                if (a.status == "verified" && b.status == "stale")
                    || (a.status == "stale" && b.status == "verified")
                {
                    contras.push(Contradiction {
                        entity: entity.to_string(),
                        conflict_type: "stale_vs_fresh".into(),
                        description: "Same entity has both verified and stale beliefs".into(),
                        severity: "medium".into(),
                    });
                }
                if (a.confidence - b.confidence).abs() > 0.4 {
                    contras.push(Contradiction {
                        entity: entity.to_string(),
                        conflict_type: "confidence_divergence".into(),
                        description: format!(
                            "Confidence diverges: {:.2} vs {:.2}",
                            a.confidence, b.confidence
                        ),
                        severity: "low".into(),
                    });
                }
            }
        }
    }
    contras
}

pub fn compute_blast_radius(
    beliefs: &[BeliefArtifact],
    changed_files: &[&str],
) -> (Vec<String>, Vec<String>, &'static str) {
    let mut affected_beliefs = Vec::new();
    let mut affected_entities = Vec::new();
    for b in beliefs {
        for cf in changed_files {
            let stem = Path::new(cf)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            let entity_lower = b.entity.to_lowercase();
            let source_match = b.sources.iter().any(|s| s.to_lowercase().contains(&stem));
            if source_match || entity_lower.contains(&stem) || b.body.to_lowercase().contains(&stem)
            {
                if !affected_beliefs.contains(&b.claim_id) {
                    affected_beliefs.push(b.claim_id.clone());
                }
                if !affected_entities.contains(&b.entity) {
                    affected_entities.push(b.entity.clone());
                }
                break;
            }
        }
    }
    let n = affected_beliefs.len();
    let risk = if n <= 2 {
        "low"
    } else if n <= 5 {
        "medium"
    } else {
        "high"
    };
    (affected_beliefs, affected_entities, risk)
}

// ═══════════════════════════════════════════════════════════════════
// Diff Analysis (Rust — zero-copy parsing)
// ═══════════════════════════════════════════════════════════════════

pub fn parse_diff(diff_text: &str, commit_msg: &str) -> ChangeSet {
    let mut cs = ChangeSet {
        files_added: Vec::new(),
        files_modified: Vec::new(),
        files_deleted: Vec::new(),
        lines_added: 0,
        lines_removed: 0,
        intent: String::new(),
        functions_changed: Vec::new(),
    };
    for line in diff_text.lines() {
        if let Some(f) = line.strip_prefix("+++ b/") {
            if f != "/dev/null" && !cs.files_modified.contains(&f.to_string()) {
                cs.files_modified.push(f.to_string());
            }
        } else if let Some(f) = line.strip_prefix("--- a/") {
            if diff_text.contains(&"+++ /dev/null".to_string()) {
                cs.files_deleted.push(f.to_string());
            }
        } else if line.starts_with('+') && !line.starts_with("+++") {
            cs.lines_added += 1;
        } else if line.starts_with('-') && !line.starts_with("---") {
            cs.lines_removed += 1;
        } else if line.starts_with("@@") {
            // Extract function names from hunk headers
            if let Some(fn_start) = line
                .find("fn ")
                .or(line.find("def "))
                .or(line.find("function "))
            {
                let rest = &line[fn_start..];
                let kw_len = if rest.starts_with("function ") {
                    9
                } else if rest.starts_with("def ") {
                    4
                } else {
                    3
                };
                let name: String = rest[kw_len..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && !cs.functions_changed.contains(&name) {
                    cs.functions_changed.push(name);
                }
            }
        }
    }
    // Classify intent
    let text = format!("{} {}", commit_msg, &diff_text[..diff_text.len().min(2000)]).to_lowercase();
    cs.intent = if text.contains("fix") || text.contains("bug") || text.contains("patch") {
        "bugfix".into()
    } else if text.contains("security") || text.contains("cve") {
        "security".into()
    } else if text.contains("test") || text.contains("spec") {
        "test".into()
    } else if text.contains("refactor") || text.contains("clean") {
        "refactor".into()
    } else if text.contains("perf") || text.contains("optim") {
        "performance".into()
    } else if text.contains("doc") || text.contains("readme") {
        "docs".into()
    } else {
        "feature".into()
    };
    cs
}

/// Review a diff for common issues (hardcoded secrets, TODOs, unsafe patterns).
pub fn review_diff(diff_text: &str) -> Vec<ReviewFinding> {
    let mut findings = Vec::new();
    let mut current_file = String::new();
    let mut current_line: usize = 0;

    let patterns: &[(&str, &str, &str, &str)] = &[
        ("password", "error", "safety", "Possible hardcoded secret"),
        ("api_key", "error", "safety", "Possible hardcoded API key"),
        ("TODO", "warning", "maintenance", "Contains TODO marker"),
        ("FIXME", "warning", "maintenance", "Contains FIXME marker"),
        ("HACK", "warning", "maintenance", "Contains HACK marker"),
        (".unwrap()", "warning", "safety", "Rust .unwrap() may panic"),
        (
            "except Exception",
            "warning",
            "safety",
            "Broad exception catch",
        ),
        (
            "except:",
            "warning",
            "safety",
            "Bare except — swallows all errors",
        ),
    ];

    for line in diff_text.lines() {
        if line.starts_with("+++ b/") {
            current_file = line[6..].to_string();
        } else if line.starts_with("@@") {
            if let Some(pos) = line.find('+') {
                let num: String = line[pos + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                current_line = num.parse().unwrap_or(0);
            }
        } else if line.starts_with('+') && !line.starts_with("+++") {
            current_line += 1;
            let content = &line[1..];
            for (pattern, severity, category, message) in patterns {
                if content.to_lowercase().contains(&pattern.to_lowercase()) {
                    findings.push(ReviewFinding {
                        severity: severity.to_string(),
                        category: category.to_string(),
                        file: current_file.clone(),
                        line: current_line,
                        message: message.to_string(),
                        suggestion: String::new(),
                    });
                }
            }
        }
    }
    findings
}

// ═══════════════════════════════════════════════════════════════════
// Routing Matrix (Rust)
// ═══════════════════════════════════════════════════════════════════

pub fn select_flow(
    intent: EpistemicIntent,
    risk: RiskLevel,
    belief_exists: bool,
    belief_fresh: bool,
    belief_verified: bool,
    is_event: bool,
    miss_count: u32,
    miss_threshold: u32,
) -> (EpistemicFlow, String) {
    if is_event {
        return (
            EpistemicFlow::ChangeDriven,
            "Event trigger → Change-Driven pipeline".into(),
        );
    }
    if miss_count >= miss_threshold {
        return (
            EpistemicFlow::SelfImprovement,
            format!(
                "Repeated miss ({}/{}) → Self-Improvement",
                miss_count, miss_threshold
            ),
        );
    }
    if !belief_exists {
        return (
            EpistemicFlow::CompileOnDemand,
            "No beliefs exist → Compile On Demand".into(),
        );
    }
    if !belief_fresh {
        return (
            EpistemicFlow::VerifyBefore,
            "Beliefs exist but stale → Verify Before Answer".into(),
        );
    }
    if !belief_verified {
        return (
            EpistemicFlow::VerifyBefore,
            "Beliefs exist but unverified → Verify Before Answer".into(),
        );
    }
    if risk == RiskLevel::High {
        return (
            EpistemicFlow::VerifyBefore,
            "High-risk domain → Verify Before Answer".into(),
        );
    }
    if intent == EpistemicIntent::CodeGeneration {
        return (
            EpistemicFlow::VerifyBefore,
            "Code generation → Verify Before Answer".into(),
        );
    }
    (
        EpistemicFlow::FastAnswer,
        "Fresh + verified + low-risk → Fast Answer".into(),
    )
}

// ═══════════════════════════════════════════════════════════════════
// PyO3 — The CogOps Engine (exposed to Python)
// ═══════════════════════════════════════════════════════════════════

#[pyclass]
pub struct CogOpsEngine {
    vault_path: PathBuf,
    miss_counts: HashMap<String, u32>,
    miss_threshold: u32,
    freshness_hours: f64,
    min_confidence: f64,
    routing_stats: HashMap<String, u32>,
}

#[pymethods]
impl CogOpsEngine {
    #[new]
    #[pyo3(signature = (vault_path, miss_threshold=3, freshness_hours=24.0, min_confidence=0.5))]
    pub fn new(
        vault_path: String,
        miss_threshold: u32,
        freshness_hours: f64,
        min_confidence: f64,
    ) -> Self {
        let vp = PathBuf::from(&vault_path);
        ensure_vault(&vp);
        CogOpsEngine {
            vault_path: vp,
            miss_counts: HashMap::new(),
            miss_threshold,
            freshness_hours,
            min_confidence,
            routing_stats: HashMap::new(),
        }
    }

    /// Classify intent from a query string.
    pub fn classify_intent(&self, query: &str) -> String {
        classify_intent(query).as_str().to_string()
    }

    /// Route a query through the epistemic routing matrix.
    pub fn route(&mut self, query: &str, is_event: bool, event_type: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let intent = classify_intent(query);
            let risk = assess_risk(query);
            let entity_key = extract_entity_key(query);

            // Check belief coverage
            let beliefs = read_all_beliefs(&self.vault_path);
            let entity_lower = entity_key.to_lowercase();
            let matching: Vec<&BeliefArtifact> = beliefs
                .iter()
                .filter(|b| b.entity.to_lowercase().contains(&entity_lower))
                .collect();

            let belief_exists = !matching.is_empty();
            let belief_fresh = matching.iter().any(|b| b.status != "stale");
            let belief_verified = matching.iter().any(|b| b.status == "verified");
            let confidence = matching
                .iter()
                .map(|b| b.confidence)
                .fold(0.0_f64, f64::max);

            let miss_count = self.miss_counts.get(&entity_key).copied().unwrap_or(0);

            let (flow, reasoning) = select_flow(
                intent,
                risk,
                belief_exists,
                belief_fresh,
                belief_verified,
                is_event || !event_type.is_empty(),
                miss_count,
                self.miss_threshold,
            );

            // Track routing stats
            *self
                .routing_stats
                .entry(flow.as_str().to_string())
                .or_insert(0) += 1;

            // If compile_on_demand or self_improvement, record miss
            if flow == EpistemicFlow::CompileOnDemand || flow == EpistemicFlow::SelfImprovement {
                *self.miss_counts.entry(entity_key.clone()).or_insert(0) += 1;
            }

            let result = PyDict::new(py);
            result.set_item("flow", flow.as_str())?;
            result.set_item("intent", intent.as_str())?;
            result.set_item("risk", risk.as_str())?;
            result.set_item("reasoning", &reasoning)?;
            result.set_item("entity_key", &entity_key)?;
            result.set_item("belief_exists", belief_exists)?;
            result.set_item("belief_fresh", belief_fresh)?;
            result.set_item("belief_verified", belief_verified)?;
            result.set_item("confidence", confidence)?;
            result.set_item("miss_count", miss_count)?;
            Ok(result.into())
        })
    }

    /// Extract entities from source code.
    pub fn extract_entities(&self, content: &str, file_path: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let entities = extract_entities(content, file_path);
            let list = PyList::empty(py);
            for e in &entities {
                let d = PyDict::new(py);
                d.set_item("name", &e.name)?;
                d.set_item("kind", &e.kind)?;
                d.set_item("file_path", &e.file_path)?;
                d.set_item("line", e.line)?;
                d.set_item("docstring", &e.docstring)?;
                d.set_item("signature", &e.signature)?;
                list.append(d)?;
            }
            Ok(list.into())
        })
    }

    /// Compile a directory of source files into belief artifacts.
    pub fn compile_beliefs(&mut self, directory: &str, max_files: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let root = Path::new(directory);
            let skip: HashSet<&str> = [
                "__pycache__",
                "node_modules",
                ".git",
                "target",
                "dist",
                "build",
                ".tox",
                ".pytest_cache",
                "venv",
                ".venv",
            ]
            .iter()
            .copied()
            .collect();
            let exts: HashSet<&str> = ["py", "rs", "ts", "tsx", "js", "jsx"]
                .iter()
                .copied()
                .collect();

            let mut files_processed = 0u32;
            let mut beliefs_written = 0u32;
            let mut entities_extracted = 0u32;
            let mut errors: Vec<String> = Vec::new();

            let mut source_files = Vec::new();
            collect_source_files(root, &skip, &exts, &mut source_files, max_files);

            for fpath in &source_files {
                match fs::read_to_string(fpath) {
                    Ok(content) => {
                        let rel = fpath
                            .strip_prefix(root)
                            .unwrap_or(fpath)
                            .to_string_lossy()
                            .to_string();
                        let entities = extract_entities(&content, &rel);
                        entities_extracted += entities.len() as u32;

                        if !entities.is_empty() {
                            let module_name = fpath
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown");
                            let lang = fpath.extension().and_then(|e| e.to_str()).unwrap_or("");
                            let loc = content.lines().count();

                            // Build body
                            let mut body =
                                format!("**Language:** {}\n**Lines of code:** {}\n\n", lang, loc);
                            let classes: Vec<&CodeEntity> = entities
                                .iter()
                                .filter(|e| {
                                    matches!(e.kind.as_str(), "class" | "struct" | "enum" | "trait")
                                })
                                .collect();
                            let funcs: Vec<&CodeEntity> =
                                entities.iter().filter(|e| e.kind == "function").collect();

                            if !classes.is_empty() {
                                body.push_str("## Types\n");
                                for c in &classes {
                                    let doc = if c.docstring.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" — {}", c.docstring)
                                    };
                                    body.push_str(&format!("- `{}`{}\n", c.signature, doc));
                                }
                            }
                            if !funcs.is_empty() {
                                body.push_str("\n## Functions\n");
                                for f in &funcs {
                                    let doc = if f.docstring.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" — {}", f.docstring)
                                    };
                                    body.push_str(&format!("- `{}`{}\n", f.signature, doc));
                                }
                            }

                            let sources: Vec<String> = entities
                                .iter()
                                .take(10)
                                .map(|e| format!("{}:{}", rel, e.line))
                                .collect();
                            let artifact = BeliefArtifact {
                                claim_id: generate_claim_id(),
                                entity: module_name.to_string(),
                                title: format!("Module: {}", module_name),
                                status: "inferred".into(),
                                confidence: 0.75,
                                sources,
                                derived_from: vec!["cogops_compiler".into(), "sast".into()],
                                body,
                            };
                            match write_belief_artifact(&self.vault_path, &artifact) {
                                Ok(_) => beliefs_written += 1,
                                Err(e) => errors.push(format!("{}: {}", rel, e)),
                            }
                        }
                        files_processed += 1;
                    }
                    Err(e) => errors.push(format!("{}: {}", fpath.display(), e)),
                }
            }

            let result = PyDict::new(py);
            result.set_item("status", "compiled")?;
            result.set_item("files_processed", files_processed)?;
            result.set_item("beliefs_written", beliefs_written)?;
            result.set_item("entities_extracted", entities_extracted)?;
            result.set_item("errors", &errors[..errors.len().min(5)])?;
            Ok(result.into())
        })
    }

    /// Run full verification pass on all beliefs.
    pub fn verify_beliefs(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let beliefs = read_all_beliefs(&self.vault_path);
            let total = beliefs.len();
            let contras = detect_contradictions(&beliefs);
            let verified = beliefs.iter().filter(|b| b.status == "verified").count();
            let stale = beliefs.iter().filter(|b| b.status == "stale").count();
            let confs: Vec<f64> = beliefs.iter().map(|b| b.confidence).collect();
            let mean_conf = if confs.is_empty() {
                0.0
            } else {
                confs.iter().sum::<f64>() / confs.len() as f64
            };
            let low_conf = confs.iter().filter(|&&c| c < self.min_confidence).count();

            let result = PyDict::new(py);
            result.set_item("total_beliefs_checked", total)?;
            result.set_item("verified_count", verified)?;
            result.set_item("stale_count", stale)?;
            result.set_item("low_confidence_count", low_conf)?;
            result.set_item("mean_confidence", (mean_conf * 1000.0).round() / 1000.0)?;
            result.set_item("contradictions", contras.len())?;
            let contra_list = PyList::empty(py);
            for c in &contras {
                let d = PyDict::new(py);
                d.set_item("entity", &c.entity)?;
                d.set_item("type", &c.conflict_type)?;
                d.set_item("description", &c.description)?;
                d.set_item("severity", &c.severity)?;
                contra_list.append(d)?;
            }
            result.set_item("contradiction_details", contra_list)?;
            Ok(result.into())
        })
    }

    /// Compute blast radius for changed files.
    pub fn blast_radius(&self, changed_files: Vec<String>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let beliefs = read_all_beliefs(&self.vault_path);
            let refs: Vec<&str> = changed_files.iter().map(|s| s.as_str()).collect();
            let (ab, ae, risk) = compute_blast_radius(&beliefs, &refs);
            let result = PyDict::new(py);
            result.set_item("affected_beliefs", &ab)?;
            result.set_item("affected_entities", &ae)?;
            result.set_item("risk_level", risk)?;
            result.set_item("description", format!("{} beliefs affected", ab.len()))?;
            Ok(result.into())
        })
    }

    /// Parse and analyze a diff.
    pub fn process_change(
        &self,
        diff_text: &str,
        commit_msg: &str,
        pr_title: &str,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let cs = parse_diff(diff_text, commit_msg);
            let findings = review_diff(diff_text);

            // Compute belief impact
            let beliefs = read_all_beliefs(&self.vault_path);
            let all_changed: Vec<&str> = cs
                .files_modified
                .iter()
                .chain(cs.files_added.iter())
                .map(|s| s.as_str())
                .collect();
            let (ab, ae, risk) = compute_blast_radius(&beliefs, &all_changed);

            let result = PyDict::new(py);
            let default_title = format!(
                "{}: {}",
                cs.intent,
                cs.files_modified
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("changes")
            );
            let title_str = if pr_title.is_empty() {
                default_title.as_str()
            } else {
                pr_title
            };
            result.set_item("title", title_str)?;
            result.set_item("intent", &cs.intent)?;
            result.set_item("files_added", &cs.files_added)?;
            result.set_item("files_modified", &cs.files_modified)?;
            result.set_item("files_deleted", &cs.files_deleted)?;
            result.set_item("lines_added", cs.lines_added)?;
            result.set_item("lines_removed", cs.lines_removed)?;
            result.set_item("functions_changed", &cs.functions_changed)?;
            result.set_item("risk_level", risk)?;
            result.set_item("affected_beliefs", &ab)?;
            result.set_item("affected_entities", &ae)?;
            result.set_item("findings_count", findings.len())?;
            let fl = PyList::empty(py);
            for f in &findings {
                let d = PyDict::new(py);
                d.set_item("severity", &f.severity)?;
                d.set_item("category", &f.category)?;
                d.set_item("file", &f.file)?;
                d.set_item("line", f.line)?;
                d.set_item("message", &f.message)?;
                fl.append(d)?;
            }
            result.set_item("findings", fl)?;
            Ok(result.into())
        })
    }

    /// Write a belief artifact to the vault.
    pub fn write_belief(
        &self,
        entity: &str,
        title: &str,
        body: &str,
        confidence: f64,
        status: &str,
        sources: Vec<String>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let artifact = BeliefArtifact {
                claim_id: generate_claim_id(),
                entity: entity.to_string(),
                title: title.to_string(),
                status: status.to_string(),
                confidence,
                sources,
                derived_from: vec!["manual".into()],
                body: body.to_string(),
            };
            match write_belief_artifact(&self.vault_path, &artifact) {
                Ok(path) => {
                    let r = PyDict::new(py);
                    r.set_item("status", "written")?;
                    r.set_item("path", &path)?;
                    r.set_item("claim_id", &artifact.claim_id)?;
                    Ok(r.into())
                }
                Err(e) => {
                    let r = PyDict::new(py);
                    r.set_item("status", "error")?;
                    r.set_item("error", &e)?;
                    Ok(r.into())
                }
            }
        })
    }

    /// Get vault status and coverage index.
    pub fn vault_status(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let beliefs = read_all_beliefs(&self.vault_path);
            let result = PyDict::new(py);
            result.set_item("total_beliefs", beliefs.len())?;
            result.set_item(
                "verified",
                beliefs.iter().filter(|b| b.status == "verified").count(),
            )?;
            result.set_item(
                "inferred",
                beliefs.iter().filter(|b| b.status == "inferred").count(),
            )?;
            result.set_item(
                "stale",
                beliefs.iter().filter(|b| b.status == "stale").count(),
            )?;
            let confs: Vec<f64> = beliefs.iter().map(|b| b.confidence).collect();
            let mean = if confs.is_empty() {
                0.0
            } else {
                confs.iter().sum::<f64>() / confs.len() as f64
            };
            result.set_item("mean_confidence", (mean * 1000.0).round() / 1000.0)?;
            // Entity list
            let entities: Vec<&str> = beliefs.iter().map(|b| b.entity.as_str()).collect();
            result.set_item("entities", &entities)?;
            // Routing stats
            let stats = PyDict::new(py);
            for (k, v) in &self.routing_stats {
                stats.set_item(k.as_str(), *v)?;
            }
            result.set_item("routing_stats", stats)?;
            Ok(result.into())
        })
    }

    /// Create a new skill from a gap report.
    pub fn create_skill(
        &self,
        entity_key: &str,
        failing_queries: Vec<String>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let skill_id = format!(
                "{:012x}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            );
            let skill_dir = self.vault_path.join("evolution/skills").join(&skill_id);
            let _ = fs::create_dir_all(skill_dir.join("tests"));

            // SKILL.md
            let now = chrono_iso();
            let skill_md = format!(
                "---\nskill_id: {}\nname: {}\nentity: {}\nstatus: draft\ncreated_at: {}\n---\n\n# {}\n\nSkill for handling {} queries.\n\n## Trigger\nActivates on queries relating to `{}`.\n\n## Steps\n1. Check source files for `{}`\n2. Extract structural information\n3. Build belief artifact\n4. Cross-reference with existing beliefs\n5. Generate answer\n",
                skill_id, entity_key, entity_key, now, entity_key, entity_key, entity_key, entity_key
            );
            let _ = fs::write(skill_dir.join("SKILL.md"), &skill_md);

            // tool.py
            let tool_py = format!(
                "\"\"\"Auto-generated skill tool: {}\"\"\"\nimport re\nTRIGGER = re.compile(r'\\b{}\\b', re.I)\ndef matches(q): return bool(TRIGGER.search(q))\ndef execute(q, ctx): return {{'status': 'executed', 'skill': '{}'}}\n",
                entity_key, entity_key, entity_key
            );
            let _ = fs::write(skill_dir.join("tool.py"), &tool_py);

            // metrics.json
            let _ = fs::write(
                skill_dir.join("metrics.json"),
                "{\"fitness_score\":0.0,\"runs\":0}",
            );

            // tests
            let tests: Vec<serde_json::Value> = failing_queries
                .iter()
                .map(|q| serde_json::json!({"input": q, "expected": "should_not_fail"}))
                .collect();
            let _ = fs::write(
                skill_dir.join("tests/test_cases.json"),
                serde_json::to_string_pretty(&tests).unwrap_or_default(),
            );

            // Update registry
            let registry_path = self.vault_path.join("evolution/registry.md");
            if let Ok(mut content) = fs::read_to_string(&registry_path) {
                content.push_str(&format!(
                    "| {} | created | {} | {} |\n",
                    skill_id,
                    &now[..10],
                    entity_key
                ));
                let _ = fs::write(&registry_path, content);
            }

            let result = PyDict::new(py);
            result.set_item("status", "created")?;
            result.set_item("skill_id", &skill_id)?;
            result.set_item("path", skill_dir.to_string_lossy().as_ref())?;
            Ok(result.into())
        })
    }

    /// List all skills.
    pub fn list_skills(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let skills_dir = self.vault_path.join("evolution/skills");
            let list = PyList::empty(py);
            if let Ok(entries) = fs::read_dir(&skills_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let d = PyDict::new(py);
                        d.set_item("skill_id", entry.file_name().to_string_lossy().as_ref())?;
                        d.set_item("path", entry.path().to_string_lossy().as_ref())?;
                        // Read SKILL.md frontmatter for status/name
                        if let Ok(skill_md) = fs::read_to_string(entry.path().join("SKILL.md")) {
                            if let Some(status) = extract_fm_value(&skill_md, "status") {
                                d.set_item("status", status)?;
                            }
                            if let Some(name) = extract_fm_value(&skill_md, "name") {
                                d.set_item("name", name)?;
                            }
                        }
                        if let Ok(metrics) = fs::read_to_string(entry.path().join("metrics.json")) {
                            if let Ok(m) = serde_json::from_str::<serde_json::Value>(&metrics) {
                                d.set_item(
                                    "fitness",
                                    m.get("fitness_score")
                                        .and_then(|v| v.as_f64())
                                        .unwrap_or(0.0),
                                )?;
                                d.set_item(
                                    "runs",
                                    m.get("runs").and_then(|v| v.as_u64()).unwrap_or(0),
                                )?;
                            }
                        }
                        list.append(d)?;
                    }
                }
            }
            Ok(list.into())
        })
    }

    /// Find source files with no corresponding belief in the vault.
    pub fn coverage_gaps(&self, directory: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let root = Path::new(directory);
            let beliefs = read_all_beliefs(&self.vault_path);
            let skip: HashSet<&str> = [
                "__pycache__",
                "node_modules",
                ".git",
                "target",
                "dist",
                "build",
                ".tox",
                ".pytest_cache",
                "venv",
                ".venv",
            ]
            .iter()
            .copied()
            .collect();
            let exts: HashSet<&str> = ["py", "rs", "ts", "tsx", "js", "jsx"]
                .iter()
                .copied()
                .collect();
            let skip_stems: HashSet<&str> = ["__init__", "mod", "lib", "main", "index"]
                .iter()
                .copied()
                .collect();

            // Build set of known entities (lowercased)
            let mut known: HashSet<String> = HashSet::new();
            for b in &beliefs {
                known.insert(b.entity.to_lowercase());
                for src in &b.sources {
                    if let Some(stem) = src
                        .split(':')
                        .next()
                        .and_then(|p| Path::new(p).file_stem())
                        .and_then(|s| s.to_str())
                    {
                        known.insert(stem.to_lowercase());
                    }
                }
            }

            let mut source_files = Vec::new();
            collect_source_files(root, &skip, &exts, &mut source_files, 1000);

            let gaps = PyList::empty(py);
            for fpath in &source_files {
                if let Some(stem) = fpath.file_stem().and_then(|s| s.to_str()) {
                    let stem_lower = stem.to_lowercase();
                    if skip_stems.contains(stem_lower.as_str()) {
                        continue;
                    }
                    if !known.contains(&stem_lower) {
                        let d = PyDict::new(py);
                        let rel = fpath
                            .strip_prefix(root)
                            .unwrap_or(fpath)
                            .to_string_lossy()
                            .to_string();
                        d.set_item("file", &rel)?;
                        d.set_item("reason", "no_belief_artifact")?;
                        d.set_item("suggested_entity", &stem_lower)?;
                        gaps.append(d)?;
                    }
                }
            }

            let result = PyDict::new(py);
            let total_gaps = gaps.len();
            result.set_item("gaps", gaps)?;
            result.set_item("total_gaps", total_gaps)?;
            Ok(result.into())
        })
    }

    /// Mark beliefs as stale after file changes.
    pub fn refresh_beliefs(&self, changed_files: Vec<String>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let beliefs_dir = self.vault_path.join("beliefs");
            let mut refreshed: Vec<String> = Vec::new();

            for cf in &changed_files {
                let stem = Path::new(cf)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if stem.is_empty() {
                    continue;
                }

                if let Ok(entries) = fs::read_dir(&beliefs_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) != Some("md") {
                            continue;
                        }
                        let fname = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if !fname.contains(&stem) {
                            continue;
                        }

                        if let Ok(content) = fs::read_to_string(&path) {
                            if content.contains("status: verified")
                                || content.contains("status: inferred")
                            {
                                let updated = content
                                    .replace("status: verified", "status: stale")
                                    .replace("status: inferred", "status: stale");
                                let _ = fs::write(&path, updated);
                                refreshed.push(fname.clone());
                            }
                        }
                    }
                }
            }

            let result = PyDict::new(py);
            result.set_item("status", "refreshed")?;
            result.set_item("stale_marked", &refreshed)?;
            result.set_item("total", refreshed.len())?;
            Ok(result.into())
        })
    }

    /// Benchmark a skill by running its test cases in a subprocess.
    pub fn benchmark_skill(&self, skill_id: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let skill_dir = self.vault_path.join("evolution/skills").join(skill_id);
            if !skill_dir.exists() {
                let r = PyDict::new(py);
                r.set_item("status", "not_found")?;
                r.set_item("skill_id", skill_id)?;
                return Ok(r.into());
            }

            let tool_path = skill_dir.join("tool.py");
            let tests_path = skill_dir.join("tests/test_cases.json");

            let mut passed: u32 = 0;
            let mut failed: u32 = 0;
            let mut errors: Vec<String> = Vec::new();

            if tool_path.exists() && tests_path.exists() {
                if let Ok(tests_json) = fs::read_to_string(&tests_path) {
                    if let Ok(tests) = serde_json::from_str::<Vec<serde_json::Value>>(&tests_json) {
                        for tc in &tests {
                            let query = tc.get("input").and_then(|v| v.as_str()).unwrap_or("");
                            // Write harness and run in subprocess
                            let tool_code = fs::read_to_string(&tool_path).unwrap_or_default();
                            let harness = format!(
                                "{}\n\nif __name__ == '__main__':\n    import json, sys\n    q = sys.argv[1] if len(sys.argv) > 1 else ''\n    r = execute(q, {{}})\n    print(json.dumps(r))\n",
                                tool_code
                            );
                            let tmp = skill_dir.join("_bench_harness.py");
                            let _ = fs::write(&tmp, &harness);
                            let output = std::process::Command::new("python")
                                .args([tmp.to_string_lossy().as_ref(), query])
                                .output();
                            let _ = fs::remove_file(&tmp);
                            match output {
                                Ok(o) if o.status.success() => passed += 1,
                                Ok(o) => {
                                    failed += 1;
                                    errors.push(format!(
                                        "{}: {}",
                                        query,
                                        String::from_utf8_lossy(&o.stderr)
                                            .chars()
                                            .take(200)
                                            .collect::<String>()
                                    ));
                                }
                                Err(e) => {
                                    failed += 1;
                                    errors.push(format!("{}: {}", query, e));
                                }
                            }
                        }
                    }
                }
            }

            let total = passed + failed;
            let fitness = if total > 0 {
                passed as f64 / total as f64
            } else {
                0.0
            };

            // Update metrics.json
            let metrics_path = skill_dir.join("metrics.json");
            let mut metrics: serde_json::Value = fs::read_to_string(&metrics_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(serde_json::json!({}));
            if let Some(obj) = metrics.as_object_mut() {
                obj.insert("fitness_score".into(), serde_json::json!(fitness));
                obj.insert(
                    "runs".into(),
                    serde_json::json!(obj.get("runs").and_then(|v| v.as_u64()).unwrap_or(0) + 1),
                );
                obj.insert(
                    "successes".into(),
                    serde_json::json!(
                        obj.get("successes").and_then(|v| v.as_u64()).unwrap_or(0) + passed as u64
                    ),
                );
                obj.insert(
                    "failures".into(),
                    serde_json::json!(
                        obj.get("failures").and_then(|v| v.as_u64()).unwrap_or(0) + failed as u64
                    ),
                );
                obj.insert("last_benchmark".into(), serde_json::json!(chrono_iso()));
            }
            let _ = fs::write(
                &metrics_path,
                serde_json::to_string_pretty(&metrics).unwrap_or_default(),
            );

            let result = PyDict::new(py);
            result.set_item("status", "benchmarked")?;
            result.set_item("skill_id", skill_id)?;
            result.set_item("fitness", fitness)?;
            result.set_item("passed", passed)?;
            result.set_item("failed", failed)?;
            result.set_item("errors", &errors[..errors.len().min(5)])?;
            Ok(result.into())
        })
    }

    /// Promote or prune a skill based on fitness score.
    pub fn promote_skill(&self, skill_id: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let skill_dir = self.vault_path.join("evolution/skills").join(skill_id);
            if !skill_dir.exists() {
                let r = PyDict::new(py);
                r.set_item("status", "not_found")?;
                return Ok(r.into());
            }

            // Read fitness from metrics
            let metrics_path = skill_dir.join("metrics.json");
            let fitness: f64 = fs::read_to_string(&metrics_path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|m| m.get("fitness_score").and_then(|v| v.as_f64()))
                .unwrap_or(0.0);

            let (action, new_status) = if fitness >= 0.7 {
                ("promoted", "promoted")
            } else if fitness <= 0.3 {
                ("pruned", "pruned")
            } else {
                ("kept", "testing")
            };

            // Update SKILL.md status
            let skill_md_path = skill_dir.join("SKILL.md");
            if let Ok(content) = fs::read_to_string(&skill_md_path) {
                let updated = content.replace(
                    &format!(
                        "status: {}",
                        extract_fm_value(&content, "status").unwrap_or_default()
                    ),
                    &format!("status: {}", new_status),
                );
                let _ = fs::write(&skill_md_path, updated);
            }

            // Update registry
            let registry_path = self.vault_path.join("evolution/registry.md");
            if let Ok(content) = fs::read_to_string(&registry_path) {
                let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                let mut found = false;
                for line in &mut lines {
                    if line.contains(skill_id) {
                        *line = format!(
                            "| {} | {} | {} | {} |",
                            skill_id,
                            action,
                            chrono_iso().get(..10).unwrap_or(""),
                            skill_id
                        );
                        found = true;
                        break;
                    }
                }
                if !found {
                    lines.push(format!(
                        "| {} | {} | {} | {} |",
                        skill_id,
                        action,
                        chrono_iso().get(..10).unwrap_or(""),
                        skill_id
                    ));
                }
                let _ = fs::write(&registry_path, lines.join("\n"));
            }

            let result = PyDict::new(py);
            result.set_item("status", action)?;
            result.set_item("skill_id", skill_id)?;
            result.set_item("fitness", fitness)?;
            result.set_item("new_status", new_status)?;
            Ok(result.into())
        })
    }

    /// Execute a full canonical epistemic flow end-to-end.
    pub fn execute_flow(
        &mut self,
        query: &str,
        diff_text: &str,
        is_event: bool,
        event_type: &str,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Step 1: Route
            let intent = classify_intent(query);
            let risk = assess_risk(query);
            let entity_key = extract_entity_key(query);
            let beliefs = read_all_beliefs(&self.vault_path);
            let entity_lower = entity_key.to_lowercase();
            let matching: Vec<&BeliefArtifact> = beliefs
                .iter()
                .filter(|b| b.entity.to_lowercase().contains(&entity_lower))
                .collect();
            let belief_exists = !matching.is_empty();
            let belief_fresh = matching.iter().any(|b| b.status != "stale");
            let belief_verified = matching.iter().any(|b| b.status == "verified");
            let miss_count = self.miss_counts.get(&entity_key).copied().unwrap_or(0);

            let (flow, reasoning) = select_flow(
                intent,
                risk,
                belief_exists,
                belief_fresh,
                belief_verified,
                is_event || !event_type.is_empty(),
                miss_count,
                self.miss_threshold,
            );

            *self
                .routing_stats
                .entry(flow.as_str().to_string())
                .or_insert(0) += 1;
            if flow == EpistemicFlow::CompileOnDemand || flow == EpistemicFlow::SelfImprovement {
                *self.miss_counts.entry(entity_key.clone()).or_insert(0) += 1;
            }

            let steps = PyList::empty(py);
            let route_step = PyDict::new(py);
            route_step.set_item("step", "route")?;
            route_step.set_item("flow", flow.as_str())?;
            route_step.set_item("reasoning", &reasoning)?;
            steps.append(route_step)?;

            let mut answer = String::new();

            match flow {
                EpistemicFlow::ChangeDriven => {
                    if !diff_text.is_empty() {
                        let cs = parse_diff(diff_text, query);
                        let findings = review_diff(diff_text);
                        let all_changed: Vec<&str> = cs
                            .files_modified
                            .iter()
                            .chain(cs.files_added.iter())
                            .map(|s| s.as_str())
                            .collect();
                        let (ab, _ae, br_risk) = compute_blast_radius(&beliefs, &all_changed);
                        let step = PyDict::new(py);
                        step.set_item("step", "change_pipeline")?;
                        step.set_item("findings", findings.len())?;
                        step.set_item("blast_radius", br_risk)?;
                        step.set_item("affected_beliefs", ab.len())?;
                        steps.append(step)?;
                        answer = format!(
                            "{}: +{}/-{}, {} findings, {} beliefs affected ({})",
                            cs.intent,
                            cs.lines_added,
                            cs.lines_removed,
                            findings.len(),
                            ab.len(),
                            br_risk
                        );
                    } else {
                        answer = "No diff provided for change-driven pipeline".into();
                    }
                }
                EpistemicFlow::CompileOnDemand => {
                    // Compile beliefs from source
                    let skip: HashSet<&str> = [
                        "__pycache__",
                        "node_modules",
                        ".git",
                        "target",
                        "dist",
                        "build",
                        "venv",
                        ".venv",
                    ]
                    .iter()
                    .copied()
                    .collect();
                    let exts: HashSet<&str> = ["py", "rs", "ts", "tsx", "js", "jsx"]
                        .iter()
                        .copied()
                        .collect();
                    let source_dir = std::env::var("ENTROLY_SOURCE").unwrap_or_else(|_| ".".into());
                    let root = Path::new(&source_dir);
                    let mut files = Vec::new();
                    collect_source_files(root, &skip, &exts, &mut files, 200);
                    let mut bw = 0u32;
                    for fpath in &files {
                        if let Ok(content) = fs::read_to_string(fpath) {
                            let rel = fpath
                                .strip_prefix(root)
                                .unwrap_or(fpath)
                                .to_string_lossy()
                                .to_string();
                            let entities = extract_entities(&content, &rel);
                            if !entities.is_empty() {
                                let module_name = fpath
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("unknown");
                                let loc = content.lines().count();
                                let mut body = format!("**LOC:** {}\n\n## Entities\n", loc);
                                for e in &entities {
                                    body.push_str(&format!("- `{}` ({})\n", e.signature, e.kind));
                                }
                                let sources: Vec<String> = entities
                                    .iter()
                                    .take(5)
                                    .map(|e| format!("{}:{}", rel, e.line))
                                    .collect();
                                let artifact = BeliefArtifact {
                                    claim_id: generate_claim_id(),
                                    entity: module_name.to_string(),
                                    title: format!("Module: {}", module_name),
                                    status: "inferred".into(),
                                    confidence: 0.75,
                                    sources,
                                    derived_from: vec!["cogops_compiler".into()],
                                    body,
                                };
                                if write_belief_artifact(&self.vault_path, &artifact).is_ok() {
                                    bw += 1;
                                }
                            }
                        }
                    }
                    // Verify
                    let fresh_beliefs = read_all_beliefs(&self.vault_path);
                    let contras = detect_contradictions(&fresh_beliefs);
                    let step = PyDict::new(py);
                    step.set_item("step", "compile_and_verify")?;
                    step.set_item("beliefs_written", bw)?;
                    step.set_item("total_beliefs", fresh_beliefs.len())?;
                    step.set_item("contradictions", contras.len())?;
                    steps.append(step)?;
                    answer = format!(
                        "Compiled {} beliefs, verified {} total, {} contradictions",
                        bw,
                        fresh_beliefs.len(),
                        contras.len()
                    );
                }
                EpistemicFlow::VerifyBefore => {
                    let contras = detect_contradictions(&beliefs);
                    let verified_n = beliefs.iter().filter(|b| b.status == "verified").count();
                    let stale_n = beliefs.iter().filter(|b| b.status == "stale").count();
                    let step = PyDict::new(py);
                    step.set_item("step", "verify")?;
                    step.set_item("checked", beliefs.len())?;
                    step.set_item("verified", verified_n)?;
                    step.set_item("stale", stale_n)?;
                    step.set_item("contradictions", contras.len())?;
                    steps.append(step)?;

                    // Assemble answer from matching beliefs
                    if !matching.is_empty() {
                        let mut parts = Vec::new();
                        for b in &matching {
                            parts.push(format!(
                                "### {} (confidence: {:.2}, status: {})\n{}",
                                b.entity,
                                b.confidence,
                                b.status,
                                b.body.chars().take(500).collect::<String>()
                            ));
                        }
                        answer = parts.join("\n\n");
                    } else {
                        answer = format!(
                            "Verified {} beliefs ({} contradictions, {} stale)",
                            beliefs.len(),
                            contras.len(),
                            stale_n
                        );
                    }
                }
                EpistemicFlow::SelfImprovement => {
                    let step = PyDict::new(py);
                    step.set_item("step", "evolution")?;
                    step.set_item("miss_count", miss_count + 1)?;
                    step.set_item("entity", &entity_key)?;
                    steps.append(step)?;
                    answer = format!(
                        "Miss recorded for '{}' (count: {}). Skill gap threshold: {}",
                        entity_key,
                        miss_count + 1,
                        self.miss_threshold
                    );

                    if miss_count + 1 >= self.miss_threshold {
                        // Auto-create skill
                        let skill_id = format!(
                            "{:012x}",
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64
                        );
                        let skill_dir = self.vault_path.join("evolution/skills").join(&skill_id);
                        let _ = fs::create_dir_all(skill_dir.join("tests"));
                        let now = chrono_iso();
                        let _ = fs::write(skill_dir.join("SKILL.md"), format!(
                            "---\nskill_id: {}\nname: {}\nentity: {}\nstatus: draft\ncreated_at: {}\n---\n\n# {}\nAuto-created from repeated failures.\n",
                            skill_id, entity_key, entity_key, now, entity_key));
                        let _ = fs::write(
                            skill_dir.join("metrics.json"),
                            "{\"fitness_score\":0.0,\"runs\":0}",
                        );
                        let _ = fs::write(skill_dir.join("tool.py"), format!(
                            "def matches(q): return '{}' in q.lower()\ndef execute(q, ctx): return {{'skill': '{}'}}\n",
                            entity_key, entity_key));
                        let skill_step = PyDict::new(py);
                        skill_step.set_item("step", "create_skill")?;
                        skill_step.set_item("skill_id", &skill_id)?;
                        steps.append(skill_step)?;
                        answer = format!("{}\nSkill created: {}", answer, skill_id);
                    }
                }
                EpistemicFlow::FastAnswer => {
                    if !matching.is_empty() {
                        let mut parts = Vec::new();
                        for b in &matching {
                            parts.push(format!(
                                "### {} (confidence: {:.2})\n{}",
                                b.entity,
                                b.confidence,
                                b.body.chars().take(500).collect::<String>()
                            ));
                        }
                        answer = parts.join("\n\n");
                    } else {
                        answer = format!(
                            "No beliefs found for '{}'. Run compile_beliefs first.",
                            entity_key
                        );
                    }
                    let step = PyDict::new(py);
                    step.set_item("step", "belief_lookup")?;
                    step.set_item("beliefs_found", matching.len())?;
                    steps.append(step)?;
                }
            }

            let result = PyDict::new(py);
            result.set_item("flow", flow.as_str())?;
            result.set_item("intent", intent.as_str())?;
            result.set_item("risk", risk.as_str())?;
            result.set_item("steps_completed", steps)?;
            result.set_item("answer", &answer)?;
            result.set_item("engine", "rust")?;
            Ok(result.into())
        })
    }

    // ═══════════════════════════════════════════════════════════════
    // Vault Search — TF-IDF full-text search across beliefs
    // ═══════════════════════════════════════════════════════════════

    /// Full-text search across all belief artifacts.
    /// Returns top_k results ranked by TF-IDF with entity-name boosting (3x).
    #[pyo3(signature = (query, top_k=5))]
    pub fn vault_search(&self, query: &str, top_k: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let beliefs = read_all_beliefs(&self.vault_path);
            if beliefs.is_empty() {
                let empty = PyList::empty(py);
                return Ok(empty.into());
            }

            let query_tokens = tokenize_search(query);
            if query_tokens.is_empty() {
                let empty = PyList::empty(py);
                return Ok(empty.into());
            }

            // Build inverted index: token → Vec<(doc_idx, zone_weight)>
            let n = beliefs.len() as f64;
            let mut doc_freq: HashMap<String, u32> = HashMap::new();
            let mut doc_tokens: Vec<HashMap<String, f64>> = Vec::with_capacity(beliefs.len());

            for belief in &beliefs {
                let mut token_scores: HashMap<String, f64> = HashMap::new();

                // Entity tokens: 3x boost
                for tok in tokenize_search(&belief.entity) {
                    *token_scores.entry(tok).or_insert(0.0) += 3.0;
                }
                // Title tokens: 2x boost
                for tok in tokenize_search(&belief.title) {
                    *token_scores.entry(tok).or_insert(0.0) += 2.0;
                }
                // Body tokens: 1x with term frequency
                let body_toks = tokenize_search(&belief.body);
                let body_len = body_toks.len().max(1) as f64;
                let mut body_counts: HashMap<String, u32> = HashMap::new();
                for tok in &body_toks {
                    *body_counts.entry(tok.clone()).or_insert(0) += 1;
                }
                for (tok, count) in &body_counts {
                    *token_scores.entry(tok.clone()).or_insert(0.0) += *count as f64 / body_len;
                }

                // Track document frequency
                for tok in token_scores.keys() {
                    *doc_freq.entry(tok.clone()).or_insert(0) += 1;
                }
                doc_tokens.push(token_scores);
            }

            // Score each document
            let mut scores: Vec<(usize, f64)> = Vec::new();
            for (idx, ts) in doc_tokens.iter().enumerate() {
                let mut score = 0.0_f64;
                for qt in &query_tokens {
                    if let Some(tf) = ts.get(qt) {
                        let df = *doc_freq.get(qt).unwrap_or(&1) as f64;
                        let idf = (n / df).ln().max(0.1);
                        score += tf * idf;
                    }
                }
                if score > 0.0 {
                    scores.push((idx, score));
                }
            }

            // Sort descending, take top_k
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(top_k);

            let results = PyList::empty(py);
            for (idx, score) in &scores {
                let b = &beliefs[*idx];
                let excerpt = extract_excerpt(&b.body, &query_tokens);
                let d = PyDict::new(py);
                d.set_item("entity", &b.entity)?;
                d.set_item("claim_id", &b.claim_id)?;
                d.set_item("score", *score)?;
                d.set_item("title", &b.title)?;
                d.set_item("excerpt", &excerpt)?;
                d.set_item("confidence", b.confidence)?;
                d.set_item("status", &b.status)?;
                results.append(d)?;
            }
            Ok(results.into())
        })
    }

    // ═══════════════════════════════════════════════════════════════
    // Doc Ingest — Compile .md docs into belief artifacts
    // ═══════════════════════════════════════════════════════════════

    /// Compile markdown documentation files into belief artifacts.
    /// Ingests README.md, ARCHITECTURE.md, docs/, CONTRIBUTING.md etc.
    /// Doc beliefs get confidence 0.80 (human-authored > machine-inferred).
    #[pyo3(signature = (directory, max_files=50))]
    pub fn compile_docs(&self, directory: &str, max_files: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let root = Path::new(directory);
            let doc_patterns = [
                "README",
                "ARCHITECTURE",
                "CONTRIBUTING",
                "CHANGELOG",
                "DESIGN",
                "API",
                "OVERVIEW",
                "GUIDE",
                "SETUP",
                "DEPLOY",
            ];
            let doc_dirs: HashSet<&str> = ["docs", "doc", "documentation", "wiki"]
                .iter()
                .copied()
                .collect();
            let skip: HashSet<&str> = [
                "node_modules",
                ".git",
                "target",
                "dist",
                "build",
                "__pycache__",
                ".venv",
                "venv",
            ]
            .iter()
            .copied()
            .collect();

            let mut md_files: Vec<PathBuf> = Vec::new();

            // Collect project-level doc files
            if let Ok(entries) = fs::read_dir(root) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            if ext.eq_ignore_ascii_case("md") {
                                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                                let stem_upper = stem.to_uppercase();
                                if doc_patterns.iter().any(|p| stem_upper.starts_with(p)) {
                                    md_files.push(path);
                                }
                            }
                        }
                    } else if path.is_dir() {
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if doc_dirs.contains(name.to_lowercase().as_str()) {
                            collect_doc_md_files(&path, &skip, &mut md_files, max_files);
                        }
                    }
                }
            }

            md_files.truncate(max_files);

            let mut docs_compiled = 0u32;
            let mut entities: Vec<String> = Vec::new();
            let now = chrono_iso();

            for md_path in &md_files {
                let content = match fs::read_to_string(md_path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if content.trim().is_empty() {
                    continue;
                }

                let rel = md_path
                    .strip_prefix(root)
                    .map(|r| r.to_string_lossy().to_string())
                    .unwrap_or_else(|_| md_path.to_string_lossy().to_string());

                let stem = md_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_lowercase();

                let entity = format!("doc/{}", stem);

                // Extract title from first # heading
                let title = content
                    .lines()
                    .find(|l| l.starts_with("# "))
                    .map(|l| l[2..].trim().to_string())
                    .unwrap_or_else(|| stem.clone());

                // Extract section structure
                let sections: Vec<String> = content
                    .lines()
                    .filter(|l| l.starts_with("## ") || l.starts_with("### "))
                    .map(|l| l.trim().to_string())
                    .collect();

                // Build belief body
                let word_count = content.split_whitespace().count();
                let section_list = if sections.is_empty() {
                    String::from("(no sections)")
                } else {
                    sections
                        .iter()
                        .map(|s| format!("- {}", s))
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                let body = format!(
                    "# Doc: {}\n\n**Source:** `{}`\n**Words:** {}\n**Type:** documentation\n\n## Sections\n{}\n",
                    title, rel, word_count, section_list
                );

                // Write belief
                let claim_id = format!("{:016x}", {
                    let mut h: u64 = 0xcbf29ce484222325;
                    for byte in entity.as_bytes() {
                        h ^= *byte as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    h
                });

                let belief_md = format!(
                    "---\nclaim_id: {}\nentity: {}\nstatus: inferred\nconfidence: 0.80\nsources:\n  - {}\nlast_checked: {}\nderived_from:\n  - doc_compiler\n---\n\n{}\n",
                    claim_id, entity, rel, now, body
                );

                let safe_name = entity.replace(['/', ' '], "_").to_lowercase();
                let out_path = self
                    .vault_path
                    .join("beliefs")
                    .join(format!("{}.md", safe_name));
                if fs::write(&out_path, belief_md).is_ok() {
                    docs_compiled += 1;
                    entities.push(entity);
                }
            }

            let result = PyDict::new(py);
            result.set_item("status", "compiled")?;
            result.set_item("docs_found", md_files.len())?;
            result.set_item("docs_compiled", docs_compiled)?;
            result.set_item("entities", entities)?;
            result.set_item("engine", "rust")?;
            Ok(result.into())
        })
    }

    // ═══════════════════════════════════════════════════════════════
    // Finetune Export — Generate training data from vault beliefs
    // ═══════════════════════════════════════════════════════════════

    /// Export vault beliefs as JSONL training data for finetuning.
    /// Generates instruction-following pairs: question about entity → belief body.
    /// Leverages PRISM scoring dimensions for quality-weighted sampling.
    #[pyo3(signature = (output_path, format="jsonl"))]
    pub fn export_training_data(&self, output_path: &str, format: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let beliefs = read_all_beliefs(&self.vault_path);

            let mut lines: Vec<String> = Vec::new();
            let mut skipped = 0u32;

            for b in &beliefs {
                // Skip low-confidence or stale beliefs — only train on verified/inferred
                if b.confidence < 0.5 || b.status == "stale" {
                    skipped += 1;
                    continue;
                }

                // Generate multiple Q&A pairs per belief for coverage
                let questions = generate_training_questions(&b.entity, &b.title, &b.body);

                for q in &questions {
                    let entry = format!(
                        "{{\"messages\":[{{\"role\":\"system\",\"content\":\"You are an expert on the {} codebase. Answer questions using your deep understanding of the architecture.\"}},{{\"role\":\"user\",\"content\":\"{}\"}},{{\"role\":\"assistant\",\"content\":\"{}\"}}]}}",
                        b.entity,
                        escape_json(q),
                        escape_json(&b.body.chars().take(2000).collect::<String>())
                    );
                    lines.push(entry);
                }
            }

            // Write output
            let content = lines.join("\n");
            let write_ok = fs::write(output_path, &content).is_ok();

            let result = PyDict::new(py);
            result.set_item("status", if write_ok { "exported" } else { "write_error" })?;
            result.set_item("output_path", output_path)?;
            result.set_item("format", format)?;
            result.set_item("beliefs_used", beliefs.len() as u32 - skipped)?;
            result.set_item("beliefs_skipped", skipped)?;
            result.set_item("training_pairs", lines.len())?;
            result.set_item("total_tokens_approx", content.split_whitespace().count())?;
            result.set_item("engine", "rust")?;
            Ok(result.into())
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn extract_entity_key(query: &str) -> String {
    let lower = query.to_lowercase();
    let stop = [
        "how", "does", "the", "what", "is", "explain", "show", "me", "can", "you", "please",
        "work", "works", "about", "for", "and", "this", "that", "with", "from", "have", "are",
    ];
    let words: Vec<&str> = lower
        .split_whitespace()
        .filter(|w| w.len() > 2 && !stop.contains(w))
        .collect();
    words.first().copied().unwrap_or("unknown").to_string()
}

fn chrono_iso() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = ts.as_secs();
    // Simple UTC ISO-8601 without chrono crate
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    // Days since 1970-01-01
    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut y = 1970;
    loop {
        let ydays = if is_leap(y) { 366 } else { 365 };
        if days < ydays {
            break;
        }
        days -= ydays;
        y += 1;
    }
    let leap = is_leap(y);
    let mdays = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 0;
    for (i, &md) in mdays.iter().enumerate() {
        if days < md {
            mo = i as u64 + 1;
            break;
        }
        days -= md;
    }
    if mo == 0 {
        mo = 12;
    }
    (y, mo, days + 1)
}

fn is_leap(y: u64) -> bool {
    y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400))
}

/// Extract a value from YAML-like frontmatter.
fn extract_fm_value(content: &str, key: &str) -> Option<String> {
    let parts: Vec<&str> = content.splitn(3, "---").collect();
    if parts.len() < 3 {
        return None;
    }
    let prefix = format!("{}:", key);
    for line in parts[1].lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            return Some(trimmed[prefix.len()..].trim().to_string());
        }
    }
    None
}

fn collect_source_files(
    dir: &Path,
    skip: &HashSet<&str>,
    exts: &HashSet<&str>,
    out: &mut Vec<PathBuf>,
    max: usize,
) {
    if out.len() >= max {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !skip.contains(name) {
                    collect_source_files(&path, skip, exts, out, max);
                }
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if exts.contains(ext) {
                    out.push(path);
                    if out.len() >= max {
                        return;
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Vault Search Helpers
// ═══════════════════════════════════════════════════════════════════

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "and", "but",
    "or", "not", "no", "if", "then", "than", "that", "this", "it", "its", "we", "you", "he", "she",
    "they", "my", "your", "how", "what", "which", "where", "when", "who", "does", "about",
];

fn tokenize_search(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let mut tokens = Vec::new();

    for word in lower.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.len() <= 1 {
            continue;
        }
        // Split snake_case
        for part in word.split('_') {
            if part.len() > 1 && !stop.contains(part) {
                tokens.push(part.to_string());
            }
        }
    }
    tokens
}

fn extract_excerpt(body: &str, query_tokens: &[String]) -> String {
    let lines: Vec<&str> = body.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Score each line by query token hits
    let mut best_idx = 0usize;
    let mut best_hits = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let hits = query_tokens
            .iter()
            .filter(|t| lower.contains(t.as_str()))
            .count();
        if hits > best_hits {
            best_hits = hits;
            best_idx = i;
        }
    }

    let start = best_idx.saturating_sub(1);
    let end = (best_idx + 3).min(lines.len());
    lines[start..end]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("\n")
}

// ═══════════════════════════════════════════════════════════════════
// Doc Ingest Helpers
// ═══════════════════════════════════════════════════════════════════

fn collect_doc_md_files(dir: &Path, skip: &HashSet<&str>, out: &mut Vec<PathBuf>, max: usize) {
    if out.len() >= max {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !skip.contains(name) {
                    collect_doc_md_files(&path, skip, out, max);
                }
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("md"))
                .unwrap_or(false)
            {
                out.push(path);
                if out.len() >= max {
                    return;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Finetune Export Helpers
// ═══════════════════════════════════════════════════════════════════

fn generate_training_questions(entity: &str, title: &str, body: &str) -> Vec<String> {
    let mut questions = Vec::new();
    let clean = entity.replace("_", " ").replace("/", " ");

    // Core questions
    questions.push(format!("What does {} do?", clean));
    questions.push(format!("Explain the {} module.", clean));
    questions.push(format!("How does {} work?", clean));

    // If title differs from entity, add title-based questions
    if !title.is_empty() && title.to_lowercase() != entity.to_lowercase() {
        questions.push(format!("What is {}?", title));
    }

    // If body mentions specific types/functions, add questions
    for line in body.lines() {
        if line.starts_with("- `class ") || line.starts_with("- `struct ") {
            let name = line
                .trim_start_matches("- `")
                .split('`')
                .next()
                .unwrap_or("")
                .split('(')
                .next()
                .unwrap_or("")
                .trim();
            if name.len() > 3 {
                questions.push(format!("What does {} in {} do?", name, clean));
            }
        }
    }

    // Cap at 6 questions per entity to avoid training data bloat
    questions.truncate(6);
    questions
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "")
        .replace('\t', "\\t")
}

// ═══════════════════════════════════════════════════════════════════
// Wiki-Link Graph Extraction
// ═══════════════════════════════════════════════════════════════════

/// Extract `[[wiki-links]]` from belief body text.
///
/// These links form the edges of the Knowledge Graph. When IOS selects
/// a belief for module A, and that belief contains `[[B]]`, module B's
/// belief should get a relevance boost — enabling transitive context
/// loading through the vault knowledge graph.
///
/// The [[wiki-link]] syntax mirrors Obsidian and is already generated
/// by the BeliefCompiler's `## Linked Beliefs` section.
///
/// O(N) single pass over the belief text. Returns unique links.
pub fn extract_wiki_links(body: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut seen = HashSet::new();
    let bytes = body.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos + 3 < len {
        if bytes[pos] == b'[' && bytes[pos + 1] == b'[' {
            let start = pos + 2;
            // Find closing ]]
            if let Some(rel_end) = body[start..].find("]]") {
                let link = body[start..start + rel_end].trim();
                if !link.is_empty() && !seen.contains(link) {
                    seen.insert(link.to_string());
                    links.push(link.to_string());
                }
                pos = start + rel_end + 2;
            } else {
                break; // unclosed [[ — no more valid links
            }
        } else {
            pos += 1;
        }
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiki_link_extraction_basic() {
        let body = "This module uses [[proxy_transform]] and [[vault]] for context.";
        let links = extract_wiki_links(body);
        assert_eq!(links, vec!["proxy_transform", "vault"]);
    }

    #[test]
    fn test_wiki_link_extraction_with_headers() {
        let body = "# Architecture\n\n## Linked Beliefs\n- [[knapsack_sds]]\n- [[channel]]\n- [[semantic_dedup]]\n";
        let links = extract_wiki_links(body);
        assert_eq!(links, vec!["knapsack_sds", "channel", "semantic_dedup"]);
    }

    #[test]
    fn test_wiki_link_extraction_empty() {
        assert!(extract_wiki_links("No links here").is_empty());
        assert!(extract_wiki_links("").is_empty());
    }

    #[test]
    fn test_wiki_link_extraction_dedup() {
        let body = "Uses [[vault]] and also [[vault]] again.";
        let links = extract_wiki_links(body);
        assert_eq!(links, vec!["vault"]); // deduplicated
    }

    #[test]
    fn test_wiki_link_extraction_unclosed() {
        let body = "Starts [[link but never closes";
        assert!(extract_wiki_links(body).is_empty());
    }

    #[test]
    fn test_wiki_link_extraction_whitespace() {
        let body = "Uses [[ spaced_link ]] here.";
        let links = extract_wiki_links(body);
        assert_eq!(links, vec!["spaced_link"]); // trimmed
    }
}
