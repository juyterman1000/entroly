/// Safety & Critical File Guardrails
///
/// Addresses three weaknesses:
///   1. Entropy scorer drops critical low-entropy files (config, schema, .env)
///   2. Context optimizer can accidentally strip safety signals
///   3. No awareness of file criticality independent of content
///
/// This module implements:
///   - **Critical file patterns**: files that must NEVER be dropped
///   - **Safety signals**: content patterns that force inclusion
///   - **Adaptive budgeting**: task-type-aware token budgets
///   - **Context ordering**: LLM-sensitive fragment ordering
///
/// Critical insight from user feedback:
///   requirements.txt, Dockerfile, .env.example have LOW entropy
///   but HIGH importance. Pure entropy scoring deletes them.
///   We need a separate importance dimension.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Criticality level — overrides entropy and relevance scoring.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Criticality {
    /// Normal fragment — subject to all optimization
    Normal,
    /// Important — boosted score, but can be dropped under pressure
    Important,
    /// Critical — always included (equivalent to pinned)
    Critical,
    /// Safety — NEVER dropped under any circumstances
    Safety,
}

/// Check if a file path matches critical file patterns.
pub fn file_criticality(path: &str) -> Criticality {
    let lower = path.to_lowercase();
    let basename = lower.rsplit('/').next().unwrap_or(&lower);

    // SAFETY: License and security files — never drop
    if matches!(basename,
        "license" | "license.md" | "license.txt"
        | "security.md" | "security.txt"
        | "codeowners"
    ) {
        return Criticality::Safety;
    }

    // CRITICAL: Config and schema files — always include
    if matches!(basename,
        "package.json" | "package-lock.json"
        | "requirements.txt" | "pyproject.toml" | "setup.py" | "setup.cfg"
        | "cargo.toml" | "cargo.lock"
        | "tsconfig.json" | "webpack.config.js" | "vite.config.ts"
        | "dockerfile" | "docker-compose.yml" | "docker-compose.yaml"
        | ".env" | ".env.example" | ".env.local"
        | "makefile" | "cmakelists.txt"
        | "go.mod" | "go.sum"
        | ".gitignore" | ".dockerignore"
    ) {
        return Criticality::Critical;
    }

    // CRITICAL: Schema and type definition files
    if basename.ends_with(".proto")
        || basename.ends_with(".graphql")
        || basename.ends_with(".schema.json")
        || basename.ends_with(".schema.ts")
        || basename.ends_with(".d.ts")
        || basename == "types.rs"
        || basename == "types.py"
        || basename == "types.ts"
        || basename == "models.py"
        || basename == "schema.py"
        || basename == "schema.rs"
    {
        return Criticality::Critical;
    }

    // IMPORTANT: Test files — high value for understanding
    if basename.starts_with("test_")
        || basename.ends_with("_test.rs")
        || basename.ends_with(".test.ts")
        || basename.ends_with(".test.js")
        || basename.ends_with("_spec.rb")
    {
        return Criticality::Important;
    }

    // IMPORTANT: API contracts and interfaces
    if basename.contains("interface")
        || basename.contains("contract")
        || basename.contains("api")
    {
        return Criticality::Important;
    }

    Criticality::Normal
}

/// Check content for safety signals that must never be stripped.
pub fn has_safety_signal(content: &str) -> bool {
    let lower = content.to_lowercase();

    // License headers
    if lower.contains("mit license")
        || lower.contains("apache license")
        || lower.contains("gnu general public")
        || lower.contains("copyright")
        || lower.contains("all rights reserved")
    {
        return true;
    }

    // Security warnings
    if lower.contains("security warning")
        || lower.contains("cve-")
        || lower.contains("vulnerability")
        || lower.contains("do not expose")
        || lower.contains("secret")
        || lower.contains("api_key")
        || lower.contains("private key")
    {
        return true;
    }

    // Sandbox/safety notes
    if lower.contains("unsafe")
        || lower.contains("⚠️")
        || lower.contains("danger")
    {
        return true;
    }

    false
}

/// Compute the criticality boost for a fragment.
/// Returns a multiplier [1.0, 10.0] for the relevance score.
pub fn criticality_boost(criticality: Criticality) -> f64 {
    match criticality {
        Criticality::Normal => 1.0,
        Criticality::Important => 2.0,
        Criticality::Critical => 5.0,
        Criticality::Safety => 10.0,
    }
}

/// Adaptive budget allocation based on task type.
///
/// Different tasks need different context volumes:
///   - Bug tracing: LARGE budget (need call chains, logs, history)
///   - Refactoring: MEDIUM budget (need interfaces, affected files)
///   - Code generation: SMALL budget (need spec + examples)
///   - Code review: MEDIUM budget (need diff + surrounding context)
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    BugTracing,
    Refactoring,
    CodeGeneration,
    CodeReview,
    Exploration,
    Testing,
    Documentation,
    Unknown,
}

impl TaskType {
    /// Classify task type from natural language query.
    pub fn classify(query: &str) -> TaskType {
        let lower = query.to_lowercase();

        if lower.contains("bug") || lower.contains("error") || lower.contains("fail")
            || lower.contains("crash") || lower.contains("fix") || lower.contains("debug")
            || lower.contains("trace") || lower.contains("broken")
        {
            return TaskType::BugTracing;
        }
        if lower.contains("refactor") || lower.contains("rename") || lower.contains("move")
            || lower.contains("extract") || lower.contains("restructure")
        {
            return TaskType::Refactoring;
        }
        if lower.contains("test") || lower.contains("spec") || lower.contains("assert") {
            return TaskType::Testing;
        }
        if lower.contains("review") || lower.contains("audit") || lower.contains("check") {
            return TaskType::CodeReview;
        }
        if lower.contains("create") || lower.contains("implement") || lower.contains("build")
            || lower.contains("add") || lower.contains("write") || lower.contains("generate")
        {
            return TaskType::CodeGeneration;
        }
        if lower.contains("doc") || lower.contains("readme") || lower.contains("comment") {
            return TaskType::Documentation;
        }
        if lower.contains("explore") || lower.contains("understand") || lower.contains("what")
            || lower.contains("how") || lower.contains("why")
        {
            return TaskType::Exploration;
        }
        TaskType::Unknown
    }

    /// Get the recommended budget multiplier for this task type.
    pub fn budget_multiplier(&self) -> f64 {
        match self {
            TaskType::BugTracing => 1.5,     // Need more context
            TaskType::Exploration => 1.3,    // Cast wide net
            TaskType::Refactoring => 1.0,
            TaskType::CodeReview => 1.0,
            TaskType::Testing => 0.8,
            TaskType::CodeGeneration => 0.7, // Need less, more focused
            TaskType::Documentation => 0.6,
            TaskType::Unknown => 1.0,
        }
    }
}

/// Context ordering strategy.
///
/// LLMs are order-sensitive. Fragment ordering affects reasoning quality.
/// We order by: pinned first → critical → high relevance → imports → rest
pub fn compute_ordering_priority(
    relevance: f64,
    criticality: Criticality,
    is_pinned: bool,
    dep_count: usize,
) -> f64 {
    let mut priority = relevance;

    // Pinned: absolute priority
    if is_pinned {
        priority += 100.0;
    }

    // Criticality boost
    priority += match criticality {
        Criticality::Safety => 50.0,
        Criticality::Critical => 30.0,
        Criticality::Important => 10.0,
        Criticality::Normal => 0.0,
    };

    // Fragments with many dependents are "foundation" — put them early
    priority += (dep_count as f64).min(20.0);

    priority
}

/// Feedback loop: record which fragments influenced a successful output.
#[derive(Serialize, Deserialize)]
pub struct FeedbackTracker {
    /// fragment_id → number of times it contributed to a successful output
    success_counts: HashMap<String, u32>,
    /// fragment_id → number of times it was present but output was bad
    failure_counts: HashMap<String, u32>,
}

impl FeedbackTracker {
    pub fn new() -> Self {
        FeedbackTracker {
            success_counts: HashMap::new(),
            failure_counts: HashMap::new(),
        }
    }

    /// Record that these fragments contributed to a successful output.
    pub fn record_success(&mut self, fragment_ids: &[String]) {
        for fid in fragment_ids {
            *self.success_counts.entry(fid.clone()).or_insert(0) += 1;
        }
    }

    /// Record that these fragments were present during a failure.
    pub fn record_failure(&mut self, fragment_ids: &[String]) {
        for fid in fragment_ids {
            *self.failure_counts.entry(fid.clone()).or_insert(0) += 1;
        }
    }

    /// Compute a learned value adjustment for a fragment.
    ///
    /// Returns a multiplier:
    ///   > 1.0 = fragment has been historically useful
    ///   < 1.0 = fragment has been historically unhelpful
    ///   = 1.0 = no data
    pub fn learned_value(&self, fragment_id: &str) -> f64 {
        let successes = *self.success_counts.get(fragment_id).unwrap_or(&0) as f64;
        let failures = *self.failure_counts.get(fragment_id).unwrap_or(&0) as f64;
        let total = successes + failures;

        if total == 0.0 {
            return 1.0;
        }

        // Wilson score lower bound — principled confidence interval
        // (used by Reddit for ranking, here for fragment attribution)
        let p = successes / total;
        let z = 1.96; // 95% confidence
        let denominator = 1.0 + z * z / total;
        let center = p + z * z / (2.0 * total);
        let spread = z * ((p * (1.0 - p) + z * z / (4.0 * total)) / total).sqrt();

        let lower_bound = (center - spread) / denominator;

        // Map [0, 1] → [0.5, 2.0] as a relevance multiplier
        0.5 + lower_bound * 1.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_files() {
        assert_eq!(file_criticality("package.json"), Criticality::Critical);
        assert_eq!(file_criticality("requirements.txt"), Criticality::Critical);
        assert_eq!(file_criticality("Dockerfile"), Criticality::Critical);
        assert_eq!(file_criticality(".env.example"), Criticality::Critical);
        assert_eq!(file_criticality("src/utils.py"), Criticality::Normal);
        assert_eq!(file_criticality("LICENSE"), Criticality::Safety);
    }

    #[test]
    fn test_safety_signals() {
        assert!(has_safety_signal("MIT License\nCopyright 2024"));
        assert!(has_safety_signal("# SECURITY WARNING: do not expose API keys"));
        assert!(!has_safety_signal("def hello(): return 'world'"));
    }

    #[test]
    fn test_task_classification() {
        assert!(matches!(TaskType::classify("fix the payment bug"), TaskType::BugTracing));
        assert!(matches!(TaskType::classify("refactor auth module"), TaskType::Refactoring));
        assert!(matches!(TaskType::classify("create a new API endpoint"), TaskType::CodeGeneration));
    }

    #[test]
    fn test_feedback_tracker() {
        let mut tracker = FeedbackTracker::new();

        // Fragment "a" succeeds 8/10 times
        for _ in 0..8 {
            tracker.record_success(&["a".into()]);
        }
        for _ in 0..2 {
            tracker.record_failure(&["a".into()]);
        }

        // Fragment "b" fails 8/10 times
        for _ in 0..2 {
            tracker.record_success(&["b".into()]);
        }
        for _ in 0..8 {
            tracker.record_failure(&["b".into()]);
        }

        let val_a = tracker.learned_value("a");
        let val_b = tracker.learned_value("b");

        assert!(val_a > val_b, "Successful fragment should be valued higher");
        assert!(val_a > 1.0, "Mostly-successful fragment should boost above 1.0");
    }
}
