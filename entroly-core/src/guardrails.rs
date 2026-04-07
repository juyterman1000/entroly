//! Safety & Critical File Guardrails
//!
//! Addresses three weaknesses:
//!   1. Entropy scorer drops critical low-entropy files (config, schema, .env)
//!   2. Context optimizer can accidentally strip safety signals
//!   3. No awareness of file criticality independent of content
//!
//! This module implements:
//!   - **Critical file patterns**: files that must NEVER be dropped
//!   - **Safety signals**: content patterns that force inclusion
//!   - **Adaptive budgeting**: task-type-aware token budgets
//!   - **Context ordering**: LLM-sensitive fragment ordering
//!
//! Critical insight from user feedback:
//!   requirements.txt, Dockerfile, .env.example have LOW entropy
//!   but HIGH importance. Pure entropy scoring deletes them.
//!   We need a separate importance dimension.

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

/// Count directory depth: how many path separators precede the basename.
/// "file:package.json" → 0, "file:src/types.ts" → 1, "file:a/b/c.ts" → 2
fn path_depth(path: &str) -> usize {
    // Strip common prefixes like "file:" before counting
    let clean = path.strip_prefix("file:").unwrap_or(path);
    clean.matches('/').count() + clean.matches('\\').count()
}

/// Check if a file path matches critical file patterns.
///
/// Monorepo-aware: in monorepos (langfuse, turborepo, nx), config files like
/// `package.json`, `tsconfig.json`, `types.ts` appear in *every* sub-package.
/// Pinning all of them blows the budget (langfuse: 90 pinned = 167K tokens,
/// budget = 8K). Only ROOT-level configs are Critical; nested copies are
/// Important (score-boosted but budget-constrained).
pub fn file_criticality(path: &str) -> Criticality {
    let lower = path.to_lowercase();
    let basename = lower.rsplit('/').next()
        .and_then(|b| b.rsplit('\\').next())
        .unwrap_or(&lower);
    let depth = path_depth(&lower);
    let is_root = depth <= 1; // root or one level deep (file:package.json)

    // SAFETY: License and security files — never drop (any depth)
    if matches!(basename,
        "license" | "license.md" | "license.txt"
        | "security.md" | "security.txt"
        | "codeowners"
    ) {
        return Criticality::Safety;
    }

    // SAFETY: Terraform state files — NEVER drop (contain real infra state)
    if basename == "terraform.tfstate" || basename == "terraform.tfstate.backup" {
        return Criticality::Safety;
    }

    // ── Root-only Critical configs ──
    // These are Critical ONLY at the project root. Nested copies in monorepo
    // sub-packages are Important (score-boosted, not force-pinned).
    let is_root_config = matches!(basename,
        "package.json" | "package-lock.json"
        | "requirements.txt" | "pyproject.toml" | "setup.py" | "setup.cfg"
        | "cargo.toml" | "cargo.lock"
        | "tsconfig.json"
        | "docker-compose.yml" | "docker-compose.yaml"
        | "makefile" | "cmakelists.txt"
        | "go.mod" | "go.sum"
        | ".gitignore" | ".dockerignore"
        | ".env" | ".env.example" | ".env.local"
        // Terraform / IaC
        | "main.tf" | "variables.tf" | "outputs.tf" | "providers.tf"
        | "terraform.tfvars" | "backend.tf"
        // Helm / K8s
        | "chart.yaml" | "values.yaml" | "values.yml"
        | "kustomization.yaml" | "kustomization.yml"
        // C# / .NET
        | "program.cs" | "startup.cs" | "appsettings.json"
        // Swift / iOS
        | "package.swift" | "podfile" | "cartfile"
    );
    if is_root_config {
        return if is_root { Criticality::Critical } else { Criticality::Important };
    }

    // ── Build tool configs: Critical at root, Important nested ──
    let is_build_config = matches!(basename,
        "webpack.config.js" | "webpack.config.ts"
        | "vite.config.ts" | "vite.config.js" | "vite.config.mts"
        | "next.config.js" | "next.config.mjs" | "next.config.ts" | "next.config.mts"
        | "angular.json" | "nuxt.config.ts" | "nuxt.config.js" | "nuxt.config.mjs"
        | "remix.config.js" | "remix.config.ts"
        | "svelte.config.js" | "svelte.config.ts" | "astro.config.mjs"
        | "tailwind.config.js" | "tailwind.config.ts" | "tailwind.config.mjs"
        | "app.vue"
        | "gatsby-config.js" | "gatsby-config.ts"
        | "postcss.config.js" | "postcss.config.mjs"
        | "jest.config.js" | "jest.config.ts"
        | "vitest.config.ts" | "vitest.config.js"
        | "babel.config.js" | "babel.config.json" | ".babelrc"
        | ".eslintrc.js" | ".eslintrc.json" | ".eslintrc.cjs"
        | "eslint.config.js" | "eslint.config.mjs"
        | "prettier.config.js" | ".prettierrc" | ".prettierrc.json"
        | "dockerfile"
    );
    if is_build_config {
        return if is_root { Criticality::Critical } else { Criticality::Important };
    }

    // ── Schema/type files: ALWAYS Important (never Critical) ──
    // In monorepos, types.ts / models.py appear in dozens of packages.
    // Pinning all of them destroys the budget. Score-boost is sufficient.
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
        return Criticality::Important;
    }

    // Terraform / HCL files (not root-level ones already handled above)
    if basename.ends_with(".tf") || basename.ends_with(".hcl") {
        return if is_root { Criticality::Critical } else { Criticality::Important };
    }

    // IMPORTANT: Test files — high value for understanding
    if basename.starts_with("test_")
        || basename.ends_with("_test.rs")
        || basename.ends_with("_test.go")
        || basename.ends_with(".test.ts")
        || basename.ends_with(".test.tsx")
        || basename.ends_with(".test.js")
        || basename.ends_with(".test.jsx")
        || basename.ends_with(".spec.ts")
        || basename.ends_with(".spec.tsx")
        || basename.ends_with(".spec.js")
        || basename.ends_with(".spec.jsx")
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
///
/// IMPORTANT: This must be TIGHT. Every match here force-pins the fragment,
/// bypassing the budget. Broad patterns like "copyright" or "⚠️" caused
/// hundreds of files to be pinned in real codebases, destroying budget
/// enforcement entirely (langfuse: 90 pinned files = 167K tokens on 8K budget).
///
/// Rule: only pin content that would be DANGEROUS to omit (active secrets,
/// private keys). License headers and copyright notices are boilerplate —
/// they get a score boost via Criticality::Important, not a force-pin.
pub fn has_safety_signal(content: &str) -> bool {
    let lower = content.to_lowercase();

    // Actual credential material — dangerous to silently strip
    if lower.contains("-----begin private key-----")
        || lower.contains("-----begin rsa private key-----")
        || lower.contains("aws_secret_access_key")
    {
        return true;
    }

    // Secret-like assignments in real code (not comments/docs)
    for line in lower.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#')
            || trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with('*')
            || trimmed.starts_with('-')
            || trimmed.starts_with("\"\"\"")
            || trimmed.starts_with("'''")
        {
            continue;
        }
        if trimmed.contains('=')
            && (trimmed.contains("secret_key")
                || trimmed.contains("api_key")
                || trimmed.contains("private_key"))
        {
            return true;
        }
    }

    false
}

/// Compute the criticality boost for a fragment.
/// Returns a multiplier [1.0, 10.0] for the relevance score.
#[allow(dead_code)]
pub(crate) fn criticality_boost(criticality: Criticality) -> f64 {
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
///
/// Extended with per-fragment Welford variance tracking for RAVEN-UCB
/// adaptive exploration (arXiv:2506.02933, June 2025).
#[derive(Serialize, Deserialize)]
pub struct FeedbackTracker {
    /// fragment_id → number of times it contributed to a successful output
    success_counts: HashMap<String, u32>,
    /// fragment_id → number of times it was present but output was bad
    failure_counts: HashMap<String, u32>,
    // ── RAVEN-UCB: Per-fragment Welford variance tracking ──
    /// fragment_id → total selection count in optimize()
    #[serde(default)]
    visit_counts: HashMap<String, u32>,
    /// fragment_id → Welford running mean of reward
    #[serde(default)]
    welford_means: HashMap<String, f64>,
    /// fragment_id → Welford M₂ accumulator (var = M₂/(n-1))
    #[serde(default)]
    welford_m2s: HashMap<String, f64>,
    /// Total optimization calls (global clock t for α_t decay)
    #[serde(default)]
    total_observations: u64,
}

impl FeedbackTracker {
    pub fn new() -> Self {
        FeedbackTracker {
            success_counts: HashMap::new(),
            failure_counts: HashMap::new(),
            visit_counts: HashMap::new(),
            welford_means: HashMap::new(),
            welford_m2s: HashMap::new(),
            total_observations: 0,
        }
    }

    /// Record that these fragments contributed to a successful output.
    pub fn record_success(&mut self, fragment_ids: &[String]) {
        for fid in fragment_ids {
            *self.success_counts.entry(fid.clone()).or_insert(0) += 1;
        }
        self.record_reward_signal(fragment_ids, 1.0);
    }

    /// Record that these fragments were present during a failure.
    pub fn record_failure(&mut self, fragment_ids: &[String]) {
        for fid in fragment_ids {
            *self.failure_counts.entry(fid.clone()).or_insert(0) += 1;
        }
        self.record_reward_signal(fragment_ids, 0.0);
    }

    /// Welford online variance update — O(1) per fragment.
    fn record_reward_signal(&mut self, fragment_ids: &[String], reward: f64) {
        self.total_observations += 1;
        for fid in fragment_ids {
            let n = self.visit_counts.entry(fid.clone()).or_insert(0);
            *n += 1;
            let count = *n as f64;
            let mean = self.welford_means.entry(fid.clone()).or_insert(0.0);
            let m2 = self.welford_m2s.entry(fid.clone()).or_insert(0.0);
            let delta1 = reward - *mean;
            *mean += delta1 / count;
            let delta2 = reward - *mean;
            *m2 += delta1 * delta2;
        }
    }

    /// Sample variance (Welford's M₂/(n-1)). Returns 1.0 for unseen fragments.
    pub fn variance(&self, fragment_id: &str) -> f64 {
        let n = *self.visit_counts.get(fragment_id).unwrap_or(&0);
        if n < 2 { return 1.0; }
        let m2 = *self.welford_m2s.get(fragment_id).unwrap_or(&0.0);
        (m2 / (n as f64 - 1.0)).clamp(0.001, 1.0)
    }

    /// Visit count for a fragment.
    pub fn visit_count(&self, fragment_id: &str) -> u32 {
        *self.visit_counts.get(fragment_id).unwrap_or(&0)
    }

    /// Welford mean reward for a fragment.
    pub fn welford_mean(&self, fragment_id: &str) -> f64 {
        *self.welford_means.get(fragment_id).unwrap_or(&0.5)
    }

    /// Current annealed exploration coefficient α_t for RAVEN-UCB.
    pub fn adaptive_exploration_rate(&self, alpha_0: f64) -> f64 {
        let t = self.total_observations as f64;
        alpha_0 / (t + std::f64::consts::E).ln()
    }

    /// RAVEN-UCB score: UCB_i = μ_i + α_t · √(σ²_i / (n_i + 1))
    /// where α_t = α₀ / ln(t + e) self-anneals exploration.
    ///
    /// Reference: arXiv:2506.02933, June 2025.
    pub fn ucb_score(&self, fragment_id: &str, alpha_0: f64) -> f64 {
        let mu = self.welford_mean(fragment_id);
        let sigma2 = self.variance(fragment_id);
        let n = self.visit_count(fragment_id) as f64;
        let alpha_t = self.adaptive_exploration_rate(alpha_0);
        mu + alpha_t * (sigma2 / (n + 1.0)).sqrt()
    }

    /// Total optimization calls (global clock).
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Compute a learned value adjustment for a fragment.
    ///
    /// Returns a multiplier:
    /// - > 1.0 = fragment has been historically useful
    /// - < 1.0 = fragment has been historically unhelpful
    /// - = 1.0 = no data
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
        // Real credential material — always pinned
        assert!(has_safety_signal("-----BEGIN PRIVATE KEY-----\nabc"));
        assert!(has_safety_signal("-----BEGIN RSA PRIVATE KEY-----\nabc"));
        assert!(has_safety_signal("AWS_SECRET_ACCESS_KEY=AKIA123"));
        assert!(has_safety_signal("SECRET_KEY = \"replace-me\""));
        assert!(has_safety_signal("api_key = \"sk-live-abc123\""));
        // License/copyright handled via file_criticality, not content scan
        assert!(!has_safety_signal("MIT License\nCopyright 2024"));
        // Normal code — never pinned
        assert!(!has_safety_signal("def hello(): return 'world'"));
        assert!(!has_safety_signal("fn dangerous() { unsafe { do_work(); } }"));
        // Comment mentioning api_key — not a real assignment
        assert!(!has_safety_signal("This vulnerability scanner checks api_key patterns."));
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

    #[test]
    fn test_ucb_variance_tracking() {
        let mut tracker = FeedbackTracker::new();
        let fragment = vec!["x".to_string()];
        let rewards = [1.0, 0.0, 1.0, 1.0, 0.0];

        for reward in rewards {
            if reward > 0.0 {
                tracker.record_success(&fragment);
            } else {
                tracker.record_failure(&fragment);
            }
        }

        let batch_mean = 3.0 / 5.0;
        let batch_rewards: [f64; 5] = [1.0, 0.0, 1.0, 1.0, 0.0];
        let batch_variance = batch_rewards
            .iter()
            .map(|value| (*value - batch_mean).powi(2))
            .sum::<f64>()
            / 4.0;

        assert!((tracker.welford_mean("x") - batch_mean).abs() < 1e-9);
        assert!((tracker.variance("x") - batch_variance).abs() < 1e-9);
    }

    #[test]
    fn test_raven_ucb_convergence() {
        let mut tracker = FeedbackTracker::new();
        let known = vec!["known".to_string()];

        for _ in 0..5 {
            tracker.record_success(&known);
        }

        let early_rate = tracker.adaptive_exploration_rate(2.0);
        let early_known_ucb = tracker.ucb_score("known", 2.0);
        let early_uncertain_ucb = tracker.ucb_score("uncertain", 2.0);
        assert!(early_uncertain_ucb > early_known_ucb);

        for _ in 0..500 {
            tracker.record_success(&known);
        }

        let late_rate = tracker.adaptive_exploration_rate(2.0);
        let late_known_ucb = tracker.ucb_score("known", 2.0);
        let late_uncertain_ucb = tracker.ucb_score("uncertain", 2.0);

        assert!(late_rate < early_rate);
        assert!(late_rate < 0.4);
        assert!(late_uncertain_ucb < late_known_ucb);
    }
}
