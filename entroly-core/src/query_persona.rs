//! Query Persona Manifold — RKHS-based Query Archetype Discovery
//!
//! Adapted from agentOS `persona_manifold.rs` for entroly's query optimization.
//!
//! # Mathematical Framework
//!
//! A query archetype IS a probability measure μ_a ∈ M(H_k) over the query
//! feature space. We represent μ_a via kernel mean embedding in RKHS:
//!
//!   μ̂_a = (1/n) Σᵢ k(xᵢ, ·)   where xᵢ are TF-IDF feature vectors
//!
//! For characteristic kernels (RBF), the embedding is injective:
//!   μ₁ ≠ μ₂ ⟹ μ̂₁ ≠ μ̂₂
//!
//! This captures distributional structure that second-moment methods miss:
//! a bimodal query distribution (debugger who alternates between Rust and Python)
//! is distinguishable from a unimodal one (full-stack developer).
//!
//! # Integration with Entroly
//!
//! 1. `query.rs` produces TF-IDF features + vagueness score → PSM embedding
//! 2. PSM assigns query to an archetype (or births a new one via Pitman-Yor)
//! 3. Each archetype stores learned PRISM weights (Recency, Frequency, Semantic, Entropy)
//! 4. `optimize()` uses per-archetype weights instead of global weights
//!
//! This means "fix the auth bug" queries get different context selection weights
//! than "explain this architecture" queries — automatically discovered, not hardcoded.
//!
//! # Core Operations
//!
//!   1. **Affinity**: A(q, a) = (1/n) Σᵢ k(q, xᵢ)  — which archetype fits this query?
//!   2. **Distance**: MMD²(μ_a, μ_b) = ||μ̂_a - μ̂_b||²_H  — are two archetypes similar?
//!   3. **Birth/Death**: Pitman-Yor process — new archetypes emerge organically
//!   4. **Per-archetype PRISM**: Each archetype learns its own 4D weight vector
//!
use serde::{Serialize, Deserialize};
use crate::prism::PrismOptimizer;

// ════════════════════════════════════════════════════════════════════
//  CONSTANTS
// ════════════════════════════════════════════════════════════════════

/// Dimensionality of query feature vectors.
/// Matches the TF-IDF + vagueness + specificity feature space.
/// We use a compact representation: top-12 TF-IDF scores + 4 meta-features.
const QUERY_DIM: usize = 16;

/// Maximum particles per archetype (Nyström budget).
const MAX_PARTICLES: usize = 100;

/// Maximum number of archetypes before forcing fusion.
const MAX_ARCHETYPES: usize = 8;

/// MMD² threshold for fusion — archetypes closer than this get merged.
const DEFAULT_FUSION_THRESHOLD: f32 = 0.02;

/// Ebbinghaus decay rate for archetype health.
const DECAY_LAMBDA: f32 = 0.005;

/// Health threshold below which an archetype dies.
const DEATH_THRESHOLD: f32 = 0.05;

// ════════════════════════════════════════════════════════════════════
//  RBF KERNEL — k(x,y) = exp(-||x-y||² / (2σ²))
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbfKernel {
    pub sigma: f32,
    inv_2sigma2: f32,
}

impl RbfKernel {
    pub fn new(sigma: f32) -> Self {
        let s = sigma.max(0.01);
        Self { sigma: s, inv_2sigma2: -1.0 / (2.0 * s * s) }
    }

    /// Adaptive bandwidth via Silverman's rule:
    /// σ = 1.06 × median_std × n^(-1/(d+4))
    pub fn silverman(particles: &[Vec<f32>]) -> Self {
        if particles.is_empty() {
            return Self::new(1.0);
        }
        let n = particles.len() as f32;
        let d = particles[0].len() as f32;

        let dim = particles[0].len();
        let mut means = vec![0.0f32; dim];
        for p in particles {
            for (i, &v) in p.iter().enumerate() {
                means[i] += v;
            }
        }
        for m in &mut means { *m /= n; }

        let mut stds = vec![0.0f32; dim];
        for p in particles {
            for (i, &v) in p.iter().enumerate() {
                let diff = v - means[i];
                stds[i] += diff * diff;
            }
        }
        for s in &mut stds { *s = (*s / n.max(1.0)).sqrt(); }

        stds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_std = stds[stds.len() / 2].max(0.01);

        let sigma = 1.06 * median_std * n.powf(-1.0 / (d + 4.0));
        Self::new(sigma.max(0.01))
    }

    /// k(x, y) = exp(-||x-y||² / (2σ²))
    #[inline]
    pub fn eval(&self, x: &[f32], y: &[f32]) -> f32 {
        let sq_dist: f32 = x.iter().zip(y.iter())
            .map(|(a, b)| { let d = a - b; d * d })
            .sum();
        (sq_dist * self.inv_2sigma2).exp()
    }
}

// ════════════════════════════════════════════════════════════════════
//  PARTICLE CLOUD — Empirical measure μ_a ≈ (1/n) Σᵢ δ_{xᵢ}
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleCloud {
    pub particles: Vec<Vec<f32>>,
    pub weights: Vec<f32>,
    pub mean: Vec<f32>,
    pub count: u64,
    pub max_particles: usize,
}

impl ParticleCloud {
    pub fn new(dim: usize, max_particles: usize) -> Self {
        Self {
            particles: Vec::new(),
            weights: Vec::new(),
            mean: vec![0.0; dim],
            count: 0,
            max_particles,
        }
    }

    /// Add a particle. Pruning uses kernel herding: removes the most
    /// redundant particle (highest total affinity to neighbors).
    pub fn add_particle(&mut self, x: &[f32]) {
        let dim = self.mean.len().min(x.len());

        // Welford mean update
        self.count += 1;
        let n = self.count as f32;
        for (i, &xi) in x.iter().enumerate().take(dim) {
            self.mean[i] += (xi - self.mean[i]) / n;
        }

        self.particles.push(x[..dim].to_vec());

        // Kernel herding pruning
        if self.particles.len() > self.max_particles {
            let prune_kernel = RbfKernel::silverman(&self.particles);
            let mut max_redundancy = f32::NEG_INFINITY;
            let mut max_idx = 0;

            for i in 0..self.particles.len() {
                let redundancy: f32 = self.particles.iter().enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, pj)| prune_kernel.eval(&self.particles[i], pj))
                    .sum();
                if redundancy > max_redundancy {
                    max_redundancy = redundancy;
                    max_idx = i;
                }
            }
            self.particles.remove(max_idx);
        }

        // Uniform weights
        let n_p = self.particles.len();
        self.weights = vec![1.0 / n_p as f32; n_p];
    }

    /// Kernel mean embedding affinity: A(t) = Σᵢ wᵢ k(t, xᵢ)
    pub fn affinity(&self, t: &[f32], kernel: &RbfKernel) -> f32 {
        if self.particles.is_empty() { return 0.0; }
        self.particles.iter().zip(self.weights.iter())
            .map(|(xi, wi)| wi * kernel.eval(t, xi))
            .sum()
    }

    /// MMD²(μ_p, μ_q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    pub fn mmd_squared(&self, other: &ParticleCloud, kernel: &RbfKernel) -> f32 {
        if self.particles.is_empty() || other.particles.is_empty() {
            return f32::MAX;
        }

        let mut term1 = 0.0f32;
        for (i, xi) in self.particles.iter().enumerate() {
            for (j, xj) in self.particles.iter().enumerate() {
                term1 += self.weights[i] * self.weights[j] * kernel.eval(xi, xj);
            }
        }

        let mut term2 = 0.0f32;
        for (i, yi) in other.particles.iter().enumerate() {
            for (j, yj) in other.particles.iter().enumerate() {
                term2 += other.weights[i] * other.weights[j] * kernel.eval(yi, yj);
            }
        }

        let mut cross = 0.0f32;
        for (i, xi) in self.particles.iter().enumerate() {
            for (j, yj) in other.particles.iter().enumerate() {
                cross += self.weights[i] * other.weights[j] * kernel.eval(xi, yj);
            }
        }

        term1 + term2 - 2.0 * cross
    }

    /// Effective support size (kernel-weighted entropy).
    pub fn effective_support(&self, kernel: &RbfKernel) -> f32 {
        if self.particles.len() < 2 { return self.particles.len() as f32; }
        let mut entropies = 0.0f32;
        for xi in &self.particles {
            let aff = self.affinity(xi, kernel);
            if aff > 1e-9 {
                entropies -= aff * aff.ln();
            }
        }
        entropies.exp().min(self.particles.len() as f32)
    }

    /// Merge another cloud into this one.
    pub fn merge(&mut self, other: &ParticleCloud) {
        for p in &other.particles {
            self.add_particle(p);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  QUERY ARCHETYPE — A discovered query pattern with its own PRISM weights
// ════════════════════════════════════════════════════════════════════

/// A query archetype = a probability measure over query feature vectors
/// + a learned PRISM weight set for context selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryArchetype {
    pub id: String,
    /// The measure: particle cloud of TF-IDF feature vectors.
    pub cloud: ParticleCloud,
    /// RBF kernel (adaptive bandwidth).
    pub kernel: RbfKernel,
    /// Per-archetype PRISM optimizer — learns specialized weights.
    pub prism: PrismOptimizer,
    /// Per-archetype scoring weights [recency, frequency, semantic, entropy].
    pub weights: [f64; 4],
    /// Pitman-Yor stick weight.
    pub stick_weight: f32,
    /// Creation tick.
    pub born_at: u64,
    /// Last assignment tick.
    pub last_used: u64,
    /// Health (Ebbinghaus decay).
    pub health: f32,
    /// Total successful optimizations using this archetype's weights.
    pub successes: u64,
    /// Total optimizations routed to this archetype.
    pub total_uses: u64,
}

impl QueryArchetype {
    pub fn new(id: String, default_weights: [f64; 4], tick: u64) -> Self {
        Self {
            id,
            cloud: ParticleCloud::new(QUERY_DIM, MAX_PARTICLES),
            kernel: RbfKernel::new(1.0),
            prism: PrismOptimizer::new(0.01),
            weights: default_weights,
            stick_weight: 0.0,
            born_at: tick,
            last_used: tick,
            health: 1.0,
            successes: 0,
            total_uses: 0,
        }
    }

    /// Kernel mean embedding affinity.
    pub fn affinity(&self, query_features: &[f32]) -> f32 {
        self.cloud.affinity(query_features, &self.kernel)
    }

    /// Record a result and update per-archetype PRISM weights.
    pub fn record_result(&mut self, gradient: &[f64; 4], success: bool) {
        self.total_uses += 1;
        if success {
            self.successes += 1;
            self.health = (self.health + 0.05).min(1.0);
        } else {
            self.health = (self.health - 0.02).max(0.0);
        }

        // Apply PRISM update to per-archetype weights
        let step = self.prism.compute_update(gradient);
        for (w, &s) in self.weights.iter_mut().zip(step.iter()) {
            *w = (*w + s).clamp(0.05, 0.60);
        }
        // Normalize weights to sum to 1.0
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Update kernel bandwidth from particles (Silverman's rule).
    pub fn update_kernel(&mut self) {
        if !self.cloud.particles.is_empty() {
            self.kernel = RbfKernel::silverman(&self.cloud.particles);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  PITMAN-YOR PROCESS — Archetype birth/death dynamics
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitmanYorProcess {
    pub alpha: f32,
    pub discount: f32,
    pub total_observations: u64,
}

impl PitmanYorProcess {
    pub fn new(alpha: f32, discount: f32) -> Self {
        Self { alpha, discount, total_observations: 0 }
    }

    /// P(new archetype | K existing, n observations).
    pub fn new_archetype_prob(&self, n_existing: usize) -> f32 {
        let n = self.total_observations as f32;
        let k = n_existing as f32;
        ((self.alpha + k * self.discount) / (n + self.alpha)).clamp(0.01, 0.99)
    }

    /// Weight for existing archetype with n_k observations.
    pub fn existing_weight(&self, n_k: u64) -> f32 {
        let n = self.total_observations as f32;
        ((n_k as f32 - self.discount) / (n + self.alpha)).max(0.001)
    }

    pub fn update_weights(&self, archetypes: &mut [QueryArchetype]) {
        let n = self.total_observations as f32 + self.alpha;
        for a in archetypes.iter_mut() {
            a.stick_weight = ((a.cloud.count as f32 - self.discount) / n).max(0.001);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  QUERY FEATURE EXTRACTION
// ════════════════════════════════════════════════════════════════════

/// Build a QUERY_DIM-dimensional feature vector from TF-IDF key terms
/// and query metadata. This is the embedding that PSM operates on.
///
/// Layout:
///   [0..11]  = top-12 TF-IDF scores (padded with 0.0 if fewer)
///   [12]     = vagueness score
///   [13]     = query length (normalized)
///   [14]     = number of key terms (normalized)
///   [15]     = needs_refinement (0.0 or 1.0)
pub fn build_query_features(
    tfidf_scores: &[f64],
    vagueness: f64,
    query_length: usize,
    num_key_terms: usize,
    needs_refinement: bool,
) -> Vec<f32> {
    let mut features = vec![0.0f32; QUERY_DIM];

    // Top-12 TF-IDF scores
    for (i, &score) in tfidf_scores.iter().take(12).enumerate() {
        features[i] = score as f32;
    }

    // Meta-features
    features[12] = vagueness as f32;
    features[13] = (query_length as f32 / 50.0).min(1.0); // normalize to ~[0,1]
    features[14] = (num_key_terms as f32 / 12.0).min(1.0);
    features[15] = if needs_refinement { 1.0 } else { 0.0 };

    features
}

// ════════════════════════════════════════════════════════════════════
//  QUERY PERSONA MANIFOLD — The main engine
// ════════════════════════════════════════════════════════════════════

/// Discovers query archetypes from TF-IDF feature vectors and maintains
/// per-archetype PRISM weight sets for specialized context selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPersonaManifold {
    archetypes: Vec<QueryArchetype>,
    py_process: PitmanYorProcess,
    id_counter: u64,
    current_tick: u64,
    total_births: u64,
    total_deaths: u64,
    total_fusions: u64,
    fusion_threshold: f32,
    /// Default weights used when birthing new archetypes.
    default_weights: [f64; 4],
    /// Whether the manifold is enabled.
    pub enabled: bool,
}

impl QueryPersonaManifold {
    pub fn new(
        default_weights: [f64; 4],
        alpha: f32,
        discount: f32,
    ) -> Self {
        Self {
            archetypes: Vec::new(),
            py_process: PitmanYorProcess::new(alpha, discount),
            id_counter: 0,
            current_tick: 0,
            total_births: 0,
            total_deaths: 0,
            total_fusions: 0,
            fusion_threshold: DEFAULT_FUSION_THRESHOLD,
            default_weights,
            enabled: true,
        }
    }

    pub fn population(&self) -> usize { self.archetypes.len() }

    /// Assign a query to an archetype. Returns (archetype_id, weights, is_new).
    ///
    /// Algorithm:
    ///   1. Compute kernel mean embedding affinity for each archetype
    ///   2. Weight by Pitman-Yor stick weights
    ///   3. Deterministic comparison: best affinity vs P(new) threshold
    ///   4. If P(new) wins → birth new archetype with default weights
    pub fn assign(&mut self, query_features: &[f32]) -> (String, [f64; 4], bool) {
        self.py_process.total_observations += 1;
        self.current_tick += 1;

        if self.archetypes.is_empty() {
            let id = self.birth_archetype(query_features);
            return (id, self.default_weights, true);
        }

        // Kernel mean embedding affinities
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, a) in self.archetypes.iter().enumerate() {
            let aff = a.affinity(query_features);
            let py_w = self.py_process.existing_weight(a.cloud.count);
            let score = aff * py_w;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        // Deterministic birth decision (no RNG needed for query routing)
        let p_new = self.py_process.new_archetype_prob(self.archetypes.len());

        // Use raw affinity (before PY weighting) for birth decision.
        // High affinity = strong match to existing archetype → don't birth.
        let best_affinity = self.archetypes.iter()
            .map(|a| a.affinity(query_features))
            .fold(f32::NEG_INFINITY, f32::max);

        // Birth if: best affinity is weak AND P(new) is high AND under cap
        if best_affinity < 0.3 && p_new > 0.3 && self.archetypes.len() < MAX_ARCHETYPES {
            let id = self.birth_archetype(query_features);
            (id, self.default_weights, true)
        } else {
            let a = &mut self.archetypes[best_idx];
            let id = a.id.clone();
            let weights = a.weights;
            a.last_used = self.current_tick;
            a.cloud.add_particle(query_features);
            // Periodically update kernel bandwidth
            if a.cloud.count.is_multiple_of(10) {
                a.update_kernel();
            }
            (id, weights, false)
        }
    }

    /// Record optimization result for an archetype.
    pub fn record_result(
        &mut self,
        archetype_id: &str,
        gradient: &[f64; 4],
        success: bool,
    ) {
        if let Some(a) = self.archetypes.iter_mut().find(|a| a.id == archetype_id) {
            a.record_result(gradient, success);
        }
        self.py_process.update_weights(&mut self.archetypes);
    }

    /// Get weights for a specific archetype. Returns default if not found.
    pub fn get_weights(&self, archetype_id: &str) -> [f64; 4] {
        self.archetypes.iter()
            .find(|a| a.id == archetype_id)
            .map(|a| a.weights)
            .unwrap_or(self.default_weights)
    }

    /// Lifecycle tick: Ebbinghaus decay + death + MMD fusion.
    pub fn lifecycle_tick(&mut self) {
        self.current_tick += 1;

        // Ebbinghaus decay
        for a in &mut self.archetypes {
            let dt = (self.current_tick - a.last_used) as f32;
            a.health *= (-DECAY_LAMBDA * dt).exp();
        }

        // Death
        let before = self.archetypes.len();
        self.archetypes.retain(|a| a.health >= DEATH_THRESHOLD);
        self.total_deaths += (before - self.archetypes.len()) as u64;

        // MMD-based fusion
        self.check_fusion();
    }

    /// Return statistics about the manifold.
    pub fn stats(&self) -> ManifoldStats {
        let total_particles: usize = self.archetypes.iter()
            .map(|a| a.cloud.particles.len()).sum();

        ManifoldStats {
            population: self.archetypes.len(),
            total_births: self.total_births,
            total_deaths: self.total_deaths,
            total_fusions: self.total_fusions,
            tick: self.current_tick,
            total_particles,
            archetypes: self.archetypes.iter().map(|a| ArchetypeInfo {
                id: a.id.clone(),
                health: a.health,
                particles: a.cloud.particles.len(),
                observations: a.cloud.count,
                stick_weight: a.stick_weight,
                effective_support: a.cloud.effective_support(&a.kernel),
                weights: a.weights,
                successes: a.successes,
                total_uses: a.total_uses,
                success_rate: if a.total_uses > 0 {
                    a.successes as f64 / a.total_uses as f64
                } else { 0.0 },
            }).collect(),
        }
    }

    // ── Internal ──

    fn birth_archetype(&mut self, query_features: &[f32]) -> String {
        self.id_counter += 1;
        let id = format!("qa_{:04}", self.id_counter);
        let mut archetype = QueryArchetype::new(
            id.clone(),
            self.default_weights,
            self.current_tick,
        );
        archetype.cloud.add_particle(query_features);
        self.archetypes.push(archetype);
        self.total_births += 1;
        id
    }

    /// Fusion via MMD² — proper metric on probability measures.
    fn check_fusion(&mut self) {
        if self.archetypes.len() < 2 { return; }

        let mut to_absorb: Vec<usize> = Vec::new();
        for i in 0..self.archetypes.len() {
            if to_absorb.contains(&i) { continue; }
            for j in (i+1)..self.archetypes.len() {
                if to_absorb.contains(&j) { continue; }

                let kernel = if self.archetypes[i].kernel.sigma > self.archetypes[j].kernel.sigma {
                    &self.archetypes[i].kernel
                } else {
                    &self.archetypes[j].kernel
                };

                let mmd2 = self.archetypes[i].cloud
                    .mmd_squared(&self.archetypes[j].cloud, kernel);

                if mmd2 < self.fusion_threshold {
                    // Merge j into i — keep the one with more observations
                    let j_cloud = self.archetypes[j].cloud.clone();
                    self.archetypes[i].cloud.merge(&j_cloud);
                    self.archetypes[i].health =
                        self.archetypes[i].health.max(self.archetypes[j].health);

                    // Merge weights: weighted average by observation count
                    let n_i = self.archetypes[i].total_uses as f64;
                    let n_j = self.archetypes[j].total_uses as f64;
                    let total = (n_i + n_j).max(1.0);
                    for k in 0..4 {
                        self.archetypes[i].weights[k] =
                            (self.archetypes[i].weights[k] * n_i
                             + self.archetypes[j].weights[k] * n_j) / total;
                    }

                    self.archetypes[i].successes += self.archetypes[j].successes;
                    self.archetypes[i].total_uses += self.archetypes[j].total_uses;
                    to_absorb.push(j);
                    self.total_fusions += 1;
                }
            }
        }

        to_absorb.sort_unstable();
        for &idx in to_absorb.iter().rev() {
            self.archetypes.remove(idx);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  STATS TYPES
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct ManifoldStats {
    pub population: usize,
    pub total_births: u64,
    pub total_deaths: u64,
    pub total_fusions: u64,
    pub tick: u64,
    pub total_particles: usize,
    pub archetypes: Vec<ArchetypeInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchetypeInfo {
    pub id: String,
    pub health: f32,
    pub particles: usize,
    pub observations: u64,
    pub stick_weight: f32,
    pub effective_support: f32,
    pub weights: [f64; 4],
    pub successes: u64,
    pub total_uses: u64,
    pub success_rate: f64,
}

// ════════════════════════════════════════════════════════════════════
//  TESTS
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(primary: usize, vagueness: f32) -> Vec<f32> {
        let mut f = vec![0.0f32; QUERY_DIM];
        if primary < 12 { f[primary] = 0.9; }
        f[12] = vagueness;
        f[13] = 0.5;
        f[14] = 0.5;
        f[15] = if vagueness > 0.5 { 1.0 } else { 0.0 };
        f
    }

    fn default_weights() -> [f64; 4] {
        [0.30, 0.25, 0.25, 0.20]
    }

    #[test]
    fn test_rbf_kernel_self() {
        let k = RbfKernel::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        assert!((k.eval(&x, &x) - 1.0).abs() < 1e-6, "k(x,x) = 1 for RBF");
    }

    #[test]
    fn test_rbf_kernel_distance() {
        let k = RbfKernel::new(1.0);
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];
        let z = vec![10.0, 0.0];
        assert!(k.eval(&x, &y) > k.eval(&x, &z), "Closer = higher kernel");
    }

    #[test]
    fn test_birth_on_empty() {
        let mut m = QueryPersonaManifold::new(default_weights(), 1.0, 0.25);
        let (id, weights, is_new) = m.assign(&make_features(0, 0.3));
        assert!(is_new);
        assert_eq!(m.population(), 1);
        assert!(id.starts_with("qa_"));
        assert_eq!(weights, default_weights());
    }

    #[test]
    fn test_repeated_assignment_converges() {
        let mut m = QueryPersonaManifold::new(default_weights(), 1.0, 0.25);
        let features = make_features(0, 0.3);
        // Seed a few observations so P(new) drops
        for _ in 0..5 {
            m.assign(&features);
        }
        // After seeding, repeated identical queries should go to existing archetypes
        let pop_before = m.population();
        for _ in 0..20 {
            m.assign(&features);
        }
        // Population should not have grown much (at most 1 more archetype)
        assert!(m.population() <= pop_before + 1,
            "Identical queries should converge: pop went from {} to {}", pop_before, m.population());
    }

    #[test]
    fn test_different_queries_can_birth() {
        let mut m = QueryPersonaManifold::new(default_weights(), 2.0, 0.1);
        // Very different feature vectors should spawn new archetypes
        m.assign(&make_features(0, 0.1));
        m.assign(&make_features(5, 0.9));
        m.assign(&make_features(10, 0.5));
        // Should have created at least 2 archetypes
        assert!(m.population() >= 2, "Different queries should birth archetypes: pop={}", m.population());
    }

    #[test]
    fn test_death_by_decay() {
        let mut m = QueryPersonaManifold::new(default_weights(), 1.0, 0.25);
        m.assign(&make_features(0, 0.3));
        assert_eq!(m.population(), 1);
        for _ in 0..1000 { m.lifecycle_tick(); }
        assert_eq!(m.population(), 0, "Should die from Ebbinghaus decay");
    }

    #[test]
    fn test_fusion_identical_archetypes() {
        let mut m = QueryPersonaManifold::new(default_weights(), 1.0, 0.25);
        m.fusion_threshold = 10.0; // Very high threshold → forces fusion

        // Force-create two archetypes with identical particles
        m.birth_archetype(&make_features(0, 0.3));
        m.birth_archetype(&make_features(0, 0.3));
        assert_eq!(m.population(), 2);

        m.lifecycle_tick();
        assert!(m.population() <= 1, "Identical measures should fuse");
    }

    #[test]
    fn test_per_archetype_weight_learning() {
        let mut m = QueryPersonaManifold::new(default_weights(), 1.0, 0.25);
        let (id, _, _) = m.assign(&make_features(0, 0.3));

        // Simulate feedback: strong recency gradient
        for _ in 0..20 {
            m.record_result(&id, &[0.5, 0.0, 0.0, 0.0], true);
        }

        let weights = m.get_weights(&id);
        // Recency weight should have increased relative to others
        assert!(weights[0] > 0.25, "Recency weight should increase with positive gradient: {:.3}", weights[0]);
    }

    #[test]
    fn test_mmd_identical_clouds() {
        let k = RbfKernel::new(1.0);
        let mut c1 = ParticleCloud::new(4, 50);
        let mut c2 = ParticleCloud::new(4, 50);
        let pts = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        for p in &pts { c1.add_particle(p); c2.add_particle(p); }
        let mmd2 = c1.mmd_squared(&c2, &k);
        assert!(mmd2 < 1e-4, "Identical clouds → MMD² ≈ 0: {}", mmd2);
    }

    #[test]
    fn test_build_query_features() {
        let tfidf = vec![0.5, 0.3, 0.1];
        let features = build_query_features(&tfidf, 0.7, 10, 3, true);
        assert_eq!(features.len(), QUERY_DIM);
        assert!((features[0] - 0.5).abs() < 1e-6);
        assert!((features[12] - 0.7).abs() < 1e-6);
        assert!((features[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manifold_stats() {
        let mut m = QueryPersonaManifold::new(default_weights(), 1.0, 0.25);
        m.assign(&make_features(0, 0.3));
        m.assign(&make_features(5, 0.8));
        let stats = m.stats();
        assert!(stats.population >= 1);
        assert!(stats.total_births >= 1);
        assert!(!stats.archetypes.is_empty());
    }

    #[test]
    fn test_max_archetypes_cap() {
        let mut m = QueryPersonaManifold::new(default_weights(), 100.0, 0.0);
        // High alpha = high P(new), try to birth many
        for i in 0..20 {
            m.assign(&make_features(i % 12, i as f32 / 20.0));
        }
        assert!(m.population() <= MAX_ARCHETYPES,
            "Should cap at {} archetypes: pop={}", MAX_ARCHETYPES, m.population());
    }

    #[test]
    fn test_pitman_yor_probability() {
        let py = PitmanYorProcess::new(1.0, 0.25);
        let p0 = py.new_archetype_prob(0);
        assert!(p0 > 0.9, "First query → high P(new): {}", p0);

        let mut py2 = PitmanYorProcess::new(1.0, 0.25);
        py2.total_observations = 100;
        let p100 = py2.new_archetype_prob(5);
        assert!(p100 < 0.1, "Many obs → low P(new): {}", p100);
    }
}
