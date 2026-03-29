//! PRISM — Anisotropic Spectral Optimizer (Generalized N-Dimensional)
//!
//! Implements quasi-second-order RL weight tuning using running covariance
//! and Eigenvalue Decomposition, adapted from "PRISM: Structured Optimization
//! via Anisotropic Spectral Shaping".
//!
//! Instead of isotropic (scalar) learning rates, this tracks the NxN covariance
//! matrix of the feature gradients. It computes the eigendecomposition
//! $C = Q \Lambda Q^T$ and applies anisotropic damping in high-variance
//! (noisy) sub-spaces: $w_{t+1} = w_t + \alpha Q \Lambda^{-1/2} Q^T g$.
//!
//! The implementation is generic over dimension N using const generics.
//! - N=4: Original 4D (Recency, Frequency, Semantic, Entropy)
//! - N=5: Extended 5D (+ Resonance — pairwise fragment interaction weight)
//!
//! Because our state space is small (4–5D), we perform exact eigendecomposition
//! (Jacobi method) rather than the approximate polar decomposition needed for
//! 100M+ parameter neural networks.

use std::f64;
use serde::{Serialize, Deserialize};

// ════════════════════════════════════════════════════════════════════
//  SYMMETRIC MATRIX — NxN with const generic dimension
// ════════════════════════════════════════════════════════════════════

/// An NxN symmetric matrix stored as a flat array for tracking gradient covariance.
///
/// Uses `N * N` flat layout (row-major) since const generic expressions like
/// `[f64; N * N]` require nightly features. The accessor methods ensure
/// symmetric access patterns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymMatrixN<const N: usize> {
    /// Flat row-major storage: element (i,j) lives at index i*N + j.
    pub data: Vec<f64>,
}

impl<const N: usize> SymMatrixN<N> {
    pub fn new() -> Self {
        SymMatrixN { data: vec![0.0; N * N] }
    }

    pub fn identity() -> Self {
        let mut m = Self::new();
        for i in 0..N { m.set(i, i, 1.0); }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * N + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * N + j] = val;
    }

    /// Update running covariance: $C = \beta C + (1-\beta) g g^T$
    pub fn update_ema(&mut self, g: &[f64], beta: f64) {
        debug_assert!(g.len() >= N);
        for i in 0..N {
            for j in 0..N {
                let idx = i * N + j;
                self.data[idx] = beta * self.data[idx] + (1.0 - beta) * (g[i] * g[j]);
            }
        }
    }

    /// Computes Eigenvalue Decomposition $C = Q \Lambda Q^T$ using the Cyclic Jacobi method.
    /// Returns (Q, Eigenvalues), where Q columns are eigenvectors.
    ///
    /// Complexity: O(N² × max_iters) — negligible for N ≤ 5.
    pub fn jacobi_eigendecomposition(&self) -> (SymMatrixN<N>, Vec<f64>) {
        let mut a = self.clone();
        let mut q = Self::identity();
        let max_iters = 100; // More headroom for 5D convergence
        let eps = 1e-9;

        for _ in 0..max_iters {
            // Find max off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut r = 1;
            for i in 0..(N - 1) {
                for j in (i + 1)..N {
                    let val = a.get(i, j).abs();
                    if val > max_val {
                        max_val = val;
                        p = i;
                        r = j;
                    }
                }
            }

            if max_val < eps {
                break; // Converged
            }

            // Compute Jacobi rotation angle
            let app = a.get(p, p);
            let arr = a.get(r, r);
            let apr = a.get(p, r);
            let theta = 0.5 * (2.0 * apr / (app - arr + 1e-15)).atan();
            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation A' = J^T A J
            for i in 0..N {
                if i != p && i != r {
                    let aip = a.get(i, p);
                    let air = a.get(i, r);
                    let new_ip = c * aip - s * air;
                    let new_ir = s * aip + c * air;
                    a.set(i, p, new_ip);
                    a.set(p, i, new_ip);
                    a.set(i, r, new_ir);
                    a.set(r, i, new_ir);
                }
            }
            a.set(p, p, c * c * app - 2.0 * s * c * apr + s * s * arr);
            a.set(r, r, s * s * app + 2.0 * s * c * apr + c * c * arr);
            a.set(p, r, 0.0);
            a.set(r, p, 0.0);

            // Apply rotation Q' = Q J
            for i in 0..N {
                let qip = q.get(i, p);
                let qir = q.get(i, r);
                q.set(i, p, c * qip - s * qir);
                q.set(i, r, s * qip + c * qir);
            }
        }

        let eigenvalues: Vec<f64> = (0..N).map(|i| a.get(i, i)).collect();
        (q, eigenvalues)
    }
}

// ════════════════════════════════════════════════════════════════════
//  TYPE ALIASES — preserve the original 4D API
// ════════════════════════════════════════════════════════════════════

/// Original 4D matrix (Recency, Frequency, Semantic, Entropy).
pub type SymMatrix4 = SymMatrixN<4>;

/// Extended 5D matrix (Recency, Frequency, Semantic, Entropy, Resonance).
#[allow(dead_code)]
pub type SymMatrix5 = SymMatrixN<5>;

// ════════════════════════════════════════════════════════════════════
//  BACKWARD-COMPATIBLE SymMatrix4 CONSTRUCTION
// ════════════════════════════════════════════════════════════════════

impl SymMatrix4 {
    /// Construct from a fixed 4x4 array (backward compat with old checkpoint format).
    pub fn from_array(data: [[f64; 4]; 4]) -> Self {
        let mut flat = vec![0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                flat[i * 4 + j] = data[i][j];
            }
        }
        SymMatrixN { data: flat }
    }

    /// Export as a fixed 4x4 array (backward compat).
    pub fn to_array(&self) -> [[f64; 4]; 4] {
        let mut out = [[0.0; 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.data[i * 4 + j];
            }
        }
        out
    }
}

// ════════════════════════════════════════════════════════════════════
//  PRISM OPTIMIZER — Generic N-Dimensional
// ════════════════════════════════════════════════════════════════════

/// Anisotropic Spectral Optimizer (PRISM-lite for N-Dimensional Context Weights)
///
/// Dimension semantics:
///   [0] Recency   — how recently was the fragment accessed?
///   [1] Frequency — how often is the fragment accessed?
///   [2] Semantic  — how similar is the fragment to the query?
///   [3] Entropy   — how information-dense is the fragment?
///   [4] Resonance — how much does this fragment amplify co-selected fragments? (5D only)
///
/// The resonance dimension captures **supermodular interaction effects**:
/// certain fragment pairs produce dramatically better LLM outputs than the sum
/// of their individual contributions. PRISM learns to weight this signal
/// relative to the four individual-fragment dimensions.
///
/// Key insight: resonance gradients are inherently noisier than individual
/// scores (they depend on which other fragments are co-selected, introducing
/// combinatorial variance). PRISM's anisotropic damping naturally handles this:
/// if dimension 4 has high gradient variance, Λ⁻¹/² shrinks its learning rate
/// automatically — no manual tuning needed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrismOptimizerN<const N: usize> {
    pub covariance: SymMatrixN<N>,
    pub beta: f64,
    pub learning_rate: f64,
    pub epsilon: f64,
}

impl<const N: usize> PrismOptimizerN<N> {
    pub fn new(learning_rate: f64) -> Self {
        let mut cov = SymMatrixN::<N>::identity();
        // Initialize with small epsilon identity to prevent division by zero
        for i in 0..N { cov.set(i, i, 1e-4); }
        PrismOptimizerN {
            covariance: cov,
            beta: 0.95,
            learning_rate,
            epsilon: 1e-6,
        }
    }

    /// Applies Anisotropic Spectral Gain to a gradient vector.
    /// Computes $P g = Q \Lambda^{-1/2} Q^T g$ and returns the update $\Delta w = \alpha P g$.
    ///
    /// For the 5D case, the resonance dimension (index 4) typically shows higher
    /// gradient variance than dimensions 0–3 because pairwise interaction signals
    /// are noisier. The spectral shaping automatically dampens this:
    ///
    ///   λ₄ >> λ₀..₃  →  λ₄⁻¹/² << λ₀..₃⁻¹/²  →  resonance gets smaller steps
    ///
    /// This is the core value of PRISM over isotropic Adam: it doesn't need a
    /// hand-tuned learning rate for resonance vs. individual scores.
    pub fn compute_update(&mut self, g: &[f64]) -> Vec<f64> {
        debug_assert!(g.len() >= N, "gradient must have at least {N} dimensions, got {}", g.len());

        // 1. Update running covariance
        self.covariance.update_ema(g, self.beta);

        // 2. Eigendecomposition: C = Q Λ Q^T
        let (q, eigenvalues) = self.covariance.jacobi_eigendecomposition();

        // 3. Spectral Shaping: Λ^{-1/2}
        // Dampens high-variance (noisy) directions, boosts clean signals.
        let lambda_inv_sqrt: Vec<f64> = eigenvalues.iter()
            .map(|&ev| 1.0 / (ev.abs() + self.epsilon).sqrt())
            .collect();

        // 4. Compute Q Λ^{-1/2} Q^T g
        // Project gradient into eigenspace: v = Q^T g
        let mut v: Vec<f64> = (0..N)
            .map(|i| (0..N).map(|j| q.get(j, i) * g[j]).sum())
            .collect();

        // Apply spectral shaping: v' = Λ^{-1/2} v
        for (vi, &scale) in v.iter_mut().zip(lambda_inv_sqrt.iter()) {
            *vi *= scale;
        }

        // Project back to feature space: step = α Q v'
        let step: Vec<f64> = (0..N)
            .map(|i| {
                let dot: f64 = (0..N).map(|j| q.get(i, j) * v[j]).sum();
                dot * self.learning_rate
            })
            .collect();

        step
    }

    /// Return the spectral condition number κ = sqrt(λ_max / λ_min).
    ///
    /// κ encodes weight-space uncertainty:
    ///   κ ≈ 1  → isotropic, well-conditioned (all dimensions equally informative)
    ///   κ >> 1 → ill-conditioned (some dimensions noisy, others stable)
    ///
    /// In 5D, expect κ to increase when resonance is first introduced (new
    /// dimension has different variance structure). This naturally triggers
    /// higher temperature via PCNT, giving the system time to calibrate
    /// resonance weights before exploiting them.
    pub fn condition_number(&self) -> f64 {
        let (_, eigenvalues) = self.covariance.jacobi_eigendecomposition();
        let max_eig = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-10);
        let min_eig = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-10);
        (max_eig / min_eig).sqrt()
    }

    /// Return current eigenvalues of the gradient covariance matrix.
    #[allow(dead_code)]
    pub fn eigenvalues(&self) -> Vec<f64> {
        let (_, eigenvalues) = self.covariance.jacobi_eigendecomposition();
        eigenvalues
    }

    /// Spectral energy per dimension: fraction of total eigenvalue mass per eigenvector.
    ///
    /// Returns N values that sum to 1.0. Useful for diagnostics:
    /// - If dimension 4 (resonance) carries <5% spectral energy, it's not contributing
    ///   meaningful signal yet — the system is still in cold-start.
    /// - If one dimension dominates (>60%), the others are being drowned out.
    #[allow(dead_code)]
    pub fn spectral_energy(&self) -> Vec<f64> {
        let (_, eigenvalues) = self.covariance.jacobi_eigendecomposition();
        let total: f64 = eigenvalues.iter().map(|ev| ev.abs()).sum();
        if total < 1e-15 {
            return vec![1.0 / N as f64; N];
        }
        eigenvalues.iter().map(|ev| ev.abs() / total).collect()
    }
}

// ════════════════════════════════════════════════════════════════════
//  TYPE ALIASES — 4D (original) and 5D (with resonance)
// ════════════════════════════════════════════════════════════════════

/// Original 4D optimizer (Recency, Frequency, Semantic, Entropy).
/// All existing call sites continue to work unchanged.
pub type PrismOptimizer = PrismOptimizerN<4>;

/// Extended 5D optimizer (Recency, Frequency, Semantic, Entropy, Resonance).
///
/// The resonance dimension weight w_resonance controls how much the pairwise
/// interaction bonus γ·Σᵢⱼ rᵢⱼ·pᵢ·pⱼ influences fragment selection.
///
/// PRISM's spectral shaping is critical here because:
/// 1. Resonance gradients have higher variance (combinatorial noise)
/// 2. Resonance is initially uncorrelated with dims 0–3 (no cross-covariance)
/// 3. As patterns emerge, cross-covariance builds up (e.g., high-entropy
///    fragments tend to resonate more → C[3][4] grows positive)
/// 4. The eigendecomposition discovers these correlations and adjusts the
///    step direction accordingly — a scalar learning rate can't do this.
pub type PrismOptimizer5D = PrismOptimizerN<5>;

// ════════════════════════════════════════════════════════════════════
//  BACKWARD-COMPATIBLE 4D API
// ════════════════════════════════════════════════════════════════════

impl PrismOptimizer {
    /// Fixed-size 4D update (backward compat with existing call sites).
    /// Delegates to the generic implementation.
    pub fn compute_update_4d(&mut self, g: &[f64; 4]) -> [f64; 4] {
        let result = self.compute_update(g.as_slice());
        [result[0], result[1], result[2], result[3]]
    }

    /// Condition number (4D). Same as generic, but named for clarity.
    pub fn condition_number_4d(&self) -> f64 {
        self.condition_number()
    }
}

// ════════════════════════════════════════════════════════════════════
//  5D RESONANCE-SPECIFIC API
// ════════════════════════════════════════════════════════════════════

/// Dimension indices for the 5D weight space.
pub mod dim {
    /// Recency dimension index.
    pub const RECENCY: usize = 0;
    /// Frequency dimension index.
    pub const FREQUENCY: usize = 1;
    /// Semantic dimension index.
    pub const SEMANTIC: usize = 2;
    /// Entropy dimension index.
    pub const ENTROPY: usize = 3;
    /// Resonance dimension index (5D only).
    pub const RESONANCE: usize = 4;
}

impl PrismOptimizer5D {
    /// Construct from an existing 4D optimizer, preserving learned covariance.
    ///
    /// The 4×4 covariance block is copied into the top-left of the 5×5 matrix.
    /// The resonance row/column is initialized to ε·I (cold-start prior):
    ///
    ///   ┌─────────────┬────┐
    ///   │   C₄ₓ₄      │ 0  │
    ///   ├─────────────┼────┤
    ///   │   0ᵀ         │ ε  │
    ///   └─────────────┴────┘
    ///
    /// This means the resonance dimension starts with the same prior variance
    /// as the initial 4D dimensions — PRISM will discover its true variance
    /// from the first few gradient updates.
    pub fn from_4d(opt4: &PrismOptimizer) -> Self {
        let mut cov = SymMatrix5::new();
        // Copy 4x4 block (dims 0..RESONANCE are the base dimensions)
        for i in 0..dim::RESONANCE {
            for j in 0..dim::RESONANCE {
                cov.set(i, j, opt4.covariance.get(i, j));
            }
        }
        // Initialize resonance dimension with cold-start prior
        cov.set(dim::RESONANCE, dim::RESONANCE, 1e-4);

        PrismOptimizerN {
            covariance: cov,
            beta: opt4.beta,
            learning_rate: opt4.learning_rate,
            epsilon: opt4.epsilon,
        }
    }

    /// Extract the resonance-specific spectral diagnostics.
    ///
    /// Returns a struct with:
    /// - resonance_eigenvalue: the eigenvalue most aligned with dim 4
    /// - resonance_energy_fraction: what % of spectral energy is in resonance
    /// - cross_correlations: covariance of resonance with dims 0–3
    /// - is_calibrated: whether enough data has flowed through resonance
    pub fn resonance_diagnostics(&self) -> ResonanceDiagnostics {
        let (q, eigenvalues) = self.covariance.jacobi_eigendecomposition();

        // Find which eigenvector is most aligned with the resonance axis
        let (resonance_eig_idx, max_alignment) = (0..5)
            .map(|col| (col, q.get(dim::RESONANCE, col).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0));

        let total_energy: f64 = eigenvalues.iter().map(|ev| ev.abs()).sum();
        let resonance_energy = if total_energy > 1e-15 {
            eigenvalues[resonance_eig_idx].abs() / total_energy
        } else {
            0.2 // uniform prior
        };

        // Cross-correlations: C[res][0..res] normalized by sqrt(C[res][res] * C[k][k])
        let c44 = self.covariance.get(dim::RESONANCE, dim::RESONANCE).max(1e-15);
        let mut cross = [0.0f64; 4];
        for (k, c) in cross.iter_mut().enumerate() {
            let ckk = self.covariance.get(k, k).max(1e-15);
            *c = self.covariance.get(dim::RESONANCE, k) / (c44 * ckk).sqrt();
        }

        // Calibrated = resonance variance has moved significantly from init
        let is_calibrated = self.covariance.get(dim::RESONANCE, dim::RESONANCE) > 5e-4;

        ResonanceDiagnostics {
            resonance_eigenvalue: eigenvalues[resonance_eig_idx],
            resonance_energy_fraction: resonance_energy,
            resonance_alignment: max_alignment,
            cross_correlations: cross,
            is_calibrated,
        }
    }
}

/// Diagnostics for the resonance dimension in 5D PRISM.
#[derive(Clone, Debug)]
pub struct ResonanceDiagnostics {
    /// Eigenvalue of the eigenvector most aligned with the resonance axis.
    pub resonance_eigenvalue: f64,
    /// Fraction of total spectral energy in the resonance-aligned eigenvector.
    /// Low (<5%) = resonance isn't contributing signal yet.
    /// High (>30%) = resonance is a dominant learning signal.
    pub resonance_energy_fraction: f64,
    /// How well the resonance eigenvector aligns with pure dim 4.
    /// 1.0 = perfectly aligned (no cross-correlation with other dims).
    /// Low = resonance has rotated into a mixed eigenvector (correlated with other dims).
    pub resonance_alignment: f64,
    /// Pearson correlation of resonance gradient with each of the 4 base dimensions.
    /// [recency, frequency, semantic, entropy]
    /// High positive: resonance and that dimension reinforce each other.
    /// Negative: they trade off (selecting for resonance deprioritizes that dimension).
    pub cross_correlations: [f64; 4],
    /// Whether enough gradient updates have flowed through the resonance
    /// dimension for the covariance estimate to be meaningful.
    pub is_calibrated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 4D Tests (backward compatibility) ────────────────────────────

    #[test]
    fn test_jacobi_eigendecomposition_diagonal_4d() {
        let mut mat = SymMatrix4::new();
        mat.set(0, 0, 5.0);
        mat.set(1, 1, 3.0);
        mat.set(2, 2, 2.0);
        mat.set(3, 3, 1.0);

        let (q, eigs) = mat.jacobi_eigendecomposition();

        // Eigenvalues should contain all diagonal entries
        let mut sorted_eigs = eigs.clone();
        sorted_eigs.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted_eigs[0] - 5.0).abs() < 1e-6);
        assert!((sorted_eigs[1] - 3.0).abs() < 1e-6);
        assert!((sorted_eigs[2] - 2.0).abs() < 1e-6);
        assert!((sorted_eigs[3] - 1.0).abs() < 1e-6);

        // Q should be orthogonal: Q^T Q ≈ I
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for k in 0..4 {
                    dot += q.get(k, i) * q.get(k, j);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-6,
                    "Q^T Q [{i}][{j}] = {dot}, expected {expected}");
            }
        }
    }

    #[test]
    fn test_anisotropic_shaping_dampens_noise_4d() {
        let mut optim = PrismOptimizer::new(0.1);

        // Flood dimension 0 with high variance (noise)
        for _ in 0..100 {
            optim.compute_update(&[10.0, 0.1, 0.1, 0.1]);
            optim.compute_update(&[-10.0, 0.1, 0.1, 0.1]);
        }

        // A unit gradient in all directions
        let step = optim.compute_update(&[1.0, 1.0, 1.0, 1.0]);

        // Dimension 0 should be heavily damped compared to 1, 2, 3
        assert!(step[0].abs() < step[1].abs() * 0.1,
            "PRISM failed to anisotropically damp the noisy dimension: step={step:?}");
    }

    #[test]
    fn test_4d_backward_compat_api() {
        let mut optim = PrismOptimizer::new(0.01);
        let g = [0.1, 0.2, 0.3, 0.4];

        // Both APIs should produce identical results
        let result_generic = optim.compute_update(&g);
        // Reset covariance to test compute_update_4d independently
        let mut optim2 = PrismOptimizer::new(0.01);
        let result_fixed = optim2.compute_update_4d(&g);

        for i in 0..4 {
            assert!((result_generic[i] - result_fixed[i]).abs() < 1e-12,
                "Generic vs fixed API mismatch at dim {i}: {} vs {}",
                result_generic[i], result_fixed[i]);
        }
    }

    // ── 5D Tests (new resonance dimension) ───────────────────────────

    #[test]
    fn test_jacobi_eigendecomposition_5d() {
        let mut mat = SymMatrix5::new();
        mat.set(0, 0, 5.0);
        mat.set(1, 1, 4.0);
        mat.set(2, 2, 3.0);
        mat.set(3, 3, 2.0);
        mat.set(4, 4, 1.0);

        let (q, eigs) = mat.jacobi_eigendecomposition();

        // Eigenvalues should contain all diagonal entries
        let mut sorted_eigs = eigs.clone();
        sorted_eigs.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted_eigs[0] - 5.0).abs() < 1e-6);
        assert!((sorted_eigs[1] - 4.0).abs() < 1e-6);
        assert!((sorted_eigs[2] - 3.0).abs() < 1e-6);
        assert!((sorted_eigs[3] - 2.0).abs() < 1e-6);
        assert!((sorted_eigs[4] - 1.0).abs() < 1e-6);

        // Q should be orthogonal: Q^T Q ≈ I
        for i in 0..5 {
            for j in 0..5 {
                let mut dot = 0.0;
                for k in 0..5 {
                    dot += q.get(k, i) * q.get(k, j);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-6,
                    "Q^T Q [{i}][{j}] = {dot}, expected {expected}");
            }
        }
    }

    #[test]
    fn test_5d_resonance_damping() {
        // Scenario: resonance (dim 4) has high gradient variance (combinatorial noise)
        // while the other 4 dims have stable gradients.
        // PRISM should automatically dampen the resonance learning rate.
        let mut optim = PrismOptimizer5D::new(0.1);

        // Simulate: dims 0–3 have consistent gradients, dim 4 oscillates wildly
        for _ in 0..100 {
            optim.compute_update(&[0.1, 0.1, 0.1, 0.1,  5.0]);
            optim.compute_update(&[0.1, 0.1, 0.1, 0.1, -5.0]);
        }

        // Now apply a unit gradient in all dimensions
        let step = optim.compute_update(&[1.0, 1.0, 1.0, 1.0, 1.0]);

        // Dimension 4 (resonance) should be damped relative to dims 0–3
        let avg_step_0_3 = (step[0].abs() + step[1].abs() + step[2].abs() + step[3].abs()) / 4.0;
        assert!(step[4].abs() < avg_step_0_3 * 0.3,
            "PRISM should damp noisy resonance dim: resonance_step={:.4}, avg_other={:.4}",
            step[4].abs(), avg_step_0_3);
    }

    #[test]
    fn test_5d_from_4d_preserves_covariance() {
        let mut opt4 = PrismOptimizer::new(0.01);

        // Train the 4D optimizer
        for _ in 0..20 {
            opt4.compute_update(&[0.3, 0.1, 0.2, 0.4]);
        }

        // Upgrade to 5D
        let opt5 = PrismOptimizer5D::from_4d(&opt4);

        // Verify the 4x4 block is preserved
        for i in 0..4 {
            for j in 0..4 {
                assert!((opt5.covariance.get(i, j) - opt4.covariance.get(i, j)).abs() < 1e-12,
                    "4x4 block mismatch at [{i}][{j}]");
            }
        }

        // Verify resonance row/column is cold-start
        for k in 0..4 {
            assert_eq!(opt5.covariance.get(4, k), 0.0,
                "Resonance cross-covariance should start at 0");
            assert_eq!(opt5.covariance.get(k, 4), 0.0,
                "Resonance cross-covariance should start at 0");
        }
        assert!((opt5.covariance.get(4, 4) - 1e-4).abs() < 1e-10,
            "Resonance variance should be epsilon-initialized");
    }

    #[test]
    fn test_5d_cross_correlation_emergence() {
        // When resonance co-varies with entropy (high-entropy fragments resonate more),
        // PRISM should discover this correlation and adjust the step direction.
        let mut optim = PrismOptimizer5D::new(0.1);

        // Simulate: entropy and resonance are positively correlated
        for _ in 0..50 {
            // When entropy gradient is high, resonance gradient is also high
            optim.compute_update(&[0.1, 0.1, 0.1, 0.5, 0.4]);
            optim.compute_update(&[0.1, 0.1, 0.1, 0.1, 0.05]);
        }

        let diag = optim.resonance_diagnostics();

        // Cross-correlation with entropy (dim 3) should be positive
        assert!(diag.cross_correlations[3] > 0.3,
            "Should discover entropy-resonance correlation: r={:.3}",
            diag.cross_correlations[3]);

        // Should be calibrated after 100 updates
        assert!(diag.is_calibrated,
            "Should be calibrated after 100 gradient updates");
    }

    #[test]
    fn test_5d_condition_number_increases_with_resonance() {
        // Start with a well-conditioned 4D system: diverse gradients → near-isotropic covariance.
        // Then show that extending to 5D with an anisotropic resonance signal breaks isotropy.
        let mut opt4 = PrismOptimizer::new(0.01);
        // Cycle through axis-aligned gradients to build isotropic covariance
        let basis: [[f64; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for i in 0..40 {
            opt4.compute_update(&basis[i % 4]);
        }
        let kappa_4 = opt4.condition_number();

        let mut opt5 = PrismOptimizer5D::from_4d(&opt4);
        // Same base gradients, but resonance dimension has 10× larger variance
        let basis5: [[f64; 5]; 4] = [
            [1.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 0.0, 1.0, 10.0],
        ];
        for i in 0..40 {
            opt5.compute_update(&basis5[i % 4]);
        }
        let kappa_5 = opt5.condition_number();

        assert!(kappa_5 > kappa_4,
            "5D κ ({kappa_5:.2}) should exceed 4D κ ({kappa_4:.2}) when resonance has different variance");
    }

    #[test]
    fn test_spectral_energy_sums_to_one() {
        let mut optim = PrismOptimizer5D::new(0.1);
        for _ in 0..20 {
            optim.compute_update(&[0.3, 0.1, 0.2, 0.4, 0.15]);
        }

        let energy = optim.spectral_energy();
        assert_eq!(energy.len(), 5);
        let total: f64 = energy.iter().sum();
        assert!((total - 1.0).abs() < 1e-6,
            "Spectral energy should sum to 1.0, got {total}");
    }

    // ── SymMatrix4 backward compat ───────────────────────────────────

    #[test]
    fn test_sym_matrix4_array_roundtrip() {
        let arr = [
            [1.0, 0.1, 0.2, 0.3],
            [0.1, 2.0, 0.4, 0.5],
            [0.2, 0.4, 3.0, 0.6],
            [0.3, 0.5, 0.6, 4.0],
        ];
        let mat = SymMatrix4::from_array(arr);
        let roundtrip = mat.to_array();
        for i in 0..4 {
            for j in 0..4 {
                assert!((arr[i][j] - roundtrip[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_5d_non_diagonal_eigendecomposition() {
        // Test with off-diagonal elements to verify rotation correctness
        let mut mat = SymMatrix5::new();
        // Create a symmetric matrix with known structure
        mat.set(0, 0, 3.0); mat.set(0, 1, 1.0); mat.set(0, 2, 0.0); mat.set(0, 3, 0.0); mat.set(0, 4, 0.0);
        mat.set(1, 0, 1.0); mat.set(1, 1, 3.0); mat.set(1, 2, 0.0); mat.set(1, 3, 0.0); mat.set(1, 4, 0.0);
        mat.set(2, 2, 2.0);
        mat.set(3, 3, 1.0);
        mat.set(4, 4, 0.5);

        let (q, eigs) = mat.jacobi_eigendecomposition();

        // Known eigenvalues: 4.0, 2.0, 2.0, 1.0, 0.5
        let mut sorted = eigs.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted[0] - 4.0).abs() < 1e-6, "Expected 4.0, got {}", sorted[0]);
        assert!((sorted[1] - 2.0).abs() < 1e-6, "Expected 2.0, got {}", sorted[1]);
        assert!((sorted[2] - 2.0).abs() < 1e-6, "Expected 2.0, got {}", sorted[2]);
        assert!((sorted[3] - 1.0).abs() < 1e-6, "Expected 1.0, got {}", sorted[3]);
        assert!((sorted[4] - 0.5).abs() < 1e-6, "Expected 0.5, got {}", sorted[4]);

        // Verify reconstruction: Q Λ Q^T ≈ A
        for i in 0..5 {
            for j in 0..5 {
                let reconstructed: f64 = eigs.iter().enumerate()
                    .map(|(k, &ek)| q.get(i, k) * ek * q.get(j, k))
                    .sum();
                let original = mat.get(i, j);
                assert!((reconstructed - original).abs() < 1e-6,
                    "Reconstruction failed at [{i}][{j}]: {reconstructed:.6} vs {original:.6}");
            }
        }
    }
}
