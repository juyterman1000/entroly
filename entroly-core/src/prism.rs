//! PRISM-Inspired Anisotropic Spectral Optimizer
//!
//! Implements quasi-second-order RL weight tuning using running covariance
//! and Eigenvalue Decomposition, adapted from "PRISM: Structured Optimization
//! via Anisotropic Spectral Shaping".
//!
//! Instead of isotropic (scalar) learning rates, this tracks the 4x4 covariance
//! matrix of the feature gradients (Recency, Frequency, Semantic, Entropy).
//! It computes the eigendecomposition $C = Q \Lambda Q^T$ and applies
//! anisotropic damping in high-variance (noisy) sub-spaces:
//! $w_{t+1} = w_t + \alpha Q \Lambda^{-1/2} Q^T g$.
//!
//! Because our state space is exactly 4D, we perform exact eigendecomposition
//! (Jacobi method) rather than the approximate polar decomposition needed for
//! 100M+ parameter neural networks.

use std::f64;
use serde::{Serialize, Deserialize};

/// A 4x4 symmetric matrix for tracking gradient covariance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymMatrix4 {
    pub data: [[f64; 4]; 4],
}

impl SymMatrix4 {
    pub fn new() -> Self {
        SymMatrix4 { data: [[0.0; 4]; 4] }
    }

    pub fn identity() -> Self {
        let mut m = Self::new();
        for i in 0..4 { m.data[i][i] = 1.0; }
        m
    }

    /// Update running covariance: $C = \beta C + (1-\beta) g g^T$
    pub fn update_ema(&mut self, g: &[f64; 4], beta: f64) {
        for i in 0..4 {
            for j in 0..4 {
                self.data[i][j] = beta * self.data[i][j] + (1.0 - beta) * (g[i] * g[j]);
            }
        }
    }

    /// Computes Eigenvalue Decomposition $C = Q \Lambda Q^T$ using the Cyclic Jacobi method.
    /// Returns (Q, Eigenvalues), where Q columns are eigenvectors.
    pub fn jacobi_eigendecomposition(&self) -> (SymMatrix4, [f64; 4]) {
        let mut a = self.clone();
        let mut q = Self::identity();
        let mut iters = 0;
        let max_iters = 50;
        let eps = 1e-9;

        while iters < max_iters {
            // Find max off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut r = 1;
            for i in 0..3 {
                for j in (i + 1)..4 {
                    let val = a.data[i][j].abs();
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

            // Compute Jacobi rotation
            let app = a.data[p][p];
            let arr = a.data[r][r];
            let apr = a.data[p][r];
            let theta = 0.5 * (2.0 * apr / (app - arr + 1e-15)).atan();
            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation A' = J^T A J
            for i in 0..4 {
                if i != p && i != r {
                    let aip = a.data[i][p];
                    let air = a.data[i][r];
                    a.data[i][p] = c * aip - s * air;
                    a.data[p][i] = a.data[i][p];
                    a.data[i][r] = s * aip + c * air;
                    a.data[r][i] = a.data[i][r];
                }
            }
            a.data[p][p] = c * c * app - 2.0 * s * c * apr + s * s * arr;
            a.data[r][r] = s * s * app + 2.0 * s * c * apr + c * c * arr;
            a.data[p][r] = 0.0;
            a.data[r][p] = 0.0;

            // Apply rotation Q' = Q J
            for i in 0..4 {
                let qip = q.data[i][p];
                let qir = q.data[i][r];
                q.data[i][p] = c * qip - s * qir;
                q.data[i][r] = s * qip + c * qir;
            }

            iters += 1;
        }

        let eigenvalues = [a.data[0][0], a.data[1][1], a.data[2][2], a.data[3][3]];
        (q, eigenvalues)
    }
}

/// Anisotropic Spectral Optimizer (PRISM-lite for 4D Context Weights)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrismOptimizer {
    pub covariance: SymMatrix4,
    pub beta: f64,
    pub learning_rate: f64,
    pub epsilon: f64,
    pub initial_weights: [f64; 4],
    pub total_updates: u32,
    pub residual_lambda: f64,
}

impl PrismOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        let mut cov = SymMatrix4::identity();
        // Initialize with small epsilon identity to prevent division by zero gracefully
        for i in 0..4 { cov.data[i][i] = 1e-4; }
        PrismOptimizer {
            covariance: cov,
            beta: 0.95, // exponential moving average decay
            learning_rate,
            epsilon: 1e-6,
            initial_weights: [0.30, 0.25, 0.25, 0.20], // Default weights
            total_updates: 0,
            residual_lambda: 0.0,
        }
    }

    /// Applies Anisotropic Spectral Gain to a gradient vector.
    /// Computes $P g = Q \Lambda^{-1/2} Q^T g$ and returns the update $\Delta w = \alpha P g$.
    pub fn compute_update(&mut self, g: &[f64; 4]) -> [f64; 4] {
        self.total_updates += 1;
        // Sigmoid warmup centered at 50 updates
        self.residual_lambda = 1.0 / (1.0 + (-0.1 * (self.total_updates as f64 - 50.0)).exp());

        // 1. Update running covariance
        self.covariance.update_ema(g, self.beta);

        // 2. Eigendecomposition: C = Q \Lambda Q^T
        let (q, eigenvalues) = self.covariance.jacobi_eigendecomposition();

        // 3. Spectral Shaping (Inverse Square Root: \Lambda^{-1/2})
        // This dampens directions with high variance (noise) and boosts clean signals.
        let mut lambda_inv_sqrt = [0.0; 4];
        for i in 0..4 {
            lambda_inv_sqrt[i] = 1.0 / (eigenvalues[i].abs() + self.epsilon).sqrt();
        }

        // 4. Compute Q \Lambda^{-1/2} Q^T g
        // First, project gradient into eigenspace: v = Q^T g
        let mut v = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                v[i] += q.data[j][i] * g[j];
            }
        }

        // Apply spectral shaping: v' = \Lambda^{-1/2} v
        for i in 0..4 {
            v[i] *= lambda_inv_sqrt[i];
        }

        // Project back to feature space: update = \alpha Q v'
        let mut step = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                step[i] += q.data[i][j] * v[j];
            }
            step[i] *= self.learning_rate;
        }

        step
    }

    /// PRISM Value Residual Mixing: Anchors the completely learned parameters (`drifted`)
    /// back to the reliable `initial_weights` using the sigmoid `residual_lambda`.
    /// When lambda ~0 (early training), returns mostly initial weights.
    /// When lambda ~1 (mature training), returns mostly learned weights.
    pub fn apply_residual(&self, drifted: &[f64; 4]) -> [f64; 4] {
        let mut mixed = [0.0; 4];
        let lam = self.residual_lambda;
        for i in 0..4 {
            mixed[i] = lam * drifted[i] + (1.0 - lam) * self.initial_weights[i];
        }
        mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_eigendecomposition_diagonal() {
        let mut mat = SymMatrix4::new();
        mat.data[0][0] = 5.0;
        mat.data[1][1] = 3.0;
        mat.data[2][2] = 2.0;
        mat.data[3][3] = 1.0;

        let (q, eigs) = mat.jacobi_eigendecomposition();
        
        // Eigenvalues should match diagonal
        assert!((eigs[0] - 5.0).abs() < 1e-6 || (eigs[1] - 5.0).abs() < 1e-6 || (eigs[2] - 5.0).abs() < 1e-6 || (eigs[3] - 5.0).abs() < 1e-6);
        
        // Q should be identity (possibly permuted/signed)
        let det_q = q.data[0][0]*q.data[1][1]*q.data[2][2]*q.data[3][3];
        assert!(det_q.abs() > 0.9);
    }

    #[test]
    fn test_anisotropic_shaping_dampens_noise() {
        let mut optim = PrismOptimizer::new(0.1);
        
        // Flood the optimizer with high variance (noise) in dimension 0
        for _ in 0..100 {
            optim.compute_update(&[10.0, 0.1, 0.1, 0.1]);
            optim.compute_update(&[-10.0, 0.1, 0.1, 0.1]);
        }
        
        // Now, a single unit gradient in all directions
        let step = optim.compute_update(&[1.0, 1.0, 1.0, 1.0]);
        
        // Dimension 0 should be heavily damped compared to 1, 2, 3
        assert!(step[0].abs() < step[1].abs() * 0.1, "PRISM failed to anisotropically damp the noisy dimension");
    }
}
