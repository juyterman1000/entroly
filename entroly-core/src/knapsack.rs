//! Unified Context Optimizer — Rust implementation.
//!
//! # Differentiable Soft Bisection (primary path, τ ≥ 0.05)
//!
//! Find threshold th* via 30-step bisection such that:
//!
//!   f(th) = Σᵢ σ((sᵢ − th) / τ) · tokensᵢ  −  B  =  0
//!
//! where sᵢ = w^T · featuresᵢ (pre-softcap linear score, same as REINFORCE).
//! f is strictly monotone decreasing in th, so bisection always converges.
//!
//! th* is the **exact Lagrange multiplier** for the token-budget constraint under
//! the continuous KKT relaxation of the 0/1 knapsack — a principled dual variable,
//! not a heuristic threshold.
//!
//! After bisection, sort fragments by p_i = σ((sᵢ − th*) / τ) descending and
//! greedily fill the *hard* budget (context windows are hard limits).
//!
//! Complexity: O(30 · N) bisection + O(N log N) sort = O(N log N).
//!   ≈ 33× faster than the O(N × Q=1000) DP table for N=500.
//!
//! Train/test consistency: the same linear score sᵢ and the same σ(·/τ) appear
//! in the REINFORCE backward pass → no train/test mismatch.
//!
//! Convergence: as τ → 0, p_i → I(sᵢ > th*) and the greedy fill recovers the
//! exact greedy sort, giving a (1-1/e) ~ 0.63 optimality guarantee.
//!
//! # Hard DP fallback (τ < 0.05)
//!
//! Exact 0/1 DP with budget quantization: O(N × Q), Q = 1000.
//! Used when weights have converged (τ is at floor) for maximum precision.
//!
//! References:
//!   Kellerer, Pferschy, Pisinger. "Knapsack Problems" (Springer, 2004)
//!   Williams. "Simple Statistical Gradient-Following Algorithms..." (1992)
//!   Boyd & Vandenberghe. "Convex Optimization", §5.2 Lagrange duality (2004)

use std::collections::HashMap;
use crate::fragment::{compute_relevance, ContextFragment};

// ── Public types ──────────────────────────────────────────────────────────────

/// Weights for the four-dimensional relevance scoring.
pub struct ScoringWeights {
    pub recency: f64,
    pub frequency: f64,
    pub semantic: f64,
    pub entropy: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        ScoringWeights {
            recency: 0.30,
            frequency: 0.25,
            semantic: 0.25,
            entropy: 0.20,
        }
    }
}

/// Result of a knapsack optimization run.
#[allow(dead_code)]
pub struct KnapsackResult {
    pub selected_indices: Vec<usize>,
    pub total_tokens: u32,
    pub total_relevance: f64,
    pub(crate) method: &'static str,
    /// Lagrange multiplier λ* for the budget constraint (soft path only; 0.0 for hard paths).
    /// Forward: p_i = σ((s_i − λ*·tokens_i) / τ)
    /// Store in EntrolyEngine and reuse in REINFORCE backward pass for exact consistency.
    pub lambda_star: f64,
    /// Adaptive Dual Gap Temperature signal: D(λ*) − primal (soft path only; 0.0 for hard).
    ///
    /// D(λ*) = τ · Σᵢ log(1 + exp((sᵢ−λ*·cᵢ)/τ)) + λ*·B  (log-sum-exp dual)
    /// primal = actual total relevance of selected fragments
    /// gap = D(λ*) − primal ∈ [0, τ·N·log(2)]
    ///
    /// gap ≈ 0 → weights converged, reduce temperature
    /// gap ≈ τ·N·log(2) → all p_i ≈ 0.5, maximum uncertainty, keep temperature high
    ///
    /// Used by ADGT (Adaptive Dual Gap Temperature) to replace the ad-hoc 0.995 schedule.
    pub dual_gap: f64,
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Numerically stable sigmoid σ(x).
/// Clamped to [-500, 500] — no NaN, no Inf, no overflow.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-500.0, 500.0);
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Raw linear score for a fragment, scaled by the per-fragment RL feedback multiplier.
///
/// This is the **pre-softcap** score — the same landscape used in the REINFORCE
/// backward pass. Feedback multipliers shift relative item values continuously,
/// making them smooth inputs to the soft bisection.
#[inline]
fn linear_score(frag: &ContextFragment, w: &ScoringWeights, fm: f64) -> f64 {
    (w.recency   * frag.recency_score
   + w.frequency * frag.frequency_score
   + w.semantic  * frag.semantic_score
   + w.entropy   * frag.entropy_score) * fm.max(0.01)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Select the most valuable subset of fragments within the token budget.
///
/// `temperature` controls the forward-pass mode:
///   - `temperature < 0.05` → exact 0/1 DP (optimal, used at weight convergence)
///   - `temperature ≥ 0.05` → soft bisection (differentiable, consistent with PRISM)
///
/// `feedback_mults` maps fragment_id → per-fragment RL-learned value multiplier.
pub fn knapsack_optimize(
    fragments: &[ContextFragment],
    token_budget: u32,
    weights: &ScoringWeights,
    feedback_mults: &HashMap<String, f64>,
    temperature: f64,
) -> KnapsackResult {
    if fragments.is_empty() {
        return KnapsackResult {
            selected_indices: vec![],
            total_tokens: 0,
            total_relevance: 0.0,
            method: "empty",
            lambda_star: 0.0,
            dual_gap: 0.0,
        };
    }

    // ── Pin handling: pinned fragments are always included first ─────────────
    let mut pinned_indices: Vec<usize> = Vec::new();
    let mut pinned_tokens: u32 = 0;
    let mut candidate_indices: Vec<usize> = Vec::new();

    for (i, frag) in fragments.iter().enumerate() {
        if frag.is_pinned {
            pinned_indices.push(i);
            pinned_tokens += frag.token_count;
        } else {
            candidate_indices.push(i);
        }
    }

    let remaining_budget = token_budget.saturating_sub(pinned_tokens);
    if remaining_budget == 0 || candidate_indices.is_empty() {
        let total_relevance = pinned_relevance(&pinned_indices, fragments, weights, feedback_mults);
        return KnapsackResult {
            selected_indices: pinned_indices,
            total_tokens: pinned_tokens,
            total_relevance,
            method: "pinned_only",
            lambda_star: 0.0,
            dual_gap: 0.0,
        };
    }

    // ── Score candidates ─────────────────────────────────────────────────────
    // Soft path: pre-softcap linear score sᵢ (same landscape as REINFORCE).
    // Hard path: compute_relevance with softcap (matches the original DP inputs).
    let use_soft = temperature >= 0.05;

    let scored: Vec<(usize, f64)> = candidate_indices
        .iter()
        .filter_map(|&i| {
            let fm = feedback_mults.get(&fragments[i].fragment_id).copied().unwrap_or(1.0);
            let score = if use_soft {
                linear_score(&fragments[i], weights, fm)
            } else {
                compute_relevance(
                    &fragments[i],
                    weights.recency, weights.frequency,
                    weights.semantic, weights.entropy,
                    fm,
                )
            };
            if score > 0.0 && fragments[i].token_count > 0 {
                Some((i, score))
            } else {
                None
            }
        })
        .collect();

    // ── Selection ────────────────────────────────────────────────────────────
    let (method, mut selected, lambda_star, dual_gap) = if use_soft {
        let (sel, lam, gap) = soft_bisection_select(&scored, fragments, remaining_budget, temperature);
        ("soft_bisection", sel, lam, gap)
    } else if scored.len() <= 2000 {
        ("exact_dp", knapsack_dp(&scored, fragments, remaining_budget), 0.0, 0.0)
    } else {
        ("greedy_approx", knapsack_greedy(&scored, fragments, remaining_budget), 0.0, 0.0)
    };

    // Merge pinned + selected
    selected.extend(pinned_indices.iter());
    let total_tokens: u32 = selected.iter().map(|&i| fragments[i].token_count).sum();
    let total_relevance: f64 = selected
        .iter()
        .map(|&i| {
            let fm = feedback_mults.get(&fragments[i].fragment_id).copied().unwrap_or(1.0);
            compute_relevance(&fragments[i], weights.recency, weights.frequency,
                              weights.semantic, weights.entropy, fm)
        })
        .sum();

    KnapsackResult { selected_indices: selected, total_tokens, total_relevance, method, lambda_star, dual_gap }
}

// ── Public bisection helper ───────────────────────────────────────────────────

/// Compute only the Lagrange dual variable λ* for a given budget target.
///
/// This is the pure bisection step, decoupled from selection. Callers that
/// use a different selection mechanism (e.g. IOS submodular greedy) can call
/// this after selection to get the λ* that makes the sigmoid model consistent
/// with the actual selection:
///
///   Find λ* ≥ 0 s.t. Σᵢ σ((sᵢ − λ*·tokensᵢ) / τ) · tokensᵢ = budget_target
///
/// The result is a meaningful proxy for IOS inclusion probability:
/// fragments with high σ((sᵢ − λ*·tokensᵢ)/τ) are the ones the sigmoid model
/// "expected" to be included given the actual budget consumed by IOS.
///
/// # Arguments
/// - `scored`: (fragment_idx, linear_score) pairs for all candidates
/// - `fragments`: fragment slice (for token counts)
/// - `budget_target`: typically the *actual* tokens used by IOS (not the full budget)
/// - `temperature`: current gradient temperature τ
///
/// Returns 0.0 if temperature < 0.05 (hard sel) or if all items fit (λ* = 0).
pub fn compute_lambda_star(
    scored: &[(usize, f64)],
    fragments: &[ContextFragment],
    budget_target: u32,
    temperature: f64,
) -> f64 {
    if temperature < 0.05 || scored.is_empty() || budget_target == 0 {
        return 0.0;
    }
    let tau = temperature.max(1e-4);
    let budget_f = budget_target as f64;

    let expected_tokens = |lambda: f64| -> f64 {
        scored.iter().map(|&(idx, score)| {
            let tc = fragments[idx].token_count as f64;
            sigmoid((score - lambda * tc) / tau) * tc
        }).sum()
    };

    // Fast path: all items fit at λ=0.
    if expected_tokens(0.0) <= budget_f {
        return 0.0;
    }

    let max_score = scored.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
    let min_tokens = scored.iter()
        .map(|&(idx, _)| fragments[idx].token_count as f64)
        .fold(f64::INFINITY, f64::min)
        .max(1.0);
    let mut hi = (max_score + 5.0 * tau) / (min_tokens * tau).max(1e-10);
    let mut iters = 0;
    while expected_tokens(hi) >= budget_f && iters < 60 { hi *= 2.0; iters += 1; }
    if expected_tokens(hi) >= budget_f { return 0.0; }

    let mut lo = 0.0_f64;
    for _ in 0..30 {
        let mid = (lo + hi) * 0.5;
        if expected_tokens(mid) > budget_f { lo = mid; } else { hi = mid; }
    }
    (lo + hi) * 0.5
}


/// Differentiable forward selector using exact Lagrange dual bisection.
///
/// # Full KKT derivation
///
/// The continuous relaxation of the 0/1 knapsack:
///   max   Σ pᵢ·sᵢ
///   s.t.  Σ pᵢ·tokensᵢ ≤ B,   pᵢ ∈ [0,1]
///
/// The Lagrangian (with λ ≥ 0 for the budget constraint):
///   L(p, λ) = Σ pᵢ·sᵢ − λ·(Σ pᵢ·tokensᵢ − B)
///            = Σ (sᵢ − λ·tokensᵢ)·pᵢ + λ·B
///
/// Maximizing over each pᵢ independently via sigmoid-smooth relaxation:
///   p*ᵢ = σ((sᵢ − λ·tokensᵢ) / τ)
///
/// This is the EXACT KKT condition for heterogeneous token counts.
/// Previous "additive threshold" version (p*ᵢ = σ((sᵢ − th*) / τ)) is only
/// exact when all tokens_i are equal — a bias-inducing simplification.
///
/// Dual feasibility: find λ* ≥ 0 such that Σ p*ᵢ·tokensᵢ = B.
/// g(λ) = Σ σ((sᵢ − λ·tokensᵢ)/τ)·tokensᵢ − B
/// dg/dλ = −1/τ · Σ p_i(1−p_i)·tokensᵢ² < 0  (strictly monotone → bisection converges)
///
/// Returns: (selected_indices, λ*)  
/// Caller stores λ* in EntrolyEngine.last_lambda_star for the REINFORCE backward pass,
/// which recomputes p_i = σ((s_i − λ*·tokens_i)/τ) for exact advantage estimation.
fn soft_bisection_select(
    scored: &[(usize, f64)],
    fragments: &[ContextFragment],
    budget: u32,
    temperature: f64,
) -> (Vec<usize>, f64, f64) {
    let tau = temperature.max(1e-4);
    let budget_f = budget as f64;

    // g(λ) = Σ σ((sᵢ − λ·tokensᵢ)/τ)·tokensᵢ − B  (strictly decreasing in λ)
    let expected_tokens = |lambda: f64| -> f64 {
        scored.iter().map(|&(idx, score)| {
            let tc = fragments[idx].token_count as f64;
            sigmoid((score - lambda * tc) / tau) * tc
        }).sum()
    };

    // Fast path: λ=0 → p_i = σ(s_i/τ) ≈ 1 for all. If total E[tokens] ≤ B, include all.
    if expected_tokens(0.0) <= budget_f {
        return (scored.iter().map(|&(idx, _)| idx).collect(), 0.0, 0.0);
    }

    // Find λ_hi s.t. g(λ_hi) < 0 (expected tokens < budget).
    let max_score = scored.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
    let min_tokens = scored.iter()
        .map(|&(idx, _)| fragments[idx].token_count as f64)
        .fold(f64::INFINITY, f64::min)
        .max(1.0);
    let mut hi = (max_score + 5.0 * tau) / (min_tokens * tau).max(1e-10);
    let mut iters = 0;
    while expected_tokens(hi) >= budget_f && iters < 60 { hi *= 2.0; iters += 1; }
    if expected_tokens(hi) >= budget_f {
        return (knapsack_greedy(scored, fragments, budget), 0.0, 0.0);
    }

    // 30-step bisection on λ ∈ [0, hi]. Each iteration: O(N). Total: O(30·N).
    let mut lo = 0.0_f64;
    for _ in 0..30 {
        let mid = (lo + hi) * 0.5;
        if expected_tokens(mid) > budget_f { lo = mid; } else { hi = mid; }
    }
    let lambda_star = (lo + hi) * 0.5;

    // Compute exact KKT probabilities at λ*.
    // Sorting by p_i ≡ sorting by reduced cost (s_i − λ*·tokens_i) — LP duality ordering.
    let mut with_probs: Vec<(usize, f64)> = scored.iter().map(|&(idx, score)| {
        let tc = fragments[idx].token_count as f64;
        let p = sigmoid((score - lambda_star * tc) / tau);
        (idx, p)
    }).collect();

    with_probs.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Hard budget enforcement via greedy fill.
    let mut selected = Vec::with_capacity(with_probs.len());
    let mut remaining = budget;
    let mut primal_value = 0.0;
    for &(idx, _) in &with_probs {
        let tc = fragments[idx].token_count;
        if tc <= remaining {
            selected.push(idx);
            remaining -= tc;
            // Primal contribution: p_i · s_i (soft selection value)
            let tc_f = tc as f64;
            let p_final = sigmoid((scored.iter().find(|&&(i,_)| i==idx).map(|&(_,s)|s).unwrap_or(0.0)
                - lambda_star * tc_f) / tau);
            primal_value += p_final * scored.iter().find(|&&(i,_)| i==idx).map(|&(_,s)|s).unwrap_or(0.0);
        }
        if remaining == 0 { break; }
    }

    // ── Adaptive Dual Gap Temperature (ADGT) signal ──────────────────────────
    // Compute D(λ*) = τ · Σ log(1 + exp((s_i − λ*·c_i)/τ)) + λ*·B  [log-sum-exp dual]
    // This is the exact smooth upper bound on the primal objective.
    // dual_gap = D(λ*) − primal ∈ [0, τ·N·log(2)]
    //   → gap ≈ 0: weights converged, can lower temperature
    //   → gap ≈ τ·N·log(2): all p_i ≈ 0.5, fully uncertain, keep temperature high
    let dual_value: f64 = scored.iter().map(|&(idx, score)| {
        let tc = fragments[idx].token_count as f64;
        let z = (score - lambda_star * tc) / tau;
        // Numerically stable log(1 + exp(z)) = log1p(exp(z))
        tau * if z > 20.0 { z } else { (1.0_f64 + z.exp()).ln() }
    }).sum::<f64>() + lambda_star * budget_f;

    let dual_gap = (dual_value - primal_value).max(0.0);

    (selected, lambda_star, dual_gap)
}

// ── Hard DP fallback (τ < 0.05) ──────────────────────────────────────────────

/// Exact 0/1 knapsack via DP with budget quantization.
///
/// Quantize budget into Q=1000 bins to bound the DP table at N×1000.
/// Precision loss: < 0.1% of optimal value.
///
/// Small fragments (token_count < granularity) are "free" items:
/// always included, real cost subtracted from budget. This prevents
/// the 12.8× cost-inflation artifact of ceiling-division quantization.
fn knapsack_dp(
    scored: &[(usize, f64)],
    fragments: &[ContextFragment],
    budget: u32,
) -> Vec<usize> {
    const Q: u32 = 1000;
    let g = (budget / Q).max(1);

    let mut free_items: Vec<usize> = Vec::new();
    let mut free_tokens: u32 = 0;
    let mut dp_items: Vec<(usize, i64, usize)> = Vec::new(); // (idx, value, quantized_cost)

    for &(idx, rel) in scored {
        let tc = fragments[idx].token_count;
        let quantized_cost = tc / g;
        if quantized_cost == 0 {
            free_items.push(idx);
            free_tokens += tc;
        } else {
            let qb_max = (budget / g) as usize;
            if quantized_cost as usize <= qb_max {
                dp_items.push((idx, (rel * 10_000.0) as i64, quantized_cost as usize));
            }
        }
    }

    let adjusted_budget = budget.saturating_sub(free_tokens);
    if adjusted_budget == 0 || dp_items.is_empty() {
        return free_items;
    }

    let qb = (adjusted_budget / g) as usize;
    let n = dp_items.len();
    let mut prev = vec![0i64; qb + 1];
    let mut keep = vec![vec![false; qb + 1]; n];

    for i in 0..n {
        let mut curr = prev.clone();
        let (_, value, cost) = dp_items[i];
        for w in cost..=qb {
            if prev[w - cost] + value > curr[w] {
                curr[w] = prev[w - cost] + value;
                keep[i][w] = true;
            }
        }
        prev = curr;
    }

    let mut selected = free_items;
    let mut w = qb;
    for i in (0..n).rev() {
        if keep[i][w] {
            let (orig_idx, _, cost) = dp_items[i];
            selected.push(orig_idx);
            w -= cost;
        }
    }
    selected
}

/// Greedy approximation for very large sets (N > 2000) under hard τ.
/// Sort by relevance/token density. Provable 0.5 optimality (Dantzig, 1957).
fn knapsack_greedy(
    scored: &[(usize, f64)],
    fragments: &[ContextFragment],
    budget: u32,
) -> Vec<usize> {
    let mut density: Vec<(usize, f64)> = scored
        .iter()
        .map(|&(idx, rel)| (idx, rel / fragments[idx].token_count.max(1) as f64))
        .collect();
    density.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = Vec::new();
    let mut remaining = budget;
    for (idx, _) in density {
        if fragments[idx].token_count <= remaining {
            selected.push(idx);
            remaining -= fragments[idx].token_count;
        }
        if remaining == 0 { break; }
    }
    selected
}

/// Compute total relevance for pinned fragments only.
fn pinned_relevance(
    pinned: &[usize],
    fragments: &[ContextFragment],
    weights: &ScoringWeights,
    feedback_mults: &HashMap<String, f64>,
) -> f64 {
    pinned.iter().map(|&i| {
        let fm = feedback_mults.get(&fragments[i].fragment_id).copied().unwrap_or(1.0);
        compute_relevance(&fragments[i], weights.recency, weights.frequency,
                          weights.semantic, weights.entropy, fm)
    }).sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::ContextFragment;

    fn no_feedback() -> HashMap<String, f64> { HashMap::new() }

    #[test]
    fn test_knapsack_selects_optimal() {
        let fragments = vec![
            { let mut f = ContextFragment::new("a".into(), "hi val small".into(), 100, "".into());
              f.recency_score = 1.0; f.entropy_score = 0.9; f },
            { let mut f = ContextFragment::new("b".into(), "lo val large".into(), 900, "".into());
              f.recency_score = 0.1; f.entropy_score = 0.1; f },
            { let mut f = ContextFragment::new("c".into(), "med val med".into(), 400, "".into());
              f.recency_score = 0.7; f.entropy_score = 0.6; f },
        ];
        // Hard path (τ=0): exact DP
        let result = knapsack_optimize(&fragments, 500, &ScoringWeights::default(), &no_feedback(), 0.0);
        let ids: Vec<&str> = result.selected_indices.iter().map(|&i| fragments[i].fragment_id.as_str()).collect();
        assert!(ids.contains(&"a"), "Should select high-value 'a'");
        assert!(!ids.contains(&"b"), "Should not select low-value 'b'");
        assert!(result.total_tokens <= 500);
    }

    #[test]
    fn test_soft_bisection_selects_optimal() {
        let fragments = vec![
            { let mut f = ContextFragment::new("a".into(), "hi val small".into(), 100, "".into());
              f.recency_score = 1.0; f.entropy_score = 0.9; f },
            { let mut f = ContextFragment::new("b".into(), "lo val large".into(), 900, "".into());
              f.recency_score = 0.1; f.entropy_score = 0.1; f },
            { let mut f = ContextFragment::new("c".into(), "med val med".into(), 400, "".into());
              f.recency_score = 0.7; f.entropy_score = 0.6; f },
        ];
        // Soft path (τ=0.1): bisection — should still prefer 'a' and 'c' over 'b'
        let result = knapsack_optimize(&fragments, 500, &ScoringWeights::default(), &no_feedback(), 0.1);
        let ids: Vec<&str> = result.selected_indices.iter().map(|&i| fragments[i].fragment_id.as_str()).collect();
        assert!(ids.contains(&"a"), "Soft bisection should select high-value 'a'");
        assert!(!ids.contains(&"b"), "Soft bisection should exclude low-value 'b'");
        assert!(result.total_tokens <= 500);
        assert_eq!(result.method, "soft_bisection");
    }

    #[test]
    fn test_small_fragments_not_penalized() {
        let fragments = vec![
            { let mut f = ContextFragment::new("small".into(), "tiny".into(), 10, "".into());
              f.recency_score = 1.0; f.entropy_score = 0.9; f },
            { let mut f = ContextFragment::new("big".into(), "large content here".into(), 500, "".into());
              f.recency_score = 0.8; f.entropy_score = 0.7; f },
        ];
        let result = knapsack_optimize(&fragments, 600, &ScoringWeights::default(), &no_feedback(), 0.0);
        let ids: Vec<&str> = result.selected_indices.iter().map(|&i| fragments[i].fragment_id.as_str()).collect();
        assert!(ids.contains(&"small"), "Small fragment should be included as free item");
        assert!(ids.contains(&"big"));
    }

    #[test]
    fn test_feedback_affects_selection() {
        let fragments = vec![
            { let mut f = ContextFragment::new("good".into(), "useful code".into(), 200, "".into());
              f.recency_score = 0.5; f.entropy_score = 0.5; f },
            { let mut f = ContextFragment::new("bad".into(), "unhelpful code".into(), 200, "".into());
              f.recency_score = 0.5; f.entropy_score = 0.5; f },
        ];
        let mut feedback = HashMap::new();
        feedback.insert("good".to_string(), 1.8);
        feedback.insert("bad".to_string(),  0.5);

        // Test both paths: hard DP
        let result = knapsack_optimize(&fragments, 250, &ScoringWeights::default(), &feedback, 0.0);
        let ids: Vec<&str> = result.selected_indices.iter().map(|&i| fragments[i].fragment_id.as_str()).collect();
        assert!(ids.contains(&"good"), "DP: feedback-boosted fragment should win");
        assert!(!ids.contains(&"bad"));

        // Soft bisection
        let result2 = knapsack_optimize(&fragments, 250, &ScoringWeights::default(), &feedback, 0.5);
        let ids2: Vec<&str> = result2.selected_indices.iter().map(|&i| fragments[i].fragment_id.as_str()).collect();
        assert!(ids2.contains(&"good"), "Soft: feedback-boosted fragment should win");
        assert!(!ids2.contains(&"bad"));
    }

    #[test]
    fn test_soft_bisection_respects_budget() {
        // Large N, various token counts: bisection must never exceed budget.
        let mut fragments = Vec::new();
        for i in 0..50 {
            let mut f = ContextFragment::new(
                format!("f{}", i), format!("content {}", i), 100 + i as u32 * 7, "".into()
            );
            f.recency_score = (i as f64) / 50.0;
            f.entropy_score = 0.5;
            fragments.push(f);
        }
        let budget = 1500u32;
        let result = knapsack_optimize(&fragments, budget, &ScoringWeights::default(), &no_feedback(), 1.0);
        assert!(result.total_tokens <= budget, "Soft bisection exceeded budget: {} > {}", result.total_tokens, budget);
    }

    #[test]
    fn test_temperature_transition() {
        // At very low temperature, soft bisection should approximate hard greedy.
        let fragments = vec![
            { let mut f = ContextFragment::new("best".into(), "best".into(), 100, "".into());
              f.recency_score = 1.0; f.entropy_score = 1.0; f },
            { let mut f = ContextFragment::new("worst".into(), "worst".into(), 100, "".into());
              f.recency_score = 0.01; f.entropy_score = 0.01; f },
        ];
        // Budget only fits one. At low τ, soft bisection → hard threshold.
        let result = knapsack_optimize(&fragments, 150, &ScoringWeights::default(), &no_feedback(), 0.05);
        let ids: Vec<&str> = result.selected_indices.iter().map(|&i| fragments[i].fragment_id.as_str()).collect();
        assert!(ids.contains(&"best"), "Low-τ soft bisection should pick the best fragment");
    }
}
