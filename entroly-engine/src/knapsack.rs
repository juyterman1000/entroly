//! Unified Context Optimizer — Rust implementation.
//!
//! # Differentiable Soft Bisection (primary path, τ ≥ 0.05)
//!
//! Find the dual variable λ* via 30-step bisection such that:
//!
//!   g(λ) = Σᵢ σ((sᵢ − λ·cᵢ) / τ) · cᵢ  −  B  =  0
//!
//! where sᵢ = w^T · featuresᵢ (pre-softcap linear score, same as REINFORCE) and
//! cᵢ = tokensᵢ. g is strictly monotone decreasing in λ, so bisection always
//! converges.
//!
//! λ* is the **exact Lagrange multiplier** for the token-budget constraint under
//! the continuous KKT relaxation of the 0/1 knapsack — a principled dual variable,
//! not a heuristic threshold. Note that it multiplies the *cost*: λ carries units
//! of value-per-token, so `sᵢ − λ·cᵢ` is a reduced cost. A constant score offset
//! `sᵢ − th` is a different rule and is only equivalent when every cᵢ is equal;
//! this header described that weaker form until now, while all three call sites
//! have always computed `σ((sᵢ − λ·cᵢ)/τ)`.
//!
//! After bisection, sort fragments by p_i = σ((sᵢ − λ*·cᵢ) / τ) descending —
//! equivalently by reduced cost, the LP-duality ordering — and greedily fill the
//! *hard* budget (context windows are hard limits).
//!
//! Complexity: O(30 · N) bisection + O(N log N) sort = O(N log N).
//!   ≈ 33× faster than the O(N × Q=1000) DP table for N=500.
//!
//! Train/test consistency: the same linear score sᵢ and the same σ(·/τ) appear
//! in the REINFORCE backward pass → no train/test mismatch.
//!
//! Convergence: as τ → 0, p_i → I(sᵢ > λ*·cᵢ) = I(sᵢ/cᵢ > λ*), so the greedy
//! fill recovers the exact density-sorted greedy — which is the ordering the
//! ½-approximation below refers to. (Under the `sᵢ − th` form it would instead
//! recover a *score*-sorted greedy, a different and weaker algorithm; that is why
//! the distinction is worth stating.) The objective here is linear (Σ sᵢ·xᵢ), i.e.
//! modular, so density-greedy on a knapsack gives the ½-approximation of
//! Dantzig-style rounding — NOT (1-1/e), which requires a submodular
//! objective. If redundancy/diversity terms are added to the score (making
//! it submodular), Sviridenko's partial-enumeration variant would be needed
//! to recover the (1-1/e - ε) bound; this file does not do that.
//!
//! # Hard DP fallback (τ < 0.05)
//!
//! Exact 0/1 DP with budget quantization: O(N × Q), Q = 1000.
//! Used when weights have converged (τ is at floor) for maximum precision.
//!
use crate::fragment::{compute_relevance, ContextFragment};
use std::collections::HashMap;

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
pub struct KnapsackResult {
    pub selected_indices: Vec<usize>,
    pub total_tokens: u32,
    pub total_relevance: f64,
    pub(crate) _method: &'static str,
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
    (w.recency * frag.recency_score
        + w.frequency * frag.frequency_score
        + w.semantic * frag.semantic_score
        + w.entropy * frag.entropy_score)
        * fm.max(0.01)
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
            _method: "empty",
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
            _method: "pinned_only",
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
            let fm = feedback_mults
                .get(&fragments[i].fragment_id)
                .copied()
                .unwrap_or(1.0);
            let score = if use_soft {
                linear_score(&fragments[i], weights, fm)
            } else {
                compute_relevance(
                    &fragments[i],
                    weights.recency,
                    weights.frequency,
                    weights.semantic,
                    weights.entropy,
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
    let (_method, mut selected, lambda_star, dual_gap) = if use_soft {
        let (sel, lam, gap) =
            soft_bisection_select(&scored, fragments, remaining_budget, temperature);
        ("soft_bisection", sel, lam, gap)
    } else if scored.len() <= 2000 {
        (
            "exact_dp",
            knapsack_dp(&scored, fragments, remaining_budget),
            0.0,
            0.0,
        )
    } else {
        (
            "greedy_approx",
            knapsack_greedy(&scored, fragments, remaining_budget),
            0.0,
            0.0,
        )
    };

    // Merge pinned + selected
    selected.extend(pinned_indices.iter());
    let total_tokens: u32 = selected.iter().map(|&i| fragments[i].token_count).sum();
    let total_relevance: f64 = selected
        .iter()
        .map(|&i| {
            let fm = feedback_mults
                .get(&fragments[i].fragment_id)
                .copied()
                .unwrap_or(1.0);
            compute_relevance(
                &fragments[i],
                weights.recency,
                weights.frequency,
                weights.semantic,
                weights.entropy,
                fm,
            )
        })
        .sum();

    KnapsackResult {
        selected_indices: selected,
        total_tokens,
        total_relevance,
        _method,
        lambda_star,
        dual_gap,
    }
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
        scored
            .iter()
            .map(|&(idx, score)| {
                let tc = fragments[idx].token_count as f64;
                sigmoid((score - lambda * tc) / tau) * tc
            })
            .sum()
    };

    // Fast path: all items fit at λ=0.
    if expected_tokens(0.0) <= budget_f {
        return 0.0;
    }

    let max_score = scored
        .iter()
        .map(|&(_, s)| s)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_tokens = scored
        .iter()
        .map(|&(idx, _)| fragments[idx].token_count as f64)
        .fold(f64::INFINITY, f64::min)
        .max(1.0);
    let mut hi = (max_score + 5.0 * tau) / (min_tokens * tau).max(1e-10);
    let mut iters = 0;
    while expected_tokens(hi) >= budget_f && iters < 60 {
        hi *= 2.0;
        iters += 1;
    }
    if expected_tokens(hi) >= budget_f {
        return 0.0;
    }

    let mut lo = 0.0_f64;
    for _ in 0..30 {
        let mid = (lo + hi) * 0.5;
        if expected_tokens(mid) > budget_f {
            lo = mid;
        } else {
            hi = mid;
        }
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
        scored
            .iter()
            .map(|&(idx, score)| {
                let tc = fragments[idx].token_count as f64;
                sigmoid((score - lambda * tc) / tau) * tc
            })
            .sum()
    };

    // Fast path: everything fits, so there is nothing to choose between.
    //
    // This tested `expected_tokens(0.0) <= budget_f`, which is
    // Σ σ(sᵢ/τ)·tokensᵢ -- a probability-weighted count. σ is strictly less
    // than 1, so the expected total is always strictly below the true total,
    // and a set that does not fit could still pass the test. It then returned
    // every item, and the caller received a hard selection over budget:
    // three fragments costing 1 + 1 + 50 against a budget of 50 gave an
    // expected 41.7, took the fast path, and reported total_tokens = 52.
    //
    // Feasibility is a property of the actual token counts, not of the soft
    // relaxation used to price them. `u64` because the sum of `u32` costs is
    // not bounded by `u32`.
    let actual_tokens: u64 = scored
        .iter()
        .map(|&(idx, _)| fragments[idx].token_count as u64)
        .sum();
    if actual_tokens <= budget as u64 {
        return (scored.iter().map(|&(idx, _)| idx).collect(), 0.0, 0.0);
    }

    // Find λ_hi s.t. g(λ_hi) < 0 (expected tokens < budget).
    let max_score = scored
        .iter()
        .map(|&(_, s)| s)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_tokens = scored
        .iter()
        .map(|&(idx, _)| fragments[idx].token_count as f64)
        .fold(f64::INFINITY, f64::min)
        .max(1.0);
    let mut hi = (max_score + 5.0 * tau) / (min_tokens * tau).max(1e-10);
    let mut iters = 0;
    while expected_tokens(hi) >= budget_f && iters < 60 {
        hi *= 2.0;
        iters += 1;
    }
    if expected_tokens(hi) >= budget_f {
        return (knapsack_greedy(scored, fragments, budget), 0.0, 0.0);
    }

    // 30-step bisection on λ ∈ [0, hi]. Each iteration: O(N). Total: O(30·N).
    let mut lo = 0.0_f64;
    for _ in 0..30 {
        let mid = (lo + hi) * 0.5;
        if expected_tokens(mid) > budget_f {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lambda_star = (lo + hi) * 0.5;

    // Compute exact KKT probabilities at λ*.
    // Sorting by p_i ≡ sorting by reduced cost (s_i − λ*·tokens_i) — LP duality ordering.
    let score_map: HashMap<usize, f64> = scored.iter().copied().collect();
    let mut with_probs: Vec<(usize, f64)> = scored
        .iter()
        .map(|&(idx, score)| {
            let tc = fragments[idx].token_count as f64;
            let p = sigmoid((score - lambda_star * tc) / tau);
            (idx, p)
        })
        .collect();

    // Ties break on fragment id, and the score comparison is a total order.
    //
    // `sort_unstable_by` does not preserve input order for equal elements, and
    // `partial_cmp(..).unwrap_or(Equal)` reported every tie -- and every NaN --
    // as equal. So when two fragments scored identically and only one fit in
    // the remaining budget, which one survived was decided by the sort's
    // partitioning of whatever order the input arrived in, not by this
    // algorithm. Two processes could then select differently from identical
    // input: `scripts/onboarding_self_dogfood.py` caught exactly that, reporting
    // total_tokens_saved 85459 against 85453 for one repository, one budget and
    // one code path. Six tokens is one boundary fragment.
    //
    // `total_cmp` also removes the NaN-equals-everything case, which made the
    // comparator non-transitive and left the resulting order unspecified.
    with_probs.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| fragments[a.0].fragment_id.cmp(&fragments[b.0].fragment_id))
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
            // Primal value: realized hard-selection objective Σ s_i.
            // Previous code used p_i·s_i (soft relaxation), which systematically
            // underestimates the primal and inflates the ADGT dual gap.
            let s_i = score_map.get(&idx).copied().unwrap_or(0.0);
            primal_value += s_i;
        }
        if remaining == 0 {
            break;
        }
    }

    // ── Adaptive Dual Gap Temperature (ADGT) signal ──────────────────────────
    // Compute D(λ*) = τ · Σ log(1 + exp((s_i − λ*·c_i)/τ)) + λ*·B  [log-sum-exp dual]
    // This is the exact smooth upper bound on the primal objective.
    // dual_gap = D(λ*) − primal ∈ [0, τ·N·log(2)]
    //   → gap ≈ 0: weights converged, can lower temperature
    //   → gap ≈ τ·N·log(2): all p_i ≈ 0.5, fully uncertain, keep temperature high
    let dual_value: f64 = scored
        .iter()
        .map(|&(idx, score)| {
            let tc = fragments[idx].token_count as f64;
            let z = (score - lambda_star * tc) / tau;
            // Numerically stable log(1 + exp(z)) = log1p(exp(z))
            tau * if z > 20.0 {
                z
            } else {
                (1.0_f64 + z.exp()).ln()
            }
        })
        .sum::<f64>()
        + lambda_star * budget_f;

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
fn knapsack_dp(scored: &[(usize, f64)], fragments: &[ContextFragment], budget: u32) -> Vec<usize> {
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

    let free_count = free_items.len();
    let mut selected = free_items;
    let mut w = qb;
    for i in (0..n).rev() {
        if keep[i][w] {
            let (orig_idx, _, cost) = dp_items[i];
            selected.push(orig_idx);
            w -= cost;
        }
    }

    // Post-DP budget guard: floor-division quantization can let the real
    // token total exceed the hard budget by up to K×(g−1) tokens. Trim
    // last-selected DP items (lowest backtrace priority) until it fits.
    while selected.len() > free_count {
        let real: u32 = selected.iter().map(|&i| fragments[i].token_count).sum();
        if real <= budget {
            break;
        }
        selected.pop();
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
    // Total order, ties broken on fragment id -- see the note in
    // `knapsack_soft_bisection`. Density ties are more likely here, not less:
    // equal-relevance fragments of equal length produce byte-identical ratios.
    density.sort_unstable_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| fragments[a.0].fragment_id.cmp(&fragments[b.0].fragment_id))
    });

    let mut selected = Vec::new();
    let mut remaining = budget;
    for (idx, _) in density {
        if fragments[idx].token_count <= remaining {
            selected.push(idx);
            remaining -= fragments[idx].token_count;
        }
        if remaining == 0 {
            break;
        }
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
    pinned
        .iter()
        .map(|&i| {
            let fm = feedback_mults
                .get(&fragments[i].fragment_id)
                .copied()
                .unwrap_or(1.0);
            compute_relevance(
                &fragments[i],
                weights.recency,
                weights.frequency,
                weights.semantic,
                weights.entropy,
                fm,
            )
        })
        .sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    /// Feasibility is a property of the real token counts, not of the soft
    /// relaxation used to price them.
    ///
    /// The fast path asked whether `Σ σ(sᵢ/τ)·tokensᵢ ≤ B` before returning
    /// every item. σ is strictly below 1, so that expected total is always
    /// below the true total and a set that does not fit could still pass.
    /// Measured: fragments costing 1 + 1 + 50 against a budget of 50 priced at
    /// an expected 41.7, took the fast path, and returned `total_tokens = 52`.
    /// Overrunning the budget is precisely what overflows a context window.
    #[test]
    fn the_include_everything_fast_path_checks_real_tokens_not_expected_tokens() {
        let weights = ScoringWeights::default();
        let mk = |id: &str, tokens: u32, score: f64| {
            let mut f = ContextFragment::new(id.into(), "c".into(), tokens, "".into());
            f.recency_score = score;
            f.frequency_score = score;
            f.semantic_score = score;
            f.entropy_score = score;
            f
        };

        let items = vec![mk("a", 1, 0.9), mk("b", 1, 0.5), mk("c", 50, 0.7)];
        let budget = 50;
        assert_eq!(
            items.iter().map(|f| f.token_count).sum::<u32>(),
            52,
            "fixture must not fit, or it proves nothing"
        );

        let soft = knapsack_optimize(&items, budget, &weights, &no_feedback(), 0.5);
        assert!(
            soft.total_tokens <= budget,
            "soft bisection returned {} tokens against a budget of {budget}",
            soft.total_tokens
        );

        // And when everything genuinely fits, the fast path must still take it.
        let fits = vec![mk("a", 1, 0.9), mk("b", 1, 0.5), mk("c", 10, 0.7)];
        let all = knapsack_optimize(&fits, 50, &weights, &no_feedback(), 0.5);
        assert_eq!(all.selected_indices.len(), 3, "a set that fits must be kept whole");
        assert_eq!(all.total_tokens, 12);
    }

    /// Degenerate inputs a caller can actually produce. Each of these is a
    /// classic way a knapsack implementation goes wrong: a zero budget, an item
    /// that cannot fit, zero-cost items (density divides by weight), and
    /// non-finite scores arriving from an upstream model.
    #[test]
    fn degenerate_inputs_never_overrun_the_budget_or_produce_nonsense() {
        let weights = ScoringWeights::default();
        let mk = |id: &str, tokens: u32, score: f64| {
            let mut f = ContextFragment::new(id.into(), "c".into(), tokens, "".into());
            f.recency_score = score;
            f.frequency_score = score;
            f.semantic_score = score;
            f.entropy_score = score;
            f
        };

        for temperature in [0.0_f64, 0.5] {
            // Zero budget: nothing may be selected.
            let items = vec![mk("a", 10, 0.9), mk("b", 20, 0.8)];
            let r = knapsack_optimize(&items, 0, &weights, &no_feedback(), temperature);
            assert_eq!(r.total_tokens, 0, "zero budget selected {} tokens", r.total_tokens);
            assert!(r.selected_indices.is_empty());

            // Every item larger than the budget: still nothing.
            let big = vec![mk("a", 500, 0.9), mk("b", 900, 0.99)];
            let r = knapsack_optimize(&big, 100, &weights, &no_feedback(), temperature);
            assert!(r.total_tokens <= 100, "overran with only oversized items");

            // Zero-token items: density is value/weight, so weight 0 is the
            // division a density-greedy solver trips over.
            let free = vec![mk("a", 0, 0.9), mk("b", 0, 0.5), mk("c", 50, 0.7)];
            let r = knapsack_optimize(&free, 50, &weights, &no_feedback(), temperature);
            assert!(r.total_tokens <= 50);
            assert!(r.total_relevance.is_finite(), "zero-cost items produced a non-finite score");

            // Non-finite scores from upstream must not poison the result.
            let poisoned = vec![
                mk("nan", 10, f64::NAN),
                mk("inf", 10, f64::INFINITY),
                mk("ok", 10, 0.5),
            ];
            let r = knapsack_optimize(&poisoned, 20, &weights, &no_feedback(), temperature);
            assert!(r.total_tokens <= 20, "overran the budget on non-finite scores");
            assert!(
                r.total_relevance.is_finite(),
                "a NaN or infinite input score reached the reported relevance"
            );
        }
    }

    /// Deterministic LCG so a failure is reproducible from the seed alone.
    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / ((1u64 << 31) as f64)
    }

    /// The soft bisection is a heuristic; the DP below 0.05 is exact. Randomised
    /// instances let the exact path referee the approximate one.
    ///
    /// Three properties, in increasing order of how much a violation would cost:
    ///   1. neither path may exceed the token budget -- overrunning it is what
    ///      blows a context window;
    ///   2. the DP must dominate, since it is the optimum by construction;
    ///   3. the heuristic must stay within the Dantzig-style 1/2 that CLAUDE.md
    ///      claims for a modular objective (explicitly *not* 1 - 1/e).
    #[test]
    fn soft_bisection_is_feasible_and_within_the_claimed_half_of_optimal() {
        let weights = ScoringWeights::default();
        let mut seed = 0x5EED_1234_u64;
        let mut worst_ratio = f64::INFINITY;

        for case in 0..60 {
            let n = 6 + (case % 9);
            let fragments: Vec<ContextFragment> = (0..n)
                .map(|k| {
                    // Mixed magnitudes on purpose. The first version of this
                    // generator drew 20..420 uniformly and never produced the
                    // shape that breaks feasibility: several very small items
                    // beside one large one, where the soft relaxation prices the
                    // set under budget while the true cost is over it.
                    let tokens = if lcg(&mut seed) < 0.4 {
                        1 + (lcg(&mut seed) * 5.0) as u32
                    } else {
                        20 + (lcg(&mut seed) * 400.0) as u32
                    };
                    let mut f = ContextFragment::new(
                        format!("f{k:03}"),
                        "x".repeat(8),
                        tokens,
                        "".into(),
                    );
                    f.recency_score = lcg(&mut seed);
                    f.frequency_score = lcg(&mut seed);
                    f.semantic_score = lcg(&mut seed);
                    f.entropy_score = lcg(&mut seed);
                    f
                })
                .collect();

            let total: u32 = fragments.iter().map(|f| f.token_count).sum();
            // A budget that binds: too large and every instance is trivial.
            let budget = (total as f64 * (0.25 + 0.4 * lcg(&mut seed))) as u32;
            if budget == 0 {
                continue;
            }

            let exact = knapsack_optimize(&fragments, budget, &weights, &no_feedback(), 0.0);
            let soft = knapsack_optimize(&fragments, budget, &weights, &no_feedback(), 0.5);

            assert!(
                exact.total_tokens <= budget,
                "case {case}: exact DP overran the budget ({} > {budget})",
                exact.total_tokens
            );
            assert!(
                soft.total_tokens <= budget,
                "case {case}: soft bisection overran the budget ({} > {budget})",
                soft.total_tokens
            );
            assert!(
                exact.total_relevance >= soft.total_relevance - 1e-9,
                "case {case}: the exact DP is optimal by construction, so it                  cannot score below the heuristic ({} < {})",
                exact.total_relevance,
                soft.total_relevance
            );

            if exact.total_relevance > 1e-9 {
                let ratio = soft.total_relevance / exact.total_relevance;
                worst_ratio = worst_ratio.min(ratio);
                assert!(
                    ratio >= 0.5 - 1e-9,
                    "case {case}: heuristic scored {ratio:.4} of optimal, below                      the Dantzig-style 1/2 claimed for a modular objective"
                );
            }
        }

        assert!(worst_ratio.is_finite(), "no binding instance was generated");
    }

    use super::*;
    use crate::fragment::ContextFragment;

    fn no_feedback() -> HashMap<String, f64> {
        HashMap::new()
    }


    /// Selection must not depend on the order the caller supplied fragments in.
    ///
    /// The selection sorts compared a float score with no secondary key, so
    /// equally-scored fragments were left in whatever relative order the sort
    /// happened to produce from the incoming sequence. That makes the result a
    /// function of input order, and input order is not a constant: upstream
    /// assembles candidates through maps and merges whose iteration order Rust
    /// seeds per process. `scripts/onboarding_self_dogfood.py` observed the
    /// consequence -- total_tokens_saved 85459 against 85453 for one
    /// repository, one budget and one code path. Six tokens is one boundary
    /// fragment changing sides.
    ///
    /// Every fragment here scores and costs identically, so the budget boundary
    /// is decided purely by the tie-break. Breaking on `fragment_id` rather than
    /// on position makes the outcome invariant to permutation, which is the
    /// property that actually holds across processes.
    #[test]
    fn selection_is_invariant_to_input_order_when_scores_tie() {
        let mk = |id: usize| {
            let mut f = ContextFragment::new(
                format!("frag-{id:02}"),
                "identical content".into(),
                100,
                "".into(),
            );
            // Identical on every scoring axis: guarantees an exact tie.
            f.recency_score = 0.5;
            f.frequency_score = 0.5;
            f.semantic_score = 0.5;
            f.entropy_score = 0.5;
            f
        };

        let weights = ScoringWeights::default();
        let budget = 1000; // exactly ten of twenty-four fragments fit

        // Soft bisection only. The exact DP below 0.05 reconstructs its
        // selection by walking the table, so among equal-value items it keeps
        // whichever the table reached first -- a positional choice this fix
        // does not touch and cannot claim. Asserting invariance there would be
        // asserting a property the code does not have.
        for temperature in [0.5_f64] {
            let ascending: Vec<ContextFragment> = (0..24).map(mk).collect();
            let baseline = knapsack_optimize(
                &ascending,
                budget,
                &weights,
                &no_feedback(),
                temperature,
            );
            let mut expected: Vec<String> = baseline
                .selected_indices
                .iter()
                .map(|&i| ascending[i].fragment_id.clone())
                .collect();
            expected.sort();
            assert_eq!(expected.len(), 10, "budget must bind at ten fragments");

            // Same set of fragments, different orders. A deterministic selector
            // must choose the same fragments every time.
            let permutations: [Vec<usize>; 3] = [
                (0..24).rev().collect(),
                (0..24).map(|i| (i * 7) % 24).collect(),
                (0..24).map(|i| (i * 5 + 3) % 24).collect(),
            ];
            for perm in permutations {
                let shuffled: Vec<ContextFragment> = perm.iter().map(|&i| mk(i)).collect();
                let got = knapsack_optimize(
                    &shuffled,
                    budget,
                    &weights,
                    &no_feedback(),
                    temperature,
                );
                let mut ids: Vec<String> = got
                    .selected_indices
                    .iter()
                    .map(|&i| shuffled[i].fragment_id.clone())
                    .collect();
                ids.sort();
                assert_eq!(
                    ids, expected,
                    "input order changed the selection (temperature {temperature})"
                );
                assert_eq!(got.total_tokens, baseline.total_tokens);
            }
        }
    }

    #[test]
    fn test_knapsack_selects_optimal() {
        let fragments = vec![
            {
                let mut f = ContextFragment::new("a".into(), "hi val small".into(), 100, "".into());
                f.recency_score = 1.0;
                f.entropy_score = 0.9;
                f
            },
            {
                let mut f = ContextFragment::new("b".into(), "lo val large".into(), 900, "".into());
                f.recency_score = 0.1;
                f.entropy_score = 0.1;
                f
            },
            {
                let mut f = ContextFragment::new("c".into(), "med val med".into(), 400, "".into());
                f.recency_score = 0.7;
                f.entropy_score = 0.6;
                f
            },
        ];
        // Hard path (τ=0): exact DP
        let result = knapsack_optimize(
            &fragments,
            500,
            &ScoringWeights::default(),
            &no_feedback(),
            0.0,
        );
        let ids: Vec<&str> = result
            .selected_indices
            .iter()
            .map(|&i| fragments[i].fragment_id.as_str())
            .collect();
        assert!(ids.contains(&"a"), "Should select high-value 'a'");
        assert!(!ids.contains(&"b"), "Should not select low-value 'b'");
        assert!(result.total_tokens <= 500);
    }

    #[test]
    fn test_soft_bisection_selects_optimal() {
        let fragments = vec![
            {
                let mut f = ContextFragment::new("a".into(), "hi val small".into(), 100, "".into());
                f.recency_score = 1.0;
                f.entropy_score = 0.9;
                f
            },
            {
                let mut f = ContextFragment::new("b".into(), "lo val large".into(), 900, "".into());
                f.recency_score = 0.1;
                f.entropy_score = 0.1;
                f
            },
            {
                let mut f = ContextFragment::new("c".into(), "med val med".into(), 400, "".into());
                f.recency_score = 0.7;
                f.entropy_score = 0.6;
                f
            },
        ];
        // Soft path (τ=0.1): bisection — should still prefer 'a' and 'c' over 'b'
        let result = knapsack_optimize(
            &fragments,
            500,
            &ScoringWeights::default(),
            &no_feedback(),
            0.1,
        );
        let ids: Vec<&str> = result
            .selected_indices
            .iter()
            .map(|&i| fragments[i].fragment_id.as_str())
            .collect();
        assert!(
            ids.contains(&"a"),
            "Soft bisection should select high-value 'a'"
        );
        assert!(
            !ids.contains(&"b"),
            "Soft bisection should exclude low-value 'b'"
        );
        assert!(result.total_tokens <= 500);
        assert_eq!(result._method, "soft_bisection");
    }

    #[test]
    fn test_small_fragments_not_penalized() {
        let fragments = vec![
            {
                let mut f = ContextFragment::new("small".into(), "tiny".into(), 10, "".into());
                f.recency_score = 1.0;
                f.entropy_score = 0.9;
                f
            },
            {
                let mut f =
                    ContextFragment::new("big".into(), "large content here".into(), 500, "".into());
                f.recency_score = 0.8;
                f.entropy_score = 0.7;
                f
            },
        ];
        let result = knapsack_optimize(
            &fragments,
            600,
            &ScoringWeights::default(),
            &no_feedback(),
            0.0,
        );
        let ids: Vec<&str> = result
            .selected_indices
            .iter()
            .map(|&i| fragments[i].fragment_id.as_str())
            .collect();
        assert!(
            ids.contains(&"small"),
            "Small fragment should be included as free item"
        );
        assert!(ids.contains(&"big"));
    }

    #[test]
    fn test_feedback_affects_selection() {
        let fragments = vec![
            {
                let mut f =
                    ContextFragment::new("good".into(), "useful code".into(), 200, "".into());
                f.recency_score = 0.5;
                f.entropy_score = 0.5;
                f
            },
            {
                let mut f =
                    ContextFragment::new("bad".into(), "unhelpful code".into(), 200, "".into());
                f.recency_score = 0.5;
                f.entropy_score = 0.5;
                f
            },
        ];
        let mut feedback = HashMap::new();
        feedback.insert("good".to_string(), 1.8);
        feedback.insert("bad".to_string(), 0.5);

        // Test both paths: hard DP
        let result = knapsack_optimize(&fragments, 250, &ScoringWeights::default(), &feedback, 0.0);
        let ids: Vec<&str> = result
            .selected_indices
            .iter()
            .map(|&i| fragments[i].fragment_id.as_str())
            .collect();
        assert!(
            ids.contains(&"good"),
            "DP: feedback-boosted fragment should win"
        );
        assert!(!ids.contains(&"bad"));

        // Soft bisection
        let result2 =
            knapsack_optimize(&fragments, 250, &ScoringWeights::default(), &feedback, 0.5);
        let ids2: Vec<&str> = result2
            .selected_indices
            .iter()
            .map(|&i| fragments[i].fragment_id.as_str())
            .collect();
        assert!(
            ids2.contains(&"good"),
            "Soft: feedback-boosted fragment should win"
        );
        assert!(!ids2.contains(&"bad"));
    }

    #[test]
    fn test_soft_bisection_respects_budget() {
        // Large N, various token counts: bisection must never exceed budget.
        let mut fragments = Vec::new();
        for i in 0..50 {
            let mut f = ContextFragment::new(
                format!("f{}", i),
                format!("content {}", i),
                100 + i as u32 * 7,
                "".into(),
            );
            f.recency_score = (i as f64) / 50.0;
            f.entropy_score = 0.5;
            fragments.push(f);
        }
        let budget = 1500u32;
        let result = knapsack_optimize(
            &fragments,
            budget,
            &ScoringWeights::default(),
            &no_feedback(),
            1.0,
        );
        assert!(
            result.total_tokens <= budget,
            "Soft bisection exceeded budget: {} > {}",
            result.total_tokens,
            budget
        );
    }

    #[test]
    fn test_quantized_dp_respects_real_token_budget() {
        let fragments = vec![
            ContextFragment::new("a".into(), "a".into(), 1001, "".into()),
            ContextFragment::new("b".into(), "b".into(), 1001, "".into()),
        ];
        let selected = knapsack_dp(&[(0, 1.0), (1, 0.9)], &fragments, 2000);
        let real_tokens: u32 = selected
            .iter()
            .map(|&index| fragments[index].token_count)
            .sum();

        assert!(real_tokens <= 2000, "selected {real_tokens} real tokens");
    }

    #[test]
    fn test_temperature_transition() {
        // At very low temperature, soft bisection should approximate hard greedy.
        let fragments = vec![
            {
                let mut f = ContextFragment::new("best".into(), "best".into(), 100, "".into());
                f.recency_score = 1.0;
                f.entropy_score = 1.0;
                f
            },
            {
                let mut f = ContextFragment::new("worst".into(), "worst".into(), 100, "".into());
                f.recency_score = 0.01;
                f.entropy_score = 0.01;
                f
            },
        ];
        // Budget only fits one. At low τ, soft bisection → hard threshold.
        let result = knapsack_optimize(
            &fragments,
            150,
            &ScoringWeights::default(),
            &no_feedback(),
            0.05,
        );
        let ids: Vec<&str> = result
            .selected_indices
            .iter()
            .map(|&i| fragments[i].fragment_id.as_str())
            .collect();
        assert!(
            ids.contains(&"best"),
            "Low-τ soft bisection should pick the best fragment"
        );
    }
}
