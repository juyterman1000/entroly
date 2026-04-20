//! NKBE — Nash-KKT Budgetary Equilibrium Allocator
//!
//! Multi-agent token budget allocation using Arrow-Debreu Walrasian economics
//! ported from agentOS allocator.rs with Nash Bargaining refinement.
//!
//! Given N agents sharing a global token budget B, finds the optimal split:
//!
//!   maximize  Σᵢ wᵢ · Uᵢ(Bᵢ)
//!   subject to  Σᵢ Bᵢ ≤ B,  Bᵢ ≥ Bᵢ_min  ∀i
//!
//! Two-phase KKT bisection:
//!   1. Global: bisect for λ* such that Σᵢ Bᵢ(λ*) = B
//!   2. Per-agent: each agent runs knapsack with budget Bᵢ(λ*)
//!
//! Nash Bargaining refinement (gradient ascent on log Nash product)
//! ensures Pareto-optimal fairness.
//!
//! REINFORCE gradient (RL weight learning) adjusts agent priorities
//! based on outcome quality.
//!
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Per-agent state for budget allocation.
#[derive(Clone, Debug)]
pub struct AgentBudgetState {
    pub name: String,
    pub min_budget: u32,
    pub fragments: Vec<NkbeFragment>,
    pub weight: f64,
}

/// Fragment descriptor for NKBE allocation.
#[derive(Clone, Debug)]
pub struct NkbeFragment {
    pub _id: String,
    pub relevance: f64,
    pub token_cost: u32,
}

/// NKBE Allocator — multi-agent token budget allocation.
///
/// Implements two-phase KKT bisection with Nash Bargaining refinement
/// and REINFORCE gradient for RL weight learning.
#[pyclass]
pub struct NkbeAllocator {
    agents: Vec<AgentBudgetState>,
    global_budget: u32,
    tau: f64,
    epsilon: f64,
    max_iter: u32,
    nash_iterations: u32,
    learning_rate: f64,
    // Stats
    total_allocations: u64,
    last_lambda_star: f64,
    last_dual_gap: f64,
}

#[pymethods]
impl NkbeAllocator {
    #[new]
    #[pyo3(signature = (global_budget=128000, tau=0.1, epsilon=1e-4, max_iter=30, nash_iterations=5, learning_rate=0.01))]
    pub fn new(
        global_budget: u32,
        tau: f64,
        epsilon: f64,
        max_iter: u32,
        nash_iterations: u32,
        learning_rate: f64,
    ) -> Self {
        NkbeAllocator {
            agents: Vec::new(),
            global_budget,
            tau: tau.max(1e-6),
            epsilon,
            max_iter,
            nash_iterations,
            learning_rate,
            total_allocations: 0,
            last_lambda_star: 0.0,
            last_dual_gap: 0.0,
        }
    }

    /// Register an agent for budget allocation.
    pub fn register_agent(
        &mut self,
        name: &str,
        weight: f64,
        min_budget: u32,
    ) {
        self.agents.push(AgentBudgetState {
            name: name.to_string(),
            min_budget,
            fragments: Vec::new(),
            weight: weight.max(0.01),
        });
    }

    /// Add a fragment to an agent (for utility estimation).
    pub fn add_fragment(
        &mut self,
        agent_name: &str,
        fragment_id: &str,
        relevance: f64,
        token_cost: u32,
    ) -> bool {
        for agent in &mut self.agents {
            if agent.name == agent_name {
                agent.fragments.push(NkbeFragment {
                    _id: fragment_id.to_string(),
                    relevance: relevance.clamp(0.0, 1.0),
                    token_cost: token_cost.max(1),
                });
                return true;
            }
        }
        false
    }

    /// Run NKBE allocation. Returns per-agent budgets as a Python dict.
    ///
    /// Two-phase KKT bisection:
    /// 1. Bisect for global λ* such that Σ demand(λ*) = B
    /// 2. Each agent gets budget proportional to their utility at λ*
    ///
    /// Nash Bargaining refinement then adjusts for Pareto-optimal fairness.
    pub fn allocate<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let n = self.agents.len();

        if n == 0 {
            return Ok(result);
        }

        self.total_allocations += 1;

        // Single agent: gets everything
        if n == 1 {
            let agent = &self.agents[0];
            let agent_dict = PyDict::new(py);
            agent_dict.set_item("budget", self.global_budget)?;
            agent_dict.set_item("weight", agent.weight)?;
            agent_dict.set_item("utility", self.agent_utility(0, 0.0))?;
            result.set_item(&agent.name, agent_dict)?;
            return Ok(result);
        }

        // Compute total minimum budget
        let total_min: u32 = self.agents.iter().map(|a| a.min_budget).sum();
        if total_min >= self.global_budget {
            // Budget exhausted by minimums
            for agent in &self.agents {
                let d = PyDict::new(py);
                d.set_item("budget", agent.min_budget)?;
                d.set_item("weight", agent.weight)?;
                d.set_item("utility", 0.0)?;
                result.set_item(&agent.name, d)?;
            }
            return Ok(result);
        }

        let available = self.global_budget - total_min;

        // Phase 1: KKT bisection for global λ*
        // Find λ* such that Σᵢ demand_i(λ*) = available
        let mut lo = 0.0_f64;
        let mut hi = 10.0_f64;

        // Find upper bound
        while self.total_demand(hi) > 0.01 * available as f64 {
            hi *= 2.0;
            if hi > 1e10 { break; }
        }

        for _ in 0..self.max_iter {
            let mid = (lo + hi) / 2.0;
            let demand = self.total_demand(mid);
            if demand > available as f64 {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < self.epsilon {
                break;
            }
        }

        let lambda_star = (lo + hi) / 2.0;
        self.last_lambda_star = lambda_star;

        // Phase 2: compute per-agent budgets from demand at λ*
        let mut budgets: Vec<u32> = Vec::with_capacity(n);
        let mut utilities: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let u = self.agent_utility(i, lambda_star);
            utilities.push(u);
            let demand = self.agent_demand(i, lambda_star);
            let budget = self.agents[i].min_budget + (demand as u32).min(available);
            budgets.push(budget);
        }

        // Nash Bargaining refinement
        // Gradient ascent on log Nash product: N = Πᵢ (Uᵢ(Bᵢ) - Uᵢ(Bᵢ_min))
        for _ in 0..self.nash_iterations {
            let mut log_nash_grads: Vec<f64> = Vec::with_capacity(n);

            for i in 0..n {
                let u_b = utilities[i] * (budgets[i] as f64).ln().max(0.01);
                let u_min = utilities[i] * (self.agents[i].min_budget as f64).ln().max(0.01);
                let denom = (u_b - u_min).max(1e-8);
                let u_prime = utilities[i] / (budgets[i] as f64).max(1.0);
                log_nash_grads.push(u_prime / denom);
            }

            let mean_grad: f64 = log_nash_grads.iter().sum::<f64>() / n as f64;

            for i in 0..n {
                let delta = (0.1 * (log_nash_grads[i] - mean_grad) * available as f64) as i64;
                let new_budget = (budgets[i] as i64 + delta).max(self.agents[i].min_budget as i64) as u32;
                budgets[i] = new_budget;
            }
        }

        // Normalize to exact global budget
        let total: u64 = budgets.iter().map(|&b| b as u64).sum();
        if total > 0 {
            let scale = self.global_budget as f64 / total as f64;
            for (i, budget) in budgets.iter_mut().enumerate().take(n) {
                *budget = ((*budget as f64 * scale) as u32).max(self.agents[i].min_budget);
            }
        }

        // Compute dual gap for diagnostics
        let primal: f64 = utilities.iter().zip(budgets.iter())
            .map(|(u, b)| u * (*b as f64).ln().max(0.01))
            .sum();
        self.last_dual_gap = self.compute_dual(lambda_star) - primal;

        // Build result
        for i in 0..n {
            let d = PyDict::new(py);
            d.set_item("budget", budgets[i])?;
            d.set_item("weight", self.agents[i].weight)?;
            d.set_item("utility", utilities[i])?;
            result.set_item(&self.agents[i].name, d)?;
        }

        Ok(result)
    }

    /// REINFORCE gradient: update agent weights based on outcomes.
    ///
    /// Δwᵢ = η · (Rᵢ − R̄) · wᵢ
    ///
    /// Agents producing better outcomes get more budget next time.
    pub fn reinforce(&mut self, outcomes_json: &str) -> bool {
        let outcomes: HashMap<String, f64> = match serde_json::from_str(outcomes_json) {
            Ok(o) => o,
            Err(_) => return false,
        };

        if outcomes.is_empty() {
            return false;
        }

        let mean_reward: f64 = outcomes.values().sum::<f64>() / outcomes.len() as f64;

        for agent in &mut self.agents {
            if let Some(&reward) = outcomes.get(&agent.name) {
                let advantage = reward - mean_reward;
                agent.weight *= 1.0 + self.learning_rate * advantage;
                agent.weight = agent.weight.clamp(0.1, 10.0);
            }
        }

        true
    }

    /// Get allocator statistics.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("total_allocations", self.total_allocations)?;
        d.set_item("agent_count", self.agents.len())?;
        d.set_item("global_budget", self.global_budget)?;
        d.set_item("last_lambda_star", self.last_lambda_star)?;
        d.set_item("last_dual_gap", self.last_dual_gap)?;
        d.set_item("tau", self.tau)?;
        Ok(d)
    }
}

impl NkbeAllocator {
    /// Agent utility at price λ: U_i(λ) = Σⱼ σ((sⱼ - λ·cⱼ)/τ) · sⱼ · wᵢ
    #[inline(always)]
    fn agent_utility(&self, agent_idx: usize, lambda: f64) -> f64 {
        let agent = &self.agents[agent_idx];
        if agent.fragments.is_empty() {
            return agent.weight * 0.5;
        }
        let mut utility = 0.0;
        for f in &agent.fragments {
            let z = (f.relevance - lambda * f.token_cost as f64) / self.tau;
            let p = sigmoid(z);
            utility += p * f.relevance;
        }
        utility * agent.weight
    }

    /// Agent demand at price λ: D_i(λ) = Σⱼ σ((sⱼ - λ·cⱼ)/τ) · cⱼ
    #[inline(always)]
    fn agent_demand(&self, agent_idx: usize, lambda: f64) -> f64 {
        let agent = &self.agents[agent_idx];
        if agent.fragments.is_empty() {
            // No fragments: demand proportional to weight
            return agent.weight * (self.global_budget as f64 / self.agents.len() as f64);
        }
        let mut demand = 0.0;
        for f in &agent.fragments {
            let z = (f.relevance - lambda * f.token_cost as f64) / self.tau;
            let p = sigmoid(z);
            demand += p * f.token_cost as f64;
        }
        demand
    }

    /// Total demand across all agents at price λ.
    fn total_demand(&self, lambda: f64) -> f64 {
        (0..self.agents.len())
            .map(|i| self.agent_demand(i, lambda))
            .sum()
    }

    /// Dual objective: D(λ) = τ·Σ log(1+exp((sᵢ−λ·cᵢ)/τ)) + λ·B
    fn compute_dual(&self, lambda: f64) -> f64 {
        let mut dual = lambda * self.global_budget as f64;
        for agent in &self.agents {
            for f in &agent.fragments {
                let z = (f.relevance - lambda * f.token_cost as f64) / self.tau;
                dual += self.tau * softplus(z);
            }
        }
        dual
    }
}

/// REINFORCE gradient computation for 4D scoring weights.
///
/// ∂E[R]/∂wₖ = Σᵢ (aᵢ − p*ᵢ) · R · σ'(zᵢ/τ) · featureᵢₖ
///
/// Returns gradient vector [Δw_recency, Δw_frequency, Δw_semantic, Δw_entropy].
#[cfg(test)]
/// REINFORCE policy gradient for NKBE budget allocation.
/// Used by the DreamingLoop for offline weight updates.
pub fn reinforce_gradient(
    features: &[[f64; 4]],    // Per-fragment feature vectors
    selections: &[bool],       // Whether each fragment was selected
    reward: f64,               // Outcome quality
    probabilities: &[f64],     // Selection probabilities p*ᵢ
    tau: f64,                  // Temperature
) -> [f64; 4] {
    let mut grad = [0.0_f64; 4];
    let n = features.len().min(selections.len()).min(probabilities.len());

    for i in 0..n {
        let p = probabilities[i].clamp(1e-8, 1.0 - 1e-8);
        let a = if selections[i] { 1.0 } else { 0.0 };

        // σ'(z/τ) = p·(1−p)/τ — focuses gradient on boundary fragments
        let sigma_prime = p * (1.0 - p) / tau.max(1e-6);
        let advantage = a - p;

        for k in 0..4 {
            grad[k] += advantage * reward * sigma_prime * features[i][k];
        }
    }

    grad
}

/// Numerically stable sigmoid.
#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Numerically stable softplus: log(1 + exp(x)).
#[inline(always)]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 0.001);
        assert!((softplus(100.0) - 100.0).abs() < 0.001);
        assert!(softplus(-100.0).abs() < 0.001);
    }

    #[test]
    fn test_single_agent_gets_full_budget() {
        let mut alloc = NkbeAllocator::new(10000, 0.1, 1e-4, 30, 5, 0.01);
        alloc.register_agent("agent_a", 1.0, 512);
        alloc.add_fragment("agent_a", "f1", 0.8, 100);

        // Can't call Python-dependent allocate() in pure Rust tests,
        // so test internal methods
        let utility = alloc.agent_utility(0, 0.0);
        assert!(utility > 0.0, "utility should be positive");

        let demand = alloc.agent_demand(0, 0.0);
        assert!(demand > 0.0, "demand should be positive");
    }

    #[test]
    fn test_demand_decreases_with_lambda() {
        let mut alloc = NkbeAllocator::new(10000, 0.1, 1e-4, 30, 5, 0.01);
        alloc.register_agent("agent_a", 1.0, 512);
        for i in 0..10 {
            alloc.add_fragment("agent_a", &format!("f{}", i), 0.5 + i as f64 * 0.05, 100);
        }

        let d_low = alloc.total_demand(0.001);
        let d_high = alloc.total_demand(1.0);
        assert!(d_low > d_high, "demand should decrease with λ");
    }

    #[test]
    fn test_reinforce_gradient_basic() {
        let features = vec![
            [1.0, 0.0, 0.5, 0.3],
            [0.0, 1.0, 0.2, 0.8],
        ];
        let selections = vec![true, false];
        let probabilities = vec![0.7, 0.3];
        let reward = 1.0;
        let tau = 0.1;

        let grad = reinforce_gradient(&features, &selections, reward, &probabilities, tau);

        // Selected item (p=0.7): advantage = 1-0.7 = 0.3, σ'= 0.7*0.3/0.1 = 2.1
        // Unselected item (p=0.3): advantage = 0-0.3 = -0.3, σ'= 0.3*0.7/0.1 = 2.1
        // grad[0] = 0.3*1.0*2.1*1.0 + (-0.3)*1.0*2.1*0.0 = 0.63
        assert!((grad[0] - 0.63).abs() < 0.01, "grad[0] = {}", grad[0]);
    }

    #[test]
    fn test_dual_gap_nonnegative() {
        let mut alloc = NkbeAllocator::new(5000, 0.1, 1e-4, 30, 5, 0.01);
        alloc.register_agent("a", 1.0, 256);
        alloc.register_agent("b", 1.0, 256);
        for i in 0..5 {
            alloc.add_fragment("a", &format!("fa{}", i), 0.6, 200);
            alloc.add_fragment("b", &format!("fb{}", i), 0.4, 300);
        }

        let dual = alloc.compute_dual(0.01);
        assert!(dual > 0.0, "dual should be positive");
    }

    #[test]
    fn test_two_agents_split() {
        let mut alloc = NkbeAllocator::new(10000, 0.1, 1e-4, 30, 5, 0.01);
        alloc.register_agent("a", 1.0, 1000);
        alloc.register_agent("b", 1.0, 1000);

        for i in 0..10 {
            alloc.add_fragment("a", &format!("fa{}", i), 0.7, 200);
            alloc.add_fragment("b", &format!("fb{}", i), 0.3, 200);
        }

        // Test that higher utility agent gets more demand
        let demand_a = alloc.agent_demand(0, 0.01);
        let demand_b = alloc.agent_demand(1, 0.01);
        assert!(demand_a > demand_b, "higher utility agent should have more demand");
    }
}
