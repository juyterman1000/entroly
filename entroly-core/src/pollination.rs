//! Pollination Engine — TD(0) RL Knowledge-Sharing + Experience Packs
//!
//! Combines two ebbiforge patterns:
//!   1. `swarm/pollination.rs` — TD(0) value function + sigmoid share probability
//!   2. `intel/pollination.rs` — Experience packs + cross-pollination
//!
//! Each agent maintains a V(share) value learned via temporal-difference RL.
//! Share probability = σ(V(share) / temperature).
//! High surprise temporarily boosts willingness to broadcast (emergency override).
//!
//! Bridge to AMP:
//!   Pollination V(share) high  →  AMP reciprocity high  →  flatter curvature  →  cheaper negotiation
//!   AMP resolution success     →  Pollination reward +1  →  V(share) increases
//!
//! References:
//!   - Sutton & Barto (2018): TD(0) temporal-difference learning
//!   - ebbiforge-core: swarm/pollination.rs, intel/pollination.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

// ════════════════════════════════════════════════════════════════════
//  EXPERIENCE PACK — Shareable Learning Unit
// ════════════════════════════════════════════════════════════════════

/// A single lesson from an agent's experience.
#[derive(Clone, Debug)]
#[allow(dead_code)] // fields are part of the public lesson schema; read by pyo3 serializers
struct Lesson {
    /// What was attempted.
    description: String,
    /// Whether it succeeded.
    success: bool,
    /// Surprise score at the time.
    surprise: f32,
    /// Domain tag (e.g. "finance", "health").
    domain: String,
    /// Tick when the lesson was learned.
    tick: u64,
}

/// An experience pack: a bundle of anonymized lessons for sharing.
#[derive(Clone, Debug)]
#[allow(dead_code)] // fields are part of the wire format
struct ExperiencePack {
    /// Source agent ID.
    source_agent: String,
    /// Creation timestamp.
    created_at: u64,
    /// The lessons bundled in this pack.
    lessons: Vec<Lesson>,
}

// ════════════════════════════════════════════════════════════════════
//  POLLINATOR STATE — Per-Agent TD(0) RL
// ════════════════════════════════════════════════════════════════════

/// Per-agent pollination state: TD(0) value function + sharing dynamics.
#[derive(Clone, Debug)]
struct PollinatorState {
    /// Raw V(share) — learned via TD(0).
    raw_eagerness: f32,
    /// Bounded share probability: σ(raw_eagerness / temperature).
    share_probability: f32,
    /// Active shares awaiting feedback: target_agent_id → tick shared.
    active_shares: HashMap<String, u64>,
    /// TD learning rate (α).
    alpha: f32,
    /// TD discount factor (γ).
    gamma: f32,
    /// Sigmoid temperature.
    temperature: f32,
    /// Max ticks after share to attribute credit.
    recency_window: u64,
    /// Surprise broadcast weight (emergency override).
    surprise_weight: f32,

    // Stats
    total_shares: u64,
    total_rewards: u64,
    total_lessons_given: u64,
    total_lessons_received: u64,
}

impl PollinatorState {
    fn new(alpha: f32, gamma: f32, temperature: f32) -> Self {
        let raw = 0.0;
        let prob = 1.0 / (1.0 + (-raw / temperature).exp());
        PollinatorState {
            raw_eagerness: raw,
            share_probability: prob,
            active_shares: HashMap::new(),
            alpha,
            gamma,
            temperature,
            recency_window: 10,
            surprise_weight: 0.5,
            total_shares: 0,
            total_rewards: 0,
            total_lessons_given: 0,
            total_lessons_received: 0,
        }
    }

    /// Should this agent share knowledge right now?
    /// High surprise temporarily boosts willingness (emergency override).
    fn should_share(&self, random_val: f32, current_surprise: f32) -> bool {
        let effective = (self.share_probability + current_surprise * self.surprise_weight).min(1.0);
        random_val < effective
    }

    /// Register a share event for future credit assignment.
    fn register_share(&mut self, target: String, tick: u64) {
        self.active_shares.insert(target, tick);
        self.total_shares += 1;
    }

    /// Apply TD(0) feedback for a previous share.
    ///
    /// V(s) ← V(s) + α × [R + γ×V(s') − V(s)]
    fn apply_feedback(&mut self, target: &str, reward: f32, current_tick: u64) {
        if let Some(share_tick) = self.active_shares.remove(target) {
            if current_tick.saturating_sub(share_tick) <= self.recency_window {
                let current_v = self.raw_eagerness;
                let next_v = current_v; // single-state approximation
                let td_error = reward + self.gamma * next_v - current_v;
                self.raw_eagerness += self.alpha * td_error;
                self.update_probability();
                self.total_rewards += 1;
            }
        }
    }

    /// Recalculate share_probability from raw_eagerness via sigmoid.
    fn update_probability(&mut self) {
        self.share_probability = 1.0 / (1.0 + (-self.raw_eagerness / self.temperature).exp());
    }
}

// ════════════════════════════════════════════════════════════════════
//  POLLINATION ENGINE — Multi-Agent Knowledge Sharing
// ════════════════════════════════════════════════════════════════════

/// Cross-Pollination Engine.
///
/// Manages TD(0) RL-driven knowledge sharing across all agents.
/// Each agent learns whether sharing is worth it through rewards.
#[pyclass]
pub struct PollinationEngine {
    /// Per-agent pollinator states.
    agents: HashMap<String, PollinatorState>,
    /// Pending experience packs.
    pending_packs: Vec<ExperiencePack>,
    /// TD learning rate.
    alpha: f32,
    /// TD discount factor.
    gamma: f32,
    /// Sigmoid temperature.
    temperature: f32,
    /// Current tick.
    current_tick: u64,

    // Global stats
    total_packs_created: u64,
    total_packs_ingested: u64,
    total_lessons_shared: u64,
}

#[pymethods]
impl PollinationEngine {
    /// Create a new pollination engine.
    ///
    /// # Arguments
    /// - `alpha`: TD(0) learning rate (default: 0.1)
    /// - `gamma`: TD discount factor (default: 0.9)
    /// - `temperature`: Sigmoid temperature for share probability (default: 1.0)
    #[new]
    #[pyo3(signature = (alpha = 0.1, gamma = 0.9, temperature = 1.0))]
    pub fn new(alpha: f32, gamma: f32, temperature: f32) -> Self {
        PollinationEngine {
            agents: HashMap::new(),
            pending_packs: Vec::new(),
            alpha,
            gamma,
            temperature,
            current_tick: 0,
            total_packs_created: 0,
            total_packs_ingested: 0,
            total_lessons_shared: 0,
        }
    }

    /// Register an agent for pollination.
    pub fn register_agent(&mut self, agent_id: String) {
        self.agents
            .entry(agent_id)
            .or_insert_with(|| PollinatorState::new(self.alpha, self.gamma, self.temperature));
    }

    /// Check whether an agent should share knowledge right now.
    ///
    /// Returns True if the agent's RL-learned probability (boosted by surprise)
    /// exceeds the random threshold.
    #[pyo3(signature = (agent_id, current_surprise = 0.0))]
    pub fn should_share(&self, agent_id: &str, current_surprise: f32) -> bool {
        if let Some(state) = self.agents.get(agent_id) {
            // Deterministic for reproducibility — use hash-based pseudo-random
            let hash = simple_hash(agent_id, self.current_tick);
            let random_val = (hash % 1000) as f32 / 1000.0;
            state.should_share(random_val, current_surprise)
        } else {
            false
        }
    }

    /// Record a learning event from an agent.
    ///
    /// Creates a lesson that can be bundled into an experience pack
    /// and shared with other agents via pollination.
    #[pyo3(signature = (agent_id, description, success, surprise = 0.0, domain = "general"))]
    pub fn record_lesson(
        &mut self,
        agent_id: &str,
        description: String,
        success: bool,
        surprise: f32,
        domain: &str,
    ) {
        // Create lesson
        let lesson = Lesson {
            description,
            success,
            surprise,
            domain: domain.to_string(),
            tick: self.current_tick,
        };

        // Add to pending pack for this agent
        let source = agent_id.to_string();
        if let Some(pack) = self
            .pending_packs
            .iter_mut()
            .find(|p| p.source_agent == source)
        {
            pack.lessons.push(lesson);
        } else {
            self.pending_packs.push(ExperiencePack {
                source_agent: source,
                created_at: self.current_tick,
                lessons: vec![lesson],
            });
        }

        if let Some(state) = self.agents.get_mut(agent_id) {
            state.total_lessons_given += 1;
        }
    }

    /// Share knowledge from one agent to another.
    ///
    /// Registers the share for future TD(0) credit assignment.
    /// Returns the number of lessons shared.
    pub fn share(&mut self, from_agent: &str, to_agent: &str) -> u32 {
        let tick = self.current_tick;

        // Find pending pack from source
        let lessons_count = self
            .pending_packs
            .iter()
            .find(|p| p.source_agent == from_agent)
            .map(|p| p.lessons.len() as u32)
            .unwrap_or(0);

        if lessons_count == 0 {
            return 0;
        }

        // Register share for TD(0) credit
        if let Some(state) = self.agents.get_mut(from_agent) {
            state.register_share(to_agent.to_string(), tick);
        }

        // Credit receiver
        if let Some(state) = self.agents.get_mut(to_agent) {
            state.total_lessons_received += lessons_count as u64;
        }

        self.total_lessons_shared += lessons_count as u64;
        lessons_count
    }

    /// Provide reward feedback for a previous share.
    ///
    /// Positive reward → agent learns sharing is valuable → V(share) ↑
    /// Negative reward → agent learns to be more selective → V(share) ↓
    pub fn reward(&mut self, sharer_id: &str, receiver_id: &str, reward: f32) {
        let tick = self.current_tick;
        if let Some(state) = self.agents.get_mut(sharer_id) {
            state.apply_feedback(receiver_id, reward, tick);
        }
    }

    /// Advance one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Get share probability for an agent (for AMP bridge).
    ///
    /// Returns the agent's current share probability [0, 1].
    /// AMP uses this as the `reciprocity` dimension in the 12D relationship vector.
    pub fn share_probability(&self, agent_id: &str) -> f32 {
        self.agents
            .get(agent_id)
            .map(|s| s.share_probability)
            .unwrap_or(0.5)
    }

    /// Get V(share) for an agent (raw TD value).
    pub fn value(&self, agent_id: &str) -> f32 {
        self.agents
            .get(agent_id)
            .map(|s| s.raw_eagerness)
            .unwrap_or(0.0)
    }

    /// Statistics for the pollination engine.
    pub fn stats(&self) -> PyObject {
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("total_agents", self.agents.len()).unwrap();
            d.set_item("total_packs_created", self.total_packs_created)
                .unwrap();
            d.set_item("total_packs_ingested", self.total_packs_ingested)
                .unwrap();
            d.set_item("total_lessons_shared", self.total_lessons_shared)
                .unwrap();
            d.set_item("current_tick", self.current_tick).unwrap();

            // Per-agent summary
            let agents = PyList::empty(py);
            for (id, state) in &self.agents {
                let a = PyDict::new(py);
                a.set_item("agent_id", id).unwrap();
                a.set_item("share_probability", state.share_probability)
                    .unwrap();
                a.set_item("raw_eagerness", state.raw_eagerness).unwrap();
                a.set_item("total_shares", state.total_shares).unwrap();
                a.set_item("total_rewards", state.total_rewards).unwrap();
                a.set_item("lessons_given", state.total_lessons_given)
                    .unwrap();
                a.set_item("lessons_received", state.total_lessons_received)
                    .unwrap();
                agents.append(a).unwrap();
            }
            d.set_item("agents", agents).unwrap();

            d.into()
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple deterministic hash for pseudo-random decisions.
fn simple_hash(id: &str, tick: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in id.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= tick;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_td_convergence() {
        pyo3::prepare_freethreaded_python();
        let mut engine = PollinationEngine::new(0.1, 0.9, 1.0);
        engine.register_agent("agent_a".to_string());
        engine.register_agent("agent_b".to_string());

        // Initial share probability should be ~0.5 (neutral)
        let p0 = engine.share_probability("agent_a");
        assert!(
            (p0 - 0.5).abs() < 0.01,
            "Initial probability should be ~0.5, got {}",
            p0
        );

        // Simulate positive reward loop: share → reward → V(share) ↑
        for _ in 0..50 {
            engine.record_lesson("agent_a", "learned something".into(), true, 0.1, "general");
            engine.share("agent_a", "agent_b");
            engine.reward("agent_a", "agent_b", 1.0); // positive feedback
            engine.tick();
        }

        let p1 = engine.share_probability("agent_a");
        assert!(
            p1 > p0,
            "After positive rewards, share probability should increase: {} → {}",
            p0,
            p1
        );
        assert!(
            p1 > 0.7,
            "After 50 positive rewards, probability should be >0.7, got {}",
            p1
        );
    }

    #[test]
    fn test_negative_reward_suppresses() {
        pyo3::prepare_freethreaded_python();
        let mut engine = PollinationEngine::new(0.1, 0.9, 1.0);
        engine.register_agent("selfish".to_string());
        engine.register_agent("other".to_string());

        for _ in 0..50 {
            engine.record_lesson("selfish", "bad share".into(), false, 0.0, "general");
            engine.share("selfish", "other");
            engine.reward("selfish", "other", -1.0); // negative feedback
            engine.tick();
        }

        let p = engine.share_probability("selfish");
        assert!(
            p < 0.5,
            "After negative rewards, share probability should decrease: got {}",
            p
        );
    }

    #[test]
    fn test_surprise_boosts_sharing() {
        pyo3::prepare_freethreaded_python();
        let _engine = PollinationEngine::new(0.1, 0.9, 1.0);

        // Create a state with low share probability
        let mut state = PollinatorState::new(0.1, 0.9, 1.0);
        state.raw_eagerness = -3.0; // very selfish
        state.update_probability();
        assert!(
            state.share_probability < 0.1,
            "Base probability should be low: {}",
            state.share_probability
        );

        // Without surprise: random_val=0.4 should NOT trigger sharing
        assert!(
            !state.should_share(0.4, 0.0),
            "Without surprise, low eagerness should not share"
        );

        // With high surprise: random_val=0.4 SHOULD trigger sharing
        // effective = 0.047 + 0.9 * 0.5 = 0.497 > 0.4
        assert!(
            state.should_share(0.4, 0.9),
            "High surprise should override low eagerness"
        );
    }

    #[test]
    fn test_recency_window() {
        pyo3::prepare_freethreaded_python();
        let mut engine = PollinationEngine::new(0.1, 0.9, 1.0);
        engine.register_agent("agent_a".to_string());
        engine.register_agent("agent_b".to_string());

        engine.record_lesson("agent_a", "lesson".into(), true, 0.0, "general");
        engine.share("agent_a", "agent_b");

        // Advance past recency window (default 10 ticks)
        for _ in 0..20 {
            engine.tick();
        }

        let v_before = engine.value("agent_a");
        engine.reward("agent_a", "agent_b", 10.0); // late reward
        let v_after = engine.value("agent_a");

        // Should NOT update because reward was too late
        assert_eq!(
            v_before, v_after,
            "Late rewards (past recency window) should be ignored"
        );
    }

    #[test]
    fn test_share_lessons_count() {
        pyo3::prepare_freethreaded_python();
        let mut engine = PollinationEngine::new(0.1, 0.9, 1.0);
        engine.register_agent("teacher".to_string());
        engine.register_agent("student".to_string());

        engine.record_lesson("teacher", "lesson 1".into(), true, 0.0, "math");
        engine.record_lesson("teacher", "lesson 2".into(), true, 0.0, "math");
        engine.record_lesson("teacher", "lesson 3".into(), false, 0.5, "math");

        let shared = engine.share("teacher", "student");
        assert_eq!(shared, 3, "Should share all 3 lessons");
    }

    #[test]
    fn test_stats() {
        pyo3::prepare_freethreaded_python();
        let mut engine = PollinationEngine::new(0.1, 0.9, 1.0);
        engine.register_agent("a".to_string());
        engine.register_agent("b".to_string());

        Python::with_gil(|py| {
            let stats: PyObject = engine.stats();
            let dict = stats.downcast_bound::<PyDict>(py).unwrap();
            let n: usize = dict
                .get_item("total_agents")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(n, 2);
        });
    }

    #[test]
    fn test_amp_bridge_reciprocity() {
        // Verify that share_probability maps directly to AMP reciprocity
        pyo3::prepare_freethreaded_python();
        let mut engine = PollinationEngine::new(0.1, 0.9, 1.0);
        engine.register_agent("cooperative".to_string());

        // Boost V(share) through positive rewards
        for _ in 0..30 {
            engine.record_lesson("cooperative", "good share".into(), true, 0.0, "general");
            engine.share("cooperative", "imaginary_partner");
            // Manually apply feedback since imaginary_partner isn't registered
            if let Some(state) = engine.agents.get_mut("cooperative") {
                state
                    .active_shares
                    .insert("imaginary_partner".to_string(), engine.current_tick);
                state.apply_feedback("imaginary_partner", 1.0, engine.current_tick);
            }
            engine.tick();
        }

        let reciprocity = engine.share_probability("cooperative");
        assert!(
            reciprocity > 0.8,
            "High V(share) should map to high AMP reciprocity: got {}",
            reciprocity
        );
    }
}
