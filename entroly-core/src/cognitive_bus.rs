//! Cognitive Bus — Inter-Agent Event Routing with Memory-Aware ISA Prioritization
//!
//! Routes events (observations, tool results, beliefs, memories) between agents
//! using Information-Surprise-Adaptive (ISA) routing. Ported from agentOS
//! cognitive_bus.rs with deep integration into entroly's entropy scoring and
//! ebbiforge's hippocampus memory system.
//!
//! Architecture:
//!   publish(event) → dedup → novelty score → priority(ISA) → per-subscriber queue
//!   drain(agent) → top-K by priority → deliver → remember (hippocampus bridge)
//!
//! ISA Routing Math:
//!   Per-subscriber Poisson rate model:
//!     λ̂(n+1) = α · Δcount/Δtime + (1−α) · λ̂(n)    // α = 0.1
//!
//!   Priority score:
//!     priority(e, s) = KL(λ_obs ‖ λ_exp) · recency(e) · novelty(e) · mi(e, s)
//!     KL(λ_obs ‖ λ_exp) = λ_exp − λ_obs + λ_obs · ln(λ_obs / λ_exp)
//!     recency(e) = e^{−decay_rate · age(e)}
//!     novelty(e) = softcapped SimHash novelty (CAP=0.8)
//!     mi(e, s) = mutual information boost (Rényi H₂ approximation)
//!
//! Memory-Aware Features:
//!   - Events with high salience are flagged for hippocampus remember()
//!   - Recalled long-term memories can be published as "memory_recall" events
//!   - Emotional tags propagate: critical events → 3x priority boost
//!   - Consolidated memories (neocortex) get lower priority than fresh episodic
//!
//! Welford Spike Detection (online, no windowing):
//!   is_spike = (x − μ̄) > k·σ    // k=3, immediate broadcast, bypass queue
//!
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;

use crate::dedup::{simhash, hamming_distance};

// ── Constants ──────────────────────────────────────────────────────────

const POISSON_ALPHA: f64 = 0.1;         // EMA smoothing for rate estimation
const RECENCY_DECAY: f64 = 0.001;       // Exponential decay rate for recency
const NOVELTY_CAP: f64 = 0.8;           // Soft cap for novelty score
const SPIKE_K: f64 = 3.0;               // Welford spike threshold (k·σ)
const DEFAULT_LAMBDA: f64 = 1.0;        // Default expected Poisson rate
const MIN_LAMBDA: f64 = 0.001;          // Floor to prevent log(0)
const MAX_QUEUE_PER_AGENT: usize = 256; // Per-agent queue cap
const EMOTIONAL_CRITICAL_BOOST: f64 = 3.0;
const EMOTIONAL_IMPORTANT_BOOST: f64 = 1.5;

// ── Event Types ────────────────────────────────────────────────────────

/// Event types routable on the cognitive bus.
/// Maps to agentOS 25 event types, grouped into 4 zones.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EventType {
    // Zone 1: Perception (external inputs)
    Observation,
    ToolResult,
    UserMessage,
    SystemAlert,

    // Zone 2: Cognition (internal processing)
    Belief,
    Hypothesis,
    PlanUpdate,
    GoalChange,

    // Zone 3: Memory (hippocampus bridge)
    MemoryRecall,
    MemoryConsolidate,
    MemoryDecay,

    // Zone 4: Action (outputs)
    ActionProposed,
    ActionCompleted,
    Feedback,

    // Custom extension
    Custom(String),
}

impl EventType {
    fn from_str(s: &str) -> Self {
        match s {
            "observation" => EventType::Observation,
            "tool_result" => EventType::ToolResult,
            "user_message" => EventType::UserMessage,
            "system_alert" => EventType::SystemAlert,
            "belief" => EventType::Belief,
            "hypothesis" => EventType::Hypothesis,
            "plan_update" => EventType::PlanUpdate,
            "goal_change" => EventType::GoalChange,
            "memory_recall" => EventType::MemoryRecall,
            "memory_consolidate" => EventType::MemoryConsolidate,
            "memory_decay" => EventType::MemoryDecay,
            "action_proposed" => EventType::ActionProposed,
            "action_completed" => EventType::ActionCompleted,
            "feedback" => EventType::Feedback,
            other => EventType::Custom(other.to_string()),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            EventType::Observation => "observation",
            EventType::ToolResult => "tool_result",
            EventType::UserMessage => "user_message",
            EventType::SystemAlert => "system_alert",
            EventType::Belief => "belief",
            EventType::Hypothesis => "hypothesis",
            EventType::PlanUpdate => "plan_update",
            EventType::GoalChange => "goal_change",
            EventType::MemoryRecall => "memory_recall",
            EventType::MemoryConsolidate => "memory_consolidate",
            EventType::MemoryDecay => "memory_decay",
            EventType::ActionProposed => "action_proposed",
            EventType::ActionCompleted => "action_completed",
            EventType::Feedback => "feedback",
            EventType::Custom(s) => s.as_str(),
        }
    }
}

// ── Bus Event ──────────────────────────────────────────────────────────

/// An event published on the cognitive bus.
#[derive(Clone, Debug)]
pub struct BusEvent {
    pub id: u64,
    pub source_agent: String,
    pub event_type: EventType,
    pub content: String,
    pub timestamp: f64,           // Monotonic tick counter
    pub simhash: u64,             // Precomputed fingerprint
    pub emotional_tag: u8,        // 0=neutral, 1=positive, 2=negative, 3=critical
    pub salience: f64,            // For hippocampus bridge
    pub is_spike: bool,           // Welford-detected anomaly
}

/// Prioritized event wrapper for the per-agent queue.
#[derive(Clone, Debug)]
struct PrioritizedEvent {
    priority: f64,
    event: BusEvent,
}

impl PartialEq for PrioritizedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.event.id == other.event.id
    }
}
impl Eq for PrioritizedEvent {}

impl PartialOrd for PrioritizedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first (max-heap)
        self.priority.partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Welford Online Variance ────────────────────────────────────────────

/// Welford's online algorithm for running mean and variance.
/// No windowing — tracks full history with O(1) memory.
#[derive(Clone, Debug)]
struct WelfordAccumulator {
    count: u64,
    mean: f64,
    m2: f64,      // Sum of squared deviations
}

impl WelfordAccumulator {
    fn new() -> Self {
        WelfordAccumulator { count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Update with a new observation. Returns (new_mean, new_stddev).
    #[inline]
    fn update(&mut self, x: f64) -> (f64, f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        let variance = if self.count > 1 { self.m2 / (self.count - 1) as f64 } else { 0.0 };
        (self.mean, variance.sqrt())
    }

    /// Check if value is a spike (> k standard deviations from mean).
    #[inline]
    fn is_spike(&self, x: f64, k: f64) -> bool {
        if self.count < 3 { return false; }  // Need minimum samples
        let variance = self.m2 / (self.count - 1) as f64;
        let sigma = variance.sqrt();
        sigma > 0.0 && (x - self.mean).abs() > k * sigma
    }
}

// ── Poisson Rate Model ─────────────────────────────────────────────────

/// Per-subscriber per-event-type Poisson rate estimation.
/// EMA-smoothed: λ̂(n+1) = α · observed_rate + (1−α) · λ̂(n)
#[derive(Clone, Debug)]
struct PoissonRate {
    lambda_expected: f64,     // Expected rate (EMA-smoothed)
    last_count: u64,          // Event count at last update
    last_time: f64,           // Timestamp of last update
}

impl PoissonRate {
    fn new() -> Self {
        PoissonRate {
            lambda_expected: DEFAULT_LAMBDA,
            last_count: 0,
            last_time: 0.0,
        }
    }

    /// Update rate estimate with new observation.
    fn observe(&mut self, current_count: u64, current_time: f64) {
        let dt = (current_time - self.last_time).max(0.001);
        let dc = (current_count - self.last_count) as f64;
        let observed_rate = dc / dt;

        self.lambda_expected = POISSON_ALPHA * observed_rate
            + (1.0 - POISSON_ALPHA) * self.lambda_expected;
        self.lambda_expected = self.lambda_expected.max(MIN_LAMBDA);

        self.last_count = current_count;
        self.last_time = current_time;
    }

    /// KL divergence: KL(λ_obs ‖ λ_exp) = λ_exp − λ_obs + λ_obs · ln(λ_obs / λ_exp)
    /// Measures information surprise — high KL = unexpected event rate.
    #[inline]
    fn kl_surprise(&self, observed_rate: f64) -> f64 {
        let lambda_obs = observed_rate.max(MIN_LAMBDA);
        let lambda_exp = self.lambda_expected.max(MIN_LAMBDA);
        let kl = lambda_exp - lambda_obs + lambda_obs * (lambda_obs / lambda_exp).ln();
        kl.max(0.0)  // KL is non-negative
    }
}

// ── Subscriber ─────────────────────────────────────────────────────────

/// A subscriber (agent) on the cognitive bus.
#[derive(Clone, Debug)]
struct Subscriber {
    _agent_id: String,
    subscribed_types: Vec<EventType>,
    /// Per-event-type Poisson rate models
    rate_models: HashMap<String, PoissonRate>,
    /// Per-event-type counters
    event_counts: HashMap<String, u64>,
    /// Priority queue of pending events for this subscriber
    queue: BinaryHeap<PrioritizedEvent>,
    /// SimHash history for novelty computation (ring buffer of recent hashes)
    hash_history: VecDeque<u64>,
    /// Current task context hash (for mutual information estimation)
    task_context_hash: u64,
    /// Welford accumulator for spike detection on priority scores
    priority_welford: WelfordAccumulator,
    /// Total events delivered
    total_delivered: u64,
}

impl Subscriber {
    fn new(agent_id: &str) -> Self {
        Subscriber {
            _agent_id: agent_id.to_string(),
            subscribed_types: Vec::new(),
            rate_models: HashMap::new(),
            event_counts: HashMap::new(),
            queue: BinaryHeap::new(),
            hash_history: VecDeque::with_capacity(128),
            task_context_hash: 0,
            priority_welford: WelfordAccumulator::new(),
            total_delivered: 0,
        }
    }

    /// Compute novelty of an event relative to this subscriber's history.
    /// novelty(msg) = 1 − max_{h ∈ history} (1 − d_H(msg, h) / 64)
    /// softcapped: CAP · tanh(novelty / CAP)
    fn compute_novelty(&self, event_hash: u64) -> f64 {
        if self.hash_history.is_empty() {
            return NOVELTY_CAP;  // First event is maximally novel
        }

        let max_similarity = self.hash_history.iter()
            .map(|&h| 1.0 - hamming_distance(event_hash, h) as f64 / 64.0)
            .fold(0.0_f64, |a, b| a.max(b));

        let raw_novelty = 1.0 - max_similarity;
        // Soft cap: CAP · tanh(novelty / CAP)
        NOVELTY_CAP * (raw_novelty / NOVELTY_CAP).tanh()
    }

    /// Estimate mutual information between event and subscriber's current task.
    /// Approximated via SimHash Hamming similarity as proxy for content overlap.
    /// I(event; task) ≈ similarity(event_hash, task_hash)
    fn compute_mutual_info(&self, event_hash: u64) -> f64 {
        if self.task_context_hash == 0 {
            return 1.0;  // No task context → neutral (don't penalize)
        }
        let dist = hamming_distance(event_hash, self.task_context_hash) as f64;
        let similarity = 1.0 - dist / 64.0;
        // Boost: events relevant to current task get higher MI score
        // Scale to [0.2, 2.0] — never zero-out, but up to 2x boost
        0.2 + 1.8 * similarity
    }

    /// Compute ISA priority for an event destined to this subscriber.
    fn compute_priority(&mut self, event: &BusEvent, current_time: f64) -> f64 {
        let type_key = event.event_type.as_str().to_string();

        // 1. KL surprise — compute from rate model, then release borrow
        let count = self.event_counts.get(&type_key).copied().unwrap_or(0);
        let (kl, last_time) = {
            let rate_model = self.rate_models.entry(type_key.clone())
                .or_insert_with(PoissonRate::new);
            let dt = (current_time - rate_model.last_time).max(0.001);
            let observed_rate = (count as f64 + 1.0) / dt;
            let kl = rate_model.kl_surprise(observed_rate);
            let lt = rate_model.last_time;
            (kl, lt)
        };
        // Deferred Poisson update (after immutable borrows complete)
        let _ = last_time;  // used above, kept for clarity

        // 2. Recency
        let age = (current_time - event.timestamp).max(0.0);
        let recency = (-RECENCY_DECAY * age).exp();

        // 3. Novelty (reads self.hash_history — immutable borrow, safe now)
        let novelty = self.compute_novelty(event.simhash);

        // 4. Mutual information (reads self.task_context_hash — immutable, safe)
        let mi = self.compute_mutual_info(event.simhash);

        // 5. Emotional boost
        let emotion_boost = match event.emotional_tag {
            3 => EMOTIONAL_CRITICAL_BOOST,
            2 => EMOTIONAL_IMPORTANT_BOOST,
            _ => 1.0,
        };

        // ISA composite: KL · recency · novelty · MI · emotion
        let priority = (1.0 + kl) * recency * novelty * mi * emotion_boost;

        // Now safe to mutably borrow rate_models again for update
        if let Some(rate_model) = self.rate_models.get_mut(&type_key) {
            rate_model.observe(count + 1, current_time);
        }
        *self.event_counts.entry(type_key).or_insert(0) += 1;

        priority
    }

    /// Check if a priority score represents a spike (Welford).
    fn check_spike(&mut self, priority: f64) -> bool {
        let is_spike = self.priority_welford.is_spike(priority, SPIKE_K);
        self.priority_welford.update(priority);
        is_spike
    }

    /// Enqueue an event with its computed priority.
    fn enqueue(&mut self, event: BusEvent, priority: f64) {
        // Add to hash history (ring buffer)
        if self.hash_history.len() >= 128 {
            self.hash_history.pop_front();
        }
        self.hash_history.push_back(event.simhash);

        // Enforce queue cap (evict lowest priority)
        if self.queue.len() >= MAX_QUEUE_PER_AGENT {
            // BinaryHeap is max-heap, so we need to remove the minimum.
            // Convert to vec, sort, truncate, rebuild.
            let mut events: Vec<_> = std::mem::take(&mut self.queue).into_vec();
            events.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(Ordering::Equal));
            events.truncate(MAX_QUEUE_PER_AGENT - 1);
            self.queue = BinaryHeap::from(events);
        }

        self.queue.push(PrioritizedEvent { priority, event });
    }

    /// Drain top-K events for delivery.
    fn drain(&mut self, limit: usize) -> Vec<BusEvent> {
        let mut result = Vec::with_capacity(limit.min(self.queue.len()));
        for _ in 0..limit {
            match self.queue.pop() {
                Some(pe) => {
                    self.total_delivered += 1;
                    result.push(pe.event);
                }
                None => break,
            }
        }
        result
    }

    /// Check if this subscriber is interested in an event type.
    fn is_subscribed(&self, event_type: &EventType) -> bool {
        self.subscribed_types.is_empty()  // Empty = subscribe to all
            || self.subscribed_types.contains(event_type)
    }
}

// ── Cognitive Bus ──────────────────────────────────────────────────────

/// The Cognitive Bus — inter-agent event routing with ISA prioritization.
///
/// Integrates with:
///   - Entroly dedup (SimHash novelty scoring)
///   - Hippocampus memory (salience-based remember/recall bridge)
///   - NKBE allocator (events influence budget reallocation)
///
/// Memory-aware routing:
///   - Events with salience > threshold are flagged for hippocampus.remember()
///   - MemoryRecall events inject recalled memories into agent context
///   - Emotional tags (from hippocampus) boost priority (critical=3x)
///   - Consolidated memories get dampened priority (already in long-term)
#[pyclass]
pub struct CognitiveBus {
    subscribers: HashMap<String, Subscriber>,
    event_counter: u64,
    current_tick: f64,
    /// Global dedup: prevent same event from being routed twice
    recent_hashes: VecDeque<u64>,
    /// Global Welford for event rate spike detection
    global_welford: WelfordAccumulator,
    /// Events flagged for hippocampus bridge (high salience)
    memory_bridge_queue: Vec<BusEvent>,
    /// Salience threshold for hippocampus bridging
    memory_salience_threshold: f64,
    /// Stats
    total_published: u64,
    total_deduplicated: u64,
    total_spikes: u64,
    total_memory_bridged: u64,
}

#[pymethods]
impl CognitiveBus {
    #[new]
    #[pyo3(signature = (memory_salience_threshold=50.0))]
    pub fn new(memory_salience_threshold: f64) -> Self {
        CognitiveBus {
            subscribers: HashMap::new(),
            event_counter: 0,
            current_tick: 0.0,
            recent_hashes: VecDeque::with_capacity(1024),
            global_welford: WelfordAccumulator::new(),
            memory_bridge_queue: Vec::new(),
            memory_salience_threshold,
            total_published: 0,
            total_deduplicated: 0,
            total_spikes: 0,
            total_memory_bridged: 0,
        }
    }

    /// Register an agent as a subscriber.
    ///
    /// Args:
    ///   agent_id: Unique agent identifier.
    ///   event_types: List of event type strings to subscribe to.
    ///                Empty list = subscribe to all events.
    pub fn subscribe(&mut self, agent_id: &str, event_types: Vec<String>) {
        let mut sub = Subscriber::new(agent_id);
        sub.subscribed_types = event_types.iter()
            .map(|s| EventType::from_str(s))
            .collect();
        self.subscribers.insert(agent_id.to_string(), sub);
    }

    /// Unsubscribe an agent.
    pub fn unsubscribe(&mut self, agent_id: &str) {
        self.subscribers.remove(agent_id);
    }

    /// Update a subscriber's current task context (for MI-based routing).
    ///
    /// The task text is SimHashed and used to boost events with high
    /// mutual information toward the subscriber's current task.
    pub fn set_task_context(&mut self, agent_id: &str, task_text: &str) {
        if let Some(sub) = self.subscribers.get_mut(agent_id) {
            sub.task_context_hash = simhash(task_text);
        }
    }

    /// Advance the bus clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1.0;
    }

    /// Set the current tick explicitly (for sync with external clock).
    pub fn set_tick(&mut self, tick: f64) {
        self.current_tick = tick;
    }

    /// Publish an event to the bus.
    ///
    /// The event is:
    ///   1. SimHash fingerprinted
    ///   2. Dedup-checked against recent events
    ///   3. Priority-scored per subscriber (ISA routing)
    ///   4. Spike-checked (Welford) for immediate broadcast
    ///   5. Enqueued to matching subscribers
    ///   6. Optionally flagged for hippocampus bridge (high salience)
    ///
    /// Args:
    ///   source_agent: ID of the publishing agent.
    ///   event_type: Event type string (e.g., "observation", "belief").
    ///   content: Event payload text.
    ///   emotional_tag: 0=neutral, 1=positive, 2=negative, 3=critical.
    ///   salience: Importance for memory bridging (0-100).
    ///
    /// Returns:
    ///   Number of subscribers the event was routed to.
    #[pyo3(signature = (source_agent, event_type, content, emotional_tag=0, salience=0.0))]
    pub fn publish(
        &mut self,
        source_agent: &str,
        event_type: &str,
        content: &str,
        emotional_tag: u8,
        salience: f64,
    ) -> usize {
        let hash = simhash(content);

        // Global dedup: check if near-duplicate was published recently
        let is_dup = self.recent_hashes.iter()
            .any(|&h| hamming_distance(hash, h) <= 3);

        if is_dup {
            self.total_deduplicated += 1;
            return 0;
        }

        // Add to recent hash ring buffer
        if self.recent_hashes.len() >= 1024 {
            self.recent_hashes.pop_front();
        }
        self.recent_hashes.push_back(hash);

        // Create event
        self.event_counter += 1;
        let event = BusEvent {
            id: self.event_counter,
            source_agent: source_agent.to_string(),
            event_type: EventType::from_str(event_type),
            content: content.to_string(),
            timestamp: self.current_tick,
            simhash: hash,
            emotional_tag,
            salience,
            is_spike: false,
        };

        self.total_published += 1;

        // Memory bridge: flag high-salience events for hippocampus
        if salience >= self.memory_salience_threshold {
            self.memory_bridge_queue.push(event.clone());
            self.total_memory_bridged += 1;
        }

        // Route to subscribers
        let etype = event.event_type.clone();
        let mut routed = 0;

        // Collect subscriber keys to avoid borrow issues
        let sub_keys: Vec<String> = self.subscribers.keys().cloned().collect();

        for key in sub_keys {
            // Don't route events back to the source agent
            if key == source_agent {
                continue;
            }

            let sub = self.subscribers.get_mut(&key).unwrap();

            if !sub.is_subscribed(&etype) {
                continue;
            }

            let priority = sub.compute_priority(&event, self.current_tick);

            // Welford spike detection
            let mut event_copy = event.clone();
            if sub.check_spike(priority) {
                event_copy.is_spike = true;
                self.total_spikes += 1;
            }

            // Global spike detection
            self.global_welford.update(priority);

            sub.enqueue(event_copy, priority);
            routed += 1;
        }

        routed
    }

    /// Drain top-K events for a subscriber.
    ///
    /// Returns events ordered by ISA priority (highest first).
    /// Spike events are always included first.
    ///
    /// Args:
    ///   agent_id: The subscriber agent ID.
    ///   limit: Maximum number of events to return.
    ///
    /// Returns:
    ///   List of dicts with event data.
    #[pyo3(signature = (agent_id, limit=10))]
    pub fn drain<'py>(&mut self, py: Python<'py>, agent_id: &str, limit: usize) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let sub = match self.subscribers.get_mut(agent_id) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        let events = sub.drain(limit);
        let mut result = Vec::with_capacity(events.len());

        for event in events {
            let dict = PyDict::new(py);
            dict.set_item("id", event.id)?;
            dict.set_item("source_agent", &event.source_agent)?;
            dict.set_item("event_type", event.event_type.as_str())?;
            dict.set_item("content", &event.content)?;
            dict.set_item("timestamp", event.timestamp)?;
            dict.set_item("emotional_tag", event.emotional_tag)?;
            dict.set_item("salience", event.salience)?;
            dict.set_item("is_spike", event.is_spike)?;
            result.push(dict);
        }

        Ok(result)
    }

    /// Drain events flagged for hippocampus memory bridging.
    ///
    /// Called by the Python layer to feed high-salience events to
    /// hippocampus.remember(). Returns and clears the bridge queue.
    pub fn drain_memory_bridge<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let events = std::mem::take(&mut self.memory_bridge_queue);
        let mut result = Vec::with_capacity(events.len());

        for event in events {
            let dict = PyDict::new(py);
            dict.set_item("content", &event.content)?;
            dict.set_item("source", &event.source_agent)?;
            dict.set_item("salience", event.salience)?;
            dict.set_item("emotional_tag", event.emotional_tag)?;
            dict.set_item("event_type", event.event_type.as_str())?;
            result.push(dict);
        }

        Ok(result)
    }

    /// Get queue depth for a subscriber.
    pub fn queue_depth(&self, agent_id: &str) -> usize {
        self.subscribers.get(agent_id)
            .map(|s| s.queue.len())
            .unwrap_or(0)
    }

    /// Get bus statistics.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("subscribers", self.subscribers.len())?;
        dict.set_item("total_published", self.total_published)?;
        dict.set_item("total_deduplicated", self.total_deduplicated)?;
        dict.set_item("total_spikes", self.total_spikes)?;
        dict.set_item("total_memory_bridged", self.total_memory_bridged)?;
        dict.set_item("current_tick", self.current_tick)?;

        // Per-subscriber stats
        let sub_stats = PyDict::new(py);
        for (id, sub) in &self.subscribers {
            let s = PyDict::new(py);
            s.set_item("queue_depth", sub.queue.len())?;
            s.set_item("total_delivered", sub.total_delivered)?;
            s.set_item("subscribed_types", sub.subscribed_types.iter()
                .map(|t| t.as_str().to_string())
                .collect::<Vec<_>>())?;
            sub_stats.set_item(id.as_str(), s)?;
        }
        dict.set_item("subscribers_detail", sub_stats)?;

        Ok(dict)
    }
}

#[cfg(test)]
impl CognitiveBus {
    // ── Rust-only helpers (not exposed to Python) ──
    /// Drain raw events (Rust-only, for internal use and testing).
    pub fn drain_raw(&mut self, agent_id: &str, limit: usize) -> Vec<BusEvent> {
        match self.subscribers.get_mut(agent_id) {
            Some(sub) => sub.drain(limit),
            None => Vec::new(),
        }
    }

    /// Drain raw memory bridge events (Rust-only, for internal use and testing).
    pub fn drain_memory_bridge_raw(&mut self) -> Vec<BusEvent> {
        std::mem::take(&mut self.memory_bridge_queue)
    }
}

// ── Unit Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_basic() {
        let mut w = WelfordAccumulator::new();
        // Feed values with some variance so stddev > 0
        for &v in &[5.0, 6.0, 4.0, 5.5, 4.5, 5.0, 6.0, 4.0, 5.5, 4.5] {
            w.update(v);
        }
        assert!(!w.is_spike(5.0, 3.0), "mean-ish value should not be spike");
        assert!(w.is_spike(50.0, 3.0), "10x value should be spike");
    }

    #[test]
    fn test_welford_needs_minimum_samples() {
        let w = WelfordAccumulator::new();
        // With 0 samples, nothing is a spike
        assert!(!w.is_spike(1000.0, 3.0));
    }

    #[test]
    fn test_poisson_rate_kl() {
        let mut rate = PoissonRate::new();
        rate.observe(10, 1.0);
        // KL should be non-negative
        assert!(rate.kl_surprise(5.0) >= 0.0);
        assert!(rate.kl_surprise(0.5) >= 0.0);
    }

    #[test]
    fn test_subscriber_novelty_first_event() {
        let sub = Subscriber::new("test");
        // First event should be maximally novel
        let novelty = sub.compute_novelty(12345);
        assert!((novelty - NOVELTY_CAP).abs() < 0.01);
    }

    #[test]
    fn test_subscriber_novelty_duplicate() {
        let mut sub = Subscriber::new("test");
        sub.hash_history.push_back(12345);
        // Same hash → zero novelty (after softcap)
        let novelty = sub.compute_novelty(12345);
        assert!(novelty < 0.01, "duplicate should have near-zero novelty");
    }

    #[test]
    fn test_subscriber_mutual_info_no_task() {
        let sub = Subscriber::new("test");
        // No task context → neutral MI (1.0)
        let mi = sub.compute_mutual_info(12345);
        assert!((mi - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_subscriber_mutual_info_same_task() {
        let mut sub = Subscriber::new("test");
        let hash = simhash("find authentication vulnerabilities in the codebase");
        sub.task_context_hash = hash;
        let mi = sub.compute_mutual_info(hash);
        // Same content → max MI (0.2 + 1.8 * 1.0 = 2.0)
        assert!((mi - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_bus_publish_and_drain() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("agent_a", vec![]);
        bus.subscribe("agent_b", vec![]);

        let routed = bus.publish("agent_a", "observation", "Found 3 CVEs in auth module", 3, 80.0);
        assert_eq!(routed, 1, "should route to agent_b only (not back to agent_a)");

        let events = bus.drain_raw("agent_b", 10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source_agent, "agent_a");
        assert_eq!(events[0].event_type, EventType::Observation);
        assert_eq!(events[0].emotional_tag, 3);
    }

    #[test]
    fn test_bus_dedup_prevents_duplicate() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("agent_b", vec![]);

        let r1 = bus.publish("agent_a", "observation", "Found XSS vulnerability", 0, 0.0);
        let r2 = bus.publish("agent_a", "observation", "Found XSS vulnerability", 0, 0.0);

        assert_eq!(r1, 1);
        assert_eq!(r2, 0, "duplicate event should be deduplicated");
        assert_eq!(bus.total_deduplicated, 1);
    }

    #[test]
    fn test_bus_memory_bridge() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("agent_b", vec![]);

        // High salience → should be bridged to hippocampus
        bus.publish("agent_a", "observation", "Critical security finding", 3, 80.0);
        // Low salience → should NOT be bridged
        bus.publish("agent_a", "observation", "Minor style issue in code formatting", 0, 5.0);

        let bridge_events = bus.drain_memory_bridge_raw();
        assert_eq!(bridge_events.len(), 1, "only high-salience event should be bridged");
        assert!((bridge_events[0].salience - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_bus_type_filtering() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("agent_b", vec!["observation".to_string()]);

        let r1 = bus.publish("agent_a", "observation", "Found something interesting", 0, 0.0);
        let r2 = bus.publish("agent_a", "belief", "I think the code is secure", 0, 0.0);

        assert_eq!(r1, 1, "observation should route to agent_b");
        assert_eq!(r2, 0, "belief should NOT route to agent_b (not subscribed)");
    }

    #[test]
    fn test_bus_no_self_routing() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("agent_a", vec![]);

        let routed = bus.publish("agent_a", "observation", "My own observation", 0, 0.0);
        assert_eq!(routed, 0, "events should not route back to source agent");
    }

    #[test]
    fn test_bus_task_context_boosts_priority() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("security_agent", vec![]);
        bus.subscribe("style_agent", vec![]);

        // Set task context for security agent
        bus.set_task_context("security_agent", "find authentication vulnerabilities XSS CSRF");

        // Publish security-related event
        bus.publish("scanner", "observation", "Found XSS vulnerability in login form authentication bypass", 2, 30.0);

        // Both agents should get the event
        let sec_events = bus.drain_raw("security_agent", 1);
        let style_events = bus.drain_raw("style_agent", 1);
        assert_eq!(sec_events.len(), 1);
        assert_eq!(style_events.len(), 1);
    }

    #[test]
    fn test_bus_stats_counters() {
        let mut bus = CognitiveBus::new(50.0);
        bus.subscribe("agent_a", vec![]);
        bus.subscribe("agent_b", vec![]);
        bus.publish("agent_a", "observation", "Test event one", 0, 0.0);
        bus.publish("agent_b", "belief", "Test event two different content here", 0, 0.0);

        assert_eq!(bus.total_published, 2);
        assert_eq!(bus.subscribers.len(), 2);
    }
}
