//! Episode — a single memory trace with Ebbinghaus-governed retention.
//!
//! Ported from ebbiforge-core/src/memory/episode.rs with adaptations
//! for agentOS multi-agent context.

/// Emotional tag controlling base salience multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EmotionalTag {
    Neutral  = 0,
    Positive = 1,
    Negative = 2,
    Critical = 3,
}

impl EmotionalTag {
    pub fn multiplier(self) -> f32 {
        match self {
            EmotionalTag::Neutral  => 1.0,
            EmotionalTag::Positive => 1.2,
            EmotionalTag::Negative => 1.5,
            EmotionalTag::Critical => 3.0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EmotionalTag::Neutral  => "neutral",
            EmotionalTag::Positive => "positive",
            EmotionalTag::Negative => "negative",
            EmotionalTag::Critical => "critical",
        }
    }
}

impl From<u8> for EmotionalTag {
    fn from(v: u8) -> Self {
        match v {
            1 => EmotionalTag::Positive,
            2 => EmotionalTag::Negative,
            3 => EmotionalTag::Critical,
            _ => EmotionalTag::Neutral,
        }
    }
}

/// Memory tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// L1: Working — current task context
    Working,
    /// L2: Episodic — session results, interactions
    Episodic,
    /// L3: Semantic — persistent patterns (never forgotten)
    Semantic,
}

impl MemoryTier {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryTier::Working  => "working",
            MemoryTier::Episodic => "episodic",
            MemoryTier::Semantic => "semantic",
        }
    }

    #[allow(clippy::should_implement_trait)] // intentional: not a FromStr impl (always returns Self, no Err)
    pub fn from_str(s: &str) -> Self {
        match s {
            "episodic" => MemoryTier::Episodic,
            "semantic" => MemoryTier::Semantic,
            _          => MemoryTier::Working,
        }
    }
}

/// A single memory trace in the hippocampal buffer.
///
/// Size: ~300 bytes per episode (compact for millions of entries).
#[derive(Clone, Debug)]
pub struct Episode {
    /// Unique monotonic ID.
    pub id: u64,
    /// Owning agent (0 = shared).
    pub agent_id: u64,
    /// The actual memory content.
    pub content: String,
    /// Memory tier.
    pub tier: MemoryTier,
    /// Quantized embedding — 64 × u16 values (128 bytes).
    pub embedding: Vec<u16>,
    /// 1024-bit binary address for Kanerva SDM + LSH indexing.
    pub binary_address: [u64; 16],
    /// Salience = direct half-life in ticks.
    ///   R(t) = e^(-age / salience)
    pub salience: f32,
    /// Creation tick.
    pub created_at: f64,
    /// Last recall tick.
    pub last_recalled: f64,
    /// Times recalled (drives spaced-recall reinforcement).
    pub recall_count: u32,
    /// Emotional intensity.
    pub emotional_tag: EmotionalTag,
    /// Whether consolidated to Kanerva SDM neocortex.
    pub consolidated: bool,
    /// Estimated token cost (~3.5 chars/token).
    pub token_cost: u32,
    /// Tags for filtering.
    pub tags: Vec<String>,
}

impl Episode {
    /// Ebbinghaus retention at a given tick.
    ///   R(t) = e^(-(current_tick - created_at) / salience)
    #[inline]
    pub fn retention(&self, current_tick: f64) -> f32 {
        let age = (current_tick - self.created_at) as f32;
        if age <= 0.0 || self.salience <= 0.0 {
            return 1.0;
        }
        (-age / self.salience).exp()
    }

    /// Spaced-recall reinforcement: each recall multiplies salience.
    #[inline]
    pub fn reinforce(&mut self, current_tick: f64, factor: f32) {
        self.recall_count += 1;
        self.last_recalled = current_tick;
        self.salience *= factor;
        self.salience = self.salience.min(10_000.0); // cap
    }

    /// Combined score for ranking: retention × (1 + ln(recalls + 1))
    #[inline]
    pub fn score(&self, current_tick: f64) -> f64 {
        let retention = self.retention(current_tick) as f64;
        let frequency = (1.0 + (self.recall_count as f64 + 1.0).ln()).min(3.0);
        retention * frequency
    }
}

// ════════════════════════════════════════════════════════════════════
//  SIMHASH + HAMMING — Embedding → Binary Address
// ════════════════════════════════════════════════════════════════════

/// Convert a quantized u16 embedding into a 1024-bit binary address
/// using SimHash (random hyperplane projections).
///
/// Preserves approximate cosine similarity: similar vectors → small
/// Hamming distance.
pub fn simhash_embedding(embedding: &[u16]) -> [u64; 16] {
    let mut address = [0u64; 16];
    let dim = embedding.len();
    if dim == 0 {
        return address;
    }

    for bit_idx in 0..1024usize {
        let mut accumulator: i64 = 0;
        for (d, &val) in embedding.iter().enumerate() {
            // Deterministic weight via golden ratio hash.
            let seed = (bit_idx as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(d as u64)
                .wrapping_mul(0x517cc1b727220a95);
            let weight: i64 = if seed & 1 == 0 { 1 } else { -1 };
            accumulator += weight * (val as i64);
        }
        if accumulator >= 0 {
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            address[word] |= 1u64 << bit;
        }
    }
    address
}

/// Hamming distance between two 1024-bit binary addresses.
/// Uses `count_ones()` → hardware POPCNT on x86-64.
#[inline]
pub fn hamming_distance(a: &[u64; 16], b: &[u64; 16]) -> u32 {
    let mut dist = 0u32;
    for i in 0..16 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Quantize f32 embedding → u16 (preserving ranking order).
///   val_u16 = clamp((val_f32 + 1.0) / 2.0 * 65535, 0, 65535)
pub fn quantize_embedding(float_vec: &[f32]) -> Vec<u16> {
    float_vec.iter()
        .map(|&v| {
            let scaled = ((v + 1.0) * 0.5 * 65535.0).round();
            scaled.clamp(0.0, 65535.0) as u16
        })
        .collect()
}

/// Generate a trigram-hash embedding from text content (no external model needed).
/// Uses FNV1a over character trigrams → 64 u16 values.
pub fn trigram_embedding(content: &str) -> Vec<u16> {
    const DIM: usize = 64;
    let mut counters = vec![0i64; DIM];
    let chars: Vec<char> = content.chars().collect();

    if chars.len() < 3 {
        return vec![32768u16; DIM]; // midpoint for very short content
    }

    for window in chars.windows(3) {
        let mut h: u64 = 0xcbf29ce484222325;
        for &ch in window {
            h ^= ch as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let idx = (h as usize) % DIM;
        let sign = if h & (1 << 32) != 0 { 1i64 } else { -1 };
        counters[idx] += sign;
    }

    // Normalize counters to u16 range
    let max_abs = counters.iter().map(|&v| v.unsigned_abs()).max().unwrap_or(1).max(1);
    counters.iter()
        .map(|&v| {
            let normalized = (v as f64 / max_abs as f64 + 1.0) * 0.5 * 65535.0;
            normalized.clamp(0.0, 65535.0) as u16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_decay() {
        let ep = Episode {
            id: 0, agent_id: 0, content: "test".into(), tier: MemoryTier::Working,
            embedding: vec![32768; 64], binary_address: [0; 16],
            salience: 1.0, created_at: 0.0, last_recalled: 0.0, recall_count: 0,
            emotional_tag: EmotionalTag::Neutral, consolidated: false,
            token_cost: 1, tags: vec![],
        };
        assert!((ep.retention(0.0) - 1.0).abs() < 0.001);
        assert!((ep.retention(1.0) - 0.3679).abs() < 0.01); // e^-1
        assert!(ep.retention(5.0) < 0.01);
    }

    #[test]
    fn test_reinforcement() {
        let mut ep = Episode {
            id: 0, agent_id: 0, content: "test".into(), tier: MemoryTier::Working,
            embedding: vec![32768; 64], binary_address: [0; 16],
            salience: 1.0, created_at: 0.0, last_recalled: 0.0, recall_count: 0,
            emotional_tag: EmotionalTag::Neutral, consolidated: false,
            token_cost: 1, tags: vec![],
        };
        ep.reinforce(5.0, 1.3);
        assert_eq!(ep.recall_count, 1);
        assert!((ep.salience - 1.3).abs() < 0.001);
    }

    #[test]
    fn test_hamming_identical() {
        let a = [0xFFu64; 16];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_opposite() {
        let a = [0u64; 16];
        let b = [u64::MAX; 16];
        assert_eq!(hamming_distance(&a, &b), 1024);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let v = vec![0.0f32, 1.0, -1.0, 0.5];
        let q = quantize_embedding(&v);
        assert_eq!(q[0], 32768);
        assert_eq!(q[1], 65535);
        assert_eq!(q[2], 0);
    }

    #[test]
    fn test_emotional_multiplier() {
        assert_eq!(EmotionalTag::Neutral.multiplier(), 1.0);
        assert_eq!(EmotionalTag::Critical.multiplier(), 3.0);
    }

    #[test]
    fn test_simhash_deterministic() {
        let emb = vec![32768u16; 64];
        let a = simhash_embedding(&emb);
        let b = simhash_embedding(&emb);
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_trigram_embedding_length() {
        let emb = trigram_embedding("hello world this is a test");
        assert_eq!(emb.len(), 64);
    }
}
