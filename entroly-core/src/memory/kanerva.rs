//! Kanerva Sparse Distributed Memory (SDM) — Neocortex
//!
//! Biologically-inspired associative memory storing patterns in
//! high-dimensional binary space. Retrieved via Hamming distance.
//!
//! Ported from ebbiforge-core/src/memory/kanerva.rs.
//!
//! - Write: O(N) where N = n_locations (fixed, small: 10K-100K)
//! - Read:  O(N) — same scan
//! - XOR+POPCNT inner loop: ~15ns per location comparison

use super::episode::hamming_distance;
use rand::RngExt;

/// Width of the content counter array.
const COUNTER_WIDTH: usize = 64;

/// A single hard location in the Kanerva SDM.
#[derive(Clone, Debug)]
struct HardLocation {
    /// 1024-bit address.
    address: [u64; 16],
    /// Distributed counter array: positive = bit was 1, negative = 0.
    counters: [i32; COUNTER_WIDTH],
    /// Number of writes to this location.
    write_count: u32,
}

/// Kanerva Sparse Distributed Memory.
///
/// O(1)-amortized associative read/write over 1024-bit binary vectors
/// with graceful degradation under noise (fuzzy matching).
#[derive(Clone, Debug)]
pub struct KanervaSDM {
    locations: Vec<HardLocation>,
    n_locations: usize,
    activation_radius: u32,
}

/// Result of a read from the SDM.
#[derive(Clone, Debug)]
pub struct SDMRecallResult {
    /// Reconstructed content vector (thresholded counters).
    pub content_vector: Vec<i32>,
    /// Number of hard locations that contributed.
    pub activated_count: usize,
    /// Mean Hamming distance of activated locations to query.
    pub mean_distance: f32,
}

impl KanervaSDM {
    /// Create a new SDM with randomly-addressed hard locations.
    ///
    /// - `n_locations`: 10,000–100,000 typical.
    /// - `activation_radius`: Max Hamming distance for activation (300–450
    ///   typical for 1024-bit addresses; ~40-45% of bits).
    pub fn new(n_locations: usize, activation_radius: u32) -> Self {
        let mut rng = rand::rng();
        let mut locations = Vec::with_capacity(n_locations);

        for _ in 0..n_locations {
            let mut address = [0u64; 16];
            for word in address.iter_mut() {
                *word = rng.random();
            }
            locations.push(HardLocation {
                address,
                counters: [0i32; COUNTER_WIDTH],
                write_count: 0,
            });
        }

        KanervaSDM {
            locations,
            n_locations,
            activation_radius,
        }
    }

    /// Write a content vector at a binary address.
    /// All hard locations within `activation_radius` are updated.
    pub fn write(&mut self, address: &[u64; 16], content: &[f32]) {
        let signs: Vec<i32> = content
            .iter()
            .take(COUNTER_WIDTH)
            .map(|&v| if v >= 0.0 { 1 } else { -1 })
            .collect();

        for loc in self.locations.iter_mut() {
            let dist = hamming_distance(address, &loc.address);
            if dist <= self.activation_radius {
                for (i, &s) in signs.iter().enumerate() {
                    if i < COUNTER_WIDTH {
                        loc.counters[i] += s;
                    }
                }
                loc.write_count += 1;
            }
        }
    }

    /// Read the content stored near a binary address.
    pub fn read(&self, address: &[u64; 16]) -> SDMRecallResult {
        let mut sum_counters = [0i64; COUNTER_WIDTH];
        let mut activated = 0usize;
        let mut total_distance = 0u64;

        for loc in &self.locations {
            let dist = hamming_distance(address, &loc.address);
            if dist <= self.activation_radius {
                for (i, sum) in sum_counters.iter_mut().enumerate() {
                    *sum += loc.counters[i] as i64;
                }
                activated += 1;
                total_distance += dist as u64;
            }
        }

        let content_vector: Vec<i32> = sum_counters
            .iter()
            .map(|&c| if c >= 0 { 1 } else { -1 })
            .collect();

        let mean_distance = if activated > 0 {
            total_distance as f32 / activated as f32
        } else {
            0.0
        };

        SDMRecallResult {
            content_vector,
            activated_count: activated,
            mean_distance,
        }
    }

    /// Number of locations that have been written to.
    pub fn occupied(&self) -> usize {
        self.locations.iter().filter(|l| l.write_count > 0).count()
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.n_locations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdm_write_read_roundtrip() {
        let mut sdm = KanervaSDM::new(100, 512); // wide radius for test
        let address = [0u64; 16];
        let content = vec![1.0f32; COUNTER_WIDTH];

        sdm.write(&address, &content);
        let result = sdm.read(&address);

        // All counters should be positive → all 1s
        assert!(result.activated_count > 0);
        assert!(result.content_vector.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_sdm_no_activation_outside_radius() {
        let sdm = KanervaSDM::new(100, 0); // radius = 0 → only exact match
        let address = [0u64; 16];
        let result = sdm.read(&address);
        // Very unlikely any random location has exact address [0;16]
        // Result should have few or no activated locations
        assert!(result.activated_count <= 1);
    }

    #[test]
    fn test_sdm_occupied() {
        let mut sdm = KanervaSDM::new(100, 512);
        assert_eq!(sdm.occupied(), 0);
        sdm.write(&[0u64; 16], &vec![1.0; COUNTER_WIDTH]);
        assert!(sdm.occupied() > 0);
    }
}
