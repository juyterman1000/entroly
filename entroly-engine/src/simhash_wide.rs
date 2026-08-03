//! Versioned, width-parameterised SimHash fingerprints.
//!
//! The Hamming estimator is `cos(pi * d / B)`. Widening reduces the pairwise
//! noise floor but does not remove maximum-selection bias; callers must retain
//! multiple-comparison correction when using similarity as a suppression rule.

use md5::{Digest, Md5};

/// Version 2 replaces Rust's unspecified `DefaultHasher` with a domain-separated
/// MD5 byte contract. MD5 is used only as a deterministic bit generator, never
/// for authentication or collision resistance.
pub const FINGERPRINT_VERSION: u16 = 2;
pub const SUPPORTED_WIDTHS: [u16; 3] = [64, 256, 1024];
const HASH_DOMAIN: &[u8] = b"entroly-simhash-v2\0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub version: u16,
    pub bits: u16,
    /// Little-endian limbs, `bits / 64` of them.
    pub words: Vec<u64>,
}

impl Fingerprint {
    pub fn limbs(bits: u16) -> usize {
        (bits as usize).div_ceil(64)
    }

    pub fn comparable(&self, other: &Fingerprint) -> bool {
        self.version == other.version && self.bits == other.bits
    }

    pub fn hamming(&self, other: &Fingerprint) -> Option<u32> {
        if !self.comparable(other) {
            return None;
        }
        Some(
            self.words
                .iter()
                .zip(other.words.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum(),
        )
    }

    pub fn cosine(&self, other: &Fingerprint) -> Option<f64> {
        let distance = self.hamming(other)? as f64;
        let width = self.bits as f64;
        Some(
            (std::f64::consts::PI * distance / width)
                .cos()
                .clamp(0.0, 1.0),
        )
    }

    pub fn noise_floor(bits: u16) -> f64 {
        std::f64::consts::PI / (2.0 * (bits as f64).sqrt())
    }
}

fn token_hash(token: &str, limb: usize) -> u64 {
    let mut hasher = Md5::new();
    hasher.update(HASH_DOMAIN);
    hasher.update(token.as_bytes());
    hasher.update((limb as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut word = [0u8; 8];
    word.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(word)
}

/// Compute a persisted, cross-platform fingerprint over whitespace tokens.
pub fn fingerprint(text: &str, bits: u16) -> Fingerprint {
    assert!(
        SUPPORTED_WIDTHS.contains(&bits),
        "unsupported SimHash width {bits}"
    );
    let limbs = Fingerprint::limbs(bits);
    let mut accumulator = vec![0i32; bits as usize];

    for token in text.split_whitespace() {
        let lower = token.to_lowercase();
        for limb in 0..limbs {
            let hash = token_hash(&lower, limb);
            for bit in 0..64 {
                let index = limb * 64 + bit;
                if index >= bits as usize {
                    break;
                }
                accumulator[index] += if (hash >> bit) & 1 == 1 { 1 } else { -1 };
            }
        }
    }

    let mut words = vec![0u64; limbs];
    for (index, sum) in accumulator.into_iter().enumerate() {
        if sum > 0 {
            words[index / 64] |= 1u64 << (index % 64);
        }
    }

    Fingerprint {
        version: FINGERPRINT_VERSION,
        bits,
        words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(seed: u64, words: usize) -> String {
        let vocab = [
            "handler", "request", "timeout", "retry", "payment", "gateway", "session",
            "token", "database", "encode", "decode", "commit", "buffer", "segment",
            "manifest", "digest", "cursor", "batch",
        ];
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..words)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                vocab[(state >> 33) as usize % vocab.len()]
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn golden_vector_pins_persisted_hash_contract() {
        let actual = fingerprint("def retry_request(req):", 256);
        assert_eq!(actual.version, 2);
        assert_eq!(
            actual.words,
            vec![
                0x0248_0180_0546_02b0,
                0x23c1_4008_1800_220b,
                0x0550_0c20_2043_e8c0,
                0x2c41_0204_0006_400c,
            ]
        );
    }

    #[test]
    fn identical_text_is_identical_at_every_width() {
        for bits in SUPPORTED_WIDTHS {
            let a = fingerprint("def retry_request(req):", bits);
            let b = fingerprint("def retry_request(req):", bits);
            assert_eq!(a, b);
            assert_eq!(a.hamming(&b), Some(0));
            assert!((a.cosine(&b).unwrap() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn widths_and_versions_are_not_silently_comparable() {
        let narrow = fingerprint("same text", 64);
        let wide = fingerprint("same text", 256);
        assert!(!narrow.comparable(&wide));
        assert_eq!(narrow.hamming(&wide), None);
        let mut next_version = fingerprint("same text", 64);
        next_version.version += 1;
        assert_eq!(narrow.hamming(&next_version), None);
    }

    #[test]
    fn noise_floor_follows_one_over_sqrt_bits() {
        let narrow = Fingerprint::noise_floor(64);
        let medium = Fingerprint::noise_floor(256);
        let wide = Fingerprint::noise_floor(1024);
        assert!((narrow - 0.196).abs() < 0.002);
        assert!((medium - 0.098).abs() < 0.002);
        assert!((wide - 0.049).abs() < 0.002);
        assert!((narrow / medium - 2.0).abs() < 0.01);
    }

    #[test]
    fn wider_fingerprints_improve_population_separation() {
        let mut margins = Vec::new();
        for bits in SUPPORTED_WIDTHS {
            let mut duplicates = Vec::new();
            let mut strangers = Vec::new();
            for seed in 0..40u64 {
                let base = doc(seed, 300);
                let mut edited: Vec<&str> = base.split_whitespace().collect();
                for index in (0..edited.len()).step_by(50) {
                    edited[index] = "sentinel";
                }
                let base_fp = fingerprint(&base, bits);
                duplicates.push(base_fp.cosine(&fingerprint(&edited.join(" "), bits)).unwrap());
                strangers.push(
                    base_fp
                        .cosine(&fingerprint(&doc(seed + 1000, 300), bits))
                        .unwrap(),
                );
            }
            let minimum_duplicate = duplicates.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum_stranger = strangers.iter().copied().fold(0.0_f64, f64::max);
            margins.push(minimum_duplicate - maximum_stranger);
        }
        assert!(margins[2] > margins[1] && margins[1] > margins[0]);
    }
}
