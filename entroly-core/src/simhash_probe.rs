//! Falsification probe for the 64-bit SimHash used in the selection path.
//!
//! NOT a product module — compiled only under `cfg(test)`.
//!
//! ## What is under test
//!
//! `dedup::simhash` builds a 64-bit fingerprint by summing a pseudo-random
//! +/-1 vector per word trigram and taking the sign of each accumulator. Two
//! consumers read it, with different failure modes:
//!
//!   1. `DedupIndex` (threshold 5) makes a HARD DROP decision, gated behind an
//!      exact 16-bit band match. A false merge silently discards a distinct
//!      fragment; a missed candidate leaks a duplicate through.
//!
//!   2. `knapsack_sds` turns Hamming distance into a CONTINUOUS similarity and
//!      multiplies it into every candidate's value. Calibration matters here,
//!      not a threshold.
//!
//! ## Ground truth
//!
//! SimHash of a feature multiset estimates the angle between the **term-
//! frequency** vectors of those features. `simhash` iterates `windows(3)`
//! WITHOUT deduplicating, so repeated trigrams contribute repeatedly. Ground
//! truth must therefore be TF-weighted cosine over the trigram multiset — a
//! deduplicated set cosine is a different quantity and would misreport the
//! estimator's error. Trigram hashing here uses FNV-1a, independent of the MD5
//! family `simhash` uses, so ground truth cannot inherit its collision
//! structure.
//!
//! These are lexical measurements, not semantic duplicate labels. Edited
//! fragments may change behaviour, and random pairs may contain duplicates.
//! Neither candidate capture nor high cosine establishes safe hard-drop.
//! Corpora come from the current working tree; record its revision and local
//! changes with results. The probe itself is excluded from the code corpus.
//!
//! Run:
//!   cargo test --manifest-path entroly-core/Cargo.toml --lib simhash_probe \
//!       -- --ignored --nocapture --test-threads=1

use crate::dedup::{hamming_distance, simhash};
use std::path::{Path, PathBuf};

/// Production `DedupIndex` banding: 4 bands x 16 bits (see `dedup.rs`).
const NUM_BANDS: usize = 4;
const BITS_PER_BAND: usize = 16;

/// Baseline production threshold. The sweep explores alternatives without
/// recommending a production change or asserting semantic safety.
const PROD_THRESHOLD: u32 = 5;

/// Fingerprint width.
const BITS: f64 = 64.0;

/// Cap on corpus size, to keep the pairwise sweep tractable.
const MAX_FRAGMENTS: usize = 1500;

struct Fragment {
    #[allow(dead_code)]
    origin: String,
    /// Raw text, retained so the controlled arm can perturb real fragments.
    body: String,
    fingerprint: u64,
    /// Trigram multiset as (hash, count), sorted by hash. This is the feature
    /// vector `simhash` actually consumes.
    tf: Vec<(u64, u32)>,
}

/// Extract the 4 band keys exactly as `DedupIndex::extract_bands` does.
fn extract_bands(fp: u64) -> [u64; NUM_BANDS] {
    let mut bands = [0u64; NUM_BANDS];
    let mask = (1u64 << BITS_PER_BAND) - 1;
    for (b, slot) in bands.iter_mut().enumerate() {
        *slot = (fp >> (b * BITS_PER_BAND)) & mask;
    }
    bands
}

/// Whether `DedupIndex::insert` would ever consider these a candidate pair.
fn shares_band(a: u64, b: u64) -> bool {
    extract_bands(a)
        .iter()
        .zip(extract_bands(b).iter())
        .any(|(x, y)| x == y)
}

/// FNV-1a, deliberately a different hash family from `simhash`'s MD5.
fn hash_trigram(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in s.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Tokenize exactly as `dedup::simhash` does.
///
/// This is a replica, and a replica that drifts is worse than no replica: it
/// reports confident numbers about a function it no longer models. It drifted
/// once already — the operator-retention fix landed in `dedup::simhash` while
/// this copy kept dropping standalone operator tokens — so
/// `tokenization_replica_still_matches_production` now pins the two together.
fn tokenize_like_production(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .map(|w| {
            let trimmed = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            // Keep the raw token when trimming would erase it: a standalone
            // operator is all punctuation and must not vanish.
            if trimmed.is_empty() {
                w
            } else {
                trimmed
            }
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word-trigram term frequencies over [`tokenize_like_production`], joining
/// adjacent triples and lowercasing the joined feature.
fn trigram_tf(text: &str) -> Vec<(u64, u32)> {
    let words = tokenize_like_production(text);

    if words.len() < 3 {
        return Vec::new();
    }

    let mut hashes: Vec<u64> = words
        .windows(3)
        .map(|w| hash_trigram(&format!("{} {} {}", w[0], w[1], w[2]).to_lowercase()))
        .collect();
    hashes.sort_unstable();

    let mut tf: Vec<(u64, u32)> = Vec::with_capacity(hashes.len());
    for h in hashes {
        match tf.last_mut() {
            Some((prev, count)) if *prev == h => *count += 1,
            _ => tf.push((h, 1)),
        }
    }
    tf
}

/// TF-weighted cosine over the trigram multiset — the quantity SimHash
/// estimates: for random-hyperplane hashing, P(bit differs) = theta/pi.
fn cosine_tf(a: &[(u64, u32)], b: &[(u64, u32)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (mut i, mut j) = (0usize, 0usize);
    let mut dot = 0.0f64;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                dot += a[i].1 as f64 * b[j].1 as f64;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let na: f64 = a
        .iter()
        .map(|(_, c)| (*c as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|(_, c)| (*c as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(0.0, 1.0)
    }
}

/// Jaccard over DISTINCT trigrams: lexical overlap, not a semantic label or
/// proof of byte equality.
fn jaccard_distinct(a: &[(u64, u32)], b: &[(u64, u32)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

// ── estimators ──────────────────────────────────────────────────────

/// The estimator currently in `knapsack_sds::diversity_factor`.
fn est_linear(d: u32) -> f64 {
    (1.0 - d as f64 / BITS).max(0.0)
}

/// The principled SimHash estimator (already used correctly in `cache.rs`).
fn est_angular(d: u32) -> f64 {
    (std::f64::consts::PI * d as f64 / BITS).cos().max(0.0)
}

// ── corpus ──────────────────────────────────────────────────────────

/// xorshift64*, so the sampling below is reproducible run to run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Cap above which the pairwise distribution is sampled rather than enumerated.
/// `MAX_FRAGMENTS` yields C(1500,2) = 1_124_250 pairs, so every corpus this
/// probe can build stays far below it and is measured exactly.
const EXHAUSTIVE_PAIR_CAP: u64 = 50_000_000;

/// The pairwise Hamming distance distribution, held as a histogram.
///
/// The arms below previously drew 50_000 index pairs *with replacement* and
/// printed the resulting minimum. Two things were wrong with that. A sampled
/// minimum is an upper bound on the true minimum and never an estimate of it,
/// so the closest pair in the corpus was very likely never examined: 50_000
/// draws cover under 5% of C(1500,2), and with replacement fewer still. And
/// the statistic the threshold question actually needs — how many distinct
/// pairs sit in the near region — is *rare*, which is precisely where
/// sampling the bulk wastes almost every draw.
///
/// 65 counters and one popcount per pair make the whole distribution exact,
/// so the statistic stops being something that would need a confidence
/// interval. Memory is constant in the number of pairs, which is why the cap
/// can be this generous.
///
/// Exactness fixes the *statistics* only. It says nothing about semantics: an
/// exact minimum over this corpus is still an exact minimum over this corpus.
/// `probe_labelled_hard_negatives` is the arm that carries labels.
struct PairDistances {
    /// Pairs at each Hamming distance 0..=64.
    all: [u64; 65],
    /// Same, restricted to pairs the band gate would surface as candidates.
    banded: [u64; 65],
    /// False when the cap forced sampling, so no caller can report a sampled
    /// minimum as though it were enumerated.
    exhaustive: bool,
}

impl PairDistances {
    fn new() -> Self {
        Self {
            all: [0; 65],
            banded: [0; 65],
            exhaustive: true,
        }
    }

    fn record(&mut self, a: u64, b: u64) {
        let d = (hamming_distance(a, b) as usize).min(64);
        self.all[d] += 1;
        if shares_band(a, b) {
            self.banded[d] += 1;
        }
    }

    /// Every unordered pair when the corpus is small enough, a reproducible
    /// sample otherwise.
    fn over(frags: &[Fragment]) -> Self {
        let n = frags.len() as u64;
        let pairs = n.saturating_mul(n.saturating_sub(1)) / 2;
        let mut out = Self::new();

        if pairs <= EXHAUSTIVE_PAIR_CAP {
            for i in 0..frags.len() {
                for j in (i + 1)..frags.len() {
                    out.record(frags[i].fingerprint, frags[j].fingerprint);
                }
            }
            return out;
        }

        out.exhaustive = false;
        let mut rng = Rng(0xD15C_0FFE_E5EE_D001);
        let mut drawn = 0usize;
        while drawn < 2_000_000 {
            let i = rng.below(frags.len());
            let j = rng.below(frags.len());
            if i == j {
                continue;
            }
            out.record(frags[i].fingerprint, frags[j].fingerprint);
            drawn += 1;
        }
        out
    }

    fn total(&self) -> u64 {
        self.all.iter().sum()
    }

    /// Smallest distance at which any pair exists; exact when `exhaustive`.
    fn min(&self) -> Option<u32> {
        self.all.iter().position(|&c| c > 0).map(|d| d as u32)
    }

    /// Smallest distance among pairs the band gate would actually surface.
    /// Pairs below this are unreachable by `DedupIndex` whatever the
    /// threshold, so this — not `min` — is the band-gated floor.
    fn min_banded(&self) -> Option<u32> {
        self.banded.iter().position(|&c| c > 0).map(|d| d as u32)
    }

    /// Share of all pairs at distance <= d, as a percentage. `gated`
    /// restricts the numerator to band-gate candidates; the denominator stays
    /// the full pair count either way, so the two columns are comparable.
    fn rate_at(&self, d: u32, gated: bool) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let src = if gated { &self.banded } else { &self.all };
        let hits: u64 = src[..=(d as usize).min(64)].iter().sum();
        100.0 * hits as f64 / total as f64
    }

    fn quantile(&self, p: f64) -> u32 {
        let total = self.total();
        if total == 0 {
            return 0;
        }
        let target = ((total as f64) * p / 100.0).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (d, &c) in self.all.iter().enumerate() {
            seen += c;
            if seen >= target {
                return d as u32;
            }
        }
        64
    }

    fn basis(&self) -> &'static str {
        if self.exhaustive {
            "exact over every pair"
        } else {
            "SAMPLED — minimum is an upper bound only"
        }
    }
}

/// Independent O(n^2) recomputation, used only to check `PairDistances`.
#[cfg(test)]
fn brute_force_min_distance(fps: &[u64]) -> Option<u32> {
    let mut best: Option<u32> = None;
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            let d = hamming_distance(fps[i], fps[j]);
            if best.is_none_or(|b| d < b) {
                best = Some(d);
            }
        }
    }
    best
}

fn chunk_source(text: &str, origin: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let is_def_start = |l: &str| {
        let indent = l.len() - l.trim_start().len();
        if indent > 4 {
            return false;
        }
        let t = l.trim_start();
        t.starts_with("fn ")
            || t.starts_with("pub fn ")
            || t.starts_with("pub async fn ")
            || t.starts_with("async fn ")
            || t.starts_with("def ")
            || t.starts_with("class ")
            || t.starts_with("pub struct ")
    };

    let mut starts: Vec<usize> = (0..lines.len())
        .filter(|&i| is_def_start(lines[i]))
        .collect();
    if starts.is_empty() {
        return Vec::new();
    }
    starts.push(lines.len());

    let mut out = Vec::new();
    for w in starts.windows(2) {
        let (s, e) = (w[0], w[1].min(w[0] + 200));
        if e - s < 5 {
            continue;
        }
        let body = lines[s..e].join("\n");
        if body.split_whitespace().count() < 20 {
            continue;
        }
        out.push((format!("{}:{}", origin, s + 1), body));
    }
    out
}

fn collect_sources(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort(); // determinism
    for p in paths {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if p.is_dir() {
            collect_sources(&p, out);
        } else if matches!(
            p.extension().and_then(|x| x.to_str()),
            Some("rs") | Some("py")
        ) {
            out.push(p);
        }
    }
}

fn build_corpus() -> Vec<Fragment> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest.parent().unwrap_or(&manifest).to_path_buf();

    let mut files = Vec::new();
    collect_sources(&manifest.join("src"), &mut files);
    collect_sources(&repo_root.join("entroly"), &mut files);
    // Editing the measurement code must not alter its own input fragments.
    files.retain(|path| path != &manifest.join("src/simhash_probe.rs"));

    let mut frags = Vec::new();
    for path in files {
        if frags.len() >= MAX_FRAGMENTS {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let origin = path
            .strip_prefix(&repo_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        for (id, body) in chunk_source(&text, &origin) {
            if frags.len() >= MAX_FRAGMENTS {
                break;
            }
            let tf = trigram_tf(&body);
            if tf.len() < 5 {
                continue;
            }
            frags.push(Fragment {
                origin: id,
                fingerprint: simhash(&body),
                tf,
                body,
            });
        }
    }
    frags
}

// ════════════════════════════════════════════════════════════════════
// Arm 1 — estimator calibration against TF cosine.
// ════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_estimator_calibration() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "corpus too small: {}", frags.len());

    const NBINS: usize = 5;
    let mut n = [0usize; NBINS];
    let (mut err_lin, mut err_ang) = ([0.0f64; NBINS], [0.0f64; NBINS]);
    let (mut sum_lin, mut sum_ang, mut sum_true) =
        ([0.0f64; NBINS], [0.0f64; NBINS], [0.0f64; NBINS]);
    let mut sum_d = [0.0f64; NBINS];

    let m = frags.len();
    for i in 0..m {
        for j in (i + 1)..m {
            let d = hamming_distance(frags[i].fingerprint, frags[j].fingerprint);
            let truth = cosine_tf(&frags[i].tf, &frags[j].tf);
            let (lin, ang) = (est_linear(d), est_angular(d));

            let b = ((truth * NBINS as f64) as usize).min(NBINS - 1);
            n[b] += 1;
            err_lin[b] += (lin - truth).abs();
            err_ang[b] += (ang - truth).abs();
            sum_lin[b] += lin;
            sum_ang[b] += ang;
            sum_true[b] += truth;
            sum_d[b] += d as f64;
        }
    }

    println!("\n=== ARM 1: estimator calibration vs TF-cosine ground truth ===");
    println!(
        "{:<13} {:>10} {:>8} {:>8} {:>9} {:>11} {:>9} {:>9}",
        "true cosine",
        "pairs",
        "mean cos",
        "mean d",
        "1-d/64",
        "cos(pi d/64)",
        "MAE lin",
        "MAE ang"
    );

    let (mut tot, mut tl, mut ta) = (0usize, 0.0, 0.0);
    for b in 0..NBINS {
        if n[b] == 0 {
            continue;
        }
        let c = n[b] as f64;
        println!(
            "{:<13} {:>10} {:>8.3} {:>8.1} {:>9.3} {:>11.3} {:>9.3} {:>9.3}",
            format!(
                "{:.1}-{:.1}",
                b as f64 / NBINS as f64,
                (b + 1) as f64 / NBINS as f64
            ),
            n[b],
            sum_true[b] / c,
            sum_d[b] / c,
            sum_lin[b] / c,
            sum_ang[b] / c,
            err_lin[b] / c,
            err_ang[b] / c
        );
        tot += n[b];
        tl += err_lin[b];
        ta += err_ang[b];
    }

    let (mae_l, mae_a) = (tl / tot as f64, ta / tot as f64);
    println!("\noverall MAE  1-d/64       : {:.4}", mae_l);
    println!("overall MAE  cos(pi d/64) : {:.4}", mae_a);
    println!("error reduction           : {:.1}x", mae_l / mae_a);

    // Sampling-noise floor. At true angle theta, d ~ Binomial(B, theta/pi),
    // so sd(d) = sqrt(B p (1-p)) and the induced similarity error is
    // |sin(theta)| * pi * sd(d) / B. At orthogonality (p = 0.5) that is
    // pi*sqrt(B/4)/B = pi/(2 sqrt(B)). Half of those estimates fall below
    // zero and are clamped to exactly 0, so the realised MAE is ~half.
    let sd_sim = std::f64::consts::PI / (2.0 * BITS.sqrt());
    println!(
        "\nnoise floor at orthogonality: sd={:.3}, expected MAE after clamping ~{:.3}",
        sd_sim,
        sd_sim * 0.798 / 2.0
    );
    println!("  -> an estimator at the floor cannot be improved by better math,");
    println!("     only by more bits. Bits needed scales as 1/sqrt(B):");
    for b in [64u32, 256, 1024] {
        println!(
            "     B={:<5} sd(similarity) = {:.4}",
            b,
            std::f64::consts::PI / (2.0 * (b as f64).sqrt())
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════════
// Arm 2 — lexical candidate capture on controlled fragment edits.
// ════════════════════════════════════════════════════════════════════

/// Replace the most frequent identifier-like token as a lexical perturbation.
/// This is a substring replacement, not a scope-aware, semantics-preserving refactor.
fn rename_identifier(text: &str) -> String {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in text.split_whitespace() {
        let t = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        if t.len() > 4 && t.chars().next().is_some_and(|c| c.is_alphabetic()) {
            *counts.entry(t).or_insert(0) += 1;
        }
    }
    match counts.iter().max_by_key(|(k, v)| (**v, **k)) {
        Some((target, _)) => text.replace(target, "renamed_symbol"),
        None => text.to_string(),
    }
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_controlled_near_duplicates() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "corpus too small: {}", frags.len());

    /// (label, perturbation) — edits with no semantic equivalence guarantee.
    type Perturbation<'a> = (&'a str, Box<dyn Fn(&str) -> String>);

    let classes: Vec<Perturbation> = vec![
        (
            "comment prepended",
            Box::new(|t: &str| format!("// NOTE: reviewed for correctness.\n{}", t)),
        ),
        (
            "whitespace reflow",
            Box::new(|t: &str| t.lines().map(|l| l.trim()).collect::<Vec<_>>().join("\n")),
        ),
        (
            "one line changed",
            Box::new(|t: &str| {
                let mut lines: Vec<String> = t.lines().map(String::from).collect();
                let mid = lines.len() / 2;
                lines[mid] = "    let adjusted_value = compute_replacement_here();".into();
                lines.join("\n")
            }),
        ),
        ("identifier renamed", Box::new(rename_identifier)),
        (
            "overlapping window (75%)",
            Box::new(|t: &str| {
                let lines: Vec<&str> = t.lines().collect();
                lines[lines.len() / 4..].join("\n")
            }),
        ),
    ];

    println!("\n=== ARM 2: lexical candidate capture on edited fragments ===");
    println!("base fragments: {}\n", frags.len());
    println!(
        "{:<26} {:>6} {:>8} {:>8} {:>7} {:>9} {:>9}",
        "perturbation", "pairs", "jaccard", "cos", "mean d", "banded%", "GATED%"
    );

    for (label, gen) in &classes {
        let (mut n, mut sj, mut sc, mut sd, mut banded, mut merged) =
            (0usize, 0.0, 0.0, 0u64, 0usize, 0usize);

        for f in frags.iter() {
            let twin = gen(&f.body);
            let tf = trigram_tf(&twin);
            if tf.len() < 5 {
                continue;
            }
            let fp = simhash(&twin);
            let d = hamming_distance(f.fingerprint, fp);
            let b = shares_band(f.fingerprint, fp);

            n += 1;
            sj += jaccard_distinct(&f.tf, &tf);
            sc += cosine_tf(&f.tf, &tf);
            sd += d as u64;
            if b {
                banded += 1;
            }
            if b && d <= PROD_THRESHOLD {
                merged += 1;
            }
        }

        if n == 0 {
            continue;
        }
        let c = n as f64;
        println!(
            "{:<26} {:>6} {:>8.3} {:>8.3} {:>7.1} {:>8.1}% {:>8.1}%",
            label,
            n,
            sj / c,
            sc / c,
            sd as f64 / c,
            100.0 * banded as f64 / c,
            100.0 * merged as f64 / c
        );
    }

    println!(
        "\n'GATED%' measures band match AND d<={}; it does not replay insertion.\n\
         Gated capture cannot exceed banded capture. Edited fragments may change\n\
         behaviour, so these rates are lexical capture, not semantic recall or\n\
         evidence that either fragment can safely be discarded.\n",
        PROD_THRESHOLD
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 4 — effect on real selection.
//
// Drives the actual `ios_select` greedy loop over real fragments and replays
// the diversity multiplier the loop used at each step, alongside what the old
// `1 - hamming/64` form would have produced at that same step. Verifies the
// change is not inert and quantifies how much the diversity term moved.
// ════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_selection_effect() {
    use crate::fragment::ContextFragment;
    use crate::knapsack_sds::{ios_select, InfoFactors};
    use std::collections::HashMap;

    let frags = build_corpus();
    let ctx: Vec<ContextFragment> = frags
        .iter()
        .take(400)
        .enumerate()
        .map(|(i, f)| {
            let tokens = (f.body.split_whitespace().count() as u32).max(1);
            let mut c =
                ContextFragment::new(format!("f{i}"), f.body.clone(), tokens, "probe.rs".into());
            // `ContextFragment::new` leaves the fragment fingerprint-less.
            // Production sets both fields together (lib.rs:752); so must this,
            // or the diversity term is skipped entirely and the arm measures
            // nothing.
            c.simhash = f.fingerprint;
            c.has_simhash = true;
            c.recency_score = 0.9;
            c.entropy_score = 0.7;
            c
        })
        .collect();

    let res = ios_select(
        &ctx,
        8000,
        0.25,
        0.25,
        0.25,
        0.25,
        &HashMap::new(),
        true,
        true,
        &InfoFactors::default(),
        0.10,
        0.0,
    );

    println!("\n=== ARM 4: effect on real ios_select ===");
    println!(
        "selected {} of {} fragments, {} tokens",
        res.selections.len(),
        ctx.len(),
        res.total_tokens
    );

    // Replay the greedy trajectory to recover the multiplier at each step.
    const FLOOR: f64 = 0.10;
    let mut chosen: Vec<u64> = Vec::new();
    let (mut sum_old, mut sum_new, mut steps, mut moved) = (0.0f64, 0.0f64, 0usize, 0usize);
    let (mut min_new, mut max_new) = (f64::MAX, 0.0f64);
    let (mut min_old, mut max_old) = (f64::MAX, 0.0f64);

    for (idx, _res_level) in &res.selections {
        let h = ctx[*idx].simhash;
        if !chosen.is_empty() {
            // New: 1 - max cos. Old: 1 - max(1 - d/64) == min(d)/64.
            // Mirrors the shipped `diversity_factor`: union-bounded LCB over
            // the k comparisons being maximised.
            let new_div = (1.0
                - chosen
                    .iter()
                    .map(|&c| crate::dedup::simhash_cosine_lcb(c, h, chosen.len(), 0.05))
                    .fold(0.0f64, f64::max))
            .max(FLOOR);
            let old_div = (chosen
                .iter()
                .map(|&c| hamming_distance(c, h) as f64 / BITS)
                .fold(f64::MAX, f64::min))
            .max(FLOOR);

            sum_new += new_div;
            sum_old += old_div;
            steps += 1;
            if (new_div - old_div).abs() > 0.01 {
                moved += 1;
            }
            min_new = min_new.min(new_div);
            max_new = max_new.max(new_div);
            min_old = min_old.min(old_div);
            max_old = max_old.max(old_div);
        }
        chosen.push(h);
    }

    if steps == 0 {
        println!("(only one fragment selected — no diversity interaction to measure)\n");
        return;
    }

    println!("\ndiversity multiplier applied during greedy selection ({steps} steps):");
    println!(
        "  old (1 - d/64)   mean {:.3}   range {:.3}..{:.3}   dynamic range {:.1}x",
        sum_old / steps as f64,
        min_old,
        max_old,
        max_old / min_old
    );
    println!(
        "  new (cosine)     mean {:.3}   range {:.3}..{:.3}   dynamic range {:.1}x",
        sum_new / steps as f64,
        min_new,
        max_new,
        max_new / min_new
    );
    println!(
        "  steps where the multiplier moved by >0.01: {}/{} ({:.0}%)",
        moved,
        steps,
        100.0 * moved as f64 / steps as f64
    );

    // User-facing reported score, old formula vs shipped.
    let n = chosen.len();
    let (mut old_sum, mut new_sum, mut pairs) = (0.0f64, 0.0f64, 0usize);
    for i in 0..n {
        for j in (i + 1)..n {
            old_sum += hamming_distance(chosen[i], chosen[j]) as f64 / BITS;
            new_sum += 1.0 - crate::dedup::simhash_cosine(chosen[i], chosen[j]);
            pairs += 1;
        }
    }
    if pairs > 0 {
        println!(
            "\nreported diversity_score for this selection:\n  \
             old formula {:.3}   shipped {:.3}   (1.0 == mutually unrelated)",
            old_sum / pairs as f64,
            new_sum / pairs as f64
        );
        println!("  ios_select returned: {:.3}", res.diversity_score);
    }
    println!();
}

// ════════════════════════════════════════════════════════════════════
// Arm 3 — fingerprint construction bias.
//
// `simhash` sets bit i when `bit_sums[i] > 0`, so an exactly-tied accumulator
// resolves to 0. Ties are common when a fragment has few trigrams, or when the
// trigram count is even. If ties are frequent, fingerprints are pulled toward
// all-zeros and unrelated sparse fragments look artificially similar — a bias
// no downstream estimator can undo.
// ════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_fingerprint_bias() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "corpus too small: {}", frags.len());

    println!("\n=== ARM 3: fingerprint balance (expect popcount ~32) ===");
    println!(
        "{:<18} {:>8} {:>12} {:>12}",
        "trigram count", "frags", "mean popcount", "mean bit-0 %"
    );

    let buckets: [(usize, usize); 4] = [(0, 50), (50, 150), (150, 400), (400, usize::MAX)];
    for (lo, hi) in buckets {
        let sel: Vec<&Fragment> = frags
            .iter()
            .filter(|f| {
                let total: u32 = f.tf.iter().map(|(_, c)| *c).sum();
                (total as usize) >= lo && (total as usize) < hi
            })
            .collect();
        if sel.is_empty() {
            continue;
        }
        let mean_pop: f64 = sel
            .iter()
            .map(|f| f.fingerprint.count_ones() as f64)
            .sum::<f64>()
            / sel.len() as f64;
        println!(
            "{:<18} {:>8} {:>12.1} {:>11.1}%",
            format!("{}..{}", lo, if hi == usize::MAX { 999999 } else { hi }),
            sel.len(),
            mean_pop,
            100.0 * (1.0 - mean_pop / BITS)
        );
    }
    println!("\na balanced fingerprint has popcount 32 (50% zeros). Systematic deficit");
    println!("means tied accumulators are resolving to 0 and inflating similarity.\n");
}

// ════════════════════════════════════════════════════════════════════
// Arm 5 — selection-set-size bias (the optimizer's curse, inside the
// greedy loop).
//
// `diversity_factor` computes `max` similarity over the ALREADY-SELECTED set.
// Taking a max over k noisy estimates is upward-biased even when every
// estimate is individually unbiased (Smith & Winkler 2006). For estimates with
// noise sd `sigma`, the max of k of them sits roughly `sigma * sqrt(2 ln k)`
// above the true max.
//
// That has a consequence specific to greedy submodular selection: k is the
// size of the selected set, so it GROWS DURING THE RUN. The redundancy penalty
// applied to the 200th candidate is systematically harsher than the one
// applied to the 5th, for no reason except that more noisy comparisons have
// been drawn. The engine invents redundancy that is not there, and invents
// more of it the longer it runs.
//
// Critically, this is NOT fixed by widening the fingerprint. More bits shrink
// `sigma`, but the `sqrt(2 ln k)` factor survives untouched.
// ════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_selection_set_size_bias() {
    let frags = build_corpus();
    assert!(frags.len() > 400, "corpus too small: {}", frags.len());
    let n = frags.len();

    // Empirical noise sd of the estimator on near-orthogonal real pairs.
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let (mut s1, mut s2, mut cnt) = (0.0f64, 0.0f64, 0usize);
    for _ in 0..20_000 {
        let (a, b) = (rng.below(n), rng.below(n));
        if a == b {
            continue;
        }
        if cosine_tf(&frags[a].tf, &frags[b].tf) > 0.05 {
            continue; // keep only genuinely unrelated pairs
        }
        let e = est_angular(hamming_distance(frags[a].fingerprint, frags[b].fingerprint));
        s1 += e;
        s2 += e * e;
        cnt += 1;
    }
    let mean = s1 / cnt as f64;
    let sigma = (s2 / cnt as f64 - mean * mean).sqrt();

    println!("\n=== ARM 5: redundancy invented by set size (optimizer's curse) ===");
    println!(
        "estimator on {} genuinely unrelated real pairs: mean {:.3}, sd {:.3}",
        cnt, mean, sigma
    );
    println!("(true cosine for these pairs is ~0, so all of this is noise)\n");

    println!(
        "{:>6} {:>10} {:>10} {:>11} {:>12} {:>11}",
        "k", "true max", "uncorrected", "invented", "CORRECTED", "residual"
    );

    const TRIALS: usize = 400;
    for k in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        let (mut est_acc, mut true_acc, mut lcb_acc) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..TRIALS {
            let q = rng.below(n);
            let (mut est_max, mut true_max, mut lcb_max) = (0.0f64, 0.0f64, 0.0f64);
            for _ in 0..k {
                let s = rng.below(n);
                if s == q {
                    continue;
                }
                let (fq, fs) = (frags[q].fingerprint, frags[s].fingerprint);
                est_max = est_max.max(est_angular(hamming_distance(fq, fs)));
                lcb_max = lcb_max.max(crate::dedup::simhash_cosine_lcb(fq, fs, k, 0.05));
                true_max = true_max.max(cosine_tf(&frags[q].tf, &frags[s].tf));
            }
            est_acc += est_max;
            true_acc += true_max;
            lcb_acc += lcb_max;
        }
        let (e, t, l) = (
            est_acc / TRIALS as f64,
            true_acc / TRIALS as f64,
            lcb_acc / TRIALS as f64,
        );
        println!(
            "{:>6} {:>10.3} {:>10.3} {:>11.3} {:>12.3} {:>11.3}",
            k,
            t,
            e,
            (e - t).max(0.0),
            l,
            (l - t).max(0.0)
        );
    }

    println!("\n'invented' is redundancy the uncorrected estimator perceives but which");
    println!("does not exist. 'CORRECTED' is the union-bounded lower confidence bound");
    println!("now used by diversity_factor; 'residual' is what it still over-states.");

    // ── Discrimination check ──
    //
    // A bound that removes invented redundancy is worthless if it also stops
    // detecting real redundancy — that would trade a noisy penalty for an
    // inert one, which is the failure this whole line of work started from.
    // So: replay genuine near-duplicates through the shipped path at a large
    // selected-set size, where the union bound is at its most conservative.
    /// (label, how to build the counterpart fragment).
    type Relationship = (&'static str, fn(&str) -> String);

    let classes: [Relationship; 4] = [
        ("identical", |t: &str| t.to_string()),
        ("comment prepended", |t: &str| {
            format!("// NOTE: reviewed.\n{t}")
        }),
        ("one line changed", |t: &str| {
            let mut lines: Vec<String> = t.lines().map(String::from).collect();
            let mid = lines.len() / 2;
            lines[mid] = "    let adjusted_value = compute_replacement_here();".into();
            lines.join("\n")
        }),
        ("unrelated fragment", |_t: &str| String::new()),
    ];

    println!("\n--- discrimination retained at k=256 (penalty must survive) ---");
    println!(
        "{:<22} {:>10} {:>14} {:>14}",
        "relationship", "true cos", "LCB penalty", "diversity mult"
    );

    for (label, gen) in classes {
        let (mut sc, mut sl, mut cnt) = (0.0f64, 0.0f64, 0usize);
        for (i, f) in frags.iter().take(300).enumerate() {
            let (twin_tf, twin_fp) = if label == "unrelated fragment" {
                // Pair against a different real fragment instead of a twin.
                let o = &frags[(i + 137) % frags.len()];
                (o.tf.clone(), o.fingerprint)
            } else {
                let t = gen(&f.body);
                let tf = trigram_tf(&t);
                if tf.len() < 5 {
                    continue;
                }
                (tf, simhash(&t))
            };
            sc += cosine_tf(&f.tf, &twin_tf);
            sl += crate::dedup::simhash_cosine_lcb(f.fingerprint, twin_fp, 256, 0.05);
            cnt += 1;
        }
        if cnt == 0 {
            continue;
        }
        let (c, l) = (sc / cnt as f64, sl / cnt as f64);
        println!(
            "{:<22} {:>10.3} {:>14.3} {:>14.3}",
            label,
            c,
            l,
            (1.0 - l).max(0.10)
        );
    }
    println!("\n(diversity mult 1.0 = no penalty; 0.10 = floor. A usable bound must");
    println!(" keep real duplicates well below 1.0 while leaving unrelated at 1.0.)\n");
}

// ════════════════════════════════════════════════════════════════════
// Arm 6 — can anything beat the band-gate ceiling at 64 bits?
//
// Compare lexical candidate capture across hash families and band geometries.
// The band gate and the distance threshold are separate filters; neither
// establishes that an edited fragment is interchangeable with its original.
//
// This arm holds the budget at 64 bits and varies both, over the same corpus
// and the same perturbations Arm 2 uses:
//
//   A. SimHash 4x16  — production, the control.
//   B. SimHash 8x8   — same signature bits and family, finer bands; index and
//                      lookup costs are measured separately.
//   C. b=2 MinHash, k=32, 8 bands x 4 rows.
//   D. b=1 MinHash, k=64, 16 bands x 4 rows.
//
// Candidate capture alone does not measure precision. Random pairs are also
// sampled, but remain unlabelled: some may be duplicates and others may differ
// in behaviour despite high lexical similarity. Report hit rates, not a
// false-positive rate or a hard-drop safety conclusion.
//
// MinHash estimates Jaccard over distinct trigrams; jaccard_distinct above is
// the matching oracle. SimHash estimates TF-cosine. They are different
// quantities, so the table prints both: a family change silently redefines
// what "similar" means for term-frequency-weighted code.
// ════════════════════════════════════════════════════════════════════

/// Odd multipliers for universal hashing; odd keeps multiply-shift bijective.
fn minhash_params(i: usize) -> (u64, u64) {
    let a = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let b = (i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1;
    (a, b)
}

/// k-permutation MinHash over the *distinct* trigrams, truncated to `b` bits.
///
/// Distinct rather than TF-weighted, so the signature estimates the same
/// quantity `jaccard_distinct` computes. Returns one value per permutation;
/// only the low `b` bits of each are ever compared.
fn minhash_bbit(tf: &[(u64, u32)], k: usize, b: u32) -> Vec<u64> {
    let mask = if b >= 64 { u64::MAX } else { (1u64 << b) - 1 };
    let mut sig = vec![u64::MAX; k];
    for &(term, _count) in tf {
        for (i, slot) in sig.iter_mut().enumerate() {
            let (a, c) = minhash_params(i);
            let h = term.wrapping_mul(a) ^ c;
            if h < *slot {
                *slot = h;
            }
        }
    }
    sig.iter().map(|h| h & mask).collect()
}

/// True when any band of `rows` consecutive signature entries matches exactly.
fn minhash_shares_band(a: &[u64], b: &[u64], rows: usize) -> bool {
    a.chunks(rows)
        .zip(b.chunks(rows))
        .any(|(x, y)| x.len() == rows && x == y)
}

/// SimHash band gate at an arbitrary geometry, so 8x8 can be compared with
/// the production 4x16 without touching `dedup.rs`.
fn simhash_shares_band_at(a: u64, b: u64, bands: usize, bits: usize) -> bool {
    (0..bands).any(|i| {
        let shift = (i * bits) as u32;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        ((a >> shift) & mask) == ((b >> shift) & mask)
    })
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_band_gate_alternatives() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "corpus too small: {}", frags.len());

    type Perturbation<'a> = (&'a str, Box<dyn Fn(&str) -> String>);
    let classes: Vec<Perturbation> = vec![
        (
            "comment prepended",
            Box::new(|t: &str| format!("// NOTE: reviewed for correctness.\n{}", t)),
        ),
        (
            "one line changed",
            Box::new(|t: &str| {
                let mut lines: Vec<String> = t.lines().map(String::from).collect();
                let mid = lines.len() / 2;
                lines[mid] = "    let adjusted_value = compute_replacement_here();".into();
                lines.join("\n")
            }),
        ),
        ("identifier renamed", Box::new(rename_identifier)),
        (
            "overlapping window (75%)",
            Box::new(|t: &str| {
                let lines: Vec<&str> = t.lines().collect();
                lines[lines.len() / 4..].join("\n")
            }),
        ),
    ];

    println!("\n=== ARM 6: band-gate alternatives at a fixed 64-bit budget ===");
    println!("base fragments: {}\n", frags.len());
    println!("Lexical candidate capture on edited fragments (not semantic duplicate recall):");
    println!(
        "{:<26} {:>6} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "perturbation", "pairs", "jacc", "cos", "SH 4x16", "SH 8x8", "MH b2k32", "MH b1k64"
    );

    for (label, gen) in &classes {
        let (mut n, mut sj, mut sc) = (0usize, 0.0f64, 0.0f64);
        let (mut a, mut b, mut c, mut d) = (0usize, 0usize, 0usize, 0usize);

        for f in frags.iter() {
            let twin = gen(&f.body);
            let tf = trigram_tf(&twin);
            if tf.len() < 5 {
                continue;
            }
            let fp = simhash(&twin);
            n += 1;
            sj += jaccard_distinct(&f.tf, &tf);
            sc += cosine_tf(&f.tf, &tf);

            if simhash_shares_band_at(f.fingerprint, fp, NUM_BANDS, BITS_PER_BAND) {
                a += 1;
            }
            if simhash_shares_band_at(f.fingerprint, fp, 8, 8) {
                b += 1;
            }
            if minhash_shares_band(&minhash_bbit(&f.tf, 32, 2), &minhash_bbit(&tf, 32, 2), 4) {
                c += 1;
            }
            if minhash_shares_band(&minhash_bbit(&f.tf, 64, 1), &minhash_bbit(&tf, 64, 1), 4) {
                d += 1;
            }
        }
        if n == 0 {
            continue;
        }
        let q = n as f64;
        println!(
            "{:<26} {:>6} {:>7.3} {:>7.3} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}%",
            label,
            n,
            sj / q,
            sc / q,
            100.0 * a as f64 / q,
            100.0 * b as f64 / q,
            100.0 * c as f64 / q,
            100.0 * d as f64 / q
        );
    }

    // Unlabelled random-pair candidate hit rate, not a precision control.
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let (mut n, mut sj, mut sc) = (0usize, 0.0f64, 0.0f64);
    let (mut a, mut b, mut c, mut d) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..20_000usize {
        let i = rng.below(frags.len());
        let j = rng.below(frags.len());
        if i == j {
            continue;
        }
        let (x, y) = (&frags[i], &frags[j]);
        n += 1;
        sj += jaccard_distinct(&x.tf, &y.tf);
        sc += cosine_tf(&x.tf, &y.tf);
        if simhash_shares_band_at(x.fingerprint, y.fingerprint, NUM_BANDS, BITS_PER_BAND) {
            a += 1;
        }
        if simhash_shares_band_at(x.fingerprint, y.fingerprint, 8, 8) {
            b += 1;
        }
        if minhash_shares_band(&minhash_bbit(&x.tf, 32, 2), &minhash_bbit(&y.tf, 32, 2), 4) {
            c += 1;
        }
        if minhash_shares_band(&minhash_bbit(&x.tf, 64, 1), &minhash_bbit(&y.tf, 64, 1), 4) {
            d += 1;
        }
    }
    let q = n as f64;
    println!(
        "\n{:<26} {:>6} {:>7.3} {:>7.3} {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}%",
        "RANDOM (unlabelled)",
        n,
        sj / q,
        sc / q,
        100.0 * a as f64 / q,
        100.0 * b as f64 / q,
        100.0 * c as f64 / q,
        100.0 * d as f64 / q
    );

    println!(
        "\nColumns measure lexical candidate capture, not semantic duplicate recall.\n\
         Random pairs are unlabelled, so their hit rate is not a false-positive\n\
         rate. Jaccard and TF-cosine describe different lexical feature metrics.\n\
         These measurements do not establish safe hard-drop or justify shipping\n\
         a hash-family, geometry, or threshold change.\n"
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 7 — gated lexical capture and candidate-set sizes for 8x8.
//
// Candidate capture and index work are measured separately on this snapshot:
//
//   1. GATED% — banded AND d <= PROD_THRESHOLD. This evaluates only those two
//      predicates, not a full DedupIndex insertion or semantic correctness.
//
//   2. Candidate-set size — the mean number of fragments sharing at least
//      one band with a given fragment, over the real corpus. This is the
//      proxy for index work, not a latency measurement or shipping gate.
// ════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, HashSet};

/// Band keys at an arbitrary geometry, mirroring `extract_bands`.
fn bands_at(fp: u64, bands: usize, bits: usize) -> Vec<u64> {
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    (0..bands)
        .map(|i| (fp >> ((i * bits) as u32)) & mask)
        .collect()
}

/// Mean and p99 of the candidate set a band index would return per fragment.
///
/// Counts distinct other fragments sharing at least one band, which is
/// a static full-corpus candidate-count proxy, not the result of replaying
/// production insertions or measuring their latency.
fn candidate_set_stats(frags: &[Fragment], bands: usize, bits: usize) -> (f64, usize, usize) {
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (idx, f) in frags.iter().enumerate() {
        for (b, key) in bands_at(f.fingerprint, bands, bits).into_iter().enumerate() {
            buckets.entry((b, key)).or_default().push(idx);
        }
    }

    let mut sizes: Vec<usize> = Vec::with_capacity(frags.len());
    for (idx, f) in frags.iter().enumerate() {
        let mut seen: HashSet<usize> = HashSet::new();
        for (b, key) in bands_at(f.fingerprint, bands, bits).into_iter().enumerate() {
            if let Some(ids) = buckets.get(&(b, key)) {
                for &other in ids {
                    if other != idx {
                        seen.insert(other);
                    }
                }
            }
        }
        sizes.push(seen.len());
    }
    sizes.sort_unstable();
    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
    let p99 = sizes[(sizes.len() * 99 / 100).min(sizes.len() - 1)];
    let max = *sizes.last().unwrap_or(&0);
    (mean, p99, max)
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_band_geometry_capture_and_candidate_counts() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "corpus too small: {}", frags.len());

    type Perturbation<'a> = (&'a str, Box<dyn Fn(&str) -> String>);
    let classes: Vec<Perturbation> = vec![
        (
            "comment prepended",
            Box::new(|t: &str| format!("// NOTE: reviewed for correctness.\n{}", t)),
        ),
        (
            "one line changed",
            Box::new(|t: &str| {
                let mut lines: Vec<String> = t.lines().map(String::from).collect();
                let mid = lines.len() / 2;
                lines[mid] = "    let adjusted_value = compute_replacement_here();".into();
                lines.join("\n")
            }),
        ),
        ("identifier renamed", Box::new(rename_identifier)),
        (
            "overlapping window (75%)",
            Box::new(|t: &str| {
                let lines: Vec<&str> = t.lines().collect();
                lines[lines.len() / 4..].join("\n")
            }),
        ),
    ];

    println!("\n=== ARM 7: gated lexical capture and candidate counts for 8x8 ===");
    println!(
        "corpus: {} fragments, threshold d <= {}\n",
        frags.len(),
        PROD_THRESHOLD
    );

    // Lexical predicate capture; edited pairs are not semantic positives.
    println!("EDITED PAIRS — GATED% (banded AND d <= {})", PROD_THRESHOLD);
    println!(
        "{:<26} {:>6} {:>10} {:>10}",
        "perturbation", "pairs", "4x16", "8x8"
    );
    for (label, gen) in &classes {
        let (mut n, mut m4, mut m8) = (0usize, 0usize, 0usize);
        for f in frags.iter() {
            let twin = gen(&f.body);
            let tf = trigram_tf(&twin);
            if tf.len() < 5 {
                continue;
            }
            let fp = simhash(&twin);
            let d = hamming_distance(f.fingerprint, fp);
            n += 1;
            if d <= PROD_THRESHOLD {
                if simhash_shares_band_at(f.fingerprint, fp, NUM_BANDS, BITS_PER_BAND) {
                    m4 += 1;
                }
                if simhash_shares_band_at(f.fingerprint, fp, 8, 8) {
                    m8 += 1;
                }
            }
        }
        if n == 0 {
            continue;
        }
        let q = n as f64;
        println!(
            "{:<26} {:>6} {:>9.1}% {:>9.1}%",
            label,
            n,
            100.0 * m4 as f64 / q,
            100.0 * m8 as f64 / q
        );
    }

    // Unlabelled pairs may include duplicates; hits are not false merges.
    //
    // Enumerated rather than sampled. Both geometries are needed here, so this
    // does not go through `PairDistances` (which models the production 4x16
    // gate only) and keeps its own counters. The gated counts are small
    // absolute numbers, which is exactly the regime where a 5%-coverage sample
    // of C(n,2) reports a different answer run to run.
    let (mut n, mut f4, mut f8, mut c4, mut c8) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for i in 0..frags.len() {
        for j in (i + 1)..frags.len() {
            let (x, y) = (&frags[i], &frags[j]);
            let d = hamming_distance(x.fingerprint, y.fingerprint);
            n += 1;
            let b4 = simhash_shares_band_at(x.fingerprint, y.fingerprint, NUM_BANDS, BITS_PER_BAND);
            let b8 = simhash_shares_band_at(x.fingerprint, y.fingerprint, 8, 8);
            if b4 {
                c4 += 1;
                if d <= PROD_THRESHOLD {
                    f4 += 1;
                }
            }
            if b8 {
                c8 += 1;
                if d <= PROD_THRESHOLD {
                    f8 += 1;
                }
            }
        }
    }
    let q = n.max(1) as f64;
    println!(
        "\nUNLABELLED PAIRS — all {} enumerated\n  \
         band hits  : 4x16 {:.2}%   8x8 {:.2}%\n  \
         gated hits : 4x16 {} ({:.4}%)   8x8 {} ({:.4}%)",
        n,
        100.0 * c4 as f64 / q,
        100.0 * c8 as f64 / q,
        f4,
        100.0 * f4 as f64 / q,
        f8,
        100.0 * f8 as f64 / q
    );

    // Static full-corpus candidate counts, not measured lookup latency.
    let (m4, p99_4, max4) = candidate_set_stats(&frags, NUM_BANDS, BITS_PER_BAND);
    let (m8, p99_8, max8) = candidate_set_stats(&frags, 8, 8);
    println!(
        "\nCANDIDATE COUNTS — static set per fragment (corpus of {})\n  \
         4x16 : mean {:.2}  p99 {}  max {}\n  \
         8x8  : mean {:.2}  p99 {}  max {}\n  \
         ratio: mean {:.1}x",
        frags.len(),
        m4,
        p99_4,
        max4,
        m8,
        p99_8,
        max8,
        if m4 > 0.0 { m8 / m4 } else { f64::INFINITY }
    );

    println!(
        "\nGated lexical capture and candidate counts describe this corpus only.\n\
         Random-pair hits are unlabelled, and counts do not measure latency.\n\
         Semantic duplicate labels and behaviour-changing negative controls\n\
         are required before making precision or hard-drop safety claims.\n"
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 8 — the threshold sweep, over two corpora.
//
// Sweep the band-plus-distance predicates on edited fragments and unlabelled
// random pairs. Gated capture is lexical, not true-positive recall; random-pair
// hit rates are not false-merge rates. Even a flat hit rate or a large sampled
// minimum cannot establish safety for behaviour-changing near neighbours.
// Both corpora are drawn from the current repository working tree, not an
// independent holdout. No historical percentages are embedded as conclusions.
// ════════════════════════════════════════════════════════════════════

fn collect_sources_ext(root: &Path, out: &mut Vec<PathBuf>, exts: &[&str]) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if p.is_dir() {
            collect_sources_ext(&p, out, exts);
        } else if p
            .extension()
            .and_then(|x| x.to_str())
            .map(|e| exts.contains(&e))
            .unwrap_or(false)
        {
            out.push(p);
        }
    }
}

/// Chunk prose on Markdown headings, falling back to fixed line windows.
///
/// `chunk_source` splits on `fn`/`def`/`class`, which Markdown does not
/// contain, so it returned 4 fragments for 153 documents -- only the code
/// blocks embedded in them. A prose corpus needs a prose chunker or the
/// second corpus silently is not measured.
fn chunk_prose(text: &str, origin: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut starts: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].trim_start().starts_with('#'))
        .collect();
    if starts.len() < 2 {
        // No headings: fixed 40-line windows so long flat documents still
        // contribute rather than being dropped.
        starts = (0..lines.len()).step_by(40).collect();
    }
    if starts.is_empty() {
        return Vec::new();
    }
    starts.push(lines.len());

    let mut out = Vec::new();
    for w in starts.windows(2) {
        let (s, e) = (w[0], w[1].min(w[0] + 200));
        if e.saturating_sub(s) < 3 {
            continue;
        }
        let body = lines[s..e].join("\n");
        if body.split_whitespace().count() < 20 {
            continue;
        }
        out.push((format!("{}:{}", origin, s + 1), body));
    }
    out
}

/// Root recursion already includes docs; visit each Markdown filename once.
fn prose_source_paths(repo_root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_sources_ext(repo_root, &mut files, &["md"]);
    files.sort();
    files.dedup();
    files
}

/// Markdown corpus: prose, short fragments, heavy boilerplate.
fn build_prose_corpus() -> Vec<Fragment> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest.parent().unwrap_or(&manifest).to_path_buf();

    let mut frags = Vec::new();
    for path in prose_source_paths(&repo_root) {
        if frags.len() >= MAX_FRAGMENTS {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let origin = path
            .strip_prefix(&repo_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        for (id, body) in chunk_prose(&text, &origin) {
            if frags.len() >= MAX_FRAGMENTS {
                break;
            }
            let tf = trigram_tf(&body);
            if tf.len() < 5 {
                continue;
            }
            frags.push(Fragment {
                origin: id,
                fingerprint: simhash(&body),
                tf,
                body,
            });
        }
    }
    frags
}

#[test]
fn corpus_origins_are_unique_and_probe_is_excluded() {
    for frags in [build_corpus(), build_prose_corpus()] {
        assert!(!frags.is_empty());
        let mut origins = HashSet::new();
        for frag in &frags {
            assert!(
                origins.insert(&frag.origin),
                "repeated origin: {}",
                frag.origin
            );
            assert!(!frag
                .origin
                .starts_with("entroly-core/src/simhash_probe.rs:"));
        }
    }
}

#[test]
fn prose_filenames_include_nested_docs_once() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("docs/nested")).unwrap();
    let expected = vec![
        root.path().join("README.md"),
        root.path().join("docs/guide.md"),
        root.path().join("docs/nested/details.md"),
    ];
    for path in &expected {
        std::fs::write(path, "# Documentation\n").unwrap();
    }
    std::fs::write(root.path().join("docs/ignored.txt"), "not Markdown").unwrap();
    assert_eq!(prose_source_paths(root.path()), expected);
}

#[test]
fn prose_chunking_splits_headings_and_preserves_text_and_origins() {
    let first = "# First section\nThese words provide enough content for the first documentation section.\nAnother line describes the expected result with useful detail for readers.";
    let second = "## Second section\nThese words provide enough content for the second documentation section.\nAnother line describes a different result with useful detail for readers.";
    let text = format!("{}\n{}\n", first, second);
    assert_eq!(
        chunk_prose(&text, "guide.md"),
        vec![
            ("guide.md:1".into(), first.into()),
            ("guide.md:4".into(), second.into())
        ],
    );
}

#[test]
fn prose_chunking_falls_back_to_fixed_windows_without_headings() {
    let lines: Vec<String> = (0..85)
        .map(|i| {
            format!(
                "Line {} provides several words of flat documentation content.",
                i
            )
        })
        .collect();
    assert_eq!(
        chunk_prose(&lines.join("\n"), "flat.md"),
        vec![
            ("flat.md:1".into(), lines[..40].join("\n")),
            ("flat.md:41".into(), lines[40..80].join("\n")),
            ("flat.md:81".into(), lines[80..].join("\n")),
        ],
    );
    assert!(chunk_prose("", "empty.md").is_empty());
}

/// The sweep itself. Shared so both corpora are measured identically.
fn run_threshold_sweep(frags: &[Fragment], corpus_label: &str) {
    type Perturbation<'a> = (&'a str, Box<dyn Fn(&str) -> String>);
    let classes: Vec<Perturbation> = vec![
        (
            "comment prepended",
            Box::new(|t: &str| format!("// NOTE: reviewed for correctness.\n{}", t)),
        ),
        (
            "one line changed",
            Box::new(|t: &str| {
                let mut lines: Vec<String> = t.lines().map(String::from).collect();
                if lines.is_empty() {
                    return t.to_string();
                }
                let mid = lines.len() / 2;
                lines[mid] = "    let adjusted_value = compute_replacement_here();".into();
                lines.join("\n")
            }),
        ),
        ("identifier renamed", Box::new(rename_identifier)),
        (
            "overlapping window (75%)",
            Box::new(|t: &str| {
                let lines: Vec<&str> = t.lines().collect();
                if lines.len() < 4 {
                    return t.to_string();
                }
                lines[lines.len() / 4..].join("\n")
            }),
        ),
    ];

    let mut per_class: Vec<(&str, Vec<(u32, bool)>)> = Vec::new();
    for (label, gen) in &classes {
        let mut rows = Vec::new();
        for f in frags.iter() {
            let twin = gen(&f.body);
            let tf = trigram_tf(&twin);
            if tf.len() < 5 {
                continue;
            }
            let fp = simhash(&twin);
            rows.push((
                hamming_distance(f.fingerprint, fp),
                shares_band(f.fingerprint, fp),
            ));
        }
        per_class.push((label, rows));
    }

    let corpus_pairs = PairDistances::over(frags);

    let pct = |rows: &[(u32, bool)], d: u32, gated: bool| -> f64 {
        if rows.is_empty() {
            return 0.0;
        }
        let hits = rows
            .iter()
            .filter(|(dist, banded)| *dist <= d && (!gated || *banded))
            .count();
        100.0 * hits as f64 / rows.len() as f64
    };

    println!(
        "\n--- {} : {} fragments, {} unlabelled pairs ({}) ---",
        corpus_label,
        frags.len(),
        corpus_pairs.total(),
        corpus_pairs.basis()
    );
    println!("Edited columns: lexical gated capture; CORPUS GATED: unlabelled pair hit rate.");
    print!("{:>3}", "d");
    for (label, _) in &per_class {
        print!(" {:>14}", &label[..label.len().min(14)]);
    }
    println!(" {:>12}", "CORPUS GATED");

    for d in [1u32, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20] {
        print!("{:>3}", d);
        for (_, rows) in &per_class {
            print!(" {:>13.1}%", pct(rows, d, true));
        }
        println!(" {:>11.4}%", corpus_pairs.rate_at(d, true));
    }

    println!(
        "unlabelled pair distance: min {} (band-gated min {}) p1 {} p5 {} p50 {}",
        corpus_pairs
            .min()
            .map_or("-".into(), |d| d.to_string()),
        corpus_pairs
            .min_banded()
            .map_or("none reachable".into(), |d| d.to_string()),
        corpus_pairs.quantile(1.0),
        corpus_pairs.quantile(5.0),
        corpus_pairs.quantile(50.0)
    );
    println!(
        "These minima are now exact rather than sampled, which removes the\n\
         sampling error and nothing else. An exact minimum over this corpus is\n\
         still an exact minimum over THIS corpus, the pairs carry no semantic\n\
         labels, and a pair at small distance may be a true duplicate that\n\
         dedup exists to collapse. This is not a false-merge rate and not a\n\
         safety bound. See probe_labelled_hard_negatives for labelled pairs."
    );
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_dedup_threshold_sweep() {
    let code = build_corpus();
    assert!(code.len() > 200, "code corpus too small: {}", code.len());
    let prose = build_prose_corpus();

    println!("\n=== ARM 8: dedup threshold sweep (4x16 band gate, production geometry) ===");
    println!("current PROD_THRESHOLD = {}", PROD_THRESHOLD);

    run_threshold_sweep(&code, "CODE corpus (.rs/.py)");

    if prose.len() < 100 {
        println!(
            "\n--- PROSE corpus: only {} fragments, too small to sweep ---",
            prose.len()
        );
    } else {
        run_threshold_sweep(&prose, "PROSE corpus (.md)");
    }

    println!(
        "\nThese are lexical predicate-capture rates on two repository corpora.\n\
         Edited pairs may change behaviour; random pairs have no semantic labels.\n\
         Their distance minima and gated rates do not bound false merges or\n\
         establish a safe production threshold.\n"
    );
}
#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_prose_gated_pair_diagnostics() {
    for (label, frags) in [
        ("PROSE (.md)", build_prose_corpus()),
        ("CODE (.rs/.py)", build_corpus()),
    ] {
        println!("\n######## corpus: {label} ########");
        gated_pair_diagnostics(&frags);
    }
}

/// What does the Hamming branch of `insert` add over the exact
/// normalized-content short circuit?
///
/// `insert` hard-drops on `same_content || hamming <= threshold`. The first
/// disjunct is exact modulo whitespace and is safe; the second is the one Arm
/// 9 showed merging labelled-distinct pairs. Splitting the gated set by byte
/// equality says what the risky disjunct actually buys, because every gated
/// pair that is already byte-identical would still be caught without it.
fn gated_pair_diagnostics(frags: &[Fragment]) {
    assert!(frags.len() > 100);
    // Enumerated, not sampled. This diagnostic hunts for the pairs that pass
    // the gate, which are rare: sampling 50_000 of C(n,2) index pairs with
    // replacement examined under 5% of them, so it missed roughly 95% of the
    // only thing it was looking for and under-reported `close` accordingly.
    // Searching for a rare event is the case where sampling the bulk is
    // maximally wasteful.
    let (mut checked, mut close, mut byte_identical) = (0usize, 0usize, 0usize);
    let mut examples: Vec<(f64, f64, bool, String, String)> = Vec::new();
    for i in 0..frags.len() {
        for j in (i + 1)..frags.len() {
            checked += 1;
            let (x, y) = (&frags[i], &frags[j]);
            if hamming_distance(x.fingerprint, y.fingerprint) <= PROD_THRESHOLD
                && shares_band(x.fingerprint, y.fingerprint)
            {
                close += 1;
                let c = cosine_tf(&x.tf, &y.tf);
                let jc = jaccard_distinct(&x.tf, &y.tf);
                let same_bytes = x.body == y.body;
                if same_bytes {
                    byte_identical += 1;
                }
                if examples.len() < 5 {
                    examples.push((c, jc, same_bytes, x.origin.clone(), y.origin.clone()));
                }
            }
        }
    }
    println!(
        "\n=== unlabelled prose pairs passing band and d <= {} ===",
        PROD_THRESHOLD
    );
    println!(
        "enumerated {checked} pairs, gated {close}, of which byte-identical: {byte_identical}"
    );
    for (c, j, same_bytes, a, b) in &examples {
        println!(
            "  cos {:.3}  jacc {:.3}  same_bytes {}   {}  <->  {}",
            c, j, same_bytes, a, b
        );
    }
    println!(
        "\nByte equality is checked directly on the chunked text. High cosine or\n\
         Jaccard alone proves neither byte equality nor semantic equivalence.\n\
         Non-identical gated pairs require independent labels; this diagnostic\n\
         does not establish that hard-dropping either fragment is safe.\n"
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 9 — labelled hard negatives.
//
// Every arm above measures unlabelled pairs and then says, correctly, that
// unlabelled hits are not false merges. None of them supplies the labels the
// hard-drop question needs. This arm does, by construction rather than by
// judgement after the fact.
//
// Construction: take a real fragment and apply one minimal edit that changes
// what the text MEANS while leaving nearly all of its trigrams intact — flip
// a comparison, change an integer literal, invert a returned boolean — or
// synthesize a templated catalogue entry differing in one subject noun. The
// pair is then a hard negative: distinct by construction, and as similar as
// the generator can make it.
//
// Why this is the right instrument, and what it still cannot do:
//
//   * Unlabelled random pairs sample the BULK of the distance distribution.
//     The threshold lives in its far-left tail. Hard negatives sample that
//     tail directly, which is why a few thousand of them carry more
//     information about a threshold than every pair in the corpus does.
//
//   * The asymmetry is the whole point. ONE distinct pair that shares a band
//     at d <= tau proves tau unsafe — an existence proof, needing no
//     sampling guarantee whatsoever. Finding none proves nothing, because
//     these generators are not exhaustive over the ways two fragments can
//     differ by one semantic token.
//
//     So: this arm can FALSIFY a threshold. It cannot CERTIFY one. Reading a
//     clean run as evidence of safety is exactly the inference the arms above
//     already warn against.
//
//   * "Distinct by construction" is a property of the generator, not a proof
//     about the strings. Each class therefore carries its rationale in code
//     so the claim can be audited rather than trusted.
//
//   * A no-op edit would make this arm silently vacuous — measuring identical
//     text and reporting d = 0 as though it were a false merge. That is the
//     failure that made an earlier version of the templated regression test
//     worthless, so no-ops are counted, excluded, and gated by a test that is
//     not #[ignore]d.
// ════════════════════════════════════════════════════════════════════

/// Replace the first occurrence of `from` with `to`; `None` when absent.
fn replace_first(text: &str, from: &str, to: &str) -> Option<String> {
    let at = text.find(from)?;
    let mut out = String::with_capacity(text.len() + to.len());
    out.push_str(&text[..at]);
    out.push_str(to);
    out.push_str(&text[at + from.len()..]);
    Some(out)
}

/// Flip the first comparison operator.
///
/// The spaces are load-bearing: without them `<` matches inside a generic
/// parameter list such as `Vec<T>`, which would rewrite a type rather than a
/// predicate and produce a pair that is not distinct in the intended way.
fn flip_comparison(text: &str) -> Option<String> {
    replace_first(text, " <= ", " >= ")
        .or_else(|| replace_first(text, " >= ", " <= "))
        .or_else(|| replace_first(text, " == ", " != "))
        .or_else(|| replace_first(text, " < ", " > "))
        .or_else(|| replace_first(text, " > ", " < "))
}

/// Invert the first returned boolean — the opposite verdict from the same
/// function.
fn invert_boolean(text: &str) -> Option<String> {
    replace_first(text, "return true", "return false")
        .or_else(|| replace_first(text, "return false", "return true"))
        .or_else(|| replace_first(text, "return True", "return False"))
        .or_else(|| replace_first(text, "return False", "return True"))
}

/// Change the first standalone integer literal.
///
/// Digits are mapped 7 <-> 3 and everything else to 7, so the replacement has
/// the same number of characters as the original. Equal width keeps the
/// surrounding trigrams aligned, which makes this a *hard* negative rather
/// than one that is easy to separate because the text got longer.
fn change_integer_literal(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let boundary = |c: u8| !(c.is_ascii_alphanumeric() || c == b'_' || c == b'.');
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let before_ok = start == 0 || boundary(b[start - 1]);
        let after_ok = i >= b.len() || boundary(b[i]);
        if before_ok && after_ok {
            let orig = &text[start..i];
            let swapped: String = orig
                .chars()
                .map(|c| if c == '7' { '3' } else { '7' })
                .collect();
            if swapped != orig {
                let mut out = String::with_capacity(text.len());
                out.push_str(&text[..start]);
                out.push_str(&swapped);
                out.push_str(&text[i..]);
                return Some(out);
            }
        }
    }
    None
}

/// Templated catalogue entries differing in exactly one subject noun.
///
/// This is the shape the earlier corpora contained no example of: a long
/// identical template around a single distinguishing word. Distinctness is not
/// subtle — a prompt about archiving a user is not a prompt about archiving a
/// workspace, and dedup is a hard drop, so collapsing them loses a string the
/// product still needs to render.
fn templated_negative_pairs() -> Vec<(String, String)> {
    let subjects = [
        "user",
        "group",
        "project",
        "workspace",
        "team",
        "webhook",
        "channel",
        "invoice",
    ];
    let verbs = ["delete", "archive", "duplicate", "restore", "export"];
    let mut out = Vec::new();
    for verb in verbs {
        let line = |s: &str| {
            format!(
                "\"confirm.{verb}\": \"Are you sure you want to {verb} this {s}? \
                 This action cannot be undone and any members will immediately \
                 lose access to its contents.\""
            )
        };
        for pair in subjects.windows(2) {
            out.push((line(pair[0]), line(pair[1])));
        }
    }
    out
}

/// Distance distribution for one labelled class, plus the bookkeeping that
/// keeps the class from being vacuous.
struct NegativeClass {
    label: &'static str,
    rationale: &'static str,
    all: [u64; 65],
    banded: [u64; 65],
    /// Edits that returned text identical to the input. Excluded from the
    /// histograms: an unchanged pair is not a negative at all.
    noop: usize,
}

impl NegativeClass {
    fn new(label: &'static str, rationale: &'static str) -> Self {
        Self {
            label,
            rationale,
            all: [0; 65],
            banded: [0; 65],
            noop: 0,
        }
    }

    fn record(&mut self, a: u64, b: u64) {
        let d = (hamming_distance(a, b) as usize).min(64);
        self.all[d] += 1;
        if shares_band(a, b) {
            self.banded[d] += 1;
        }
    }

    fn pairs(&self) -> u64 {
        self.all.iter().sum()
    }

    fn min(&self) -> Option<u32> {
        self.all.iter().position(|&c| c > 0).map(|d| d as u32)
    }

    /// Smallest distance among pairs `DedupIndex` would actually consider.
    fn min_gated(&self) -> Option<u32> {
        self.banded.iter().position(|&c| c > 0).map(|d| d as u32)
    }

    /// Pairs this class proves would be merged at `tau`: band-shared AND
    /// within the threshold, which is exactly `DedupIndex::insert`.
    fn merged_at(&self, tau: u32) -> u64 {
        self.banded[..=(tau as usize).min(64)].iter().sum()
    }

    /// Distinct text that collided to the same fingerprint. Unreachable by
    /// raising or lowering a threshold, so it is reported separately.
    fn fingerprint_collisions(&self) -> u64 {
        self.all[0]
    }
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_labelled_hard_negatives() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "code corpus too small: {}", frags.len());

    type Gen = fn(&str) -> Option<String>;
    let edits: [(&'static str, &'static str, Gen); 3] = [
        (
            "comparison flipped",
            "`a < b` and `a > b` are different predicates",
            flip_comparison,
        ),
        (
            "integer literal changed",
            "a different constant is a different program",
            change_integer_literal,
        ),
        (
            "returned boolean inverted",
            "the opposite verdict from the same function",
            invert_boolean,
        ),
    ];

    let mut classes: Vec<NegativeClass> = Vec::new();

    for (label, rationale, gen) in edits {
        let mut class = NegativeClass::new(label, rationale);
        for f in frags.iter() {
            let Some(edited) = gen(&f.body) else {
                continue;
            };
            if edited == f.body {
                class.noop += 1;
                continue;
            }
            let tf = trigram_tf(&edited);
            if tf.len() < 5 {
                continue;
            }
            class.record(f.fingerprint, simhash(&edited));
        }
        classes.push(class);
    }

    let mut templated = NegativeClass::new(
        "templated subject swapped",
        "a prompt about a user is not a prompt about a workspace",
    );
    for (a, b) in templated_negative_pairs() {
        if a == b {
            templated.noop += 1;
            continue;
        }
        templated.record(simhash(&a), simhash(&b));
    }
    classes.push(templated);

    const TAUS: [u32; 4] = [3, 5, 8, 12];

    println!("\n=== ARM 9: labelled hard negatives (production 4x16 band gate) ===");
    println!("current PROD_THRESHOLD = {PROD_THRESHOLD}");
    println!(
        "\nEach pair below is semantically distinct BY CONSTRUCTION, so a merge\n\
         is a true false merge rather than an unlabelled hit. MERGED@tau counts\n\
         pairs that share a band and sit within tau — the exact DedupIndex\n\
         predicate.\n"
    );

    print!("{:<28} {:>6} {:>5} {:>7} {:>6}", "class", "pairs", "min", "minGate", "fpCol");
    for tau in TAUS {
        print!(" {:>9}", format!("MERGED@{tau}"));
    }
    println!();

    let mut falsified: Vec<(u32, &'static str, u64)> = Vec::new();
    for c in &classes {
        print!(
            "{:<28} {:>6} {:>5} {:>7} {:>6}",
            c.label,
            c.pairs(),
            c.min().map_or("-".to_string(), |d| d.to_string()),
            c.min_gated().map_or("-".to_string(), |d| d.to_string()),
            c.fingerprint_collisions()
        );
        for tau in TAUS {
            let m = c.merged_at(tau);
            print!(" {:>9}", m);
            if m > 0 {
                falsified.push((tau, c.label, m));
            }
        }
        println!();
    }

    println!("\nrationale per class (the basis for calling these pairs distinct):");
    for c in &classes {
        println!(
            "  {:<28} {}   [{} no-op edits excluded]",
            c.label, c.rationale, c.noop
        );
    }

    println!("\nFALSIFICATION SUMMARY");
    let mut any = false;
    for tau in TAUS {
        let total: u64 = falsified
            .iter()
            .filter(|(t, _, _)| *t == tau)
            .map(|(_, _, m)| *m)
            .sum();
        if total > 0 {
            any = true;
            let worst = falsified
                .iter()
                .filter(|(t, _, _)| *t == tau)
                .max_by_key(|(_, _, m)| *m)
                .map(|(_, label, _)| *label)
                .unwrap_or("?");
            println!(
                "  tau = {tau:>2}: UNSAFE — {total} distinct pair(s) merge, worst class \"{worst}\""
            );
        } else {
            println!("  tau = {tau:>2}: not falsified by these generators");
        }
    }
    if !any {
        println!("  (no class merged at any tau tried)");
    }

    println!(
        "\nRead this one way only. A tau marked UNSAFE is falsified: a pair of\n\
         distinct strings demonstrably collapses, and dedup is a hard drop, so\n\
         one of them is discarded at ingest. A tau \"not falsified\" is NOT\n\
         shown safe — these four generators cover a handful of the ways two\n\
         fragments can differ by one semantic token, and absence of a\n\
         counterexample from them is not evidence of absence. Certifying a\n\
         threshold needs labelled near-duplicates from an independent holdout,\n\
         which this repository does not contain.\n"
    );
}

/// The arm is worthless if its generators return the input unchanged: it would
/// then compare identical text, report distance 0, and present that as a false
/// merge. An earlier version of the templated regression test failed exactly
/// this way — it passed at every threshold because the strings it picked did
/// not actually collide, so it asserted nothing.
///
/// Not #[ignore]d: this runs in CI so the probe cannot rot into vacuity.
#[test]
fn every_hard_negative_generator_changes_the_text() {
    let code = "fn pick(limit: usize) -> bool {\n    if limit < 42 {\n        return true;\n    }\n    return false;\n}";

    let flipped = flip_comparison(code).expect("comparison present");
    assert_ne!(flipped, code, "flip_comparison returned the input unchanged");
    assert!(
        flipped.contains(" > 42"),
        "expected the predicate to invert, got: {flipped}"
    );

    let renumbered = change_integer_literal(code).expect("integer literal present");
    assert_ne!(
        renumbered, code,
        "change_integer_literal returned the input unchanged"
    );
    assert!(
        renumbered.contains("77"),
        "expected 42 -> 77 (equal width), got: {renumbered}"
    );
    assert_eq!(
        renumbered.len(),
        code.len(),
        "replacement must preserve width so trigrams stay aligned"
    );

    let inverted = invert_boolean(code).expect("boolean return present");
    assert_ne!(inverted, code, "invert_boolean returned the input unchanged");

    // A literal inside an identifier is not a standalone literal.
    assert!(
        change_integer_literal("let sha256_digest = hash();").is_none(),
        "digits inside an identifier must not be treated as a literal"
    );
    // A generic parameter list is not a comparison.
    assert!(
        flip_comparison("let v: Vec<String> = Vec::new();").is_none(),
        "generic brackets must not be read as a comparison operator"
    );

    for (a, b) in templated_negative_pairs() {
        assert_ne!(a, b, "templated pair generator produced identical strings");
    }
}

/// `PairDistances` replaced a 50_000-draw sample, so the thing worth pinning
/// is that it agrees with an independent O(n^2) recomputation — otherwise the
/// arms would report an exact-looking number that is simply wrong.
#[test]
fn pair_distances_are_exact_over_every_pair() {
    let fingerprints: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001, // d = 1 from the first
        0xFFFF_FFFF_FFFF_FFFF,
        0x0F0F_0F0F_0F0F_0F0F,
        0x0000_0000_0000_0003,
    ];
    let frags: Vec<Fragment> = fingerprints
        .iter()
        .enumerate()
        .map(|(i, &fp)| Fragment {
            origin: format!("synthetic:{i}"),
            body: String::new(),
            fingerprint: fp,
            tf: Vec::new(),
        })
        .collect();

    let dist = PairDistances::over(&frags);
    let n = frags.len() as u64;

    assert!(dist.exhaustive, "a 5-fragment corpus must be enumerated");
    assert_eq!(
        dist.total(),
        n * (n - 1) / 2,
        "every unordered pair must be counted exactly once"
    );
    assert_eq!(
        dist.min(),
        brute_force_min_distance(&fingerprints),
        "histogram minimum must match the independent recomputation"
    );
    assert_eq!(dist.min(), Some(1), "0x0 and 0x1 differ in one bit");

    // The band-gated floor cannot sit below the ungated floor.
    if let (Some(lo), Some(gated)) = (dist.min(), dist.min_banded()) {
        assert!(gated >= lo, "gated minimum {gated} below ungated {lo}");
    }

    // rate_at is monotone in d and reaches 100% of pairs ungated.
    let mut prev = 0.0;
    for d in 0..=64u32 {
        let r = dist.rate_at(d, false);
        assert!(r >= prev - f64::EPSILON, "rate_at not monotone at d={d}");
        prev = r;
    }
    assert!(
        (dist.rate_at(64, false) - 100.0).abs() < 1e-9,
        "every pair is within distance 64"
    );
}

/// The replica above must keep matching `dedup::simhash`.
///
/// It did not, once: PR #620 taught production to retain standalone operator
/// tokens and left this copy dropping them, so the probe modelled a function
/// that no longer existed while still printing confident numbers about it.
///
/// Direction matters. Equal token streams produce equal trigrams and therefore
/// *must* produce equal fingerprints; the converse is not true, because two
/// different token streams may still collide in 64 bits. So the implication
/// checked here is the sound one, and it is exactly the one that fails when
/// the tokenizers disagree about operators.
#[test]
fn tokenization_replica_still_matches_production() {
    let cases = [
        ("if limit < 42 { go(); }", "if limit > 42 { go(); }"),
        ("a == b", "a != b"),
        ("value = compute(x);", "value = compute(x)"),
        ("alpha beta gamma", "alpha beta gamma"),
    ];
    for (left, right) in cases {
        if trigram_tf(left) == trigram_tf(right) {
            assert_eq!(
                simhash(left),
                simhash(right),
                "replica says {left:?} and {right:?} have identical features, but \
                 production disagrees -- tokenize_like_production has drifted"
            );
        }
    }
    // And the specific drift that happened: operators must survive tokenization.
    assert!(
        tokenize_like_production("if a < b").contains(&"<"),
        "a standalone operator token must not be dropped"
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 10 — does fingerprint WIDTH fix what the threshold cannot?
//
// Arm 9 falsified every tau in [3, 12]: a one-token semantic change in a long
// fragment moves too few of 64 bits to be told apart from a true duplicate at
// any threshold. That is a resolution limit, so the only remaining knob is
// width.
//
// Why this needs its own fingerprint rather than `simhash_wide::fingerprint`:
// that function is width-parameterised but tokenizes into *unigrams* with no
// punctuation handling. A one-token edit moves 1 of n unigram features but 3
// of n trigram features, and unigrams are order-blind entirely. Measuring
// width on unigrams would confound a width change with a feature-space
// downgrade, so this mirrors production's trigram features at arbitrary width.
// The hash family differs from production's MD5, which is fine: only the
// relative behaviour ACROSS widths is read here, never the absolute rate.
//
// Both directions are measured, because removing false merges is trivial if
// you are allowed to push everything apart — that also destroys the duplicate
// capture dedup exists for. A width only wins if false merges fall WHILE
// near-duplicate capture holds.
//
// Thresholds are scaled to hold d/B constant: tau_B = tau_64 * B / 64. Band
// gating is deliberately omitted; it only ever removes merges, so an ungated
// count is an upper bound on what the gated production path would do, and the
// band geometry for a wider fingerprint is a separate design choice.
// ════════════════════════════════════════════════════════════════════

fn hash_feature_limb(feature: &str, limb: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (limb as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for byte in feature.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Production's trigram features at an arbitrary width.
fn wide_trigram_fingerprint(text: &str, bits: usize) -> Vec<u64> {
    let limbs = bits.div_ceil(64);
    let words = tokenize_like_production(text);
    let features: Vec<String> = if words.len() >= 3 {
        words
            .windows(3)
            .map(|w| format!("{} {} {}", w[0], w[1], w[2]).to_lowercase())
            .collect()
    } else {
        words.iter().map(|w| w.to_lowercase()).collect()
    };

    let mut acc = vec![0i32; bits];
    for feature in &features {
        for limb in 0..limbs {
            let h = hash_feature_limb(feature, limb);
            for bit in 0..64 {
                let index = limb * 64 + bit;
                if index >= bits {
                    break;
                }
                acc[index] += if (h >> bit) & 1 == 1 { 1 } else { -1 };
            }
        }
    }

    let mut out = vec![0u64; limbs];
    for (index, sum) in acc.into_iter().enumerate() {
        if sum > 0 {
            out[index / 64] |= 1u64 << (index % 64);
        }
    }
    out
}

fn wide_hamming(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_fingerprint_width_sweep() {
    let frags = build_corpus();
    assert!(frags.len() > 200, "code corpus too small: {}", frags.len());

    const WIDTHS: [usize; 3] = [64, 256, 1024];
    const BASE_TAUS: [u32; 3] = [3, 5, 8];

    type Gen = fn(&str) -> Option<String>;
    // Distinct by construction — a merge here is a false merge.
    let negatives: [(&str, Gen); 3] = [
        ("comparison flipped", flip_comparison),
        ("integer literal changed", change_integer_literal),
        ("returned boolean inverted", invert_boolean),
    ];
    // Near-duplicates — a merge here is the behaviour dedup exists for.
    type Perturb = fn(&str) -> String;
    let positives: [(&str, Perturb); 2] = [
        ("comment prepended", |t: &str| {
            format!("// NOTE: reviewed for correctness.\n{t}")
        }),
        ("one line changed", |t: &str| {
            let mut lines: Vec<String> = t.lines().map(String::from).collect();
            if lines.is_empty() {
                return t.to_string();
            }
            let mid = lines.len() / 2;
            lines[mid] = "    let adjusted_value = compute_replacement_here();".into();
            lines.join("\n")
        }),
    ];

    println!("\n=== ARM 10: fingerprint width sweep (trigram features, ungated) ===");
    println!(
        "tau scaled to hold d/B constant: tau_B = tau_64 * B/64.\n\
         FALSE = labelled-distinct pairs merged (want down).\n\
         CAPTURE = near-duplicate pairs merged (want held).\n\
         Ungated: band gating only removes merges, so FALSE is an upper bound.\n"
    );

    for bits in WIDTHS {
        // Distance distributions, computed once per width.
        let mut neg: Vec<(&str, Vec<u32>)> = Vec::new();
        for (label, gen) in negatives {
            let mut rows = Vec::new();
            for f in frags.iter() {
                let Some(edited) = gen(&f.body) else { continue };
                if edited == f.body {
                    continue;
                }
                rows.push(wide_hamming(
                    &wide_trigram_fingerprint(&f.body, bits),
                    &wide_trigram_fingerprint(&edited, bits),
                ));
            }
            neg.push((label, rows));
        }
        let mut pos: Vec<(&str, Vec<u32>)> = Vec::new();
        for (label, gen) in positives {
            let mut rows = Vec::new();
            for f in frags.iter() {
                let twin = gen(&f.body);
                rows.push(wide_hamming(
                    &wide_trigram_fingerprint(&f.body, bits),
                    &wide_trigram_fingerprint(&twin, bits),
                ));
            }
            pos.push((label, rows));
        }

        let pct = |rows: &[u32], tau: u32| -> f64 {
            if rows.is_empty() {
                return 0.0;
            }
            100.0 * rows.iter().filter(|&&d| d <= tau).count() as f64 / rows.len() as f64
        };
        let count = |rows: &[u32], tau: u32| rows.iter().filter(|&&d| d <= tau).count();
        let min = |rows: &[u32]| rows.iter().copied().min().unwrap_or(0);

        println!(
            "--- {bits} bits (noise floor {:.4}) ---",
            std::f64::consts::PI / (2.0 * (bits as f64).sqrt())
        );
        print!("{:<28} {:>6}", "class", "min");
        for t in BASE_TAUS {
            print!(" {:>12}", format!("tau64={t}"));
        }
        println!();

        for (label, rows) in &neg {
            print!("{:<28} {:>6}", format!("FALSE {label}"), min(rows));
            for t in BASE_TAUS {
                let tau = t * bits as u32 / 64;
                print!(" {:>12}", count(rows, tau));
            }
            println!();
        }
        for (label, rows) in &pos {
            print!("{:<28} {:>6}", format!("CAPTURE {label}"), min(rows));
            for t in BASE_TAUS {
                let tau = t * bits as u32 / 64;
                print!(" {:>11.1}%", pct(rows, tau));
            }
            println!();
        }
        println!();
    }

    println!(
        "Width wins only if FALSE counts fall while CAPTURE holds. If FALSE\n\
         falls and CAPTURE falls with it, the fingerprint merely pushed every\n\
         pair apart, which any width-independent threshold cut would also do\n\
         and which removes the duplicate detection dedup exists to provide.\n\
         Absolute rates are not comparable to Arm 9: the hash family differs\n\
         and no band gate is applied here. Read the trend across widths only.\n"
    );
}
