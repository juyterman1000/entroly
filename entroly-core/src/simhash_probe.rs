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
//!   1. `DedupIndex` (threshold 3) makes a HARD DROP decision, gated behind an
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
//! Run:
//!   cargo test --manifest-path entroly-core/Cargo.toml --lib simhash_probe \
//!       -- --ignored --nocapture --test-threads=1

use crate::dedup::{hamming_distance, simhash};
use std::path::{Path, PathBuf};

/// Production `DedupIndex` banding: 4 bands x 16 bits (see `dedup.rs`).
const NUM_BANDS: usize = 4;
const BITS_PER_BAND: usize = 16;

/// Production dedup threshold (`lib.rs`: `hamming_threshold=8`).
///
/// Was 3. Arm 8 swept d over this corpus: unrelated pairs have min 14 and
/// p1 23, so 3 sat far below the empty band between true near-duplicates
/// (d 5-13) and unrelated content, costing recall for nothing. At 8 the
/// identifier-rename class merges 12.0% against 1.2%, with zero false
/// merges in 49,962 unrelated pairs.
const PROD_THRESHOLD: u32 = 8;

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

/// Word-trigram term frequencies, tokenized exactly as `dedup::simhash` does:
/// split on whitespace, trim non-alphanumeric (keeping `_`), drop empties,
/// join adjacent triples, lowercase the joined feature.
fn trigram_tf(text: &str) -> Vec<(u64, u32)> {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'))
        .filter(|w| !w.is_empty())
        .collect();

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
/// estimates. Charikar (2002): P(bit differs) = theta/pi.
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

/// Jaccard over the DISTINCT trigrams. Used only to label "is this pair really
/// a near-duplicate", which is a set question, not a TF question.
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
// Arm 2 — controlled near-duplicates (dedup recall).
// ════════════════════════════════════════════════════════════════════

/// Rename the most frequent identifier-like token, simulating a refactor.
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

    /// (label, perturbation) — a named way of producing a near-duplicate.
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

    println!("\n=== ARM 2: near-duplicate recall ===");
    println!("base fragments: {}\n", frags.len());
    println!(
        "{:<26} {:>6} {:>8} {:>8} {:>7} {:>9} {:>9}",
        "perturbation", "pairs", "jaccard", "cos", "mean d", "banded%", "MERGED%"
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
        "\n'MERGED%' is what DedupIndex::insert actually collapses at d<={} (band gate AND\n\
         threshold), excluding the exact-content-hash short circuit. MERGED% <= banded%\n\
         always, and banded% does not depend on the threshold — so raising the threshold\n\
         cannot lift recall past the band-gate ceiling.\n",
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
// Arm 2 measured the binding constraint: banded% falls to ~17% once an edit
// drops TF-cosine to ~0.68, and it does not depend on the merge threshold,
// so no amount of tuning lifts it. Arm 1 measured the other half: the
// angular estimator already sits at the orthogonality noise floor (sd 0.196
// at 64 bits), so better arithmetic cannot help either. Only the hash family
// or the band geometry can.
//
// This arm holds the budget at 64 bits and varies both, over the same corpus
// and the same perturbations Arm 2 uses:
//
//   A. SimHash 4x16  — production, the control.
//   B. SimHash 8x8   — same bits, same family, finer bands. Costs no storage
//                      and no migration, so it is the bar a family change
//                      has to clear.
//   C. b=2 MinHash, k=32, 8 bands x 4 rows.
//   D. b=1 MinHash, k=64, 16 bands x 4 rows.
//
// Recall alone would be a misleading score: a scheme that bands everything
// reaches 100% and destroys the index, because DedupIndex collapsing two
// distinct fragments silently discards one. Every scheme is therefore also
// measured on random unrelated pairs, and the two numbers are reported
// together.
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
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
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

    // Precision control. This is the number that decides whether higher
    // recall is worth anything: a banded unrelated pair is a candidate the
    // distance threshold must then reject, and if it does not, DedupIndex
    // discards a distinct fragment.
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
        "UNRELATED (false cand.)",
        n,
        sj / q,
        sc / q,
        100.0 * a as f64 / q,
        100.0 * b as f64 / q,
        100.0 * c as f64 / q,
        100.0 * d as f64 / q
    );

    println!(
        "\nRead the last row first. Recall above it is worth only what precision\n\
         here allows: a false candidate that also passes the distance threshold\n\
         makes DedupIndex discard a distinct fragment, which is strictly worse\n\
         than missing a duplicate. 8x8 costs no storage and no migration, so a\n\
         MinHash family change is justified only if it beats that column on\n\
         recall WITHOUT losing on this row. jacc and cos are printed together\n\
         because MinHash estimates Jaccard over distinct trigrams while SimHash\n\
         estimates TF-cosine; a family change silently redefines what similar\n\
         means for term-frequency-weighted code.\n"
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 7 — the two numbers that decide whether 8x8 can ship.
//
// Arm 6 showed 8x8 lifts worst-case candidate recall from 17.9% to 76.1%
// on the same 64 bits, at a false-candidate rate of 3.22% against 0.01%.
// Candidate recall is not the shipped behaviour, and a 300x rise in false
// candidates is not free. Two things still have to hold:
//
//   1. MERGED% — banded AND d <= PROD_THRESHOLD, which is what DedupIndex
//      actually collapses. Recall that the threshold rejects is worthless,
//      and a false candidate the threshold ACCEPTS is a distinct fragment
//      silently discarded. Both directions are measured here.
//
//   2. Candidate-set size — the mean number of fragments sharing at least
//      one band with a given fragment, over the real corpus. This is the
//      per-lookup work the band gate exists to bound, so it is the latency
//      cost of the finer geometry.
// ════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, HashSet};

/// Band keys at an arbitrary geometry, mirroring `extract_bands`.
fn bands_at(fp: u64, bands: usize, bits: usize) -> Vec<u64> {
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    (0..bands)
        .map(|i| (fp >> ((i * bits) as u32)) & mask)
        .collect()
}

/// Mean and p99 of the candidate set a band index would return per fragment.
///
/// Counts distinct other fragments sharing at least one band, which is
/// exactly the set `DedupIndex` would hand to the Hamming check.
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
fn probe_band_geometry_shipping_gates() {
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

    println!("\n=== ARM 7: shipping gates for 8x8 ===");
    println!("corpus: {} fragments, threshold d <= {}\n", frags.len(), PROD_THRESHOLD);

    // ── Gate 1: what actually merges ──
    println!("GATE 1 — MERGED% (banded AND d <= {})", PROD_THRESHOLD);
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

    // ── Gate 1b: the harm direction ──
    //
    // A false MERGE, not a false candidate. This is the number that would
    // make 8x8 unsafe: an unrelated pair that bands AND passes the Hamming
    // threshold causes a distinct fragment to be discarded.
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let (mut n, mut f4, mut f8, mut c4, mut c8) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for _ in 0..50_000usize {
        let i = rng.below(frags.len());
        let j = rng.below(frags.len());
        if i == j {
            continue;
        }
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
    let q = n as f64;
    println!(
        "\nGATE 1b — unrelated pairs ({} sampled)\n  \
         false candidates : 4x16 {:.2}%   8x8 {:.2}%\n  \
         FALSE MERGES     : 4x16 {} ({:.4}%)   8x8 {} ({:.4}%)",
        n,
        100.0 * c4 as f64 / q,
        100.0 * c8 as f64 / q,
        f4,
        100.0 * f4 as f64 / q,
        f8,
        100.0 * f8 as f64 / q
    );

    // ── Gate 2: per-lookup work ──
    let (m4, p99_4, max4) = candidate_set_stats(&frags, NUM_BANDS, BITS_PER_BAND);
    let (m8, p99_8, max8) = candidate_set_stats(&frags, 8, 8);
    println!(
        "\nGATE 2 — candidate set per fragment (corpus of {})\n  \
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
        "\n8x8 ships only if GATE 1 shows MERGED% strictly improved, GATE 1b shows\n\
         false merges not materially worse, and GATE 2 shows the candidate set\n\
         still bounded — the band gate exists to keep the Hamming check off the\n\
         whole corpus, and a geometry that returns everything has no value even\n\
         at perfect recall.\n"
    );
}

// ════════════════════════════════════════════════════════════════════
// Arm 8 — the threshold sweep, over two corpora.
//
// Arm 7 established that the band gate is not the binding constraint:
// 8x8 lifted candidate recall 17.9% -> 76.1% and moved MERGED% by zero,
// because production merged at d <= 3 while an identifier rename sits at
// d ~ 12.6. Every extra candidate was rejected by the Hamming check.
//
// So the question is the threshold, not the geometry. This sweeps d and
// reports, at each value:
//
//   * MERGED% per perturbation class — the true positives gained.
//   * FALSE MERGE% on unrelated pairs — the cost, and the thing that
//     actually breaks the index, since a false merge silently discards a
//     distinct fragment.
//
// A threshold is only safe below the left tail of the unrelated-pair
// distance distribution, so that distribution is printed too.
//
// One corpus cannot establish this. The code corpus is `.rs`/`.py`; the
// prose corpus is Markdown, which has a smaller trigram alphabet, shorter
// fragments, and far more shared boilerplate — exactly where the gap
// between near-duplicates and unrelated content should be narrowest.
// Both are swept by the same function so the numbers are comparable.
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
        let body = lines[s..e].join("
");
        if body.split_whitespace().count() < 20 {
            continue;
        }
        out.push((format!("{}:{}", origin, s + 1), body));
    }
    out
}

/// Markdown corpus: prose, short fragments, heavy boilerplate.
fn build_prose_corpus() -> Vec<Fragment> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest.parent().unwrap_or(&manifest).to_path_buf();

    let mut files = Vec::new();
    collect_sources_ext(&repo_root.join("docs"), &mut files, &["md"]);
    collect_sources_ext(&repo_root, &mut files, &["md"]);

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

    let mut rng = Rng(0xD15C_0FFE_E5EE_D001);
    let mut unrelated: Vec<(u32, bool)> = Vec::new();
    for _ in 0..50_000usize {
        let i = rng.below(frags.len());
        let j = rng.below(frags.len());
        if i == j {
            continue;
        }
        let (x, y) = (&frags[i], &frags[j]);
        unrelated.push((
            hamming_distance(x.fingerprint, y.fingerprint),
            shares_band(x.fingerprint, y.fingerprint),
        ));
    }

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
        "\n--- {} : {} fragments, {} unrelated pairs ---",
        corpus_label,
        frags.len(),
        unrelated.len()
    );
    print!("{:>3}", "d");
    for (label, _) in &per_class {
        print!(" {:>14}", &label[..label.len().min(14)]);
    }
    println!(" {:>12}", "FALSE MERGE");

    for d in [1u32, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20] {
        print!("{:>3}", d);
        for (_, rows) in &per_class {
            print!(" {:>13.1}%", pct(rows, d, true));
        }
        println!(" {:>11.4}%", pct(&unrelated, d, true));
    }

    let mut dists: Vec<u32> = unrelated.iter().map(|(d, _)| *d).collect();
    dists.sort_unstable();
    let q = |p: usize| dists[(dists.len() * p / 100).min(dists.len().saturating_sub(1))];
    println!(
        "unrelated-pair distance: min {} p1 {} p5 {} p50 {}   <- safe thresholds live below min",
        dists.first().copied().unwrap_or(0),
        q(1),
        q(5),
        q(50)
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
        "\nA threshold generalises only if the unrelated-pair minimum stays above\n\
         it on BOTH corpora. Prose has a smaller trigram alphabet and more shared\n\
         boilerplate than code, so its unrelated pairs sit closer together; that\n\
         is the corpus that decides the ceiling, not the code one.\n"
    );
}
#[test]
#[ignore = "measurement probe; run explicitly with --ignored --nocapture"]
fn probe_prose_zero_distance_pairs_are_real_duplicates() {
    let frags = build_prose_corpus();
    assert!(frags.len() > 100);
    let mut rng = Rng(0xD15C_0FFE_E5EE_D001);
    let (mut checked, mut close, mut truly_similar) = (0usize, 0usize, 0usize);
    let mut examples: Vec<(f64, f64, String, String)> = Vec::new();
    for _ in 0..50_000usize {
        let i = rng.below(frags.len());
        let j = rng.below(frags.len());
        if i == j {
            continue;
        }
        checked += 1;
        let (x, y) = (&frags[i], &frags[j]);
        if hamming_distance(x.fingerprint, y.fingerprint) <= PROD_THRESHOLD
            && shares_band(x.fingerprint, y.fingerprint)
        {
            close += 1;
            let c = cosine_tf(&x.tf, &y.tf);
            let jc = jaccard_distinct(&x.tf, &y.tf);
            if c > 0.8 {
                truly_similar += 1;
            }
            if examples.len() < 5 {
                examples.push((c, jc, x.origin.clone(), y.origin.clone()));
            }
        }
    }
    println!("\n=== prose pairs that WOULD merge at d <= {} ===", PROD_THRESHOLD);
    println!("sampled {checked}, would-merge {close}, of which cosine > 0.8: {truly_similar}");
    for (c, j, a, b) in &examples {
        println!("  cos {:.3}  jacc {:.3}   {}  <->  {}", c, j, a, b);
    }
    println!(
        "\nIf cosine is high these are genuine duplicate prose (shared boilerplate\n\
         across documents), so collapsing them is correct behaviour and the rate\n\
         is not a false-merge rate. If cosine is low they are real errors.\n"
    );
}
