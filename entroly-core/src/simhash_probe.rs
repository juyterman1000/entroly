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

/// Production dedup threshold (`lib.rs`: `hamming_threshold=3`).
const PROD_THRESHOLD: u32 = 3;

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
            let new_div = (1.0
                - chosen
                    .iter()
                    .map(|&c| crate::dedup::simhash_cosine(c, h))
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
