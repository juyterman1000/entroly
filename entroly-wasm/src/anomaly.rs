//! Entropy Anomaly Scanner
//!
//! Flips the entropy lens: instead of "what context helps the LLM?",
//! asks "what code surprises the model?" Code that is statistically
//! unusual relative to its neighbors correlates with:
//!
//!   - Copy-paste errors (pattern breaks)
//!   - Security anti-patterns (unusual auth flows)
//!   - Dead logic (contradicts surrounding invariants)
//!   - Misunderstood APIs (deviates from every other callsite)
//!
//! Algorithm: Robust Z-score using MAD (Median Absolute Deviation)
//! on per-directory entropy groups. MAD is resistant to up to 50%
//! contamination — a single outlier won't mask other anomalies.
//!
//! Mathematical foundation:
//!   M = median(entropy_scores)
//!   MAD = median(|eᵢ - M|)
//!   zᵢ = 0.6745 × (eᵢ - M) / MAD
//!
//! The constant 0.6745 normalizes MAD to be consistent with σ
//! for normally distributed data: MAD ≈ 0.6745σ.
//!
//! Nobody does this. Static analyzers use rules. Linters use patterns.
//! This uses information theory against the codebase itself.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::fragment::ContextFragment;
use crate::entropy::boilerplate_ratio;

/// Minimum fragments in a directory group to detect anomalies.
/// Below this, we don't have enough statistical mass.
const MIN_GROUP_SIZE: usize = 5;

/// Z-score threshold for flagging anomalies. 2.5 corresponds to
/// roughly p < 0.006 on each tail for normal data.
const Z_THRESHOLD: f64 = 2.5;

/// Consistency constant: for normal data, MAD ≈ 0.6745 × σ.
const MAD_CONSISTENCY: f64 = 0.6745;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Entropy significantly higher than neighbors — complex, unusual,
    /// or foreign code pasted into a simple module.
    Spike,
    /// Entropy significantly lower than neighbors — dead stub,
    /// placeholder, or accidentally committed empty function.
    Drop,
}

impl AnomalyType {
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            AnomalyType::Spike => "ENTROPY_SPIKE",
            AnomalyType::Drop  => "ENTROPY_DROP",
        }
    }
}

/// A single entropy anomaly detected in the codebase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyAnomaly {
    pub fragment_id: String,
    pub source: String,
    pub directory: String,
    pub anomaly_type: AnomalyType,
    /// Modified Z-score: how many MAD-adjusted deviations from the median.
    pub z_score: f64,
    /// Fragment's entropy score [0, 1].
    pub entropy_score: f64,
    /// Group median entropy for context.
    pub group_median: f64,
    /// Boilerplate ratio — high boilerplate + spike = copy-paste anomaly.
    pub boilerplate_ratio: f64,
    /// Confidence [0, 1] — higher z-score and larger group = more confident.
    pub confidence: f64,
    pub recommendation: String,
}

/// Full anomaly scan report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyReport {
    pub fragments_scanned: usize,
    pub groups_analyzed: usize,
    pub anomalies: Vec<EntropyAnomaly>,
    pub summary: String,
}

/// Compute the median of a sorted slice.
fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Extract directory from a source path.
/// "src/handlers/auth.rs" → "src/handlers"
/// "auth.rs" → ""
fn directory_of(source: &str) -> String {
    // Normalize backslashes (Windows paths)
    let normed = source.replace('\\', "/");
    match normed.rfind('/') {
        Some(pos) => normed[..pos].to_string(),
        None => String::new(), // root-level file
    }
}

/// Scan all fragments for entropy anomalies using robust Z-scores.
///
/// Groups fragments by directory, computes MAD-based Z-scores within
/// each group, and flags fragments with |z| > 2.5.
pub fn scan_anomalies(fragments: &[&ContextFragment]) -> AnomalyReport {
    if fragments.is_empty() {
        return AnomalyReport {
            fragments_scanned: 0,
            groups_analyzed: 0,
            anomalies: vec![],
            summary: "No fragments to scan.".into(),
        };
    }

    // ── Group by directory ──
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, frag) in fragments.iter().enumerate() {
        let dir = directory_of(&frag.source);
        groups.entry(dir).or_default().push(i);
    }

    let mut anomalies: Vec<EntropyAnomaly> = Vec::new();
    let mut groups_analyzed = 0usize;

    for (dir, indices) in &groups {
        if indices.len() < MIN_GROUP_SIZE {
            continue;
        }
        groups_analyzed += 1;

        // Collect entropy scores for this group
        let mut entropies: Vec<f64> = indices.iter()
            .map(|&i| fragments[i].entropy_score)
            .collect();
        entropies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let med = median(&entropies);

        // Compute MAD = median(|eᵢ - M|)
        let mut abs_devs: Vec<f64> = entropies.iter()
            .map(|&e| (e - med).abs())
            .collect();
        abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = median(&abs_devs);

        // MAD ≈ 0 means all entropies are nearly identical — no anomalies
        if mad < 1e-9 {
            continue;
        }

        let group_size = indices.len() as f64;

        for &idx in indices {
            let frag = fragments[idx];
            let z = MAD_CONSISTENCY * (frag.entropy_score - med) / mad;

            if z.abs() < Z_THRESHOLD {
                continue;
            }

            let anomaly_type = if z > 0.0 { AnomalyType::Spike } else { AnomalyType::Drop };
            let bp = boilerplate_ratio(&frag.content);

            // Confidence: scales with |z| and group size
            // Larger groups → more statistical power → higher confidence
            let size_factor = (group_size / 10.0).min(1.0);
            let z_factor = ((z.abs() - Z_THRESHOLD) / 2.0).min(1.0);
            let confidence = (0.5 * size_factor + 0.5 * z_factor).clamp(0.0, 1.0);

            let recommendation = match anomaly_type {
                AnomalyType::Spike => {
                    if bp > 0.5 {
                        format!(
                            "Copy-paste anomaly: '{}' has high entropy ({:.2}) with {:.0}% boilerplate \
                             in a low-entropy directory. Likely foreign code pasted in. Review for \
                             correctness and style consistency.",
                            basename(&frag.source),
                            frag.entropy_score,
                            bp * 100.0,
                        )
                    } else {
                        format!(
                            "Complexity anomaly: '{}' has unusually high information density ({:.2}, \
                             z={:.1}) compared to its neighbors (median {:.2}). May contain \
                             non-obvious logic, unusual patterns, or security-sensitive code worth reviewing.",
                            basename(&frag.source),
                            frag.entropy_score,
                            z,
                            med,
                        )
                    }
                }
                AnomalyType::Drop => {
                    format!(
                        "Dead logic suspect: '{}' has unusually low entropy ({:.2}, z={:.1}) in a \
                         directory where median is {:.2}. May be a stub, placeholder, or \
                         accidentally committed empty implementation.",
                        basename(&frag.source),
                        frag.entropy_score,
                        z,
                        med,
                    )
                }
            };

            anomalies.push(EntropyAnomaly {
                fragment_id: frag.fragment_id.clone(),
                source: frag.source.clone(),
                directory: dir.clone(),
                anomaly_type,
                z_score: (z * 100.0).round() / 100.0,
                entropy_score: frag.entropy_score,
                group_median: (med * 1000.0).round() / 1000.0,
                boilerplate_ratio: (bp * 100.0).round() / 100.0,
                confidence: (confidence * 100.0).round() / 100.0,
                recommendation,
            });
        }
    }

    // Sort by |z_score| descending — most anomalous first
    anomalies.sort_unstable_by(|a, b| {
        b.z_score.abs().partial_cmp(&a.z_score.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let spikes = anomalies.iter().filter(|a| a.anomaly_type == AnomalyType::Spike).count();
    let drops = anomalies.iter().filter(|a| a.anomaly_type == AnomalyType::Drop).count();

    let summary = if anomalies.is_empty() {
        format!(
            "No entropy anomalies detected across {} fragments in {} directory groups.",
            fragments.len(), groups_analyzed,
        )
    } else {
        format!(
            "{} entropy anomalies found ({} spikes, {} drops) across {} fragments in {} groups.",
            anomalies.len(), spikes, drops, fragments.len(), groups_analyzed,
        )
    };

    AnomalyReport {
        fragments_scanned: fragments.len(),
        groups_analyzed,
        anomalies,
        summary,
    }
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next()
        .and_then(|s| if s.is_empty() { None } else { Some(s) })
        .unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::ContextFragment;

    fn make_frag(id: &str, source: &str, entropy: f64, content: &str) -> ContextFragment {
        let mut f = ContextFragment::new(id.into(), content.into(), 50, source.into());
        f.entropy_score = entropy;
        f
    }

    #[test]
    fn test_spike_detection() {
        // 6 fragments in same directory: 5 normal + 1 outlier high
        let frags = [
            make_frag("a", "src/handlers/auth.rs", 0.50, "fn auth() { validate(); }"),
            make_frag("b", "src/handlers/user.rs", 0.48, "fn get_user() { query(); }"),
            make_frag("c", "src/handlers/item.rs", 0.52, "fn list_items() { fetch(); }"),
            make_frag("d", "src/handlers/cart.rs", 0.49, "fn add_cart() { insert(); }"),
            make_frag("e", "src/handlers/order.rs", 0.51, "fn place_order() { save(); }"),
            // Outlier: entropy 0.95 vs group median ~0.50
            make_frag("f", "src/handlers/hack.rs", 0.95, "fn x(){let a=b^c;d(e(f(g(h))));}"),
        ];
        let refs: Vec<&ContextFragment> = frags.iter().collect();
        let report = scan_anomalies(&refs);

        assert!(!report.anomalies.is_empty(), "Should detect at least one anomaly");
        assert!(report.anomalies.iter().any(|a| a.fragment_id == "f"),
            "Fragment 'f' should be flagged as anomaly");
        assert!(report.anomalies.iter().any(|a| a.anomaly_type == AnomalyType::Spike),
            "Anomaly should be a spike");
    }

    #[test]
    fn test_drop_detection() {
        // 6 fragments: 5 high entropy + 1 outlier low
        let frags = [
            make_frag("a", "lib/core/engine.rs", 0.80, "complex algorithmic code here"),
            make_frag("b", "lib/core/parser.rs", 0.78, "complex parsing logic there"),
            make_frag("c", "lib/core/scorer.rs", 0.82, "scoring with many branches"),
            make_frag("d", "lib/core/graph.rs", 0.79, "graph traversal algorithms"),
            make_frag("e", "lib/core/index.rs", 0.81, "indexing and retrieval code"),
            // Outlier: entropy 0.10 vs group median ~0.80
            make_frag("f", "lib/core/stub.rs", 0.10, "pass"),
        ];
        let refs: Vec<&ContextFragment> = frags.iter().collect();
        let report = scan_anomalies(&refs);

        assert!(report.anomalies.iter().any(|a|
            a.fragment_id == "f" && a.anomaly_type == AnomalyType::Drop
        ), "Fragment 'f' should be flagged as entropy drop");
    }

    #[test]
    fn test_small_group_skipped() {
        // Only 3 fragments in a directory — below MIN_GROUP_SIZE
        let frags = [
            make_frag("a", "tiny/a.rs", 0.50, "normal"),
            make_frag("b", "tiny/b.rs", 0.50, "normal"),
            make_frag("c", "tiny/outlier.rs", 0.99, "anomalous"),
        ];
        let refs: Vec<&ContextFragment> = frags.iter().collect();
        let report = scan_anomalies(&refs);

        assert!(report.anomalies.is_empty(),
            "Groups smaller than {} should not produce anomalies", MIN_GROUP_SIZE);
        assert_eq!(report.groups_analyzed, 0);
    }

    #[test]
    fn test_uniform_group_no_anomalies() {
        // All fragments have identical entropy — MAD = 0, no anomalies
        let frags: Vec<ContextFragment> = (0..6)
            .map(|i| make_frag(
                &format!("f{}", i),
                &format!("src/uniform/{}.rs", i),
                0.60,
                "all same entropy",
            ))
            .collect();
        let refs: Vec<&ContextFragment> = frags.iter().collect();
        let report = scan_anomalies(&refs);

        assert!(report.anomalies.is_empty(),
            "Uniform entropy group should produce no anomalies");
    }

    #[test]
    fn test_cross_directory_isolation() {
        // High entropy in dir_a is normal, should not affect dir_b
        let mut frags = Vec::new();
        for i in 0..5 {
            frags.push(make_frag(
                &format!("a{}", i), &format!("high/h{}.rs", i), 0.85, "high entropy code",
            ));
        }
        for i in 0..5 {
            frags.push(make_frag(
                &format!("b{}", i), &format!("low/l{}.rs", i), 0.30, "low entropy code",
            ));
        }
        let refs: Vec<&ContextFragment> = frags.iter().collect();
        let report = scan_anomalies(&refs);

        // No anomalies — each directory is internally consistent
        assert!(report.anomalies.is_empty(),
            "Internally consistent directories should not produce cross-contamination anomalies");
    }
}
