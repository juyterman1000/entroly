//! Archetype Engine — Codebase Structural Fingerprinting
//!
//! Novel contribution: Instead of one global set of scoring weights,
//! Entroly detects **what kind** of codebase it's running inside and
//! loads archetype-specific optimized weights.
//!
//! Mathematical grounding:
//!   - The fingerprint is a d=16 dimensional vector F ∈ ℝ¹⁶ computed
//!     from structural invariants of the codebase:
//!     F = [lang_dist(4), struct_metrics(5), topology(4), entropy_stats(3)]
//!
//!   - Archetypes are discovered via online k-means clustering:
//!     argmin_C Σᵢ ||Fᵢ - μ_c(i)||²
//!     with adaptive k: clusters split when variance exceeds θ_split
//!     and merge when centroids are within θ_merge.
//!
//!   - Classification is O(k·d) per query — microseconds for k≤32, d=16.
//!
//! Key properties:
//!   - **Invariant to file naming** — uses extensions + AST, not paths
//!   - **Stable across commits** — structural ratios change slowly
//!   - **High signal** — discriminates React frontends from Rust libraries
//!     from Django backends from monorepos
//!
//! Usage:
//!   let engine = ArchetypeEngine::new();
//!   let fp = engine.fingerprint(&file_stats);
//!   let archetype_id = engine.classify(&fp);
//!   let weights = engine.get_weights(archetype_id);

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dimensionality of the fingerprint vector.
pub const FINGERPRINT_DIM: usize = 16;

/// Maximum number of archetypes the engine will maintain.
const MAX_ARCHETYPES: usize = 32;

/// Minimum samples before a cluster can split.
const MIN_SPLIT_SAMPLES: usize = 10;

/// Variance threshold to trigger a cluster split (empirical).
const SPLIT_VARIANCE_THRESHOLD: f64 = 0.25;

/// Distance threshold below which two centroids merge.
const MERGE_DISTANCE_THRESHOLD: f64 = 0.08;

// ═══════════════════════════════════════════════════════════════════
// Data Structures
// ═══════════════════════════════════════════════════════════════════

/// A 16-dimensional structural fingerprint of a codebase.
///
/// Dimensions:
///   [0]  lang_python:    fraction of files that are .py
///   [1]  lang_rust:      fraction of files that are .rs
///   [2]  lang_js_ts:     fraction of files that are .js/.ts/.jsx/.tsx
///   [3]  lang_other:     fraction of files in other languages
///   [4]  avg_file_size:  mean lines per file (normalized log)
///   [5]  func_density:   functions per 100 lines (normalized)
///   [6]  class_ratio:    classes / (classes + functions)
///   [7]  import_density: imports per file (normalized log)
///   [8]  test_ratio:     fraction of files in test directories
///   [9]  graph_density:  edges / (nodes * (nodes-1)) of dep graph
///   [10] max_depth:      max dependency chain depth (normalized)
///   [11] module_count:   number of top-level modules (normalized log)
///   [12] coupling:       avg edges per node (normalized)
///   [13] entropy_mean:   mean Kolmogorov entropy across all files
///   [14] entropy_var:    variance of entropy (high = heterogeneous)
///   [15] ffi_ratio:      fraction of files with cross-language FFI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub dims: [f64; FINGERPRINT_DIM],
}

impl Fingerprint {
    pub fn zero() -> Self {
        Fingerprint {
            dims: [0.0; FINGERPRINT_DIM],
        }
    }

    /// L2 (Euclidean) distance to another fingerprint.
    #[inline]
    pub fn distance(&self, other: &Fingerprint) -> f64 {
        let mut sum = 0.0;
        for i in 0..FINGERPRINT_DIM {
            let d = self.dims[i] - other.dims[i];
            sum += d * d;
        }
        sum.sqrt()
    }

    /// Cosine similarity ∈ [-1, 1].
    #[inline]
    pub fn cosine_similarity(&self, other: &Fingerprint) -> f64 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..FINGERPRINT_DIM {
            dot += self.dims[i] * other.dims[i];
            norm_a += self.dims[i] * self.dims[i];
            norm_b += other.dims[i] * other.dims[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            dot / denom
        }
    }
}

/// Raw statistics collected from a codebase scan.
/// The fingerprinter converts these into a normalized Fingerprint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodebaseStats {
    pub total_files: usize,
    pub python_files: usize,
    pub rust_files: usize,
    pub js_ts_files: usize,
    pub other_files: usize,
    pub total_lines: usize,
    pub total_functions: usize,
    pub total_classes: usize,
    pub total_imports: usize,
    pub test_files: usize,
    pub graph_nodes: usize,
    pub graph_edges: usize,
    pub max_dep_depth: usize,
    pub module_count: usize,
    pub entropy_values: Vec<f64>,
    pub ffi_files: usize,
}

/// A scoring weight profile for a specific archetype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightProfile {
    pub w_recency: f64,
    pub w_frequency: f64,
    pub w_semantic: f64,
    pub w_entropy: f64,
    pub decay_half_life: f64,
    pub min_relevance: f64,
    pub exploration_rate: f64,
}

impl Default for WeightProfile {
    fn default() -> Self {
        WeightProfile {
            w_recency: 0.30,
            w_frequency: 0.25,
            w_semantic: 0.25,
            w_entropy: 0.20,
            decay_half_life: 15.0,
            min_relevance: 0.05,
            exploration_rate: 0.10,
        }
    }
}

/// An archetype cluster: centroid + optimized weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Archetype {
    pub id: u32,
    pub label: String,
    pub centroid: Fingerprint,
    pub weights: WeightProfile,
    pub sample_count: usize,
    /// Running sum of squared distances to centroid (for variance).
    pub variance_sum: f64,
    /// Confidence score [0, 1]: higher = more stable cluster.
    pub confidence: f64,
}

impl Archetype {
    /// Intra-cluster variance.
    pub fn variance(&self) -> f64 {
        if self.sample_count < 2 {
            0.0
        } else {
            self.variance_sum / self.sample_count as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Archetype Engine
// ═══════════════════════════════════════════════════════════════════

/// The core archetype detection and classification engine.
///
/// Maintains an online k-means clustering of codebase fingerprints.
/// Each cluster (archetype) has its own optimized weight profile
/// that the DreamingLoop evolves independently.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeEngine {
    archetypes: Vec<Archetype>,
    next_id: u32,
}

impl ArchetypeEngine {
    /// Create a new engine with seed archetypes.
    ///
    /// Seeds 5 initial archetypes from well-known codebase patterns.
    /// These centroids are empirically derived from analysis of
    /// popular open-source repositories.
    pub fn new() -> Self {
        let mut engine = ArchetypeEngine {
            archetypes: Vec::new(),
            next_id: 0,
        };

        // Seed archetype 0: Python Backend (Django/FastAPI/Flask)
        engine.add_seed(
            "python_backend",
            [
                0.70, 0.00, 0.05, 0.25, 0.45, 0.60, 0.35, 0.55, 0.25, 0.15, 0.35, 0.40, 0.25, 0.50,
                0.15, 0.05,
            ],
            WeightProfile {
                w_recency: 0.35,
                w_frequency: 0.25,
                w_semantic: 0.20,
                w_entropy: 0.20,
                ..Default::default()
            },
        );

        // Seed archetype 1: Rust Systems Library
        engine.add_seed(
            "rust_systems",
            [
                0.00, 0.80, 0.05, 0.15, 0.55, 0.70, 0.20, 0.45, 0.20, 0.25, 0.50, 0.30, 0.35, 0.65,
                0.20, 0.15,
            ],
            WeightProfile {
                w_recency: 0.20,
                w_frequency: 0.20,
                w_semantic: 0.30,
                w_entropy: 0.30,
                ..Default::default()
            },
        );

        // Seed archetype 2: JS/TS Frontend (React/Vue/Angular)
        engine.add_seed(
            "js_frontend",
            [
                0.00, 0.00, 0.80, 0.20, 0.35, 0.45, 0.40, 0.65, 0.15, 0.10, 0.25, 0.50, 0.20, 0.40,
                0.25, 0.05,
            ],
            WeightProfile {
                w_recency: 0.40,
                w_frequency: 0.20,
                w_semantic: 0.25,
                w_entropy: 0.15,
                ..Default::default()
            },
        );

        // Seed archetype 3: Full-Stack Monorepo
        engine.add_seed(
            "fullstack_monorepo",
            [
                0.30, 0.05, 0.40, 0.25, 0.40, 0.50, 0.30, 0.55, 0.20, 0.20, 0.45, 0.65, 0.30, 0.50,
                0.30, 0.10,
            ],
            WeightProfile {
                w_recency: 0.30,
                w_frequency: 0.25,
                w_semantic: 0.25,
                w_entropy: 0.20,
                decay_half_life: 20.0,
                ..Default::default()
            },
        );

        // Seed archetype 4: Data Science / ML
        engine.add_seed(
            "data_science",
            [
                0.75, 0.00, 0.05, 0.20, 0.50, 0.40, 0.15, 0.60, 0.10, 0.08, 0.20, 0.25, 0.15, 0.55,
                0.35, 0.05,
            ],
            WeightProfile {
                w_recency: 0.25,
                w_frequency: 0.30,
                w_semantic: 0.25,
                w_entropy: 0.20,
                ..Default::default()
            },
        );

        engine
    }

    fn add_seed(&mut self, label: &str, dims: [f64; FINGERPRINT_DIM], weights: WeightProfile) {
        self.archetypes.push(Archetype {
            id: self.next_id,
            label: label.to_string(),
            centroid: Fingerprint { dims },
            weights,
            sample_count: 1,
            variance_sum: 0.0,
            confidence: 0.5, // prior
        });
        self.next_id += 1;
    }

    /// Compute a normalized fingerprint from raw codebase statistics.
    ///
    /// All dimensions are normalized to [0, 1] for stable clustering.
    /// Uses log-normalization for heavy-tailed distributions (file counts,
    /// import counts) and linear normalization for ratios.
    pub fn fingerprint(&self, stats: &CodebaseStats) -> Fingerprint {
        let total = stats.total_files.max(1) as f64;
        let total_lines = stats.total_lines.max(1) as f64;

        // Language distribution [0-3]: fractions summing to ~1.0
        let lang_py = stats.python_files as f64 / total;
        let lang_rs = stats.rust_files as f64 / total;
        let lang_js = stats.js_ts_files as f64 / total;
        let lang_other = stats.other_files as f64 / total;

        // Structural metrics [4-8]
        let avg_file_size = log_norm(total_lines / total, 500.0);
        let func_density = log_norm(
            stats.total_functions as f64 / (total_lines / 100.0).max(1.0),
            20.0,
        );
        let class_ratio = if stats.total_classes + stats.total_functions > 0 {
            stats.total_classes as f64 / (stats.total_classes + stats.total_functions) as f64
        } else {
            0.0
        };
        let import_density = log_norm(stats.total_imports as f64 / total, 30.0);
        let test_ratio = stats.test_files as f64 / total;

        // Topology [9-12]
        let nodes = stats.graph_nodes.max(1) as f64;
        let max_possible_edges = nodes * (nodes - 1.0);
        let graph_density = if max_possible_edges > 0.0 {
            (stats.graph_edges as f64 / max_possible_edges).min(1.0)
        } else {
            0.0
        };
        let max_depth = log_norm(stats.max_dep_depth as f64, 15.0);
        let module_count = log_norm(stats.module_count as f64, 50.0);
        let coupling = log_norm(stats.graph_edges as f64 / nodes, 10.0);

        // Entropy [13-15]
        let (entropy_mean, entropy_var) = if stats.entropy_values.is_empty() {
            (0.5, 0.1)
        } else {
            let n = stats.entropy_values.len() as f64;
            let mean = stats.entropy_values.iter().sum::<f64>() / n;
            let var = stats
                .entropy_values
                .iter()
                .map(|e| (e - mean) * (e - mean))
                .sum::<f64>()
                / n;
            (mean.clamp(0.0, 1.0), var.clamp(0.0, 1.0))
        };
        let ffi_ratio = stats.ffi_files as f64 / total;

        Fingerprint {
            dims: [
                lang_py,
                lang_rs,
                lang_js,
                lang_other,
                avg_file_size,
                func_density,
                class_ratio,
                import_density,
                test_ratio,
                graph_density,
                max_depth,
                module_count,
                coupling,
                entropy_mean,
                entropy_var,
                ffi_ratio,
            ],
        }
    }

    /// Classify a fingerprint into the nearest archetype.
    ///
    /// Returns (archetype_id, distance, confidence).
    /// O(k·d) where k=num_archetypes, d=16.
    pub fn classify(&self, fp: &Fingerprint) -> (u32, f64, f64) {
        if self.archetypes.is_empty() {
            return (0, f64::MAX, 0.0);
        }

        let mut best_id = 0u32;
        let mut best_dist = f64::MAX;
        let mut second_dist = f64::MAX;

        for arch in &self.archetypes {
            let dist = fp.distance(&arch.centroid);
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_id = arch.id;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        // Confidence: ratio of second-best to best distance.
        // High ratio = clear winner. Ratio near 1.0 = ambiguous.
        // Silhouette-inspired: (second - best) / max(second, best)
        let confidence = if second_dist > 0.0 && best_dist < f64::MAX {
            ((second_dist - best_dist) / second_dist.max(best_dist)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        (best_id, best_dist, confidence)
    }

    /// Update the archetype centroids with a new observation.
    ///
    /// Online k-means update: μ_new = μ_old + (1/n) · (x - μ_old)
    /// Also triggers split/merge maintenance.
    pub fn observe(&mut self, fp: &Fingerprint) -> u32 {
        let (arch_id, dist, _confidence) = self.classify(fp);

        // Find the archetype and update
        if let Some(arch) = self.archetypes.iter_mut().find(|a| a.id == arch_id) {
            arch.sample_count += 1;
            let n = arch.sample_count as f64;

            // Online centroid update: μ += (1/n)(x - μ)
            for i in 0..FINGERPRINT_DIM {
                arch.centroid.dims[i] += (fp.dims[i] - arch.centroid.dims[i]) / n;
            }

            // Update variance running sum
            arch.variance_sum += dist * dist;

            // Update confidence (EMA of classification margin)
            arch.confidence = 0.9 * arch.confidence + 0.1 * (1.0 - dist.min(1.0));
        }

        // Periodic maintenance: split/merge every 50 observations
        let total_samples: usize = self.archetypes.iter().map(|a| a.sample_count).sum();
        if total_samples.is_multiple_of(50) {
            self.maybe_split();
            self.maybe_merge();
        }

        arch_id
    }

    /// Split high-variance clusters into two.
    fn maybe_split(&mut self) {
        if self.archetypes.len() >= MAX_ARCHETYPES {
            return;
        }

        let mut to_split = Vec::new();
        for arch in &self.archetypes {
            if arch.sample_count >= MIN_SPLIT_SAMPLES && arch.variance() > SPLIT_VARIANCE_THRESHOLD
            {
                to_split.push(arch.id);
            }
        }

        for id in to_split {
            if self.archetypes.len() >= MAX_ARCHETYPES {
                break;
            }
            if let Some(idx) = self.archetypes.iter().position(|a| a.id == id) {
                let parent = self.archetypes[idx].clone();

                // Create child by perturbing the centroid along the
                // dimension with highest contribution to variance.
                // Simple heuristic: perturb the largest dimension.
                let mut child_dims = parent.centroid.dims;
                let max_dim = child_dims
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                child_dims[max_dim] = (child_dims[max_dim] + 0.15).min(1.0);

                // Shrink parent toward opposite direction
                self.archetypes[idx].centroid.dims[max_dim] =
                    (self.archetypes[idx].centroid.dims[max_dim] - 0.10).max(0.0);
                self.archetypes[idx].sample_count /= 2;
                self.archetypes[idx].variance_sum /= 2.0;

                let child = Archetype {
                    id: self.next_id,
                    label: format!("{}_split_{}", parent.label, self.next_id),
                    centroid: Fingerprint { dims: child_dims },
                    weights: parent.weights.clone(), // inherit parent weights
                    sample_count: parent.sample_count / 2,
                    variance_sum: parent.variance_sum / 4.0,
                    confidence: 0.3, // low initial confidence
                };
                self.next_id += 1;
                self.archetypes.push(child);
            }
        }
    }

    /// Merge clusters whose centroids are too close.
    fn maybe_merge(&mut self) {
        let n = self.archetypes.len();
        if n < 3 {
            return;
        } // keep at least 2

        let mut to_merge: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.archetypes[i]
                    .centroid
                    .distance(&self.archetypes[j].centroid);
                if dist < MERGE_DISTANCE_THRESHOLD {
                    to_merge.push((i, j));
                }
            }
        }

        // Process merges (only one per pass to avoid index invalidation)
        if let Some(&(i, j)) = to_merge.first() {
            let total = self.archetypes[i].sample_count + self.archetypes[j].sample_count;
            let wi = self.archetypes[i].sample_count as f64 / total as f64;
            let wj = self.archetypes[j].sample_count as f64 / total as f64;

            // Weighted centroid merge
            for d in 0..FINGERPRINT_DIM {
                self.archetypes[i].centroid.dims[d] = wi * self.archetypes[i].centroid.dims[d]
                    + wj * self.archetypes[j].centroid.dims[d];
            }

            // Weighted weight merge
            let wj_profile = self.archetypes[j].weights.clone();
            let wi_p = &mut self.archetypes[i].weights;
            wi_p.w_recency = wi * wi_p.w_recency + wj * wj_profile.w_recency;
            wi_p.w_frequency = wi * wi_p.w_frequency + wj * wj_profile.w_frequency;
            wi_p.w_semantic = wi * wi_p.w_semantic + wj * wj_profile.w_semantic;
            wi_p.w_entropy = wi * wi_p.w_entropy + wj * wj_profile.w_entropy;

            // Normalize weights to sum=1
            let wsum = wi_p.w_recency + wi_p.w_frequency + wi_p.w_semantic + wi_p.w_entropy;
            if wsum > 0.0 {
                wi_p.w_recency /= wsum;
                wi_p.w_frequency /= wsum;
                wi_p.w_semantic /= wsum;
                wi_p.w_entropy /= wsum;
            }

            self.archetypes[i].sample_count = total;
            self.archetypes[i].variance_sum += self.archetypes[j].variance_sum;
            self.archetypes[i].confidence =
                (self.archetypes[i].confidence + self.archetypes[j].confidence) / 2.0;

            self.archetypes.remove(j);
        }
    }

    /// Get the optimized weight profile for an archetype.
    pub fn get_weights(&self, archetype_id: u32) -> WeightProfile {
        self.archetypes
            .iter()
            .find(|a| a.id == archetype_id)
            .map(|a| a.weights.clone())
            .unwrap_or_default()
    }

    /// Update the weight profile for an archetype (called by DreamingLoop).
    pub fn set_weights(&mut self, archetype_id: u32, weights: WeightProfile) {
        if let Some(arch) = self.archetypes.iter_mut().find(|a| a.id == archetype_id) {
            arch.weights = weights;
        }
    }

    /// Get all archetypes for serialization / dashboard display.
    pub fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    /// Summary statistics for monitoring.
    pub fn stats(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("archetype_count".into(), self.archetypes.len() as f64);
        m.insert(
            "total_samples".into(),
            self.archetypes.iter().map(|a| a.sample_count as f64).sum(),
        );
        m.insert(
            "avg_confidence".into(),
            if self.archetypes.is_empty() {
                0.0
            } else {
                self.archetypes.iter().map(|a| a.confidence).sum::<f64>()
                    / self.archetypes.len() as f64
            },
        );
        m
    }
}

// ═══════════════════════════════════════════════════════════════════
// PyO3 Bindings — Exposes ArchetypeEngine to Python
// ═══════════════════════════════════════════════════════════════════

impl Default for ArchetypeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl ArchetypeEngine {
    /// Create a new ArchetypeEngine with 5 seed archetypes.
    #[new]
    pub fn py_new() -> Self {
        ArchetypeEngine::new()
    }

    /// Fingerprint + classify a codebase from raw stats dict.
    ///
    /// Args:
    ///   stats: dict with keys matching CodebaseStats fields
    ///
    /// Returns:
    ///   dict with archetype_id, label, confidence, weights
    pub fn classify_codebase(
        &self,
        py: Python<'_>,
        stats: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let cs = CodebaseStats {
            total_files: stats
                .get_item("total_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            python_files: stats
                .get_item("python_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            rust_files: stats
                .get_item("rust_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            js_ts_files: stats
                .get_item("js_ts_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            other_files: stats
                .get_item("other_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            total_lines: stats
                .get_item("total_lines")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            total_functions: stats
                .get_item("total_functions")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            total_classes: stats
                .get_item("total_classes")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            total_imports: stats
                .get_item("total_imports")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            test_files: stats
                .get_item("test_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            graph_nodes: stats
                .get_item("graph_nodes")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            graph_edges: stats
                .get_item("graph_edges")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            max_dep_depth: stats
                .get_item("max_dep_depth")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            module_count: stats
                .get_item("module_count")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            entropy_values: stats
                .get_item("entropy_values")?
                .and_then(|v| v.extract::<Vec<f64>>().ok())
                .unwrap_or_default(),
            ffi_files: stats
                .get_item("ffi_files")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
        };

        let fp = self.fingerprint(&cs);
        let (arch_id, dist, confidence) = self.classify(&fp);
        let weights = self.get_weights(arch_id);
        let label = self
            .archetypes()
            .iter()
            .find(|a| a.id == arch_id)
            .map(|a| a.label.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let result = PyDict::new(py);
        result.set_item("archetype_id", arch_id)?;
        result.set_item("label", &label)?;
        result.set_item("confidence", confidence)?;
        result.set_item("distance", dist)?;
        result.set_item("w_recency", weights.w_recency)?;
        result.set_item("w_frequency", weights.w_frequency)?;
        result.set_item("w_semantic", weights.w_semantic)?;
        result.set_item("w_entropy", weights.w_entropy)?;
        result.set_item("decay_half_life", weights.decay_half_life)?;
        result.set_item("min_relevance", weights.min_relevance)?;
        result.set_item("exploration_rate", weights.exploration_rate)?;
        Ok(result.into())
    }

    /// Get summary stats as a Python dict.
    pub fn py_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let m = self.stats();
        let result = PyDict::new(py);
        for (k, v) in &m {
            result.set_item(k, v)?;
        }
        result.set_item(
            "archetype_labels",
            self.archetypes()
                .iter()
                .map(|a| a.label.clone())
                .collect::<Vec<_>>(),
        )?;
        Ok(result.into())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════

/// Log-normalize a value to [0, 1] given a reference maximum.
///
/// f(x) = ln(1 + x) / ln(1 + max_ref)
///
/// Properties:
///   - f(0) = 0
///   - f(max_ref) ≈ 1
///   - Compresses heavy tails (file counts, import counts)
///   - Monotonically increasing
#[inline]
fn log_norm(value: f64, max_ref: f64) -> f64 {
    let v = value.max(0.0);
    let m = max_ref.max(1.0);
    (1.0 + v).ln() / (1.0 + m).ln()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_distance() {
        let a = Fingerprint {
            dims: [0.0; FINGERPRINT_DIM],
        };
        let b = Fingerprint {
            dims: [1.0; FINGERPRINT_DIM],
        };
        let dist = a.distance(&b);
        assert!((dist - 4.0).abs() < 0.001); // sqrt(16) = 4.0
    }

    #[test]
    fn test_log_norm() {
        assert!((log_norm(0.0, 100.0)).abs() < 0.001);
        assert!((log_norm(100.0, 100.0) - 1.0).abs() < 0.001);
        assert!(log_norm(50.0, 100.0) > 0.5); // log compression
    }

    #[test]
    fn test_seed_archetypes() {
        let engine = ArchetypeEngine::new();
        assert_eq!(engine.archetypes.len(), 5);
    }

    #[test]
    fn test_classify_python_backend() {
        let engine = ArchetypeEngine::new();
        let stats = CodebaseStats {
            total_files: 100,
            python_files: 70,
            rust_files: 0,
            js_ts_files: 5,
            other_files: 25,
            total_lines: 15000,
            total_functions: 400,
            total_classes: 80,
            total_imports: 1200,
            test_files: 25,
            graph_nodes: 100,
            graph_edges: 300,
            max_dep_depth: 5,
            module_count: 15,
            entropy_values: vec![0.5; 100],
            ffi_files: 2,
        };
        let fp = engine.fingerprint(&stats);
        let (id, _dist, _conf) = engine.classify(&fp);
        // Should classify as python_backend (id=0) or data_science (id=4)
        assert!(id == 0 || id == 4);
    }

    #[test]
    fn test_observe_updates_centroid() {
        let mut engine = ArchetypeEngine::new();
        let stats = CodebaseStats {
            total_files: 50,
            python_files: 0,
            rust_files: 45,
            js_ts_files: 0,
            other_files: 5,
            total_lines: 10000,
            total_functions: 300,
            total_classes: 20,
            total_imports: 200,
            test_files: 10,
            graph_nodes: 50,
            graph_edges: 150,
            max_dep_depth: 6,
            module_count: 8,
            entropy_values: vec![0.65; 50],
            ffi_files: 5,
        };
        let fp = engine.fingerprint(&stats);
        let id = engine.observe(&fp);
        assert_eq!(id, 1); // rust_systems
        let arch = engine.archetypes.iter().find(|a| a.id == 1).unwrap();
        assert!(arch.sample_count >= 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Fingerprint {
            dims: [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        };
        let b = Fingerprint {
            dims: [
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        };
        assert!((a.cosine_similarity(&b)).abs() < 0.001); // orthogonal
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 0.001); // identical
    }
}
