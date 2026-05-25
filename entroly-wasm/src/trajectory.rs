use crate::dedup::{hamming_distance, simhash};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryTransition {
    pub status: String,
    pub current_hash: u64,
    pub similarity: f64,
    pub hamming: u32,
    pub elapsed_s: f64,
}

pub fn classify_query_transition(
    prev_hash: u64,
    current_query: &str,
    elapsed_s: f64,
    time_window_s: f64,
    rephrase_threshold: f64,
    topic_change_threshold: Option<f64>,
) -> QueryTransition {
    let current_hash = simhash(current_query);
    if elapsed_s < 0.0 {
        return QueryTransition {
            status: "invalid_time".to_string(),
            current_hash,
            similarity: 0.0,
            hamming: 64,
            elapsed_s,
        };
    }
    if elapsed_s > time_window_s {
        return QueryTransition {
            status: "expired".to_string(),
            current_hash,
            similarity: 0.0,
            hamming: 64,
            elapsed_s,
        };
    }

    let hamming = hamming_distance(current_hash, prev_hash);
    let similarity = 1.0 - (f64::from(hamming) / 64.0);
    let status = if similarity > rephrase_threshold {
        "rephrase"
    } else if topic_change_threshold
        .map(|threshold| similarity < threshold)
        .unwrap_or(true)
    {
        "topic_change"
    } else {
        "ambiguous"
    };

    QueryTransition {
        status: status.to_string(),
        current_hash,
        similarity,
        hamming,
        elapsed_s,
    }
}

