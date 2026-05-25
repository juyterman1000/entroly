use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

const WEIGHT_KEYS: [&str; 4] = ["w_r", "w_f", "w_s", "w_e"];
const DECAY_GAMMA: f64 = 0.995;
const WARMUP_EPISODES: f64 = 8.0;
const MAX_BLEND_RATE: f64 = 0.5;
const MIN_WEIGHT: f64 = 0.05;
const MAX_WEIGHT: f64 = 0.80;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEpisode {
    #[serde(default)]
    pub t: f64,
    #[serde(default)]
    pub r: f64,
    #[serde(default)]
    pub w: Value,
}

fn weight_from_value(w: &Value, key: &str, alias: &str, default: f64) -> f64 {
    w.get(key)
        .or_else(|| w.get(alias))
        .and_then(Value::as_f64)
        .unwrap_or(default)
}

fn extract_weights(w: &Value) -> HashMap<&'static str, f64> {
    HashMap::from([
        ("w_r", weight_from_value(w, "w_r", "R", 0.30)),
        ("w_f", weight_from_value(w, "w_f", "F", 0.25)),
        ("w_s", weight_from_value(w, "w_s", "S", 0.25)),
        ("w_e", weight_from_value(w, "w_e", "E", 0.20)),
    ])
}

fn normalize_weights(w: &HashMap<&'static str, f64>) -> Value {
    let mut out: HashMap<&'static str, f64> = HashMap::new();
    for key in WEIGHT_KEYS {
        let v = w.get(key).copied().unwrap_or(0.25).clamp(MIN_WEIGHT, MAX_WEIGHT);
        out.insert(key, v);
    }
    let sum: f64 = out.values().sum();
    if sum > 0.0 {
        for key in WEIGHT_KEYS {
            if let Some(v) = out.get_mut(key) {
                *v = ((*v / sum) * 10_000.0).round() / 10_000.0;
            }
        }
    }
    json!({
        "w_r": out["w_r"],
        "w_f": out["w_f"],
        "w_s": out["w_s"],
        "w_e": out["w_e"],
    })
}

pub fn reward_weighted_optimize(
    episodes: &[LearningEpisode],
    current_weights: &Value,
) -> Option<Value> {
    if episodes.len() < 3 {
        return None;
    }

    let rewards: Vec<f64> = episodes.iter().map(|e| e.r).collect();
    let n = rewards.len();
    let mu = rewards.iter().sum::<f64>() / n as f64;
    let sigma = (rewards.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / n as f64).sqrt();
    let advantages: Vec<f64> = if sigma < 1e-6 {
        rewards
            .iter()
            .map(|r| {
                if *r > 0.0 {
                    1.0
                } else if *r < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        rewards.iter().map(|r| (r - mu) / (sigma + 1e-8)).collect()
    };

    let mut sorted: Vec<(usize, &LearningEpisode)> = episodes.iter().enumerate().collect();
    sorted.sort_by(|(_, a), (_, b)| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    let mut attract: HashMap<&'static str, f64> = WEIGHT_KEYS.map(|k| (k, 0.0)).into_iter().collect();
    let mut repel: HashMap<&'static str, f64> = WEIGHT_KEYS.map(|k| (k, 0.0)).into_iter().collect();
    let mut attract_sum = 0.0;
    let mut repel_sum = 0.0;

    for (rank, (original_idx, ep)) in sorted.iter().enumerate() {
        let weights = extract_weights(&ep.w);
        let adv = advantages[*original_idx];
        let decay = DECAY_GAMMA.powf((n - 1 - rank) as f64);
        if adv > 0.0 {
            let weight = decay * adv;
            for key in WEIGHT_KEYS {
                *attract.get_mut(key).unwrap() += weight * weights[key];
            }
            attract_sum += weight;
        } else if adv < 0.0 {
            let weight = decay * adv.abs();
            for key in WEIGHT_KEYS {
                *repel.get_mut(key).unwrap() += weight * weights[key];
            }
            repel_sum += weight;
        }
    }

    if attract_sum <= 0.0 {
        return None;
    }
    for key in WEIGHT_KEYS {
        *attract.get_mut(key).unwrap() /= attract_sum;
        if repel_sum > 0.0 {
            *repel.get_mut(key).unwrap() /= repel_sum;
        }
    }

    let current = extract_weights(current_weights);
    let mut dim_std: HashMap<&'static str, f64> = HashMap::new();
    let mut dim_snr: HashMap<&'static str, f64> = HashMap::new();
    for key in WEIGHT_KEYS {
        let values: Vec<f64> = sorted.iter().map(|(_, ep)| extract_weights(&ep.w)[key]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64)
            .sqrt()
            .max(0.01);
        dim_std.insert(key, std);
        dim_snr.insert(key, (attract[key] - current[key]).abs() / std);
    }

    let confidence = (n as f64 / WARMUP_EPISODES).min(1.0);
    let base_alpha = confidence * MAX_BLEND_RATE;
    let beta = if repel_sum > 0.0 {
        0.15 * (repel_sum / attract_sum).min(1.0)
    } else {
        0.0
    };

    let mut optimal: HashMap<&'static str, f64> = HashMap::new();
    let mut blended: HashMap<&'static str, f64> = HashMap::new();
    for key in WEIGHT_KEYS {
        let repel_delta = if repel_sum > 0.0 { repel[key] - current[key] } else { 0.0 };
        let opt = attract[key] - beta * repel_delta;
        optimal.insert(key, opt);

        let nat_grad_scale = 1.0 / (dim_std[key].powi(2) + 0.01);
        let sigmoid_snr = 1.0 / (1.0 + (-2.0 * (dim_snr[key] - 0.5)).exp());
        let alpha_k = base_alpha * sigmoid_snr;
        let direction = ((opt - current[key]) * nat_grad_scale).clamp(-0.1, 0.1);
        blended.insert(key, current[key] + alpha_k * direction);
    }

    let mut polyak: HashMap<&'static str, f64> = WEIGHT_KEYS.map(|k| (k, 0.0)).into_iter().collect();
    for (_, ep) in sorted {
        let weights = extract_weights(&ep.w);
        for key in WEIGHT_KEYS {
            *polyak.get_mut(key).unwrap() += weights[key];
        }
    }
    for key in WEIGHT_KEYS {
        *polyak.get_mut(key).unwrap() /= n as f64;
    }

    let positive: Vec<f64> = episodes.iter().filter(|e| e.r > 0.0).map(|e| e.r).collect();
    let avg_positive = if positive.is_empty() {
        0.0
    } else {
        positive.iter().sum::<f64>() / positive.len() as f64
    };
    let estimated_regret = (((avg_positive - mu) * n as f64).max(0.0) * 100.0).round() / 100.0;

    Some(json!({
        "optimal": normalize_weights(&optimal),
        "blended": normalize_weights(&blended),
        "polyak": normalize_weights(&polyak),
        "confidence": confidence,
        "success_count": episodes.iter().filter(|e| e.r > 0.0).count(),
        "failure_count": episodes.iter().filter(|e| e.r < 0.0).count(),
        "total_episodes": n,
        "estimated_regret": estimated_regret,
    }))
}

pub fn reward_weighted_optimize_json(
    episodes_json: &str,
    current_weights_json: &str,
) -> Result<Option<Value>, serde_json::Error> {
    let episodes: Vec<LearningEpisode> = serde_json::from_str(episodes_json)?;
    let current_weights: Value = serde_json::from_str(current_weights_json)?;
    Ok(reward_weighted_optimize(&episodes, &current_weights))
}

