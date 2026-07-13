//! Rust-native model intelligence resolver.
//!
//! This module mirrors the Python control-plane semantics while remaining
//! independently packageable inside `entroly-core`. The checked-in snapshot is
//! generated from `entroly/models/registry.json`; repository CI enforces
//! byte-for-byte parity between the two package-local copies.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const SNAPSHOT_JSON: &str = include_str!("../../model_registry.json");
const DEFAULT_FALLBACK_CONTEXT: u64 = 128_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegistryTrust {
    Verified,
    Discovered,
    User,
    Announced,
    Fallback,
}

impl RegistryTrust {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Discovered => "discovered",
            Self::User => "user",
            Self::Announced => "announced",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCapability {
    pub id: String,
    pub provider: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub supports_tools: Option<bool>,
    #[serde(default)]
    pub supports_vision: Option<bool>,
    #[serde(default)]
    pub supports_reasoning: Option<bool>,
    #[serde(default)]
    pub reasoning_levels: Vec<String>,
    #[serde(default)]
    pub input_price_per_million: Option<f64>,
    #[serde(default)]
    pub output_price_per_million: Option<f64>,
    pub trust: RegistryTrust,
    pub source: String,
    #[serde(default)]
    pub verified_at: Option<String>,
    #[serde(default)]
    pub observed_at: Option<String>,
}

impl ModelCapability {
    fn normalize(mut self) -> Result<Self, String> {
        self.id = normalize_name(&self.id);
        self.provider = normalize_name(&self.provider);
        self.aliases = self
            .aliases
            .into_iter()
            .map(|alias| normalize_name(&alias))
            .filter(|alias| !alias.is_empty())
            .collect();

        if self.id.is_empty() {
            return Err("model capability id must be non-empty".into());
        }
        if self.provider.is_empty() {
            return Err(format!("provider must be non-empty for {:?}", self.id));
        }
        if matches!(self.context_window, Some(0)) {
            return Err(format!("invalid context_window for {:?}", self.id));
        }
        if matches!(self.max_output_tokens, Some(0)) {
            return Err(format!("invalid max_output_tokens for {:?}", self.id));
        }
        if let (Some(context), Some(output)) = (self.context_window, self.max_output_tokens) {
            if output >= context {
                return Err(format!(
                    "max_output_tokens must be smaller than context_window for {:?}",
                    self.id
                ));
            }
        }
        if self
            .input_price_per_million
            .is_some_and(|price| !price.is_finite() || price < 0.0)
            || self
                .output_price_per_million
                .is_some_and(|price| !price.is_finite() || price < 0.0)
        {
            return Err(format!("invalid price metadata for {:?}", self.id));
        }
        Ok(self)
    }

    pub fn estimated_cost_usd(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let input_price = self.input_price_per_million?;
        let output_price = self.output_price_per_million?;
        Some(
            (input_tokens as f64 * input_price + output_tokens as f64 * output_price) / 1_000_000.0,
        )
    }

    fn fingerprint_value(&self) -> Value {
        let mut aliases = self.aliases.clone();
        aliases.sort();
        json!({
            "aliases": aliases,
            "context_window": self.context_window,
            "id": self.id,
            "input_price_per_million": self.input_price_per_million,
            "max_output_tokens": self.max_output_tokens,
            "observed_at": self.observed_at,
            "output_price_per_million": self.output_price_per_million,
            "provider": self.provider,
            "reasoning_levels": self.reasoning_levels,
            "source": self.source,
            "supports_reasoning": self.supports_reasoning,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "trust": self.trust,
            "verified_at": self.verified_at,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelResolution {
    pub requested_model: String,
    pub capability: Option<ModelCapability>,
    pub context_window: u64,
    pub exact: bool,
    pub trust: RegistryTrust,
    pub warning: Option<String>,
    pub registry_digest: String,
    pub base_registry_digest: String,
}

impl ModelResolution {
    pub fn model_id(&self) -> &str {
        self.capability
            .as_ref()
            .map(|capability| capability.id.as_str())
            .unwrap_or(self.requested_model.as_str())
    }

    pub fn effective_input_budget(
        &self,
        requested_output_tokens: Option<u64>,
        safety_fraction: f64,
        minimum_safety_tokens: u64,
    ) -> Result<u64, String> {
        if !(0.0..1.0).contains(&safety_fraction) {
            return Err("safety_fraction must be in [0, 1)".into());
        }

        let model_output = self
            .capability
            .as_ref()
            .and_then(|capability| capability.max_output_tokens);
        let output_reserve = match (requested_output_tokens, model_output) {
            (Some(requested), Some(model_max)) => requested.min(model_max),
            (Some(requested), None) => requested,
            (None, Some(model_max)) => model_max,
            (None, None) => 0,
        };
        let safety = minimum_safety_tokens
            .max((self.context_window as f64 * safety_fraction).floor() as u64);
        Ok(self
            .context_window
            .saturating_sub(output_reserve)
            .saturating_sub(safety)
            .max(1))
    }
}

#[derive(Debug, Deserialize)]
struct RegistryDocument {
    models: Vec<ModelCapability>,
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    fallback_context_window: u64,
    by_id: HashMap<String, ModelCapability>,
    aliases: HashMap<String, String>,
    aliases_by_id: HashMap<String, HashSet<String>>,
    registry_digest: String,
    base_registry_digest: String,
}

impl ModelRegistry {
    pub fn bundled() -> Result<Self, String> {
        Self::from_snapshot(SNAPSHOT_JSON, DEFAULT_FALLBACK_CONTEXT)
    }

    pub fn from_snapshot(snapshot: &str, fallback_context_window: u64) -> Result<Self, String> {
        if fallback_context_window == 0 {
            return Err("fallback_context_window must be positive".into());
        }
        let document: RegistryDocument = serde_json::from_str(snapshot)
            .map_err(|error| format!("invalid model registry JSON: {error}"))?;
        let base_registry_digest = sha256_hex(snapshot.as_bytes());
        let mut registry = Self {
            fallback_context_window,
            by_id: HashMap::new(),
            aliases: HashMap::new(),
            aliases_by_id: HashMap::new(),
            registry_digest: String::new(),
            base_registry_digest,
        };
        let mut seen = HashSet::new();
        for capability in document.models {
            let capability = capability.normalize()?;
            if !seen.insert(capability.id.clone()) {
                return Err(format!("duplicate model id {:?}", capability.id));
            }
            registry.install(capability);
        }
        registry.registry_digest = registry.compute_effective_digest()?;
        Ok(registry)
    }

    pub fn install_override(&mut self, capability: ModelCapability) -> Result<(), String> {
        self.install(capability.normalize()?);
        self.registry_digest = self.compute_effective_digest()?;
        Ok(())
    }

    fn install(&mut self, capability: ModelCapability) {
        if let Some(old_aliases) = self.aliases_by_id.remove(&capability.id) {
            for alias in old_aliases {
                if self.aliases.get(&alias) == Some(&capability.id) {
                    self.aliases.remove(&alias);
                }
            }
        }

        let mut aliases: HashSet<String> = capability.aliases.iter().cloned().collect();
        aliases.insert(capability.id.clone());
        for alias in &aliases {
            self.aliases.insert(alias.clone(), capability.id.clone());
        }
        self.aliases_by_id.insert(capability.id.clone(), aliases);
        self.by_id.insert(capability.id.clone(), capability);
    }

    pub fn all(&self) -> Vec<&ModelCapability> {
        let mut models: Vec<&ModelCapability> = self.by_id.values().collect();
        models.sort_by(|left, right| left.id.cmp(&right.id));
        models
    }

    pub fn registry_digest(&self) -> &str {
        &self.registry_digest
    }

    pub fn base_registry_digest(&self) -> &str {
        &self.base_registry_digest
    }

    pub fn resolve(&self, model: &str) -> ModelResolution {
        let requested = normalize_name(model);
        let mut exact = false;
        let canonical = if let Some(canonical) = self.aliases.get(&requested) {
            exact = true;
            Some(canonical.clone())
        } else {
            let matches: Vec<(&String, &String)> = self
                .aliases
                .iter()
                .filter(|(alias, _)| {
                    is_prefix_alias(alias) && requested.starts_with(alias.as_str())
                })
                .collect();
            if matches.is_empty() {
                None
            } else {
                let longest = matches
                    .iter()
                    .map(|(alias, _)| alias.len())
                    .max()
                    .unwrap_or(0);
                let candidate_ids: HashSet<String> = matches
                    .into_iter()
                    .filter(|(alias, _)| alias.len() == longest)
                    .map(|(_, canonical)| canonical.clone())
                    .collect();
                if candidate_ids.len() == 1 {
                    candidate_ids.into_iter().next()
                } else {
                    return self.fallback_resolution(
                        model,
                        format!(
                            "Ambiguous model prefix {:?}; matched {}. Add an exact alias override.",
                            model,
                            sorted_join(candidate_ids)
                        ),
                    );
                }
            }
        };

        let Some(canonical) = canonical else {
            return self.fallback_resolution(
                model,
                format!(
                    "Unknown model {:?}; using conservative {}-token fallback. Add verified metadata or an override.",
                    model, self.fallback_context_window
                ),
            );
        };
        let capability = self
            .by_id
            .get(&canonical)
            .expect("alias index must reference an installed capability")
            .clone();
        if let Some(context_window) = capability.context_window {
            ModelResolution {
                requested_model: model.to_string(),
                capability: Some(capability.clone()),
                context_window,
                exact,
                trust: capability.trust,
                warning: None,
                registry_digest: self.registry_digest.clone(),
                base_registry_digest: self.base_registry_digest.clone(),
            }
        } else {
            ModelResolution {
                requested_model: model.to_string(),
                capability: Some(capability.clone()),
                context_window: self.fallback_context_window,
                exact,
                trust: capability.trust,
                warning: Some(format!(
                    "Model {:?} is recognized but its context window is unverified; using conservative {}-token fallback.",
                    capability.id, self.fallback_context_window
                )),
                registry_digest: self.registry_digest.clone(),
                base_registry_digest: self.base_registry_digest.clone(),
            }
        }
    }

    fn fallback_resolution(&self, model: &str, warning: String) -> ModelResolution {
        ModelResolution {
            requested_model: model.to_string(),
            capability: None,
            context_window: self.fallback_context_window,
            exact: false,
            trust: RegistryTrust::Fallback,
            warning: Some(warning),
            registry_digest: self.registry_digest.clone(),
            base_registry_digest: self.base_registry_digest.clone(),
        }
    }

    fn compute_effective_digest(&self) -> Result<String, String> {
        let models: Vec<Value> = self
            .all()
            .into_iter()
            .map(ModelCapability::fingerprint_value)
            .collect();
        let canonical = serde_json::to_vec(&json!({
            "fallback_context_window": self.fallback_context_window,
            "models": models,
        }))
        .map_err(|error| format!("cannot serialize model registry fingerprint: {error}"))?;
        Ok(sha256_hex(&canonical))
    }
}

fn normalize_name(value: &str) -> String {
    value.trim().to_lowercase()
}

fn is_prefix_alias(alias: &str) -> bool {
    matches!(
        alias.as_bytes().last(),
        Some(b'-') | Some(b'/') | Some(b':') | Some(b'.')
    )
}

fn sorted_join(values: HashSet<String>) -> String {
    let mut values: Vec<String> = values.into_iter().collect();
    values.sort();
    values.join(", ")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_exact_and_explicit_prefix_aliases() {
        let registry = ModelRegistry::bundled().unwrap();
        let exact = registry.resolve("gemini-2.5-pro");
        assert!(exact.exact);
        assert_eq!(exact.context_window, 1_048_576);
        assert_eq!(exact.trust, RegistryTrust::Verified);

        let dated = registry.resolve("gpt-4o-2024-08-06");
        assert_eq!(dated.model_id(), "openai/gpt-4o");
        assert_eq!(dated.context_window, 128_000);
        assert!(!dated.exact);

        let unrelated = registry.resolve("gpt-4xyz");
        assert_eq!(unrelated.trust, RegistryTrust::Fallback);
        assert!(unrelated.capability.is_none());
    }

    #[test]
    fn resolves_verified_gpt_5_6_metadata() {
        let registry = ModelRegistry::bundled().unwrap();
        let resolution = registry.resolve("gpt-5.6-sol");
        assert_eq!(resolution.model_id(), "openai/gpt-5.6-sol");
        assert_eq!(resolution.trust, RegistryTrust::Verified);
        assert_eq!(resolution.context_window, 1_050_000);
        assert!(resolution.warning.is_none());
    }

    #[test]
    fn resolves_nemotron_native_context_safely() {
        let registry = ModelRegistry::bundled().unwrap();
        let resolution = registry.resolve("nemotron-3-ultra");
        assert_eq!(resolution.model_id(), "nvidia/nemotron-3-ultra-550b-a55b");
        assert_eq!(resolution.context_window, 262_144);

        let provisional = registry.resolve("nemotron-super");
        assert_eq!(provisional.trust, RegistryTrust::Announced);
        assert_eq!(provisional.context_window, DEFAULT_FALLBACK_CONTEXT);
    }

    #[test]
    fn reserves_output_capacity_and_safety_margin() {
        let registry = ModelRegistry::bundled().unwrap();
        let resolution = registry.resolve("openai/o1");
        assert_eq!(
            resolution
                .effective_input_budget(Some(20_000), 0.05, 512)
                .unwrap(),
            170_000
        );
    }

    #[test]
    fn fingerprints_are_stable_sha256_values() {
        let first = ModelRegistry::bundled().unwrap();
        let second = ModelRegistry::bundled().unwrap();
        assert_eq!(first.registry_digest(), second.registry_digest());
        assert_eq!(first.registry_digest().len(), 64);
        assert_eq!(first.base_registry_digest().len(), 64);
        assert!(first
            .registry_digest()
            .chars()
            .all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn override_replaces_stale_alias_and_changes_effective_digest() {
        let mut registry = ModelRegistry::bundled().unwrap();
        let before = registry.registry_digest().to_string();
        registry
            .install_override(ModelCapability {
                id: "openai/gpt-4".into(),
                provider: "openai".into(),
                aliases: vec!["gpt-four".into()],
                context_window: Some(16_384),
                max_output_tokens: None,
                supports_tools: Some(true),
                supports_vision: None,
                supports_reasoning: None,
                reasoning_levels: vec![],
                input_price_per_million: None,
                output_price_per_million: None,
                trust: RegistryTrust::User,
                source: "unit-test".into(),
                verified_at: None,
                observed_at: None,
            })
            .unwrap();

        assert_eq!(registry.resolve("gpt-four").context_window, 16_384);
        assert_eq!(registry.resolve("gpt-4").trust, RegistryTrust::Fallback);
        assert_ne!(before, registry.registry_digest());
    }
}
