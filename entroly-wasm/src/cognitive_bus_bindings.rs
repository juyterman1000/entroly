//! Thin wasm-bindgen boundary over the shared Cognitive Bus.
//!
//! No routing, novelty, spike-detection or salience logic belongs here. The ISA
//! priority model, Poisson rate estimation, Welford spike detection, per-agent
//! queue caps and the hippocampus bridge all live in
//! `entroly_engine::cognitive_bus`, and this file converts to and from JS.
//!
//! JavaScript receives the same canonical JSON that Python receives through
//! PyO3, because both sides call the engine's own serializers rather than
//! building their own payloads. That is the property that makes section 14
//! parity checkable instead of assumed.

use entroly_engine::cognitive_bus::CognitiveBus;
use wasm_bindgen::prelude::*;

fn js_err(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn to_json_string(value: &serde_json::Value) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(js_err)
}

#[wasm_bindgen]
pub struct WasmCognitiveBus {
    inner: CognitiveBus,
}

#[wasm_bindgen]
impl WasmCognitiveBus {
    /// `memorySalienceThreshold` matches the PyO3 default of 50.0.
    #[wasm_bindgen(constructor)]
    pub fn new(memory_salience_threshold: Option<f64>) -> WasmCognitiveBus {
        Self {
            inner: CognitiveBus::new(memory_salience_threshold.unwrap_or(50.0)),
        }
    }

    /// `eventTypesJSON` is a JSON array of event-type strings. Passing them as
    /// JSON rather than a `Box<[JsString]>` keeps one decoding path shared with
    /// every other binding here.
    #[wasm_bindgen(js_name = subscribe)]
    pub fn subscribe(&mut self, agent_id: &str, event_types_json: &str) -> Result<(), JsValue> {
        let event_types: Vec<String> = serde_json::from_str(event_types_json).map_err(js_err)?;
        self.inner.subscribe(agent_id, event_types);
        Ok(())
    }

    #[wasm_bindgen(js_name = unsubscribe)]
    pub fn unsubscribe(&mut self, agent_id: &str) {
        self.inner.unsubscribe(agent_id);
    }

    #[wasm_bindgen(js_name = setTaskContext)]
    pub fn set_task_context(&mut self, agent_id: &str, task_text: &str) {
        self.inner.set_task_context(agent_id, task_text);
    }

    #[wasm_bindgen(js_name = tick)]
    pub fn tick(&mut self) {
        self.inner.tick();
    }

    #[wasm_bindgen(js_name = setTick)]
    pub fn set_tick(&mut self, tick: f64) {
        self.inner.set_tick(tick);
    }

    /// Returns the number of subscribers the event was routed to, exactly as
    /// the PyO3 `publish` does.
    #[wasm_bindgen(js_name = publish)]
    pub fn publish(
        &mut self,
        source_agent: &str,
        event_type: &str,
        content: &str,
        emotional_tag: Option<u8>,
        salience: Option<f64>,
    ) -> usize {
        self.inner.publish(
            source_agent,
            event_type,
            content,
            emotional_tag.unwrap_or(0),
            salience.unwrap_or(0.0),
        )
    }

    /// Full event shape: `id`, `source_agent`, `event_type`, `content`,
    /// `timestamp`, `emotional_tag`, `salience`, `is_spike`.
    #[wasm_bindgen(js_name = drainJSON)]
    pub fn drain_json(&mut self, agent_id: &str, limit: Option<usize>) -> Result<String, JsValue> {
        let events = self.inner.drain(agent_id, limit.unwrap_or(10));
        to_json_string(&serde_json::Value::Array(events))
    }

    /// Hippocampus bridge shape: `content`, `source`, `salience`,
    /// `emotional_tag`, `event_type`. Deliberately narrower than `drainJSON`.
    #[wasm_bindgen(js_name = drainMemoryBridgeJSON)]
    pub fn drain_memory_bridge_json(&mut self) -> Result<String, JsValue> {
        let events = self.inner.drain_memory_bridge();
        to_json_string(&serde_json::Value::Array(events))
    }

    #[wasm_bindgen(js_name = queueDepth)]
    pub fn queue_depth(&self, agent_id: &str) -> usize {
        self.inner.queue_depth(agent_id)
    }

    #[wasm_bindgen(js_name = statsJSON)]
    pub fn stats_json(&self) -> Result<String, JsValue> {
        to_json_string(&self.inner.stats())
    }
}
