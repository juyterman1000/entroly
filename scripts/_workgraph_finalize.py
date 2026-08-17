from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    target.write_text(text + addition, encoding="utf-8")


# PyO3: binding exists, but it must be declared and registered in the module users import.
replace_once(
    "entroly-core/src/lib.rs",
    "mod witness;\n",
    "mod witness;\nmod work_graph_bindings;\n",
)
replace_once(
    "entroly-core/src/lib.rs",
    "    m.add_class::<PyDedupIndex>()?;\n",
    "    m.add_class::<PyDedupIndex>()?;\n    work_graph_bindings::register(m)?;\n",
)

# WASM: make the thin boundary part of the actual crate root, not an orphan source file.
replace_once(
    "entroly-wasm/src/lib.rs",
    "pub(crate) use entroly_engine::utilization;\n",
    "pub(crate) use entroly_engine::utilization;\nmod work_graph_bindings;\npub use work_graph_bindings::WasmWorkGraph;\n",
)

# JS orchestration must stay Number-friendly even though wasm-bindgen maps raw i64 to BigInt.
replace_once(
    "entroly-wasm/src/work_graph_bindings.rs",
    "use wasm_bindgen::prelude::*;\n",
    "use wasm_bindgen::prelude::*;\n\nfn js_safe_i64(value: f64, name: &str) -> Result<i64, JsValue> {\n    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;\n    if !value.is_finite() || value.fract() != 0.0 || value.abs() > MAX_SAFE_INTEGER {\n        return Err(JsValue::from_str(&format!(\n            \"{name} must be a finite JavaScript-safe integer\"\n        )));\n    }\n    Ok(value as i64)\n}\n",
)
replace_once(
    "entroly-wasm/src/work_graph_bindings.rs",
    "    pub fn coordination_json(&self, now_ms: i64, pretty: bool) -> Result<String, JsValue> {\n        self.inner.coordination_json(now_ms, pretty).map_err(js_err)\n",
    "    pub fn coordination_json(&self, now_ms: f64, pretty: bool) -> Result<String, JsValue> {\n        let now_ms = js_safe_i64(now_ms, \"now_ms\")?;\n        self.inner.coordination_json(now_ms, pretty).map_err(js_err)\n",
)
replace_once(
    "entroly-wasm/src/work_graph_bindings.rs",
    "        generated_at_ms: i64,\n        pretty: bool,\n    ) -> Result<String, JsValue> {\n        self.inner\n            .handoff_json(\n                workstream_id,\n                from_agent,\n                to_agent,\n                generated_at_ms,\n",
    "        generated_at_ms: f64,\n        pretty: bool,\n    ) -> Result<String, JsValue> {\n        let generated_at_ms = js_safe_i64(generated_at_ms, \"generated_at_ms\")?;\n        self.inner\n            .handoff_json(\n                workstream_id,\n                from_agent,\n                to_agent,\n                generated_at_ms,\n",
)

# Node wrapper: transport validation only. Work-state semantics remain in Rust.
replace_once(
    "entroly-wasm/js/work_graph.js",
    "function fromJSONText(value) {\n  return JSON.parse(value);\n}\n",
    "function fromJSONText(value) {\n  return JSON.parse(value);\n}\n\nfunction requireSafeInteger(value, name, { min = Number.MIN_SAFE_INTEGER } = {}) {\n  const number = Number(value);\n  if (!Number.isSafeInteger(number) || number < min) {\n    throw new TypeError(`${name} must be a safe integer${min === 0 ? ' >= 0' : ''}`);\n  }\n  return number;\n}\n",
)
replace_once(
    "entroly-wasm/js/work_graph.js",
    "    return fromJSONText(this._inner.resumeJSON(id, Number(maxEvidence), Boolean(pretty)));\n",
    "    return fromJSONText(this._inner.resumeJSON(\n      id,\n      requireSafeInteger(maxEvidence, 'maxEvidence', { min: 0 }),\n      Boolean(pretty),\n    ));\n",
)
replace_once(
    "entroly-wasm/js/work_graph.js",
    "    return fromJSONText(this._inner.coordinationJSON(Number(nowMs), Boolean(pretty)));\n",
    "    return fromJSONText(this._inner.coordinationJSON(\n      requireSafeInteger(nowMs, 'nowMs'),\n      Boolean(pretty),\n    ));\n",
)
replace_once(
    "entroly-wasm/js/work_graph.js",
    "      Number(generatedAtMs),\n      Boolean(pretty),\n",
    "      requireSafeInteger(generatedAtMs, 'generatedAtMs'),\n      Boolean(pretty),\n",
)

# Root npm surface: users should not need a deep import.
replace_once(
    "entroly-wasm/index.js",
    "const { EntrolyConfig } = require('./js/config');\n",
    "const { EntrolyConfig } = require('./js/config');\nconst { WorkGraph } = require('./js/work_graph');\n",
)
replace_once(
    "entroly-wasm/index.js",
    "  EntrolyEngine: WasmEntrolyEngine,\n  WasmEntrolyEngine,\n",
    "  EntrolyEngine: WasmEntrolyEngine,\n  WasmEntrolyEngine,\n  WorkGraph,\n",
)
replace_once(
    "entroly-wasm/index.d.ts",
    'export * from "./pkg/entroly_wasm";\n',
    'export * from "./pkg/entroly_wasm";\nexport * from "./js/work_graph";\n',
)

# Published npm tarball and actual npm test contract.
package_path = ROOT / "entroly-wasm/package.json"
package = json.loads(package_path.read_text(encoding="utf-8"))
files = package.setdefault("files", [])
for name in ("js/work_graph.js", "js/work_graph.d.ts"):
    if name not in files:
        files.append(name)
package.setdefault("scripts", {})["test"] = "node test_wasm_e2e.js && node test_work_graph.js"
package_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# Work Graph npm contract must fail on unsafe numeric transport, not silently round timestamps.
replace_once(
    "entroly-wasm/test_work_graph.js",
    "assert(!first.verifyHandoff(receipt), 'tampered receipt passed graph-bound verification');\n\nconsole.log('Work Graph npm contract: PASS');\n",
    "assert(!first.verifyHandoff(receipt), 'tampered receipt passed graph-bound verification');\n\nlet unsafeTimestampRejected = false;\ntry { first.coordination(Number.MAX_SAFE_INTEGER + 1); }\ncatch (_) { unsafeTimestampRejected = true; }\nassert(unsafeTimestampRejected, 'unsafe timestamp was accepted by npm/WASM boundary');\n\nlet negativeEvidenceRejected = false;\ntry { first.resume(unfinished[0].node_id, -1); }\ncatch (_) { negativeEvidenceRejected = true; }\nassert(negativeEvidenceRejected, 'negative maxEvidence was accepted');\n\nconsole.log('Work Graph npm contract: PASS');\n",
)

# Python root discoverability. The wrapper itself tolerates native absence so fallback installs import safely.
append_once(
    "entroly/__init__.py",
    "AI Work Graph — shared Rust temporal work-state engine",
    "\n# AI Work Graph — shared Rust temporal work-state engine. Python remains a\n# thin orchestration layer over entroly-engine.\ntry:\n    from .work_graph import WorkGraph, WorkGraphUnavailableError  # noqa: F401\nexcept ImportError:\n    pass\n",
)

print("Work Graph binding finalization applied successfully")
