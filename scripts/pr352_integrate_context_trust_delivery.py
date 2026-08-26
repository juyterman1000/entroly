#!/usr/bin/env python3
"""Guarded one-shot integration of PR #352 Context/Trust Rust seams.

Shared semantics already exist in entroly-engine. This transform makes those
semantics reachable from Python and npm without reimplementing them in host
languages. Every replacement is anchor-count guarded; a concurrent semantic
edit aborts rather than being guessed over.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    wg = read("entroly-engine/src/work_graph.rs")
    if "pub fn context_scope_json(" in wg:
        print("Context/Trust delivery integration already applied")
        return 0

    # 1. Rust Work Graph -> bounded Context contract seam.
    wg = replace_once(
        wg,
        "use crate::coordination_index::{candidate_pairs, CoordinationScope};\n",
        "use crate::coordination_index::{candidate_pairs, CoordinationScope};\n"
        "use crate::engine_contracts::WorkScope;\n",
        "work graph context import",
    )
    anchor = '''    pub fn resume_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> Result<String, WorkGraphError> {
        let view = self.resume(workstream_id, max_evidence)?;
        if pretty {
            serde_json::to_string_pretty(&view).map_err(Into::into)
        } else {
            serde_json::to_string(&view).map_err(Into::into)
        }
    }

'''
    addition = anchor + '''    /// Derive the bounded, text-light Context/Trust integration scope from the
    /// exact Rust-owned resume view. Raw decision/failure prose and context
    /// bytes remain in their owning stores and never become graph payload here.
    pub fn context_scope(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
    ) -> Result<WorkScope, WorkGraphError> {
        let view = self.resume(workstream_id, max_evidence)?;
        WorkScope::from_resume(&view)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))
    }

    pub fn context_scope_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> Result<String, WorkGraphError> {
        let scope = self.context_scope(workstream_id, max_evidence)?;
        if pretty {
            serde_json::to_string_pretty(&scope).map_err(Into::into)
        } else {
            serde_json::to_string(&scope).map_err(Into::into)
        }
    }

'''
    wg = replace_once(wg, anchor, addition, "context scope methods")
    write("entroly-engine/src/work_graph.rs", wg)

    # 2. Keep criticality label conversion canonical in Rust Trust Engine.
    trust = read("entroly-engine/src/trust_engine.rs")
    trust = replace_once(
        trust,
        '''    pub fn file_criticality(&self, path: &str) -> Criticality {
        file_criticality(path)
    }

''',
        '''    pub fn file_criticality(&self, path: &str) -> Criticality {
        file_criticality(path)
    }

    /// Stable lowercase transport label for file criticality. Bindings call
    /// this rather than duplicating enum-to-string policy in Python and WASM.
    pub fn file_criticality_label(&self, path: &str) -> &'static str {
        match self.file_criticality(path) {
            Criticality::Normal => "normal",
            Criticality::Important => "important",
            Criticality::Critical => "critical",
            Criticality::Safety => "safety",
        }
    }

''',
        "Trust criticality label",
    )
    trust = replace_once(
        trust,
        '''    fn guardrail_facade_preserves_existing_policy() {
        let engine = TrustEngine::default();
        assert_eq!(
            engine.file_criticality("file:SECURITY.md"),
            file_criticality("file:SECURITY.md")
        );
        assert_eq!(
            engine.has_safety_signal("AWS_SECRET_ACCESS_KEY=example"),
            has_safety_signal("AWS_SECRET_ACCESS_KEY=example")
        );
    }
''',
        '''    fn guardrail_facade_preserves_existing_policy() {
        let engine = TrustEngine::default();
        assert_eq!(
            engine.file_criticality("file:SECURITY.md"),
            file_criticality("file:SECURITY.md")
        );
        assert_eq!(engine.file_criticality_label("file:SECURITY.md"), "safety");
        assert_eq!(
            engine.has_safety_signal("AWS_SECRET_ACCESS_KEY=example"),
            has_safety_signal("AWS_SECRET_ACCESS_KEY=example")
        );
    }
''',
        "Trust label regression",
    )
    write("entroly-engine/src/trust_engine.rs", trust)

    # 3. PyO3 Work Graph context-scope method.
    core_wg = read("entroly-core/src/work_graph_bindings.rs")
    core_wg = replace_once(
        core_wg,
        '''    #[pyo3(signature = (now_ms, pretty = false))]
    fn coordination_json(&self, now_ms: i64, pretty: bool) -> PyResult<String> {
''',
        '''    #[pyo3(signature = (workstream_id = None, max_evidence = 128, pretty = false))]
    fn context_scope_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> PyResult<String> {
        self.inner
            .context_scope_json(workstream_id, max_evidence, pretty)
            .map_err(py_err)
    }

    #[pyo3(signature = (now_ms, pretty = false))]
    fn coordination_json(&self, now_ms: i64, pretty: bool) -> PyResult<String> {
''',
        "PyO3 context scope",
    )
    write("entroly-core/src/work_graph_bindings.rs", core_wg)

    # 4. Dedicated thin PyO3 Trust binding.
    write("entroly-core/src/trust_engine_bindings.rs", r'''//! Thin PyO3 boundary over `entroly_engine::trust_engine`.
//!
//! Claim support, policy validation, commitments, and guardrail semantics stay
//! in Rust. This file only converts transport values and serializes results.

use entroly_engine::trust_engine::TrustEngine;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_err(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "TrustEngine", module = "entroly_core")]
pub(crate) struct PyTrustEngine {
    inner: TrustEngine,
}

#[pymethods]
impl PyTrustEngine {
    #[new]
    #[pyo3(signature = (profile = "rag"))]
    fn new(profile: &str) -> PyResult<Self> {
        Ok(Self {
            inner: TrustEngine::try_new(profile).map_err(py_err)?,
        })
    }

    fn assess_claim_json(&self, evidence: &str, claim: &str) -> PyResult<String> {
        serde_json::to_string(&self.inner.assess_claim_support(evidence, claim)).map_err(py_err)
    }

    fn file_criticality(&self, path: &str) -> String {
        self.inner.file_criticality_label(path).to_string()
    }

    fn has_safety_signal(&self, content: &str) -> bool {
        self.inner.has_safety_signal(content)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrustEngine>()
}
''')

    core_lib = read("entroly-core/src/lib.rs")
    core_lib = replace_once(
        core_lib,
        "mod telemetry;\npub(crate) use entroly_engine::trajectory;\n",
        "mod telemetry;\nmod trust_engine_bindings;\npub(crate) use entroly_engine::trajectory;\n",
        "core Trust module",
    )
    core_lib = replace_once(
        core_lib,
        "    work_graph_bindings::register(m)?;\n",
        "    work_graph_bindings::register(m)?;\n    trust_engine_bindings::register(m)?;\n",
        "core Trust registration",
    )
    write("entroly-core/src/lib.rs", core_lib)

    # 5. WASM Work Graph context scope and Trust binding.
    wasm_wg = read("entroly-wasm/src/work_graph_bindings.rs")
    wasm_wg = replace_once(
        wasm_wg,
        '''    #[wasm_bindgen(js_name = coordinationJSON)]
    pub fn coordination_json(&self, now_ms: f64, pretty: bool) -> Result<String, JsValue> {
''',
        '''    #[wasm_bindgen(js_name = contextScopeJSON)]
    pub fn context_scope_json(
        &self,
        workstream_id: Option<String>,
        max_evidence: usize,
        pretty: bool,
    ) -> Result<String, JsValue> {
        self.inner
            .context_scope_json(workstream_id.as_deref(), max_evidence, pretty)
            .map_err(js_err)
    }

    #[wasm_bindgen(js_name = coordinationJSON)]
    pub fn coordination_json(&self, now_ms: f64, pretty: bool) -> Result<String, JsValue> {
''',
        "WASM context scope",
    )
    write("entroly-wasm/src/work_graph_bindings.rs", wasm_wg)

    write("entroly-wasm/src/trust_engine_bindings.rs", r'''//! Thin wasm-bindgen boundary over `entroly_engine::trust_engine`.

use entroly_engine::trust_engine::TrustEngine;
use wasm_bindgen::prelude::*;

fn js_err(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
pub struct WasmTrustEngine {
    inner: TrustEngine,
}

#[wasm_bindgen]
impl WasmTrustEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(profile: &str) -> Result<WasmTrustEngine, JsValue> {
        Ok(Self {
            inner: TrustEngine::try_new(profile).map_err(js_err)?,
        })
    }

    #[wasm_bindgen(js_name = assessClaimJSON)]
    pub fn assess_claim_json(&self, evidence: &str, claim: &str) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.assess_claim_support(evidence, claim)).map_err(js_err)
    }

    #[wasm_bindgen(js_name = fileCriticality)]
    pub fn file_criticality(&self, path: &str) -> String {
        self.inner.file_criticality_label(path).to_string()
    }

    #[wasm_bindgen(js_name = hasSafetySignal)]
    pub fn has_safety_signal(&self, content: &str) -> bool {
        self.inner.has_safety_signal(content)
    }
}
''')

    wasm_lib = read("entroly-wasm/src/lib.rs")
    wasm_lib = replace_once(
        wasm_lib,
        "pub(crate) use entroly_engine::trajectory;\npub(crate) use entroly_engine::utilization;\nmod work_graph_bindings;\n",
        "pub(crate) use entroly_engine::trajectory;\npub(crate) use entroly_engine::utilization;\nmod trust_engine_bindings;\nmod work_graph_bindings;\n",
        "WASM Trust module",
    )
    write("entroly-wasm/src/lib.rs", wasm_lib)

    # 6. Python host wrappers remain conversion-only.
    py_wg = read("entroly/work_graph.py")
    py_wg = replace_once(
        py_wg,
        '''    def coordination(self, now_ms: int, *, pretty: bool = False) -> dict[str, Any]:
        return _json_value(str(self._inner.coordination_json(now_ms, pretty)))

''',
        '''    def context_scope(
        self,
        workstream_id: str | None = None,
        *,
        max_evidence: int = 128,
        pretty: bool = False,
    ) -> dict[str, Any]:
        """Bounded Rust-owned scope for Context/Trust decisions."""
        return _json_value(
            str(self._inner.context_scope_json(workstream_id, max_evidence, pretty))
        )

    def coordination(self, now_ms: int, *, pretty: bool = False) -> dict[str, Any]:
        return _json_value(str(self._inner.coordination_json(now_ms, pretty)))

''',
        "Python context scope",
    )
    write("entroly/work_graph.py", py_wg)

    write("entroly/trust.py", '''"""Thin Python surface for Entroly's shared Rust Trust Engine.

No evidence scoring, profile policy, commitment, or criticality semantics live
here. Python validates native capability, converts JSON, and returns Rust-owned
results.
"""

from __future__ import annotations

import json
from typing import Any

from .native_status import native_status, native_status_message

_TRUST_STATUS = native_status(("TrustEngine",))
_RustTrustEngine = (
    getattr(_TRUST_STATUS.module, "TrustEngine", None) if _TRUST_STATUS.ok else None
)


class TrustEngineUnavailableError(RuntimeError):
    """Raised when the native shared Trust Engine is unavailable."""


def _require_native() -> type:
    if _RustTrustEngine is None:
        raise TrustEngineUnavailableError(
            native_status_message(_TRUST_STATUS, feature="the Entroly Trust Engine")
        )
    return _RustTrustEngine


class TrustEngine:
    """Evidence-bounded trust facade backed entirely by shared Rust semantics."""

    __slots__ = ("_inner",)

    def __init__(self, profile: str = "rag") -> None:
        self._inner = _require_native()(profile)

    def assess_claim(self, evidence: str, claim: str) -> dict[str, Any]:
        return json.loads(str(self._inner.assess_claim_json(evidence, claim)))

    def file_criticality(self, path: str) -> str:
        return str(self._inner.file_criticality(path))

    def has_safety_signal(self, content: str) -> bool:
        return bool(self._inner.has_safety_signal(content))


__all__ = ["TrustEngine", "TrustEngineUnavailableError"]
''')

    py_init = read("entroly/__init__.py")
    py_init = replace_once(
        py_init,
        '''# EICV — Evidence-Invariant Causal Verification.
''',
        '''# Evidence-bounded Trust Engine — shared Rust truth with thin Python transport.
try:
    from .trust import TrustEngine, TrustEngineUnavailableError  # noqa: F401
except ImportError:
    pass

# EICV — Evidence-Invariant Causal Verification.
''',
        "Python root Trust export",
    )
    write("entroly/__init__.py", py_init)

    # 7. Node thin wrappers/types/root export.
    js_wg = read("entroly-wasm/js/work_graph.js")
    js_wg = replace_once(
        js_wg,
        '''  coordination(nowMs = Date.now(), pretty = false) {
''',
        '''  contextScope(workstreamId = null, maxEvidence = 128, pretty = false) {
    const id = workstreamId == null ? undefined : String(workstreamId);
    return fromJSONText(this._inner.contextScopeJSON(
      id,
      requireSafeInteger(maxEvidence, 'maxEvidence', { min: 0 }),
      Boolean(pretty),
    ));
  }

  coordination(nowMs = Date.now(), pretty = false) {
''',
        "Node context scope",
    )
    write("entroly-wasm/js/work_graph.js", js_wg)

    wg_dts = read("entroly-wasm/js/work_graph.d.ts")
    wg_dts = replace_once(
        wg_dts,
        '''export interface WorkGraphCoordinationConflict {
''',
        '''export interface WorkGraphContextScope {
  repo_id: string;
  graph_revision: number;
  graph_commitment: string;
  workstream_id: string;
  task_ids: string[];
  agent_ids: string[];
  changed_paths: string[];
  commit_ids: string[];
  evidence_ids: string[];
}

export interface WorkGraphCoordinationConflict {
''',
        "Context scope TypeScript type",
    )
    wg_dts = replace_once(
        wg_dts,
        '''  resume(workstreamId?: string | null, maxEvidence?: number, pretty?: boolean): WorkGraphResumeView;
  coordination(nowMs?: number, pretty?: boolean): WorkGraphCoordinationReport;
''',
        '''  resume(workstreamId?: string | null, maxEvidence?: number, pretty?: boolean): WorkGraphResumeView;
  contextScope(workstreamId?: string | null, maxEvidence?: number, pretty?: boolean): WorkGraphContextScope;
  coordination(nowMs?: number, pretty?: boolean): WorkGraphCoordinationReport;
''',
        "Context scope TypeScript method",
    )
    write("entroly-wasm/js/work_graph.d.ts", wg_dts)

    write("entroly-wasm/js/trust_engine.js", ''''use strict';

// Thin Node transport over the shared Rust/WASM Trust Engine. All profile,
// scoring, commitment, and guardrail semantics stay in entroly-engine.
function wasmTrustEngine() {
  const { WasmTrustEngine } = require('../pkg/entroly_wasm');
  if (!WasmTrustEngine) {
    throw new Error('Rust Trust Engine is unavailable; rebuild entroly-wasm');
  }
  return WasmTrustEngine;
}

class TrustEngine {
  constructor(profile = 'rag') {
    this._inner = new (wasmTrustEngine())(String(profile));
  }

  assessClaim(evidence, claim) {
    return JSON.parse(this._inner.assessClaimJSON(String(evidence), String(claim)));
  }

  fileCriticality(path) {
    return this._inner.fileCriticality(String(path));
  }

  hasSafetySignal(content) {
    return Boolean(this._inner.hasSafetySignal(String(content)));
  }
}

module.exports = { TrustEngine };
''')
    write("entroly-wasm/js/trust_engine.d.ts", '''export type EvidenceSupportStatus = "supported" | "unsupported" | "unknown";

export interface ClaimEvidenceAssessment {
  status: EvidenceSupportStatus;
  support_density: number;
  unsupported_fraction: number;
  contradiction_fraction: number;
  evidence_commitment: string;
}

export type FileCriticality = "normal" | "important" | "critical" | "safety";

export class TrustEngine {
  constructor(profile?: "rag" | "qa" | "summarization" | "dialogue" | "fact_check" | "default" | string);
  assessClaim(evidence: string, claim: string): ClaimEvidenceAssessment;
  fileCriticality(path: string): FileCriticality;
  hasSafetySignal(content: string): boolean;
}
''')

    index_js = read("entroly-wasm/index.js")
    index_js = replace_once(
        index_js,
        "let WasmEntrolyEngine;\nlet WasmWorkGraph;\n",
        "let WasmEntrolyEngine;\nlet WasmWorkGraph;\nlet WasmTrustEngine;\n",
        "npm wasm Trust variable",
    )
    index_js = replace_once(
        index_js,
        '''    WasmEntrolyEngine,
    WasmWorkGraph,
''',
        '''    WasmEntrolyEngine,
    WasmWorkGraph,
    WasmTrustEngine,
''',
        "npm wasm Trust binding",
    )
    index_js = replace_once(
        index_js,
        '''    !WasmWorkGraph ||
    !classifyQueryTransitionRust ||
''',
        '''    !WasmWorkGraph ||
    !WasmTrustEngine ||
    !classifyQueryTransitionRust ||
''',
        "npm stale pkg Trust gate",
    )
    index_js = replace_once(
        index_js,
        '''const {
  WorkGraph,
  RepositoryDiscoveryError,
''',
        '''const { TrustEngine } = require('./js/trust_engine');
const {
  WorkGraph,
  RepositoryDiscoveryError,
''',
        "npm Trust require",
    )
    index_js = replace_once(
        index_js,
        '''  // Shared Rust AI Work Graph with Node-only repository/persistence glue.
  WorkGraph,
''',
        '''  // Shared Rust evidence-bounded Trust Engine.
  TrustEngine,

  // Shared Rust AI Work Graph with Node-only repository/persistence glue.
  WorkGraph,
''',
        "npm Trust export",
    )
    write("entroly-wasm/index.js", index_js)

    index_dts = read("entroly-wasm/index.d.ts")
    index_dts = replace_once(
        index_dts,
        '''export * from "./js/work_graph";
''',
        '''export * from "./js/trust_engine";
export * from "./js/work_graph";
''',
        "npm Trust types export",
    )
    write("entroly-wasm/index.d.ts", index_dts)

    package = read("entroly-wasm/package.json")
    package = replace_once(
        package,
        '''    "js/evolution_daemon.js",
    "js/work_graph.js",
''',
        '''    "js/evolution_daemon.js",
    "js/trust_engine.js",
    "js/trust_engine.d.ts",
    "js/work_graph.js",
''',
        "npm Trust package files",
    )
    package = replace_once(
        package,
        '''"node test_wasm_e2e.js && node test_work_graph.js && node test_work_graph_repo.js && node test_work_graph_store.js && node test_work_graph_content_digest.js && node test_work_graph_continuity.js && node test_work_graph_root_exports.js"''',
        '''"node test_wasm_e2e.js && node test_work_graph.js && node test_work_graph_repo.js && node test_work_graph_store.js && node test_work_graph_content_digest.js && node test_work_graph_continuity.js && node test_context_trust_delivery.js && node test_work_graph_root_exports.js"''',
        "npm delivery test command",
    )
    write("entroly-wasm/package.json", package)

    # 8. Persistent Python and Node delivery tests.
    write("tests/test_context_trust_delivery.py", r'''"""Production delivery tests for the Rust-owned Context/Trust seams."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from entroly.trust import TrustEngine
from entroly.work_graph import WorkGraph


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _dirty_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Entroly Test")
    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text("def auth():\n    return True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    (repo / "src" / "auth.py").write_text("def auth():\n    return False\n", encoding="utf-8")
    return repo


def test_work_graph_context_scope_is_bounded_and_text_light(tmp_path: Path) -> None:
    repo = _dirty_repo(tmp_path)
    graph = WorkGraph.from_repository(
        str(repo),
        agent_id="agent:test",
        session_id="session:test",
        task_hint={
            "task_id": "task:auth",
            "title": "repair auth",
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": ["run tests"],
            "source_kind": "user_statement",
            "source_ref": "test://task",
        },
        include_checkpoint=False,
        observed_at_ms=1234,
    )
    scope = graph.context_scope()
    assert scope["repo_id"] == graph.repo_id
    assert scope["graph_revision"] == graph.revision
    assert scope["graph_commitment"] == graph.graph_commitment
    assert scope["workstream_id"]
    assert "src/auth.py" in scope["changed_paths"]
    assert scope["task_ids"] == sorted(set(scope["task_ids"]))
    assert scope["agent_ids"] == sorted(set(scope["agent_ids"]))
    payload = json.dumps(scope, sort_keys=True)
    assert "repair auth" not in payload
    assert "run tests" not in payload
    assert "selected_context" not in payload


def test_trust_engine_is_evidence_bounded_and_fail_closed() -> None:
    evidence = "The service retries a request three times before returning an error."
    claim = "The service retries a request three times."
    engine = TrustEngine("rag")
    assessment = engine.assess_claim(evidence, claim)
    assert assessment["status"] in {"supported", "unsupported", "unknown"}
    assert assessment["evidence_commitment"] == "sha256:" + hashlib.sha256(
        evidence.encode("utf-8")
    ).hexdigest()
    assert 0.0 <= assessment["support_density"] <= 1.0
    assert engine.file_criticality("file:SECURITY.md") == "safety"
    assert engine.has_safety_signal("AWS_SECRET_ACCESS_KEY=example")
    with pytest.raises(ValueError):
        TrustEngine("rga")
''')

    write("entroly-wasm/test_context_trust_delivery.js", r'''#!/usr/bin/env node
'use strict';

const assert = require('assert');
const crypto = require('crypto');
const { TrustEngine, WorkGraph } = require('./index');

function observation() {
  return {
    repo_id: 'repo:context-trust-test',
    observed_at_ms: 1234,
    repository_label: 'demo',
    agent_id: 'agent:test',
    session_id: 'session:test',
    task_hint: {
      task_id: 'task:auth',
      title: 'repair auth',
      trust: 'observed',
      explicit_status: 'in_progress',
      remaining_work: ['run tests'],
      source_kind: 'user_statement',
      source_ref: 'test://task',
    },
    branch: {
      name: 'main', head_sha: 'abc', base_ref: 'refs/heads/main', default_branch: 'main',
      ahead_by: 0, behind_by: 0, merge_in_progress: false, rebase_in_progress: false,
      detached: false,
    },
    changes: [{ path: 'src/auth.py', kind: 'modified', staged: false, conflicted: false, old_path: '' }],
    commits: [], verifications: [], decisions: [], claims: [], leases: [], model_executions: [],
  };
}

const graph = new WorkGraph('repo:context-trust-test');
graph.observeRepository(observation());
const scope = graph.contextScope();
assert.strictEqual(scope.repo_id, graph.repoId);
assert.strictEqual(scope.graph_revision, graph.revision);
assert.strictEqual(scope.graph_commitment, graph.graphCommitment);
assert.ok(scope.changed_paths.includes('src/auth.py'));
const scopeText = JSON.stringify(scope);
assert.ok(!scopeText.includes('repair auth'));
assert.ok(!scopeText.includes('run tests'));

const evidence = 'The service retries a request three times before returning an error.';
const claim = 'The service retries a request three times.';
const trust = new TrustEngine('rag');
const assessment = trust.assessClaim(evidence, claim);
const digest = crypto.createHash('sha256').update(evidence, 'utf8').digest('hex');
assert.strictEqual(assessment.evidence_commitment, `sha256:${digest}`);
assert.ok(['supported', 'unsupported', 'unknown'].includes(assessment.status));
assert.strictEqual(trust.fileCriticality('file:SECURITY.md'), 'safety');
assert.strictEqual(trust.hasSafetySignal('AWS_SECRET_ACCESS_KEY=example'), true);
assert.throws(() => new TrustEngine('rga'));
console.log('Context/Trust delivery tests passed');
''')

    print("applied guarded Context/Trust delivery integration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
