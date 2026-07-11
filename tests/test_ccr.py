import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.routing import Route

from entroly.ccr import (
    CompressedContextStore,
    capture_ranked_recovery_candidates,
    capture_recoverable_fragments,
    get_ccr_store,
    hierarchical_context_fragments,
    slice_recovery_content,
)
from entroly.proxy import _context_retrieve
from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig
from entroly.proxy_transform import (
    format_context_block,
    format_hierarchical_context,
)


def test_content_addressed_handles_preserve_historical_versions():
    store = CompressedContextStore()

    old_handle = store.store(
        source="file:auth.py",
        original_content="def login(): return 'old'",
        compressed_content="def login(): ...",
        original_tokens=8,
        compressed_tokens=4,
    )
    new_handle = store.store(
        source="file:auth.py",
        original_content="def login(): return 'new'",
        compressed_content="def login(): ...",
        original_tokens=8,
        compressed_tokens=4,
    )

    assert old_handle != new_handle
    assert store.retrieve(old_handle)["original"].endswith("'old'")
    assert store.retrieve(new_handle)["original"].endswith("'new'")
    assert store.retrieve("file:auth.py")["retrieval_handle"] == new_handle


def test_source_lookup_falls_back_when_refreshed_history_evicts_latest():
    store = CompressedContextStore(max_entries=2)
    old_handle = store.store("source", "old", "old", 1, 1)
    new_handle = store.store("source", "new", "new", 1, 1)
    assert store.retrieve(old_handle) is not None

    store.store("other", "other", "other", 1, 1)

    assert store.retrieve(new_handle) is None
    assert store.retrieve("source")["retrieval_handle"] == old_handle


def test_slice_recovery_content_preserves_lead_and_query_local_exact_window():
    content = (
        "Configuration reference\n"
        + ("introductory material\n" * 30)
        + "rotation_interval = 45\n"
        + ("trailing material\n" * 20)
    )

    sliced, was_sliced = slice_recovery_content(
        content,
        query="find rotation_interval",
        token_budget=64,
    )

    assert was_sliced is True
    assert len(sliced) <= 64 * 4
    assert sliced.startswith("Configuration reference")
    assert "rotation_interval = 45" in sliced
    assert "[... exact excerpt gap; retrieve full source by handle ...]" in sliced


def test_capture_recoverable_fragments_attaches_exact_handle():
    store = CompressedContextStore()
    originals = {
        "auth-1": {
            "id": "auth-1",
            "source": "file:auth.py",
            "content": "def login(token):\n    return validate(token)\n",
            "token_count": 12,
            "entropy_score": 0.8,
        }
    }
    selected = [
        {
            "id": "auth-1",
            "source": "file:auth.py",
            "content": "def login(token): ...",
            "token_count": 5,
            "variant": "skeleton",
            "relevance": 0.9,
            "entropy_score": 0.8,
        }
    ]

    handles = capture_recoverable_fragments(
        selected, originals.get, store=store
    )

    assert handles == [selected[0]["retrieval_handle"]]
    assert selected[0]["recoverable"] is True
    recovered = store.retrieve(handles[0])
    assert recovered["original"] == originals["auth-1"]["content"]
    assert recovered["content_sha256"] == hashlib.sha256(
        originals["auth-1"]["content"].encode("utf-8")
    ).hexdigest()


def test_lazy_materialization_recovers_visible_hierarchical_source():
    store = CompressedContextStore()

    def lookup(key):
        if key == "file:routes.py":
            return {
                "fragment_id": "routes-1",
                "source": key,
                "content": "def create_route(path): return Route(path)\n",
                "token_count": 11,
            }
        return None

    recovered = store.retrieve_or_materialize("file:routes.py", lookup)

    assert recovered["source"] == "file:routes.py"
    assert recovered["resolution"] == "reference"
    assert recovered["original"].startswith("def create_route")
    assert store.retrieve(recovered["retrieval_handle"]) == recovered


def test_hierarchical_context_fragments_only_uses_rendered_hierarchy():
    hcc_result = {
        "level3_fragments": [
            {"id": "full-1", "source": "file:full.py", "content": "full"}
        ],
        "level2_fragments": [
            {
                "id": "skel-1",
                "source": "file:skel.py",
                "content": "def work(): ...",
                "variant": "skeleton",
            }
        ],
    }

    selected = hierarchical_context_fragments(hcc_result)

    assert [fragment["id"] for fragment in selected] == ["full-1", "skel-1"]


def test_capture_ranked_recovery_candidates_registers_omitted_sources():
    store = CompressedContextStore()
    originals = {
        "omitted-1": {
            "id": "omitted-1",
            "source": "file:omitted.py",
            "content": "def omitted_detail():\n    return 17\n",
            "token_count": 9,
        },
    }

    candidates = capture_ranked_recovery_candidates(
        [{
            "id": "omitted-1",
            "source": "file:omitted.py",
            "scores": {"semantic": 0.9, "composite": 0.7},
        }],
        originals.get,
        store=store,
    )

    assert len(candidates) == 1
    assert candidates[0]["recovery_candidate"] is True
    assert candidates[0]["variant"] == "reference"
    recovered = store.retrieve(candidates[0]["retrieval_handle"])
    assert recovered["original"] == originals["omitted-1"]["content"]


def test_capture_ranked_recovery_candidates_keeps_other_chunk_from_selected_source():
    store = CompressedContextStore()
    originals = {
        "auth-detail-2": {
            "id": "auth-detail-2",
            "source": "file:auth.py",
            "content": "ROTATION_INTERVAL = 17\n",
            "token_count": 6,
        },
    }

    candidates = capture_ranked_recovery_candidates(
        [{
            "id": "auth-detail-2",
            "source": "file:auth.py",
            "scores": {"semantic": 0.9, "composite": 0.8},
        }],
        originals.get,
        selected_keys={"auth-detail-1"},
        store=store,
    )

    assert [candidate["id"] for candidate in candidates] == ["auth-detail-2"]
    assert store.retrieve(candidates[0]["retrieval_handle"])["original"] == (
        "ROTATION_INTERVAL = 17\n"
    )


def test_native_hierarchical_result_exposes_rendered_level2_recovery_candidates():
    entroly_core = pytest.importorskip("entroly_core")
    EntrolyEngine = entroly_core.EntrolyEngine

    engine = EntrolyEngine()
    fragments = [
        (
            "def validate_token(token):\n"
            "    return token == 'known'\n\n"
            "def login(token):\n"
            "    return validate_token(token)\n",
            "file:auth.py",
        ),
        (
            "def create_invoice(amount):\n"
            "    return {'amount': amount}\n\n"
            "def charge(amount):\n"
            "    return create_invoice(amount)\n",
            "file:billing.py",
        ),
        (
            "def read_hits(key):\n"
            "    return []\n\n"
            "def allow(key):\n"
            "    return len(read_hits(key)) < 5\n",
            "file:limits.py",
        ),
        (
            "def render_dashboard(user):\n"
            "    return {'user': user}\n\n"
            "def dashboard(user):\n"
            "    return render_dashboard(user)\n",
            "file:dashboard.py",
        ),
        (
            "def connect_db(url):\n"
            "    return url\n\n"
            "def query(sql):\n"
            "    return connect_db(sql)\n",
            "file:db.py",
        ),
    ]
    for content, source in fragments:
        engine.ingest(content, source, 30, False)

    result = engine.hierarchical_compress(100, "validate auth token login")
    l2 = result["level2_fragments"]
    l3_ids = {fragment["id"] for fragment in result["level3_fragments"]}

    assert l2
    assert all(fragment["variant"] == "skeleton" for fragment in l2)
    assert all(fragment["id"] not in l3_ids for fragment in l2)
    assert all(
        f"## {fragment['source']}" in result["level2_cluster"]
        for fragment in l2
    )


def test_native_hierarchical_result_prefers_explicit_query_conditioned_seeds():
    entroly_core = pytest.importorskip("entroly_core")
    EntrolyEngine = entroly_core.EntrolyEngine

    engine = EntrolyEngine()
    auth = engine.ingest(
        "def validate_token(token):\n"
        "    return token == 'known'\n\n"
        "def login(token):\n"
        "    return validate_token(token)\n",
        "file:auth.py",
        30,
        False,
    )
    billing = engine.ingest(
        "def create_invoice(amount):\n"
        "    return {'amount': amount}\n\n"
        "def charge(amount):\n"
        "    return create_invoice(amount)\n",
        "file:billing.py",
        30,
        False,
    )

    result = engine.hierarchical_compress(
        100,
        "unrelated fallback query",
        [billing["fragment_id"]],
    )

    assert result["cluster_ids"][0] == billing["fragment_id"]
    assert result["cluster_ids"][0] != auth["fragment_id"]


def test_proxy_pipeline_seeds_hierarchy_from_optimizer_and_captures_rendered_ccr():
    store = get_ccr_store()
    store.clear()

    class Rust:
        def __init__(self):
            self.calls = []
            self.recall_calls = []

        def hierarchical_compress(self, token_budget, query, seed_ids):
            self.calls.append((token_budget, query, seed_ids))
            return {
                "status": "compressed",
                "level1_map": "file:rendered.py",
                "level1_tokens": 2,
                "level2_cluster": "## file:rendered.py\ndef rendered(): ...",
                "level2_tokens": 4,
                "level2_fragments": [{
                    "id": "rendered-1",
                    "source": "file:rendered.py",
                    "content": "def rendered(): ...",
                    "preview": "def rendered(): ...",
                    "token_count": 4,
                    "variant": "skeleton",
                    "relevance": 0.9,
                }],
                "level3_fragments": [],
                "level3_tokens": 0,
                "coverage": {
                    "level1_files": 1,
                    "level2_cluster_files": 1,
                    "level3_full_files": 0,
                },
            }

        def classify_task(self, _query):
            return {"task_type": "Code"}

        def recall_auto(self, query, top_k):
            self.recall_calls.append((query, top_k))
            return [{
                "fragment_id": "recall-1",
                "source": "file:recall.py",
                "relevance": 0.95,
            }]

    class Engine:
        def __init__(self):
            self._rust = Rust()
            self._turn_counter = 0
            self._ltm = None

        def advance_turn(self):
            return None

        def optimize_context(self, _token_budget, _query):
            return {
                "selected_fragments": [{
                    "id": "optimized-seed",
                    "source": "file:optimized.py",
                    "content": "def optimized(): ...",
                    "token_count": 4,
                    "variant": "skeleton",
                }],
                "query_analysis": {},
            }

        def explain_selection(self):
            return {
                "excluded": [{
                    "id": "omitted-1",
                    "source": "file:omitted.py",
                    "scores": {"semantic": 0.8, "composite": 0.7},
                }]
            }

        def _get_fragment(self, key):
            if key in {"rendered-1", "file:rendered.py"}:
                return {
                    "id": "rendered-1",
                    "source": "file:rendered.py",
                    "content": "def rendered():\n    return 'exact'\n",
                    "token_count": 8,
                }
            if key in {"omitted-1", "file:omitted.py"}:
                return {
                    "id": "omitted-1",
                    "source": "file:omitted.py",
                    "content": "def omitted():\n    return 'standby'\n",
                    "token_count": 8,
                }
            if key in {"recall-1", "file:recall.py"}:
                return {
                    "id": "recall-1",
                    "source": "file:recall.py",
                    "content": "def recalled():\n    return 'query-ranked'\n",
                    "token_count": 9,
                }
            return None

    engine = Engine()
    proxy = PromptCompilerProxy(
        engine,
        ProxyConfig(
            witness_mode="off",
            enable_adaptive_budget=False,
            enable_dynamic_budget=False,
            enable_hierarchical_compression=True,
            enable_ltm=False,
            enable_security_scan=False,
            enable_context_scaffold=False,
            enable_prompt_directives=False,
            enable_passive_feedback=False,
        ),
    )

    result = proxy._run_pipeline("render the selected code", {"model": "test"})

    assert engine._rust.calls[0][2] == ["optimized-seed"]
    assert engine._rust.recall_calls == [("render the selected code", 8)]
    assert [fragment["id"] for fragment in result["selected_fragments"]] == [
        "rendered-1"
    ]
    recovered = store.retrieve("file:rendered.py")
    assert recovered["original"] == "def rendered():\n    return 'exact'\n"
    assert recovered["compressed"] == "def rendered(): ..."
    assert [
        fragment["id"] for fragment in result["recoverable_fragments"]
    ] == ["rendered-1", "recall-1", "omitted-1"]
    assert "file:omitted.py" not in result["context"]
    assert "file:recall.py" not in result["context"]
    assert store.retrieve("file:omitted.py")["original"].endswith("'standby'\n")
    assert store.retrieve("file:recall.py")["original"].endswith("'query-ranked'\n")


def test_flat_context_exposes_retrieval_handle():
    block = format_context_block(
        [{
            "source": "file:auth.py",
            "content": "def login(token): ...",
            "token_count": 5,
            "variant": "skeleton",
            "retrieval_handle": "ccr:1234",
        }],
        [],
        [],
        None,
    )

    assert "entroly_retrieve(source_or_handle)" in block
    assert "retrieve: ccr:1234" in block


def test_hierarchical_context_explains_source_recovery():
    block = format_hierarchical_context(
        {
            "status": "compressed",
            "level1_map": "file:auth.py",
            "coverage": {"level1_files": 1},
        },
        [],
        [],
        None,
    )

    assert "entroly_retrieve(source)" in block
    assert "file:auth.py" in block


def test_http_retrieve_endpoint_lazily_materializes_source():
    store = get_ccr_store()
    store.clear()

    class Engine:
        def _get_fragment(self, key):
            if key != "file:db.py":
                return None
            return {
                "id": "db-1",
                "source": key,
                "content": "class DB:\n    pass\n",
                "token_count": 6,
            }

    app = Starlette(routes=[Route("/retrieve", _context_retrieve)])
    app.state.proxy = SimpleNamespace(engine=Engine())

    async def fetch():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.get("/retrieve", params={"source": "file:db.py"})

    response = asyncio.run(fetch())
    payload = response.json()

    assert response.status_code == 200
    assert payload["source"] == "file:db.py"
    assert payload["retrieval_handle"].startswith("ccr:")
    assert payload["original_content"] == "class DB:\n    pass\n"
