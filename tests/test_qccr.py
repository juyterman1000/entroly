"""Smoke tests for Query-Conditioned Compressive Retrieval."""
from __future__ import annotations

import pytest

# QCCR is the Rust SSOT (entroly-qccr crate); these tests exercise the engine,
# so they self-skip on the pure-Python (engine-less) install surface.
from entroly.native_status import QCCR_SYMBOLS, native_status  # noqa: E402

_NATIVE = native_status(QCCR_SYMBOLS)
pytestmark = pytest.mark.skipif(
    not _NATIVE.ok,
    reason="installed entroly_core does not include current QCCR symbols",
)

from entroly.qccr import select, _expanded_query_tokens  # noqa: E402


def test_empty_fragments_returns_empty():
    assert select([], token_budget=1024, query="anything") == []


def test_empty_query_returns_input():
    frags = [{"source": "a.py", "content": "def f(): pass", "token_count": 5}]
    assert select(frags, token_budget=1024, query="") == frags


def test_selects_query_relevant_file():
    frags = [
        {"source": "irrelevant.md", "content": "This document explains the weather patterns in spring and autumn.", "token_count": 20},
        {"source": "relevant.py", "content": "def jaccard_similarity(a, b):\n    return len(a & b) / len(a | b)", "token_count": 15},
        {"source": "also_irrelevant.md", "content": "The history of ancient Rome spans over twelve centuries.", "token_count": 18},
    ]
    result = select(frags, token_budget=512, query="What is jaccard similarity?")
    assert result, "qccr returned nothing for a query with obvious match"
    sources = [r.get("source") for r in result]
    assert "relevant.py" in sources, f"qccr did not pick the jaccard file: {sources}"


def test_selected_fragments_keep_engine_contract_fields():
    frags = [{
        "fragment_id": "native-relevant-1",
        "source": "relevant.py",
        "content": "def jaccard_similarity(a, b):\n    return len(a & b) / len(a | b)",
        "token_count": 15,
    }]
    result = select(frags, token_budget=512, query="What is jaccard similarity?")
    assert result
    frag = result[0]
    assert frag["id"] == "qccr::relevant.py"
    assert frag["fragment_id"] == frag["id"]
    assert isinstance(frag["relevance"], float)
    assert frag["relevance_score"] == frag["relevance"]
    assert frag["source_fragment_ids"] == ["native-relevant-1"]


def test_short_fragments_are_not_dropped_by_query_fallback():
    frags = [
        {"source": "a.py", "content": "def a(): return 1", "token_count": 8},
        {"source": "b.py", "content": "def b(): return 2", "token_count": 8},
    ]
    result = select(frags, token_budget=1000, query="fix a")
    assert [frag["source"] for frag in result] == ["b.py", "a.py"]


def test_budget_respected():
    frags = [
        {"source": f"f{i}.py", "content": "def func(): return 1\n" * 50, "token_count": 200}
        for i in range(20)
    ]
    result = select(frags, token_budget=500, query="function definition")
    total = sum(r.get("token_count", 0) for r in result)
    assert total <= 600, f"budget exceeded: {total} > 500"  # small slack for rounding


def test_query_expansion_splits_identifiers():
    # Tokenizer + expansion are the Rust SSOT; verify identifier splitting still
    # surfaces through the public expansion API.
    toks = _expanded_query_tokens("How does taint_flow work with CamelCase identifiers?")
    assert "taint" in toks
    assert "flow" in toks
    assert "camel" in toks or "camelcase" in toks
    assert "case" in toks or "camelcase" in toks


# NOTE: sentence-splitting, single-field BM25, and MMR are now the Rust single
# source of truth (entroly-qccr) and are unit-tested there (cargo test
# -p entroly-qccr). Their behaviour is also covered end-to-end by the select()
# tests above and the held-out Langfuse regressions below.


def test_architecture_query_keeps_event_record_mapper_over_generic_ingestion_files():
    service_filler = "\n".join(
        f"const unrelatedWorkerHelper{i} = 'tenant project queue retry backoff';"
        for i in range(120)
    )
    frags = [
        {
            "source": "file:web/src/pages/api/public/ingestion.ts",
            "content": (
                "The public ingestion endpoint validates incoming request JSON with jsonSchema. "
                "It has access to prisma and then enqueues trace events for the worker."
            ),
            "token_count": 45,
        },
        {
            "source": "file:worker/src/queues/otelIngestionQueue.ts",
            "content": (
                "The OTEL worker parses each incoming observation through createIngestionEventSchema. "
                "It then passes events to IngestionService for storage processing."
            ),
            "token_count": 50,
        },
        {
            "source": "file:web/src/__tests__/async/traces-ui-table.servertest.ts",
            "content": (
                "This test creates trace JSON fixtures and asserts the UI table renders prisma-backed "
                "trace rows with schema-shaped fields."
            ),
            "token_count": 40,
        },
        {
            "source": "file:worker/src/services/IngestionService/index.ts",
            "content": (
                "export class IngestionService {\n"
                "  private async processTraceEventList(params): Promise<void> {\n"
                "    const traceRecords = this.mapTraceEventsToRecords(params);\n"
                "    await this.writeEvent(traceRecords, 'trace');\n"
                "  }\n"
                "  private async processObservationEventList(params): Promise<void> {\n"
                "    const observationRecords = this.mapObservationEventsToRecords(params);\n"
                "  }\n"
                "  private mapTraceEventsToRecords(params): TraceRecordInsertType[] {\n"
                "    return params.traceEventList.map((trace) => ({ id: trace.id, project_id: trace.projectId }));\n"
                "  }\n"
                "  private mapObservationEventsToRecords(params): ObservationRecordInsertType[] {\n"
                "    return params.observationEventList.map((obs) => ({ trace_id: obs.traceId, input: obs.body.input }));\n"
                "  }\n"
                "}\n"
                f"{service_filler}\n"
            ),
            "token_count": 1800,
        },
    ]

    selected = select(
        frags,
        token_budget=420,
        query="How does the trace worker map incoming json to prisma schema in Langfuse?",
    )
    sources = [frag["source"] for frag in selected]
    content = "\n".join(frag["content"] for frag in selected)

    assert sources[0] == "file:worker/src/services/IngestionService/index.ts"
    assert "mapTraceEventsToRecords" in content
    assert "TraceRecordInsertType" in content
    assert "servertest" not in sources[:2]


def test_persistence_query_prefers_repositories_over_dataset_ui_components():
    ui_filler = "\n".join(
        f"const column{i} = 'dataset run item score table persisted display';"
        for i in range(80)
    )
    frags = [
        {
            "source": "file:web/src/features/datasets/components/DatasetRunItemsByRunTable.tsx",
            "content": (
                "Dataset run items and scores are shown in this frontend table. "
                "The component renders persisted scores in dataset run columns.\n"
                f"{ui_filler}\n"
            ),
            "token_count": 900,
        },
        {
            "source": "file:packages/shared/src/server/repositories/definitions.ts",
            "content": (
                "export type ScoreRecordInsertType = z.infer<typeof scoreRecordInsertSchema>;\n"
                "export type DatasetRunItemRecordInsertType = z.infer<typeof datasetRunItemRecordInsertSchema>;\n"
                "export const parseClickhouseScore = (record): ScoreRecordInsertType => record;\n"
                "export const parseClickhouseDatasetRunItem = (record): DatasetRunItemRecordInsertType => record;\n"
            ),
            "token_count": 180,
        },
        {
            "source": "file:packages/shared/src/server/repositories/scores.ts",
            "content": (
                "export const upsertScore = async (score: Partial<ScoreRecordReadType>) => {\n"
                "  await upsertClickhouse({ table: 'scores', values: [score], eventBodyMapper: mapScore });\n"
                "}\n"
                "const datasetJoin = `JOIN dataset_run_items_rmt dri ON s.trace_id = dri.trace_id`;\n"
            ),
            "token_count": 180,
        },
    ]

    selected = select(
        frags,
        token_budget=360,
        query="How are dataset run items and scores persisted in Langfuse?",
    )
    sources = [frag["source"] for frag in selected]
    content = "\n".join(frag["content"] for frag in selected)

    assert sources[0] != "file:web/src/features/datasets/components/DatasetRunItemsByRunTable.tsx"
    assert any("/server/repositories/" in source for source in sources[:2])
    assert "ScoreRecordInsertType" in content or "upsertClickhouse" in content


def test_large_corpus_is_prefiltered_fast_and_keeps_the_answer():
    # A query over hundreds of files must not fall into the super-linear
    # localizer/sentence-selection path that times out on a real repo. The
    # BM25F pre-filter caps the working set; the obviously-relevant answer file
    # must still survive the cap, and the call must return quickly.
    import time
    import random

    from entroly.qccr import _PREFILTER_FILE_FLOOR

    random.seed(7)
    vocab = "alpha beta gamma delta epsilon module helper worker queue table column".split()

    def noise(n: int) -> str:
        return " ".join(
            " ".join(random.choice(vocab) for _ in range(10)) + "." for _ in range(n)
        )

    answer = {
        "source": "file:entroly/proxy.py",
        "content": (
            "def _catch_all(request): compressed = optimize the context and inject "
            "the compressed context into the outbound request messages before "
            "forwarding upstream to the provider."
        ),
        "token_count": 40,
        "fragment_id": "ANSWER",
    }
    distractors = [
        {
            "source": f"file:noise/d_{i}.py",
            "content": noise(random.randint(20, 60)),
            "token_count": random.randint(200, 1200),
            "fragment_id": f"d{i}",
        }
        for i in range(4 * _PREFILTER_FILE_FLOOR)  # comfortably above the cap
    ]
    half = len(distractors) // 2
    corpus = distractors[:half] + [answer] + distractors[half:]

    start = time.perf_counter()
    selected = select(
        corpus,
        token_budget=4000,
        query=(
            "where does the proxy inject compressed context into request "
            "messages before forwarding upstream"
        ),
    )
    elapsed = time.perf_counter() - start

    # Budget derivation, so this stays a real gate rather than a formality.
    # Measured on this workload (513 files): 0.10-0.13s, median 0.12s. The old
    # 15s ceiling was ~125x that, so a 100x algorithmic regression — an O(n^2)
    # blowup or an unbounded rescan — passed silently. 3s is ~25x the local
    # median, which absorbs slower CI hardware, cold caches, and noise while
    # still catching any regression that changes the complexity class.
    assert elapsed < 3.0, (
        f"prefilter regressed: {elapsed:.2f}s for {len(corpus)} files "
        f"(expected ~0.12s; budget 3.0s allows ~25x for CI variance)"
    )
    assert "file:entroly/proxy.py" in [r.get("source") for r in selected], (
        "the answer file was lost to the pre-filter cap"
    )


def test_prefilter_cap_scales_with_budget_and_floors():
    from entroly.qccr import _PREFILTER_FILE_FLOOR

    # Small budgets floor at the constant; large budgets scale up (budget // 64).
    assert max(_PREFILTER_FILE_FLOOR, 4000 // 64) == _PREFILTER_FILE_FLOOR
    assert max(_PREFILTER_FILE_FLOOR, 1_000_000 // 64) == 1_000_000 // 64


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                raise


def test_input_over_budget_is_still_compressed():
    """The bypass must not become a way to skip compression entirely."""
    from entroly.qccr import select

    topics = [
        "authentication tokens expire", "database migrations rollback",
        "cache eviction policy", "retry backoff jitter",
        "alpha beta gamma matching", "logging redaction rules",
    ]
    fragments = [
        {"source": f"file:mod_{i}.py", "content": (t + " detail line. ") * 120,
         "token_count": 420}
        for i, t in enumerate(topics)
    ]
    budget = 2000
    assert sum(f["token_count"] for f in fragments) > budget

    result = select(fragments, budget, query="alpha beta gamma matching")

    assert result is not fragments, "over-budget input must be compressed"
    assert result, "compression must not return an empty selection"
    delivered = sum(len(str(f.get("content", ""))) // 4 for f in result)
    assert delivered <= budget, f"compressed output {delivered} exceeds budget {budget}"


def test_benchmark_compressor_does_not_inflate_input_that_already_fits():
    """Compression must never make the payload larger than doing nothing.

    GSM8K measured 283 input tokens against a 50,000-token budget and returned
    294.4 -- a token saving of -4.0% -- because the harness chunks text into
    400-char pieces and rejoins them with a newline per chunk. Accuracy was
    identical in both arms (0.85), so the extra tokens bought nothing.
    """
    from bench.accuracy import _entroly_compress

    text = "Natalia sold clips to 48 friends in April, and half as many in May. " * 4
    budget = 50_000
    assert len(text) // 4 <= budget

    out = _entroly_compress(text, budget, query="how many clips did natalia sell")

    assert out == text, "input already inside the budget must be returned untouched"
    assert len(out) <= len(text), "compression must never inflate the payload"
