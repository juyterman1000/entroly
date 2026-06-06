"""Smoke tests for Query-Conditioned Compressive Retrieval."""
from __future__ import annotations

from entroly.qccr import select, _query_tokens, _split_sentences, _mmr_select, _bm25_corpus


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


def test_tokenization_splits_identifiers():
    toks = _query_tokens("How does taint_flow work with CamelCase identifiers?")
    assert "taint" in toks
    assert "flow" in toks
    assert "camel" in toks or "camelcase" in toks
    assert "case" in toks or "camelcase" in toks


def test_sentence_split_code_breaks():
    text = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
    sents = _split_sentences(text)
    assert len(sents) >= 1


def test_mmr_selects_diverse():
    sentences = [
        "Jaccard similarity measures set overlap.",
        "Jaccard similarity measures set overlap completely.",  # near-duplicate
        "The weather is sunny today.",
    ]
    tf, _, _, _ = _bm25_corpus(sentences)
    rel = [2.0, 1.9, 0.0]  # first two relevant, third not
    chosen = _mmr_select(sentences, tf, rel, budget_tokens=100)
    # Should pick index 0 (highest rel); index 2 has rel=0 so excluded.
    assert 0 in chosen
    assert 2 not in chosen


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                raise
