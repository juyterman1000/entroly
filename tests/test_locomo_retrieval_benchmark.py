from benchmarks.locomo_retrieval import run


def test_memory_and_bm25_score_the_same_annotated_turns_under_budget():
    dialogue = {
        "sample_id": "fixture",
        "conversation": {
            "session_1": [
                {"dia_id": "D1:1", "speaker": "A", "text": "Orchid"},
                {"dia_id": "D1:2", "speaker": "B", "text": "Marble"},
            ],
        },
        "qa": [
            {"category": 1, "question": "Orchid", "evidence": ["D1:1"]},
            {"category": 2, "question": "Unlinked", "evidence": ["D1:3"]},
            {"category": 5, "question": "Unanswerable", "evidence": []},
        ],
    }

    report = run([dialogue], budget=3)

    assert report["scored_queries"] == 1
    assert [row["reason"] for row in report["exclusions"]] == ["unresolvable_evidence", "out_of_scope_category"]
    for method in ("memory_os", "bm25"):
        result = report["rows"][0][method]
        assert result["recovered_evidence"] == ["D1:1"]
        assert result["estimated_tokens"] <= 3
        assert report["summary"][method]["all_evidence_queries"] == 1
