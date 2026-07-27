from entroly.context_receipts import run_receipt_pipeline


def test_no_match_withholding_is_not_credited_as_token_savings():
    receipt = run_receipt_pipeline(
        [("source.md", "alpha beta gamma delta")],
        query="quasar zebra unrelated",
        token_budget=100,
        prefer_rust=False,
    )

    ratio = receipt["compression_ratio"]
    assert receipt["selected_context"] == []
    assert ratio["tokens_withheld"] == ratio["source_tokens"]
    assert ratio["tokens_saved"] == 0
    assert ratio["savings_eligible"] is False
    assert ratio["savings_status"] == "not_credited_no_relevance_evidence"
    assert ratio["reduction_pct"] == 0.0
    assert ratio["withheld_pct"] > 0.0
