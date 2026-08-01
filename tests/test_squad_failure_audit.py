from bench.squad_failure_audit import classify_failure


def test_failure_taxonomy_distinguishes_ranking_trim_and_utilization() -> None:
    common = {
        "raw_context": "The Dutch name is Rhijn.",
        "answers": ["Rhijn"],
        "baseline_correct": True,
        "treatment_correct": False,
    }
    ranking = classify_failure(
        selected_pre_trim="The Dutch name is",
        emitted_post_trim="The Dutch name is",
        **common,
    )
    trimming = classify_failure(
        selected_pre_trim="The Dutch name is Rhijn.",
        emitted_post_trim="The Dutch name is",
        **common,
    )
    utilization = classify_failure(
        selected_pre_trim="The Dutch name is Rhijn.",
        emitted_post_trim="The Dutch name is Rhijn.",
        **common,
    )
    assert ranking.category == "retrieval_or_ranking_loss"
    assert trimming.category == "trim_or_boundary_loss"
    assert utilization.category == "utilization_order_or_generation_failure"
