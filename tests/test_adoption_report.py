from __future__ import annotations

from scripts import adoption_report


def test_adoption_report_keeps_downloads_and_opt_in_observations_distinct(monkeypatch):
    def fake_fetch(url: str, **_kwargs):
        if "pypistats" in url:
            return {"data": {"last_month": 800}}
        if "npmjs" in url:
            return {"downloads": 200}
        return {
            "activation_monthly_pseudonyms": 50,
            "active_monthly_pseudonyms": 80,
            "commands": {
                "observations": 100,
                "failed_observations": 4,
                "observed_error_rate": 0.04,
            },
            "benefit": {
                "monthly_pseudonyms_with_positive_reduction": 60,
                "observed_benefit_rate_among_active_pseudonyms": 0.75,
                "money_savings_verified": False,
            },
        }

    monkeypatch.setattr(adoption_report, "_fetch_json", fake_fetch)

    report = adoption_report.build_report(
        collector_summary_url="https://telemetry.example/v1/summary",
        admin_token="not-rendered",
    )

    assert report["registry_downloads"]["combined_last_month"] == 1000
    assert report["observed_activations_per_1000_downloads"] == 50.0
    assert report["observed_benefited_pseudonyms_per_1000_downloads"] == 60.0
    assert report["observed_benefit_rate_among_active_pseudonyms"] == 0.75
    assert report["interpretation"]["actual_unique_user_adoption_rate_known"] is False
    assert report["interpretation"]["money_savings_are_provider_verified"] is False
    assert (
        report["consented_product_health"]["commands"]["observed_error_rate"]
        == 0.04
    )


def test_adoption_report_works_before_collector_is_deployed(monkeypatch):
    monkeypatch.setattr(
        adoption_report,
        "_fetch_json",
        lambda url, **_kwargs: (
            {"data": {"last_month": 11}}
            if "pypistats" in url
            else {"downloads": 7}
        ),
    )

    report = adoption_report.build_report()

    assert report["registry_downloads"]["combined_last_month"] == 18
    assert report["consented_product_health"] is None
    assert report["observed_activations_per_1000_downloads"] is None
    assert report["observed_benefited_pseudonyms_per_1000_downloads"] is None
