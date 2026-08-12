"""Build an honest Entroly adoption and product-health snapshot.

Registry downloads are aggregate package fetches, not people or successful
activations. Product-health events are explicit-consent observations, not a
census. The report keeps those denominators separate and exposes only an
"observed activations per 1,000 downloads" diagnostic, never a fabricated
unique-user conversion rate. Benefit observations are separately limited to
coarse, consented before/after reductions.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from typing import Any


PYPI_RECENT_URL = "https://pypistats.org/api/packages/entroly/recent"
NPM_MONTH_URL = "https://api.npmjs.org/downloads/point/last-month/entroly"


def _fetch_json(url: str, *, token: str = "", timeout: float = 10.0) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("report endpoints must be absolute HTTPS URLs")
    headers = {"Accept": "application/json", "User-Agent": "entroly-adoption-report/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {parsed.hostname}")
    return payload


def registry_downloads() -> dict[str, int]:
    pypi = _fetch_json(PYPI_RECENT_URL)
    npm = _fetch_json(NPM_MONTH_URL)
    pypi_month = int((pypi.get("data") or {}).get("last_month", 0) or 0)
    npm_month = int(npm.get("downloads", 0) or 0)
    return {
        "pypi_last_month": max(0, pypi_month),
        "npm_last_month": max(0, npm_month),
        "combined_last_month": max(0, pypi_month) + max(0, npm_month),
    }


def build_report(
    *,
    collector_summary_url: str | None = None,
    admin_token: str = "",
) -> dict[str, Any]:
    downloads = registry_downloads()
    health: dict[str, Any] | None = None
    if collector_summary_url:
        health = _fetch_json(collector_summary_url, token=admin_token)

    observed_activations = (
        int(health.get("activation_monthly_pseudonyms", 0) or 0)
        if health is not None
        else None
    )
    observed_benefited = None
    observed_benefit_rate = None
    observed_exit_responses = None
    if health is not None:
        benefit = health.get("benefit") or {}
        observed_benefited = int(
            benefit.get("monthly_pseudonyms_with_positive_reduction", 0) or 0
        )
        observed_benefit_rate = benefit.get(
            "observed_benefit_rate_among_active_pseudonyms"
        )
        observed_exit_responses = int(
            (health.get("exit_feedback") or {}).get("responses", 0) or 0
        )
    denominator = downloads["combined_last_month"]
    per_thousand = None
    benefited_per_thousand = None
    if observed_activations is not None and denominator > 0:
        per_thousand = round((observed_activations / denominator) * 1_000, 3)
    if observed_benefited is not None and denominator > 0:
        benefited_per_thousand = round((observed_benefited / denominator) * 1_000, 3)

    return {
        "schema_version": "entroly.adoption-report.v1",
        "registry_downloads": downloads,
        "consented_product_health": health,
        "observed_activations_per_1000_downloads": per_thousand,
        "observed_benefited_pseudonyms_per_1000_downloads": benefited_per_thousand,
        "observed_benefit_rate_among_active_pseudonyms": observed_benefit_rate,
        "observed_structured_exit_feedback_responses": observed_exit_responses,
        "interpretation": {
            "actual_unique_user_adoption_rate_known": False,
            "downloads_are_unique_users": False,
            "telemetry_is_a_census": False,
            "direct_uninstalls_are_observable": False,
            "money_savings_are_provider_verified": False,
            "reason": (
                "Registry counts include repeat, CI, and automated downloads; "
                "product-health events include only explicitly consenting installations. "
                "Cost signals are modeled from coarse provider-bound reductions, not bills."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collector-summary-url",
        default=None,
        help="Optional HTTPS /v1/summary URL for the deployed Entroly collector",
    )
    parser.add_argument(
        "--admin-token-env",
        default="ENTROLY_TELEMETRY_ADMIN_TOKEN",
        help="Environment variable containing the collector admin token",
    )
    args = parser.parse_args()
    token = os.environ.get(args.admin_token_env, "")
    report = build_report(
        collector_summary_url=args.collector_summary_url,
        admin_token=token,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
