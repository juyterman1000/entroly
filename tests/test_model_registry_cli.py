from __future__ import annotations

import json

from entroly.models.__main__ import main


def test_model_cli_resolve_emits_budget_and_provenance(capsys):
    assert main(["resolve", "gemini-2.5-pro", "--output-tokens", "10000"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["model_id"] == "google/gemini-2.5-pro"
    assert payload["context_window"] == 1_048_576
    assert payload["effective_input_budget"] < payload["context_window"]
    assert payload["trust"] == "verified"
    assert len(payload["registry_digest"]) == 64


def test_model_cli_diagnostics_emits_trust_counts(capsys):
    assert main(["diagnostics"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] > 0
    assert payload["trust_counts"]["verified"] > 0
