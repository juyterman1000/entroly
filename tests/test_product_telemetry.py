from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import entroly.product_telemetry as telemetry
from entroly.telemetry_collector import (
    TelemetryStore,
    create_app,
    validate_batch_payload,
    validate_deletion_payload,
)


@pytest.fixture(autouse=True)
def isolated_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_TELEMETRY_TESTING", "1")
    monkeypatch.delenv("ENTROLY_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.delenv("ENTROLY_TELEMETRY_ENDPOINT", raising=False)
    monkeypatch.delenv("ENTROLY_TELEMETRY_TOKEN", raising=False)
    monkeypatch.delenv("CI", raising=False)
    telemetry._SEEN_DAILY.clear()


def _queue() -> list[dict]:
    path = telemetry._queue_path()
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_disabled_by_default_is_network_and_filesystem_silent(tmp_path: Path):
    report = telemetry.status()

    assert report["enabled"] is False
    assert report["upload_configured"] is False
    assert report["queued_events"] == 0
    assert not (tmp_path / "state").exists()
    assert telemetry.capture_surface_started("mcp") is False


def test_explicit_enable_queues_only_closed_schema():
    report = telemetry.enable()
    config_text = telemetry._config_path().read_text()
    seed = json.loads(config_text)["pseudonym_seed"]
    queue_text = telemetry._queue_path().read_text()

    assert report["enabled"] is True
    assert report["upload_configured"] is False
    assert report["queued_events"] == 1
    assert seed not in queue_text
    assert _queue()[0]["event_name"] == "activation"


def test_environment_cannot_turn_local_consent_into_upload(
    monkeypatch: pytest.MonkeyPatch,
):
    telemetry.enable()
    monkeypatch.setenv(
        "ENTROLY_TELEMETRY_ENDPOINT", "https://surprise.example/v1/events"
    )

    assert telemetry.status()["upload_configured"] is False
    assert telemetry.flush() == {"status": "not_configured", "sent": 0}


def test_unknown_and_sensitive_properties_are_never_serialized():
    telemetry.enable()

    assert telemetry.capture(
        "command",
        {
            "command": "verify-claims",
            "result": "error",
            "duration_bucket": "lt_1s",
            "error_type": "ValueError",
            "prompt": "TOP SECRET PROMPT",
            "path": "C:/private/customer/repo.py",
            "message": "api_key=never-store-this",
        },
    )
    assert telemetry.capture_surface_error(
        "mcp", ValueError("customer prompt and /private/path")
    )

    body = telemetry._queue_path().read_text()
    assert "TOP SECRET" not in body
    assert "customer" not in body
    assert "repo.py" not in body
    assert "api_key" not in body
    command = [event for event in _queue() if event["event_name"] == "command"][0]
    assert command["properties"] == {
        "command": "verify-claims",
        "duration_bucket": "lt_1s",
        "error_type": "ValueError",
        "result": "error",
    }


def test_value_signal_uses_only_coarse_buckets():
    telemetry.enable()

    assert telemetry.capture_optimization_outcome(
        "sdk_compress",
        before_tokens=12_345,
        after_tokens=3_200,
        measurement_scope="local_estimate",
        cost_evidence="not_available",
    )

    body = telemetry._queue_path().read_text()
    assert "12345" not in body
    assert "3200" not in body
    event = [
        item for item in _queue() if item["event_name"] == "optimization_outcome"
    ][0]
    assert event["properties"] == {
        "cost_evidence": "not_available",
        "measurement_scope": "local_estimate",
        "reduction_percent_bucket": "70_89",
        "surface": "sdk_compress",
        "tokens_saved_bucket": "1k_9k",
    }


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (0, 0, "none"),
        (100, 100, "none"),
        (100, 95, "lt_10"),
        (100, 80, "10_29"),
        (100, 60, "30_49"),
        (100, 40, "50_69"),
        (100, 20, "70_89"),
        (100, 5, "90_plus"),
    ],
)
def test_reduction_percent_buckets(before: int, after: int, expected: str):
    assert telemetry.reduction_percent_bucket(before, after) == expected


def test_error_categories_can_be_disabled_independently():
    telemetry.enable(error_events=False)

    assert telemetry.capture_surface_error("mcp", ValueError("secret")) is False
    assert telemetry.capture_cli_result(
        "doctor", result="error", elapsed_seconds=0.2, error=ValueError("secret")
    )

    command = [event for event in _queue() if event["event_name"] == "command"][0]
    assert command["properties"]["result"] == "error"
    assert "error_type" not in command["properties"]


def test_command_and_error_categories_do_not_report_usage_frequency():
    telemetry.enable()

    assert telemetry.capture_cli_result(
        "doctor", result="error", elapsed_seconds=0.2, error=ValueError("one")
    )
    assert not telemetry.capture_cli_result(
        "doctor", result="error", elapsed_seconds=0.2, error=ValueError("two")
    )
    assert telemetry.capture_surface_error("mcp", OSError("one"))
    assert not telemetry.capture_surface_error("mcp", OSError("two"))

    assert len([item for item in _queue() if item["event_name"] == "command"]) == 1
    assert len(
        [item for item in _queue() if item["event_name"] == "surface_error"]
    ) == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://telemetry.example/events",
        "ftp://telemetry.example/events",
        "https://user:pass@telemetry.example/events",
        "https://telemetry.example/events?token=secret",
    ],
)
def test_remote_endpoint_policy_rejects_unsafe_urls(endpoint: str):
    with pytest.raises(ValueError):
        telemetry.validate_endpoint(endpoint)


def test_endpoint_policy_allows_https_and_loopback_http():
    assert telemetry.validate_endpoint("https://telemetry.example/v1/events") == (
        "https://telemetry.example/v1/events"
    )
    assert telemetry.validate_endpoint("http://127.0.0.1:9381/v1/events") == (
        "http://127.0.0.1:9381/v1/events"
    )


def test_air_gap_and_hard_disable_override_stored_consent(monkeypatch: pytest.MonkeyPatch):
    telemetry.enable()
    monkeypatch.setenv("ENTROLY_AIR_GAP", "1")

    assert telemetry.is_enabled() is False
    assert telemetry.capture_surface_started("proxy") is False
    assert telemetry.flush() == {"status": "disabled", "sent": 0}
    assert telemetry.status()["hard_disabled"] is True


def test_daily_surface_event_and_monthly_identifier_rotation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(telemetry, "_today", lambda: "2026-08-11")
    telemetry.enable()
    assert telemetry.capture_surface_started("sdk_compress") is True
    assert telemetry.capture_surface_started("sdk_compress") is False
    august_id = _queue()[-1]["installation_id"]

    monkeypatch.setattr(telemetry, "_today", lambda: "2026-09-01")
    assert telemetry.capture_surface_started("sdk_compress") is True
    september_id = _queue()[-1]["installation_id"]

    assert august_id != september_id


class _Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status


class _Opener:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return _Response()


def test_successful_flush_sends_bounded_schema_and_dequeues(monkeypatch: pytest.MonkeyPatch):
    opener = _Opener()
    monkeypatch.setattr(telemetry, "_opener", lambda: opener)
    telemetry.enable(endpoint="https://telemetry.example/v1/events")
    telemetry.capture_cli_result("doctor", result="success", elapsed_seconds=0.3)

    result = telemetry.flush()

    assert result == {"status": "sent", "sent": 2}
    assert telemetry.status()["queued_events"] == 0
    request, timeout = opener.requests[0]
    assert 0.1 <= timeout <= 3.0
    payload = json.loads(request.data)
    assert len(validate_batch_payload(payload)) == 2
    assert request.headers["Content-type"] == "application/json"

    assert telemetry.capture_cli_result(
        "audit", result="success", elapsed_seconds=0.3
    )
    assert telemetry.flush() == {"status": "deferred", "sent": 0}
    assert telemetry.flush(force=True) == {"status": "sent", "sent": 1}


def test_failed_flush_keeps_queue_and_exposes_only_error_category(
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingOpener:
        def open(self, _request, _timeout):
            raise OSError("secret internal proxy and customer path")

    monkeypatch.setattr(telemetry, "_opener", lambda: FailingOpener())
    telemetry.enable(endpoint="https://telemetry.example/v1/events")

    result = telemetry.flush()

    assert result == {"status": "error", "sent": 0, "error_type": "OSError"}
    assert telemetry.status()["queued_events"] == 1
    status_text = telemetry._status_path().read_text()
    assert "secret" not in status_text
    assert "customer" not in status_text


def test_disable_withdraws_consent_and_purges_identity_and_queue():
    telemetry.enable()
    assert telemetry._config_path().exists()
    assert telemetry._queue_path().exists()

    report = telemetry.disable_and_purge()

    assert report["enabled"] is False
    assert not telemetry._config_path().exists()
    assert not telemetry._queue_path().exists()
    assert telemetry.status()["enabled"] is False


def test_disable_requests_remote_pseudonym_deletion_before_local_purge(
    monkeypatch: pytest.MonkeyPatch,
):
    opener = _Opener()
    monkeypatch.setattr(telemetry, "_opener", lambda: opener)
    telemetry.enable(endpoint="https://telemetry.example/v1/events")
    seed = json.loads(telemetry._config_path().read_text())["pseudonym_seed"]

    report = telemetry.disable_and_purge()

    assert report["remote_deletion"] == "deleted"
    request, _timeout = opener.requests[0]
    assert request.method == "DELETE"
    payload = json.loads(request.data)
    installation_ids = validate_deletion_payload(payload)
    assert len(installation_ids) == 4
    assert seed not in request.data.decode()
    assert not telemetry._config_path().exists()


def test_collector_rejects_extra_properties_and_summarizes_without_raw_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(telemetry, "_platform_family", lambda: "windows")
    telemetry.enable()
    telemetry.capture_cli_result("doctor", result="success", elapsed_seconds=0.2)
    telemetry.capture_optimization_outcome(
        "proxy",
        before_tokens=5_000,
        after_tokens=1_000,
        measurement_scope="provider_bound_estimate",
        cost_evidence="modeled_positive",
    )
    events = _queue()
    payload = {
        "schema_version": telemetry.BATCH_SCHEMA_VERSION,
        "events": events,
    }
    assert validate_batch_payload(payload) == events

    hostile = json.loads(json.dumps(payload))
    hostile["events"][0]["properties"]["prompt"] = "private"
    with pytest.raises(ValueError):
        validate_batch_payload(hostile)

    incomplete = json.loads(json.dumps(payload))
    incomplete["events"][0]["properties"] = {}
    with pytest.raises(ValueError):
        validate_batch_payload(incomplete)

    future = json.loads(json.dumps(payload))
    future["events"][0]["occurred_on"] = "2999-01-01"
    with pytest.raises(ValueError):
        validate_batch_payload(future)

    store = TelemetryStore(tmp_path / "collector" / "events.db")
    assert store.ingest(events) == 3
    assert store.ingest(events) == 0
    summary = store.summary(days=30)
    rendered = json.dumps(summary)

    assert summary["active_monthly_pseudonyms"] == 1
    assert summary["activation_monthly_pseudonyms"] == 1
    assert summary["commands"]["successful_observations"] == 1
    assert summary["benefit"]["monthly_pseudonyms_with_positive_reduction"] == 1
    assert summary["benefit"]["money_savings_verified"] is False
    assert summary["benefit"]["cost_evidence"] == {"modeled_positive": 1}
    assert summary["platforms"]["windows"]["benefited_monthly_pseudonyms"] == 1
    assert summary["privacy"]["exact_tokens_or_costs_stored"] is False
    assert summary["privacy"]["usage_volume_claim_allowed"] is False
    assert events[0]["installation_id"] not in rendered
    assert summary["privacy"]["unique_user_claim_allowed"] is False

    assert store.delete_installations([events[0]["installation_id"]]) == 3
    assert store.summary(days=30)["active_monthly_pseudonyms"] == 0


def test_collector_http_surface_is_authenticated_and_aggregate_only(tmp_path: Path):
    from starlette.testclient import TestClient

    telemetry.enable()
    telemetry.capture_cli_result("doctor", result="success", elapsed_seconds=0.2)
    events = _queue()
    app = create_app(
        db_path=tmp_path / "collector.db",
        ingest_token="ingest-secret",
        admin_token="admin-secret",
    )
    client = TestClient(app)
    payload = {"schema_version": telemetry.BATCH_SCHEMA_VERSION, "events": events}

    assert client.post("/v1/events", json=payload).status_code == 401
    response = client.post(
        "/v1/events",
        json=payload,
        headers={"Authorization": "Bearer ingest-secret"},
    )
    assert response.json() == {"accepted": 2, "inserted": 2}
    assert client.get("/v1/summary").status_code == 404
    response = client.get(
        "/v1/summary",
        headers={"Authorization": "Bearer admin-secret"},
    )
    rendered = response.text
    assert response.status_code == 200
    assert events[0]["installation_id"] not in rendered

    deletion = {
        "schema_version": telemetry.DELETE_SCHEMA_VERSION,
        "installation_ids": [events[0]["installation_id"]],
    }
    response = client.request(
        "DELETE",
        "/v1/events",
        json=deletion,
        headers={"Authorization": "Bearer ingest-secret"},
    )
    assert response.json() == {"deleted": 2}


def test_cli_telemetry_preview_is_complete_and_content_blind(capsys):
    from entroly import cli

    rc = cli.cmd_telemetry(
        SimpleNamespace(
            action="preview",
            endpoint=None,
            no_error_events=False,
            json_output=False,
        )
    )
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output["enabled_by_default"] is False
    assert "prompts" in output["never_collected"]
    assert "exception_messages" in output["never_collected"]
    assert "exact_token_counts" in output["never_collected"]
    assert "model_identifiers" in output["never_collected"]
