"""Online training surface for WITNESS.

The trainer only consumes external labels. It never treats the model's own
verdict as ground truth; callers must provide a source such as CI, tests,
user acceptance, or an explicit review decision.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .witness import extract_claims, select_evidence_windows
from .witness_features import extract_features
from .witness_risk_model import RiskModel


SAFE_LABELS = {
    "safe",
    "grounded",
    "accepted",
    "user_accepted",
    "ci_passed",
    "test_passed",
    "tests_passed",
    "command_succeeded",
    "false_positive",
}
HALU_LABELS = {
    "hallucinated",
    "unsupported",
    "rejected",
    "user_rejected",
    "ci_failed",
    "test_failed",
    "tests_failed",
    "command_failed",
    "false_negative",
    "witness_correct",
}


@dataclass(frozen=True)
class WitnessTrainingRecord:
    ts: float
    source: str
    label: str
    y: int
    profile: str
    n_claims: int
    mean_risk_before: float
    mean_risk_after: float
    model_updates: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_model_path() -> Path:
    raw = os.environ.get("ENTROLY_WITNESS_MODEL_PATH")
    if raw:
        return Path(raw)
    return Path.home() / ".entroly" / "witness" / "risk_model.json"


def label_to_y(label: str) -> int:
    normalized = str(label or "").strip().lower()
    if normalized in SAFE_LABELS:
        return 0
    if normalized in HALU_LABELS:
        return 1
    raise ValueError(f"unsupported witness training label: {label!r}")


def _split_question_context(context: str) -> tuple[str, str | None]:
    if "\n\nQuestion:" not in context:
        return context, None
    knowledge, question = context.split("\n\nQuestion:", 1)
    return knowledge.strip() or context, question.strip() or None


class WitnessTrainingStore:
    """Thread-safe online SGD trainer for the WITNESS risk model."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path is not None else default_model_path()
        self.audit_path = self.model_path.with_suffix(self.model_path.suffix + ".training.jsonl")
        self._lock = threading.Lock()
        self.model = RiskModel.load(self.model_path)

    def record(
        self,
        *,
        context: str,
        output: str,
        label: str,
        profile: str = "auto",
        source: str = "manual",
    ) -> WitnessTrainingRecord:
        y = label_to_y(label)
        claims = extract_claims(output, force_python=True)
        knowledge, question = _split_question_context(context)
        before: list[float] = []
        after: list[float] = []

        with self._lock:
            for claim in claims:
                windows, adequacy = select_evidence_windows(knowledge, claim.text)
                evidence_text = "\n".join(w.text for w in windows) if windows else knowledge
                features = extract_features(
                    claim.text,
                    evidence_text,
                    adequacy=adequacy,
                    question=question,
                )
                before.append(self.model.predict(features))
                after.append(self.model.update(features, y=y))

            self.model.save(self.model_path)

        record = WitnessTrainingRecord(
            ts=time.time(),
            source=source,
            label=str(label),
            y=y,
            profile=profile,
            n_claims=len(claims),
            mean_risk_before=sum(before) / max(len(before), 1),
            mean_risk_after=sum(after) / max(len(after), 1),
            model_updates=self.model.updates,
        )
        self._append_audit(record)
        return record

    def record_ravs_outcome(self, payload: dict[str, Any]) -> WitnessTrainingRecord | None:
        label = _label_from_ravs_payload(payload)
        context = str(payload.get("context") or payload.get("prompt") or payload.get("query") or "")
        output = str(payload.get("output") or payload.get("response") or payload.get("answer") or "")
        if label is None or not context or not output:
            return None
        return self.record(
            context=context,
            output=output,
            label=label,
            profile=str(payload.get("profile") or "auto"),
            source=str(payload.get("source") or "ravs"),
        )

    def _append_audit(self, record: WitnessTrainingRecord) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")


def _label_from_ravs_payload(payload: dict[str, Any]) -> str | None:
    final_label = str(payload.get("final_label") or "").strip().lower()
    if final_label in {
        "accepted",
        "ci_passed",
        "command_succeeded",
        "escalated_and_accepted",
        "escalated_and_kept_original",
    }:
        return "safe"
    if final_label in {"rejected", "ci_failed", "command_failed"}:
        return "hallucinated"

    events = payload.get("outcome_events")
    if not isinstance(events, list):
        return None
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        typ = str(event.get("type") or event.get("event_type") or "").strip().lower()
        value = str(event.get("value") or event.get("outcome") or event.get("label") or "").strip().lower()
        signal = f"{typ}:{value}"
        if typ in {"ci_passed", "command_succeeded", "accepted"} or value in {"passed", "success", "accepted"}:
            return "safe"
        if typ in {"ci_failed", "command_failed", "rejected"} or value in {"failed", "failure", "rejected"}:
            return "hallucinated"
        if signal in {"test_result:passed", "ci_result:passed"}:
            return "safe"
        if signal in {"test_result:failed", "ci_result:failed"}:
            return "hallucinated"
    return None
