"""Bounded, read-only index for context receipts and session chains."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from entroly.context_receipts.models import stable_hash
from entroly.models import get_model_registry
from entroly.session_intelligence import SessionReceiptChain

_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_FILES = 512
_MAX_DEPTH = 5
_MAX_RECEIPT_ITEMS = 200
_MAX_EXCERPT = 600


def _bounded_text(value: object, limit: int = _MAX_EXCERPT) -> str:
    text = str(value or "").replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "…"


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _nonnegative_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _positive_int(value: object) -> int | None:
    result = _nonnegative_int(value)
    return result if result is not None and result > 0 else None


def _model_name(receipt: Mapping[str, Any]) -> str:
    metadata = _mapping(receipt.get("metadata"))
    for value in (
        receipt.get("model"),
        receipt.get("model_id"),
        metadata.get("model"),
        metadata.get("model_id"),
    ):
        if value:
            return str(value).strip()
    return ""


def _receipt_integrity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    expected_hash = str(receipt.get("reproducibility_hash") or "")
    expected_id = str(receipt.get("receipt_id") or "")
    payload = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "reproducibility_hash"}
    }
    actual_hash = stable_hash(payload)
    issues: list[str] = []
    if not expected_hash:
        issues.append("missing reproducibility_hash")
    elif not secrets_compare(expected_hash, actual_hash):
        issues.append("reproducibility_hash mismatch")
    if expected_id != f"cr_{actual_hash[:12]}":
        issues.append("receipt_id is not hash-derived")
    return {"valid": not issues, "issues": issues, "actual_hash": actual_hash}


def secrets_compare(left: str, right: str) -> bool:
    # Kept local to avoid exposing receipt hashes to timing shortcuts if the
    # dashboard is ever proxied beyond loopback.
    import hmac

    return hmac.compare_digest(left, right)


def _usage(receipt: Mapping[str, Any]) -> tuple[int | None, int | None]:
    usage = _mapping(receipt.get("usage"))
    input_tokens = next(
        (
            value
            for value in (
                _nonnegative_int(receipt.get("input_tokens")),
                _nonnegative_int(usage.get("input_tokens")),
                _nonnegative_int(usage.get("prompt_tokens")),
            )
            if value is not None
        ),
        None,
    )
    output_tokens = next(
        (
            value
            for value in (
                _nonnegative_int(receipt.get("output_tokens")),
                _nonnegative_int(usage.get("output_tokens")),
                _nonnegative_int(usage.get("completion_tokens")),
            )
            if value is not None
        ),
        None,
    )
    return input_tokens, output_tokens


def _model_and_cost(receipt: Mapping[str, Any], *, selected_tokens: int) -> dict[str, Any]:
    model = _model_name(receipt)
    if not model:
        return {
            "model": None,
            "context_window": _positive_int(receipt.get("context_window")),
            "trust": "unknown",
            "warning": "Receipt does not record a model; cost is unavailable.",
            "input_price_per_million": None,
            "output_price_per_million": None,
            "context_input_estimate_usd": None,
            "request_estimate_usd": None,
            "pricing_note": "No model provenance was recorded.",
        }

    resolution = get_model_registry().resolve(model)
    capability = resolution.capability
    input_price = capability.input_price_per_million if capability else None
    output_price = capability.output_price_per_million if capability else None
    input_tokens, output_tokens = _usage(receipt)
    request_estimate = None
    if capability and input_tokens is not None and output_tokens is not None:
        request_estimate = capability.estimated_cost_usd(input_tokens, output_tokens)
    context_estimate = (
        selected_tokens * input_price / 1_000_000
        if input_price is not None
        else None
    )
    return {
        "model": resolution.model_id,
        "context_window": resolution.context_window,
        "trust": resolution.trust.value,
        "warning": resolution.warning,
        "input_price_per_million": input_price,
        "output_price_per_million": output_price,
        "context_input_estimate_usd": context_estimate,
        "request_estimate_usd": request_estimate,
        "pricing_note": (
            "Base registry rates only; cache, long-context, provider, and negotiated "
            "pricing adjustments are not included."
            if input_price is not None
            else "Pricing is not available for this model."
        ),
    }


def _receipt_view(receipt: Mapping[str, Any]) -> dict[str, Any]:
    selected_raw = _items(receipt.get("selected_context"))[:_MAX_RECEIPT_ITEMS]
    omitted_raw = _items(receipt.get("omitted_context"))[:_MAX_RECEIPT_ITEMS]
    ratio = _mapping(receipt.get("compression_ratio"))
    selected_tokens = _nonnegative_int(ratio.get("selected_tokens"))
    if selected_tokens is None:
        selected_tokens = sum(_nonnegative_int(item.get("token_count")) or 0 for item in selected_raw)
    source_tokens = _nonnegative_int(ratio.get("source_tokens"))
    omitted_item_tokens = sum(
        _nonnegative_int(item.get("token_count")) or 0 for item in omitted_raw
    )
    if source_tokens is None:
        source_tokens = selected_tokens + omitted_item_tokens
    omitted_tokens = max(0, source_tokens - selected_tokens)
    token_budget = _nonnegative_int(receipt.get("token_budget")) or 0

    selected = [
        {
            "chunk_id": _bounded_text(item.get("chunk_id"), 160),
            "source_path": _bounded_text(item.get("source_path"), 500),
            "section": _bounded_text(item.get("section_heading"), 240),
            "token_count": _nonnegative_int(item.get("token_count")) or 0,
            "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            "reasons": [_bounded_text(reason, 240) for reason in item.get("reasons", [])[:8]]
            if isinstance(item.get("reasons"), list)
            else [],
            "excerpt": _bounded_text(item.get("text")),
            "fingerprint": _bounded_text(item.get("fingerprint"), 100),
        }
        for item in selected_raw
    ]
    omitted = [
        {
            "chunk_id": _bounded_text(item.get("chunk_id"), 160),
            "source_path": _bounded_text(item.get("source_path"), 500),
            "section": _bounded_text(item.get("section_heading"), 240),
            "token_count": _nonnegative_int(item.get("token_count")) or 0,
            "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            "omission_reason": _bounded_text(item.get("omission_reason"), 400),
            "reasons": [_bounded_text(reason, 240) for reason in item.get("reasons", [])[:8]]
            if isinstance(item.get("reasons"), list)
            else [],
            "excerpt": _bounded_text(item.get("text_preview")),
            "fingerprint": _bounded_text(item.get("fingerprint"), 100),
            "recoverable": bool(item.get("recoverable") or item.get("recovery_reference")),
        }
        for item in omitted_raw
    ]
    return {
        "receipt_id": _bounded_text(receipt.get("receipt_id"), 160),
        "query": _bounded_text(receipt.get("query"), 1000),
        "token_budget": token_budget,
        "selected_tokens": selected_tokens,
        "source_tokens": source_tokens,
        "omitted_tokens": omitted_tokens,
        "selected_count": len(selected_raw),
        "omitted_count": len(omitted_raw),
        "budget_utilization": selected_tokens / token_budget if token_budget else None,
        "selection_ratio": selected_tokens / source_tokens if source_tokens else None,
        "selected": selected,
        "omitted": omitted,
        "risk_summary": _mapping(receipt.get("risk_summary")),
        "warnings": [
            _bounded_text(warning, 500)
            for warning in receipt.get("warnings", [])[:20]
        ] if isinstance(receipt.get("warnings"), list) else [],
        "integrity": _receipt_integrity(receipt),
        "model": _model_and_cost(receipt, selected_tokens=selected_tokens),
    }


@dataclass(frozen=True, slots=True)
class _Document:
    payload: dict[str, Any]
    source: str
    modified_at: float


class ContextSessionIndex:
    """A bounded snapshot of local context-control artifacts."""

    def __init__(self, roots: Iterable[str | Path] | None = None):
        self.roots = self._normalize_roots(roots)
        self.diagnostics: list[dict[str, str]] = []
        self._sessions: dict[str, dict[str, Any]] = {}
        self._build()

    @staticmethod
    def _normalize_roots(roots: Iterable[str | Path] | None) -> tuple[Path, ...]:
        if roots is None:
            candidates: list[Path] = []
            explicit = os.environ.get("ENTROLY_DIR")
            if explicit:
                candidates.append(Path(explicit))
            candidates.append(Path.cwd() / ".entroly")
        else:
            candidates = [Path(root) for root in roots]
        normalized: list[Path] = []
        for candidate in candidates:
            root = candidate.expanduser().resolve()
            if root not in normalized:
                normalized.append(root)
        return tuple(normalized)

    def _candidate_files(self) -> Iterable[tuple[Path, Path]]:
        yielded = 0
        for root in self.roots:
            if not root.is_dir():
                continue
            for current, directories, files in os.walk(root, followlinks=False):
                current_path = Path(current)
                try:
                    depth = len(current_path.relative_to(root).parts)
                except ValueError:
                    directories[:] = []
                    continue
                directories[:] = sorted(
                    directory
                    for directory in directories[:64]
                    if not (current_path / directory).is_symlink()
                )
                if depth >= _MAX_DEPTH:
                    directories[:] = []
                for filename in sorted(files):
                    if yielded >= _MAX_FILES:
                        self.diagnostics.append(
                            {"type": "scan_limit", "message": f"Stopped after {_MAX_FILES} JSON files."}
                        )
                        return
                    path = current_path / filename
                    if path.suffix.lower() != ".json":
                        continue
                    if not (
                        "receipt" in filename.lower()
                        or "session" in filename.lower()
                        or current_path.name == "receipts"
                    ):
                        continue
                    yielded += 1
                    yield root, path

    def _read_documents(self) -> list[_Document]:
        documents: list[_Document] = []
        for root, path in self._candidate_files():
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
                size = resolved.stat().st_size
                if size > _MAX_FILE_BYTES:
                    raise ValueError(f"file exceeds {_MAX_FILE_BYTES} byte safety limit")
                payload = json.loads(resolved.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                relative = str(resolved.relative_to(root)).replace("\\", "/")
                documents.append(_Document(payload, relative, resolved.stat().st_mtime))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.diagnostics.append(
                    {
                        "type": type(exc).__name__,
                        "source": path.name,
                        "message": _bounded_text(exc, 300),
                    }
                )
        return documents

    @staticmethod
    def _key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    def _build(self) -> None:
        documents = self._read_documents()
        receipts: dict[str, _Document] = {}
        chains: list[_Document] = []
        for document in documents:
            payload = document.payload
            if payload.get("schema_version") == SessionReceiptChain.schema_version:
                chains.append(document)
            elif payload.get("receipt_id") and (
                "selected_context" in payload or "omitted_context" in payload
            ):
                receipt_id = str(payload["receipt_id"])
                existing = receipts.get(receipt_id)
                if existing is not None and existing.payload != payload:
                    self.diagnostics.append(
                        {
                            "type": "duplicate_receipt",
                            "source": document.source,
                            "message": f"Conflicting receipt id {receipt_id}; newest copy selected.",
                        }
                    )
                if existing is None or document.modified_at >= existing.modified_at:
                    receipts[receipt_id] = document

        consumed: set[str] = set()
        for document in chains:
            try:
                chain = SessionReceiptChain.from_dict(document.payload)
                integrity = chain.verify_integrity()
            except (TypeError, ValueError) as exc:
                self.diagnostics.append(
                    {"type": type(exc).__name__, "source": document.source, "message": _bounded_text(exc)}
                )
                continue
            views: list[dict[str, Any]] = []
            missing: list[str] = []
            for link in chain.links[-_MAX_RECEIPT_ITEMS:]:
                lookup_id = link.source_receipt_id or link.receipt_id
                record = receipts.get(lookup_id)
                if record is None:
                    missing.append(lookup_id)
                    views.append(
                        {
                            "receipt_id": link.receipt_id,
                            "query": _bounded_text(link.query, 1000),
                            "token_budget": link.token_budget,
                            "selected_tokens": 0,
                            "source_tokens": 0,
                            "omitted_tokens": 0,
                            "selected_count": 0,
                            "omitted_count": 0,
                            "selected": [],
                            "omitted": [],
                            "warnings": ["Receipt body is unavailable; chain metadata only."],
                            "risk_summary": dict(link.risk_summary),
                            "integrity": {"valid": False, "issues": ["receipt body missing"]},
                            "model": _model_and_cost({}, selected_tokens=0),
                            "created_at": link.created_at,
                            "turn_index": link.turn_index,
                        }
                    )
                    continue
                consumed.add(lookup_id)
                view = _receipt_view(record.payload)
                view.update(created_at=link.created_at, turn_index=link.turn_index)
                views.append(view)
            if missing:
                integrity = dict(integrity)
                integrity["valid"] = False
                integrity["issues"] = [
                    *integrity.get("issues", []),
                    f"{len(missing)} receipt body/bodies missing",
                ]
            session_key = self._key(f"chain:{chain.session_id}:{document.source}")
            self._sessions[session_key] = self._session_payload(
                key=session_key,
                session_id=chain.session_id,
                receipts=views,
                integrity=integrity,
                source=document.source,
                modified_at=document.modified_at,
            )

        for receipt_id, document in receipts.items():
            if receipt_id in consumed:
                continue
            view = _receipt_view(document.payload)
            view.update(created_at=document.modified_at, turn_index=0)
            session_key = self._key(f"receipt:{receipt_id}:{document.source}")
            self._sessions[session_key] = self._session_payload(
                key=session_key,
                session_id=f"receipt:{receipt_id}",
                receipts=[view],
                integrity=view["integrity"],
                source=document.source,
                modified_at=document.modified_at,
            )

    @staticmethod
    def _session_payload(
        *,
        key: str,
        session_id: str,
        receipts: list[dict[str, Any]],
        integrity: Mapping[str, Any],
        source: str,
        modified_at: float,
    ) -> dict[str, Any]:
        latest = receipts[-1] if receipts else _receipt_view({})
        created_at = latest.get("created_at") or modified_at
        summary = {
            "key": key,
            "session_id": _bounded_text(session_id, 300),
            "query": latest.get("query", ""),
            "turn_count": len(receipts),
            "created_at": created_at,
            "receipt_id": latest.get("receipt_id", ""),
            "token_budget": latest.get("token_budget", 0),
            "selected_tokens": latest.get("selected_tokens", 0),
            "omitted_tokens": latest.get("omitted_tokens", 0),
            "selected_count": latest.get("selected_count", 0),
            "omitted_count": latest.get("omitted_count", 0),
            "budget_utilization": latest.get("budget_utilization"),
            "selection_ratio": latest.get("selection_ratio"),
            "model": latest.get("model", {}),
            "integrity": dict(integrity),
        }
        return {
            "summary": summary,
            "source": source,
            "receipts": receipts,
            "integrity": dict(integrity),
        }

    def list_sessions(self, *, query: str = "", limit: int = 50) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 100))
        needle = query.strip().casefold()
        summaries = [session["summary"] for session in self._sessions.values()]
        if needle:
            summaries = [
                summary
                for summary in summaries
                if needle
                in " ".join(
                    str(summary.get(field, ""))
                    for field in ("session_id", "query", "receipt_id")
                ).casefold()
                or needle in str(_mapping(summary.get("model")).get("model") or "").casefold()
            ]
        summaries.sort(key=lambda item: (float(item.get("created_at") or 0), item["key"]), reverse=True)
        return {
            "sessions": summaries[:safe_limit],
            "total": len(summaries),
            "diagnostics": self.diagnostics[:20],
        }

    def get_session(self, key: str) -> dict[str, Any] | None:
        if not re_key(key):
            return None
        return self._sessions.get(key)


def re_key(value: str) -> bool:
    return len(value) == 24 and all(character in "0123456789abcdef" for character in value)
