"""Local artifact store for Context Receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_STORE = Path(".entroly") / "receipts"
DEFAULT_INDEX = DEFAULT_STORE / "index.json"
LATEST_POINTER = DEFAULT_STORE / "latest_receipt.txt"


def ensure_store(path: Path = DEFAULT_STORE) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def default_receipt_path(receipt_id: str) -> Path:
    return ensure_store() / f"{receipt_id}.json"


def default_report_path(receipt_id: str) -> Path:
    return ensure_store() / f"{receipt_id}.md"


def set_latest_receipt(path: str | Path) -> None:
    ensure_store()
    LATEST_POINTER.write_text(str(Path(path)), encoding="utf-8")


def latest_receipt_path() -> Path | None:
    if not LATEST_POINTER.exists():
        return None
    raw = LATEST_POINTER.read_text(encoding="utf-8").strip()
    return Path(raw) if raw else None
