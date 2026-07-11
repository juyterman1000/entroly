#!/usr/bin/env python3
"""One-time guarded patch for model decision receipt integration.

The script is intentionally exact and idempotent. It changes only the import
surface and the return block of ``stable_request_fingerprint``.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("entroly/control_plane.py")
IMPORT_NEEDLE = "from typing import Any, Literal\n"
IMPORT_REPLACEMENT = (
    "from typing import Any, Literal\n\n"
    "from .model_decision_receipt import model_decision_tags\n"
)
RETURN_NEEDLE = '''    return {
        "entroly_request_fingerprint": _fingerprint(payload),
        "entroly_protocol_fingerprint": _fingerprint(protocol),
        "entroly_tool_fingerprint": _fingerprint(tool_contract),
        "entroly_header_fingerprint": _fingerprint(sticky_headers),
    }
'''
RETURN_REPLACEMENT = '''    tags = {
        "entroly_request_fingerprint": _fingerprint(payload),
        "entroly_protocol_fingerprint": _fingerprint(protocol),
        "entroly_tool_fingerprint": _fingerprint(tool_contract),
        "entroly_header_fingerprint": _fingerprint(sticky_headers),
    }
    tags.update(
        model_decision_tags(
            payload,
            provider=detected_provider,
            path=path,
        )
    )
    return tags
'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    if "from .model_decision_receipt import model_decision_tags" not in text:
        if text.count(IMPORT_NEEDLE) != 1:
            raise SystemExit("control_plane import anchor changed; refusing unsafe patch")
        text = text.replace(IMPORT_NEEDLE, IMPORT_REPLACEMENT, 1)

    if "model_decision_tags(" not in text.split("def stable_request_fingerprint", 1)[-1]:
        if text.count(RETURN_NEEDLE) != 1:
            raise SystemExit("stable_request_fingerprint anchor changed; refusing unsafe patch")
        text = text.replace(RETURN_NEEDLE, RETURN_REPLACEMENT, 1)

    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
