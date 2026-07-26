from __future__ import annotations

from pathlib import Path

path = Path("scripts/verify_readme.py")
text = path.read_text(encoding="utf-8")
old = '''check(
    "README developer Context OS identity",
    lambda: require(
        all(
            phrase in README_TEXT
            for phrase in (
                "Entroly — The Open-Source Context OS for AI Agents",
                "Keep your agent. Give it a Context OS.",
                "recoverable compression",
                "Context Receipts",
                "local-first",
            )
        ),
        "README no longer states the developer and AI-agent Context OS contract",
    ),
)
'''
new = '''check(
    "README developer Context Assurance and Context OS identity",
    lambda: require(
        all(
            phrase in README_TEXT
            for phrase in (
                "Entroly — Drop-In Context Assurance to Lower AI Operational Cost",
                "Reduce unnecessary context without losing control of critical evidence.",
                "content-addressed evidence",
                "recoverable compression",
                "Context OS",
                "Context Receipts",
                "local-first",
            )
        ),
        "README no longer states the cost-led Context Assurance and AI-agent Context OS contract",
    ),
)
'''
if new in text:
    raise RuntimeError("README identity verifier is already updated")
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected exactly one stale identity contract, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
