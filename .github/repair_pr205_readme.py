from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def ensure_literal(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one old match, found {count}")
    return text.replace(old, new, 1)


def ensure_regex(text: str, pattern: str, replacement: str, marker: str, label: str) -> str:
    if marker in text:
        return text
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one bounded match, found {count}")
    return updated


text = README.read_text(encoding="utf-8")
text = ensure_literal(
    text,
    "Entroly uses budgeted context selection, content-addressed recovery, and auditable receipts to lower provider-bound inference expenditure—without rewriting your codebase or agent architecture.",
    "Entroly uses budgeted context selection, content-addressed evidence, exact recovery, and auditable receipts to lower provider-bound inference expenditure—without rewriting your codebase or agent architecture.",
    "identity wording",
)
text = ensure_literal(
    text,
    '  <a href="#see-value-in-your-first-session"><b>See value</b></a> ·\n',
    '  <a href="#see-value-in-your-first-session"><b>See value</b></a> ·\n  <a href="docs/ai-cost-optimization.html"><b>AI cost guide</b></a> ·\n',
    "canonical AI cost link",
)
text = ensure_literal(
    text,
    'alt="Entroly repository and GitHub stars"',
    'alt="Entroly GitHub stars"',
    "canonical GitHub stars badge alt",
)
text = ensure_regex(
    text,
    r"<sub>These are offline exact-evidence pilots on frozen SQuAD v2 subsets,.*?</sub>",
    "<sub>These are offline exact-evidence pilots on frozen SQuAD v2 subsets. They\n"
    "do not measure generated answers and are not downstream answer-quality, latency,\n"
    "production-savings, or general neural superiority claims. PRISM-R is an opt-in\n"
    "research prototype, is not the default compressor, and remains opt-in research code.</sub>",
    "PRISM-R is an opt-in research prototype",
    "PRISM-R scope",
)
if "lobehub.com/badge/" in text:
    text, count = re.subn(
        r'<a href="https://lobehub\.com/mcp/juyterman1000-entroly"><img src="https://lobehub\.com/badge/mcp/juyterman1000-entroly" alt="Current external Entroly status on LobeHub"></a>\s*',
        "",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"external marketplace badge: expected one match, found {count}")

required = (
    "content-addressed evidence",
    "docs/ai-cost-optimization.html",
    'alt="Entroly GitHub stars"',
    "PRISM-R is an opt-in research prototype",
    "do not measure generated answers",
    "not the default compressor",
    "remains opt-in research code",
)
for marker in required:
    if marker not in text:
        raise RuntimeError(f"README missing required marker: {marker}")
if "lobehub.com/badge/" in text:
    raise RuntimeError("README still contains an external marketplace badge")

README.write_text(text, encoding="utf-8")
