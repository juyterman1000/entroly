from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


text = README.read_text(encoding="utf-8")
text = replace_once(
    text,
    "Entroly uses budgeted context selection, content-addressed recovery, and auditable receipts to lower provider-bound inference expenditure—without rewriting your codebase or agent architecture.",
    "Entroly uses budgeted context selection, content-addressed evidence, exact recovery, and auditable receipts to lower provider-bound inference expenditure—without rewriting your codebase or agent architecture.",
    "identity wording",
)
text = replace_once(
    text,
    '  <a href="#see-value-in-your-first-session"><b>See value</b></a> ·\n',
    '  <a href="#see-value-in-your-first-session"><b>See value</b></a> ·\n  <a href="docs/ai-cost-optimization.html"><b>AI cost guide</b></a> ·\n',
    "canonical AI cost link",
)
text = replace_once(
    text,
    'alt="Entroly repository and GitHub stars"',
    'alt="Entroly GitHub stars"',
    "canonical GitHub stars badge alt",
)
text = replace_once(
    text,
    "<sub>These are offline exact-evidence pilots on frozen SQuAD v2 subsets, not\ndownstream answer-quality, latency, production-savings, or general neural\nsuperiority claims. PRISM-R remains opt-in research code.</sub>",
    "<sub>These are offline exact-evidence pilots on frozen SQuAD v2 subsets. They\ndo not measure generated answers and are not downstream answer-quality, latency,\nproduction-savings, or general neural superiority claims. PRISM-R is an opt-in\nresearch prototype, is not the default compressor, and remains opt-in research code.</sub>",
    "PRISM-R scope",
)
text = replace_once(
    text,
    '<a href="https://lobehub.com/mcp/juyterman1000-entroly"><img src="https://lobehub.com/badge/mcp/juyterman1000-entroly" alt="Current external Entroly status on LobeHub"></a>\n\n',
    "",
    "external marketplace badge",
)

required = (
    "content-addressed evidence",
    "docs/ai-cost-optimization.html",
    'alt="Entroly GitHub stars"',
    "PRISM-R is an opt-in research prototype",
    "do not measure generated answers",
    "not the default compressor",
)
for marker in required:
    if marker not in text:
        raise RuntimeError(f"README missing required marker: {marker}")
if "lobehub.com/badge/" in text:
    raise RuntimeError("README still contains an external marketplace badge")

README.write_text(text, encoding="utf-8")
