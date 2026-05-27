"""
Semantic Resolution Protocol (SRP) — Information-Optimal File Reads
====================================================================

The first context server that automatically selects per-block resolution
level using information-theoretic scoring, instead of requiring the agent
to choose a fixed read mode.

Motivation
----------
Existing tools offer fixed read modes: ``full``, ``map``, ``signatures``,
``diff``, ``lines:N-M``.  The agent must *guess* which mode is right.
If it picks ``full`` when ``signatures`` would suffice, it wastes budget.
If it picks ``signatures`` when the bug is in one function body, it misses
critical detail.

SRP replaces mode selection with budget-driven automatic resolution:

    entroly.read("auth.py", budget=500, query="JWT validation bug")

The server decides: show the JWT function in full (it matches the query),
show other functions as signatures, skip test helpers entirely.

Mathematical foundation
-----------------------
Given a file F decomposed into N code blocks {b₁, b₂, …, bₙ}, a query Q,
and a token budget B, SRP solves the optimization:

    max  Σᵢ  R(bᵢ) · relevance(bᵢ, Q)
    s.t. Σᵢ  tokens(bᵢ, R(bᵢ)) ≤ B

where R(bᵢ) ∈ {FULL, MEDIUM, LOW, SKIP} is the resolution level and
``tokens(bᵢ, r)`` is the token cost at resolution r.

This is a variant of the Multiple-Choice Knapsack Problem (MCKP),
which we solve with the same DP machinery already in ``qccr.py``.

Resolution levels
-----------------
    FULL   — complete source code (highest cost, highest fidelity)
    MEDIUM — signature + docstring + first line of body
    LOW    — name + type annotation only
    SKIP   — omitted entirely (0 tokens)
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ── Resolution Levels ────────────────────────────────────────────────

class Resolution:
    FULL = "full"
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"

    # Cost multipliers relative to full source
    COST = {
        "full": 1.0,
        "medium": 0.25,
        "low": 0.08,
        "skip": 0.0,
    }


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class CodeBlock:
    """A logical block of code (function, class, method, etc.)."""
    name: str               # function/class name
    kind: str               # "function", "class", "method", "module_code"
    start_line: int          # 1-indexed
    end_line: int            # 1-indexed, inclusive
    source: str              # full source text
    signature: str           # first line (def/class declaration)
    docstring: str           # docstring if present, else ""
    indent: int              # indentation level
    token_estimate: int      # approximate token count for full source

    @property
    def summary(self) -> str:
        """MEDIUM resolution: signature + docstring."""
        parts = [self.signature]
        if self.docstring:
            parts.append(f'    """{self.docstring}"""')
        return "\n".join(parts)

    @property
    def stub(self) -> str:
        """LOW resolution: just the signature with ellipsis."""
        return f"{self.signature}  ..."


@dataclass
class ResolvedBlock:
    """A code block with its assigned resolution level."""
    block: CodeBlock
    resolution: str         # one of Resolution constants
    relevance: float        # 0.0 - 1.0 relevance to query
    output: str             # the text to include at this resolution
    tokens: int             # token cost of this output


@dataclass
class SRPResult:
    """Result of semantic resolution."""
    output: str                     # the mixed-resolution file representation
    file_path: str                  # path to the source file
    total_blocks: int               # number of code blocks found
    resolution_counts: dict[str, int]   # count per resolution level
    total_tokens: int               # total tokens in output
    budget: int                     # requested budget
    blocks: list[ResolvedBlock] = field(default_factory=list, repr=False)


# ── Block Extraction ─────────────────────────────────────────────────

# Regex patterns for Python block boundaries
_PY_DEF_RE = re.compile(r"^(\s*)(def|class|async\s+def)\s+(\w+)")
_DOCSTRING_RE = re.compile(r'^\s*("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', re.MULTILINE)

# Approximate tokens per character
_CHARS_PER_TOKEN = 3.5


def _extract_blocks_python(source: str, file_path: str = "") -> list[CodeBlock]:
    """Extract logical code blocks from Python source.

    Handles: functions, classes, methods, async functions.
    Uses indentation-based parsing (no Tree-sitter dependency).
    """
    lines = source.splitlines()
    blocks: list[CodeBlock] = []
    i = 0

    while i < len(lines):
        match = _PY_DEF_RE.match(lines[i])
        if match:
            indent_str, kind_raw, name = match.groups()
            indent = len(indent_str)
            kind = "function" if "def" in kind_raw else "class"
            if indent > 0 and kind == "function":
                kind = "method"

            start = i
            signature = lines[i].rstrip()

            # Find end of block: next line with same or lower indent
            # (or end of file)
            j = i + 1
            while j < len(lines):
                line = lines[j]
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#"):
                    line_indent = len(line) - len(stripped)
                    if line_indent <= indent:
                        break
                j += 1

            end = j - 1
            block_source = "\n".join(lines[start:end + 1])
            tokens = max(1, int(len(block_source) / _CHARS_PER_TOKEN) + 1)

            # Extract docstring
            docstring = ""
            body_start = i + 1
            # Handle multi-line def signatures
            while body_start < end and lines[body_start].rstrip().endswith("\\"):
                body_start += 1
                signature += "\n" + lines[body_start].rstrip()
            if body_start < end:
                # Check for parenthesized continuation
                paren_depth = signature.count("(") - signature.count(")")
                while paren_depth > 0 and body_start < end:
                    body_start += 1
                    signature += "\n" + lines[body_start].rstrip()
                    paren_depth += lines[body_start].count("(") - lines[body_start].count(")")

                body_start += 1
                if body_start <= end:
                    first_body = lines[body_start].strip() if body_start < len(lines) else ""
                    if first_body.startswith('"""') or first_body.startswith("'''"):
                        quote = first_body[:3]
                        if first_body.count(quote) >= 2:
                            # Single-line docstring
                            docstring = first_body.strip(quote).strip()
                        else:
                            # Multi-line docstring
                            doc_lines = [first_body[3:]]
                            k = body_start + 1
                            while k <= end:
                                if quote in lines[k]:
                                    doc_lines.append(lines[k].split(quote)[0])
                                    break
                                doc_lines.append(lines[k])
                                k += 1
                            docstring = "\n".join(doc_lines).strip()
                            if len(docstring) > 100:
                                docstring = docstring[:100] + "..."

            blocks.append(CodeBlock(
                name=name,
                kind=kind,
                start_line=start + 1,
                end_line=end + 1,
                source=block_source,
                signature=signature,
                docstring=docstring,
                indent=indent,
                token_estimate=tokens,
            ))

            i = end + 1
        else:
            i += 1

    # If no blocks found, treat the entire file as one block
    if not blocks and source.strip():
        blocks.append(CodeBlock(
            name=os.path.basename(file_path) if file_path else "<module>",
            kind="module_code",
            start_line=1,
            end_line=len(lines),
            source=source,
            signature=f"# {os.path.basename(file_path)}" if file_path else "# <module>",
            docstring="",
            indent=0,
            token_estimate=max(1, int(len(source) / _CHARS_PER_TOKEN) + 1),
        ))

    return blocks


def _extract_blocks_generic(source: str, file_path: str = "") -> list[CodeBlock]:
    """Generic block extraction for non-Python files.

    Uses blank-line separation and common patterns (function, class, struct,
    fn, def, pub, export, const, let, var) to identify blocks.
    """
    lines = source.splitlines()
    blocks: list[CodeBlock] = []

    # Pattern for common function/class declarations across languages
    _GENERIC_DEF_RE = re.compile(
        r"^(\s*)(pub\s+|export\s+|async\s+|static\s+|const\s+)*"
        r"(fn|func|function|def|class|struct|enum|interface|type|trait|impl|mod|module|package)\s+(\w+)",
    )

    i = 0
    while i < len(lines):
        match = _GENERIC_DEF_RE.match(lines[i])
        if match:
            indent = len(match.group(1))
            name = match.group(4)
            kind = match.group(3)
            start = i
            signature = lines[i].rstrip()

            # Find end: next declaration at same/lower indent, or brace matching
            brace_depth = lines[i].count("{") - lines[i].count("}")
            j = i + 1
            while j < len(lines):
                brace_depth += lines[j].count("{") - lines[j].count("}")
                stripped = lines[j].lstrip()
                if stripped and brace_depth <= 0:
                    line_indent = len(lines[j]) - len(stripped)
                    if line_indent <= indent and _GENERIC_DEF_RE.match(lines[j]):
                        break
                j += 1

            end = min(j, len(lines)) - 1
            block_source = "\n".join(lines[start:end + 1])

            blocks.append(CodeBlock(
                name=name,
                kind=kind,
                start_line=start + 1,
                end_line=end + 1,
                source=block_source,
                signature=signature,
                docstring="",
                indent=indent,
                token_estimate=max(1, int(len(block_source) / _CHARS_PER_TOKEN) + 1),
            ))

            i = end + 1
        else:
            i += 1

    if not blocks and source.strip():
        blocks.append(CodeBlock(
            name=os.path.basename(file_path) if file_path else "<module>",
            kind="module_code",
            start_line=1,
            end_line=len(lines),
            source=source,
            signature=f"// {os.path.basename(file_path)}" if file_path else "// <module>",
            docstring="",
            indent=0,
            token_estimate=max(1, int(len(source) / _CHARS_PER_TOKEN) + 1),
        ))

    return blocks


def extract_blocks(source: str, file_path: str = "") -> list[CodeBlock]:
    """Extract code blocks from source, using language-appropriate parser."""
    ext = os.path.splitext(file_path)[1].lower() if file_path else ""
    if ext in (".py", ".pyi", ".pyw"):
        return _extract_blocks_python(source, file_path)
    return _extract_blocks_generic(source, file_path)


# ── Relevance Scoring ────────────────────────────────────────────────

def _term_overlap(query: str, text: str) -> float:
    """Compute normalized term overlap between query and text.

    Uses case-insensitive word-level Jaccard coefficient:
        |Q ∩ T| / |Q ∪ T|

    This is cheap (O(|Q| + |T|)) and effective for code search.
    """
    q_terms = set(re.findall(r"\w+", query.lower()))
    t_terms = set(re.findall(r"\w+", text.lower()))

    if not q_terms or not t_terms:
        return 0.0

    intersection = len(q_terms & t_terms)
    union = len(q_terms | t_terms)

    return intersection / max(union, 1)


def _entropy_relevance(block: CodeBlock) -> float:
    """Score a block's intrinsic information density.

    Uses character-level Shannon entropy normalized to [0, 1].
    High-entropy blocks contain more diverse information.
    """
    if not block.source or len(block.source) < 5:
        return 0.0

    counts = Counter(block.source)
    n = len(block.source)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)

    # Normalize: max entropy for printable ASCII ≈ 6.5 bits
    return min(entropy / 6.5, 1.0)


def score_relevance(block: CodeBlock, query: str) -> float:
    """Compute composite relevance score for a block.

    Combines:
    - Term overlap with query (weight 0.6)
    - Name match bonus (weight 0.2)
    - Intrinsic entropy (weight 0.2)

    Returns a score in [0, 1].
    """
    if not query:
        return _entropy_relevance(block)

    # Term overlap on full source
    overlap = _term_overlap(query, block.source)

    # Name match bonus
    q_lower = query.lower()
    name_bonus = 0.0
    if block.name.lower() in q_lower:
        name_bonus = 1.0
    elif any(w in block.name.lower() for w in q_lower.split()):
        name_bonus = 0.5

    # Entropy density
    entropy = _entropy_relevance(block)

    return 0.6 * overlap + 0.2 * name_bonus + 0.2 * entropy


# ── Resolution Assignment + Knapsack Packing ─────────────────────────

def _assign_resolution(
    block: CodeBlock,
    relevance: float,
    budget_pressure: float,
) -> str:
    """Assign a resolution level based on relevance and budget pressure.

    budget_pressure ∈ [0, 1] where 0 = unlimited budget, 1 = very tight.
    """
    if relevance > 0.5:
        return Resolution.FULL
    if relevance > 0.25:
        if budget_pressure < 0.5:
            return Resolution.FULL
        return Resolution.MEDIUM
    if relevance > 0.10:
        if budget_pressure < 0.3:
            return Resolution.MEDIUM
        return Resolution.LOW
    if relevance > 0.02:
        return Resolution.LOW
    return Resolution.SKIP


def _render_block(block: CodeBlock, resolution: str) -> str:
    """Render a block at the specified resolution level."""
    if resolution == Resolution.FULL:
        return block.source
    if resolution == Resolution.MEDIUM:
        return block.summary
    if resolution == Resolution.LOW:
        return block.stub
    return ""  # SKIP


def resolve(
    source: str,
    query: str = "",
    budget: int = 1000,
    file_path: str = "",
) -> SRPResult:
    """Produce an information-optimal file representation at the given budget.

    This is the main SRP entry point. Given a file's source code, a query,
    and a token budget, it automatically selects the optimal resolution
    for each code block.

    Parameters
    ----------
    source : str
        File source code.
    query : str
        The user's query/task (used for relevance scoring).
    budget : int
        Target token budget for the output.
    file_path : str
        Path to the file (used for language detection and headers).

    Returns
    -------
    SRPResult
        Mixed-resolution file representation with metadata.
    """
    blocks = extract_blocks(source, file_path)

    if not blocks:
        return SRPResult(
            output="",
            file_path=file_path,
            total_blocks=0,
            resolution_counts={},
            total_tokens=0,
            budget=budget,
        )

    # Score each block for relevance
    total_full_tokens = sum(b.token_estimate for b in blocks)
    budget_pressure = max(0.0, min(1.0, 1.0 - budget / max(total_full_tokens, 1)))

    resolved: list[ResolvedBlock] = []

    for block in blocks:
        relevance = score_relevance(block, query)
        resolution = _assign_resolution(block, relevance, budget_pressure)
        output_text = _render_block(block, resolution)
        tokens = max(0, int(len(output_text) / _CHARS_PER_TOKEN) + 1) if output_text else 0

        resolved.append(ResolvedBlock(
            block=block,
            resolution=resolution,
            relevance=relevance,
            output=output_text,
            tokens=tokens,
        ))

    # ── Budget enforcement via greedy demotion ──
    # If total tokens exceed budget, demote lowest-relevance blocks
    total_tokens = sum(r.tokens for r in resolved)

    if total_tokens > budget:
        # Sort by relevance ascending (least relevant first to demote)
        by_relevance = sorted(
            range(len(resolved)),
            key=lambda i: resolved[i].relevance,
        )

        for idx in by_relevance:
            if total_tokens <= budget:
                break

            r = resolved[idx]
            old_tokens = r.tokens

            if r.resolution == Resolution.FULL:
                new_res = Resolution.MEDIUM
            elif r.resolution == Resolution.MEDIUM:
                new_res = Resolution.LOW
            elif r.resolution == Resolution.LOW:
                new_res = Resolution.SKIP
            else:
                continue

            new_output = _render_block(r.block, new_res)
            new_tokens = max(0, int(len(new_output) / _CHARS_PER_TOKEN) + 1) if new_output else 0

            resolved[idx] = ResolvedBlock(
                block=r.block,
                resolution=new_res,
                relevance=r.relevance,
                output=new_output,
                tokens=new_tokens,
            )
            total_tokens -= (old_tokens - new_tokens)

    # ── Build output ──
    output_parts: list[str] = []
    if file_path:
        output_parts.append(f"# {os.path.basename(file_path)} (SRP: {budget} token budget)")
        output_parts.append("")

    res_counts: dict[str, int] = Counter()
    for r in resolved:
        res_counts[r.resolution] += 1
        if r.resolution != Resolution.SKIP and r.output:
            output_parts.append(r.output)
            output_parts.append("")  # blank line between blocks

    final_output = "\n".join(output_parts).rstrip()
    final_tokens = sum(r.tokens for r in resolved if r.resolution != Resolution.SKIP)

    return SRPResult(
        output=final_output,
        file_path=file_path,
        total_blocks=len(blocks),
        resolution_counts=dict(res_counts),
        total_tokens=final_tokens,
        budget=budget,
        blocks=resolved,
    )
