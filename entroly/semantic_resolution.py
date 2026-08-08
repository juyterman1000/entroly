"""
Semantic Resolution Protocol (SRP) — Information-Optimal File Reads
====================================================================

A context-reading protocol that selects per-block resolution using
information-theoretic scoring instead of requiring the agent to choose a
single fixed read mode.

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
    DIFF   — signature + unified diff vs ``previous_source`` (1-line context)
    LOW    — name + type annotation only
    SKIP   — omitted entirely (0 tokens)

DIFF is enabled per-block when the caller passes ``previous_source``
to ``resolve()`` — typically a change-driven flow (post-commit /
post-edit) where the agent needs to see *what changed* without
re-paying for unchanged code. Modified blocks become eligible for
DIFF; unchanged blocks fall through to the standard ladder.
"""
from __future__ import annotations

import difflib
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
    DIFF = "diff"
    STRUCTURE = "structure"
    LOW = "low"
    SKIP = "skip"

    # Cost multipliers relative to full source. DIFF sits between LOW and
    # MEDIUM: it's cheaper than emitting the whole signature+docstring,
    # but slightly richer than a stub because it conveys the *delta*.
    COST = {
        "full": 1.0,
        "medium": 0.25,
        "diff": 0.15,
        "structure": 0.20,
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
    # Optional previous version of this block's source. Populated by
    # ``resolve(..., previous_source=...)`` when SRP is invoked in a
    # change-driven flow. Used to render the DIFF resolution.
    previous_source: str = ""

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

    @property
    def diff(self) -> str:
        """DIFF resolution: signature + compact unified diff vs previous_source.

        Returns an empty string when no previous version is attached or
        when the block is unchanged — callers should fall back to a
        sibling resolution in that case (`_render_block` handles it).

        Diff context is intentionally tight (n=1) because the agent reads
        DIFF blocks to learn *what changed*, not to re-derive structure
        from surrounding context — SRP already provides structure via
        the signature line prepended to the diff.
        """
        if not self.previous_source or self.previous_source == self.source:
            return ""
        diff_lines = list(difflib.unified_diff(
            self.previous_source.splitlines(),
            self.source.splitlines(),
            lineterm="",
            n=1,
            fromfile=f"{self.name}~",
            tofile=self.name,
        ))
        return "\n".join(diff_lines)

    @property
    def is_modified(self) -> bool:
        """True iff a previous_source is attached and differs from current."""
        return bool(self.previous_source) and self.previous_source != self.source


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
    # Set when the caller pinned a resolution with `resolve(resolution=...)`.
    # A pinned resolution is honoured exactly and is NOT demoted to fit, so the
    # output can exceed `budget`; `over_budget` says whether it did. Silently
    # truncating a resolution the caller explicitly asked for would defeat the
    # purpose of asking.
    forced_resolution: str | None = None
    over_budget: bool = False
    # Inclusive, 1-indexed line range for exact range reads. ``None`` means
    # the normal semantic-resolution path was used.
    line_range: tuple[int, int] | None = None
    # Identifies the implementation used for caller-forced STRUCTURE reads.
    # ``native-skeleton`` is the shared Rust engine; ``full-fallback`` means
    # the engine could not produce a useful smaller outline, so source was
    # returned losslessly instead of inventing or corrupting structure.
    structure_backend: str | None = None


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


def _extract_structure(source: str, file_path: str) -> tuple[str, str]:
    """Return a native structural outline, failing open to exact source.

    Importability is not a sufficient native capability check: an older wheel
    can import successfully while lacking this newer export. The explicit
    ``getattr`` keeps editable/source checkouts and stale wheels safe.
    """
    try:
        import entroly_core

        extractor = getattr(entroly_core, "extract_skeleton", None)
        if callable(extractor):
            outline = extractor(source, file_path)
            if isinstance(outline, str) and outline.strip():
                return outline, "native-skeleton"
    except Exception as exc:
        logger.debug("Native structure extraction unavailable: %s", exc)

    # The native extractor deliberately declines unknown languages, tiny files,
    # and outlines that would retain more than 70% of the source. Returning the
    # exact source is honest and lossless in all three cases.
    return source, "full-fallback"


# ── Relevance Scoring ────────────────────────────────────────────────

def _term_overlap(query: str, text: str) -> float:
    """Query-term coverage: fraction of the query's terms present in the text.

        |Q ∩ T| / |Q|

    NOT Jaccard (|Q ∩ T| / |Q ∪ T|): Jaccard is symmetric and dominated by the
    larger set, so for a real code block (hundreds of tokens) against a short
    query it collapses to ≈0.01–0.02 for *every* block — the query signal is
    drowned out and no block is ever scored as relevant. Coverage measures "how
    much of what I asked for does this block contain", which is the correct
    relevance signal and is not diluted by block size.

    Query terms shorter than 3 chars (``or``, ``of``, ``to``) are dropped so
    common glue words don't inflate coverage; if that empties the query we fall
    back to the full term set.
    """
    q_terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3}
    if not q_terms:
        q_terms = set(re.findall(r"\w+", query.lower()))
    t_terms = set(re.findall(r"\w+", text.lower()))

    if not q_terms or not t_terms:
        return 0.0

    return len(q_terms & t_terms) / len(q_terms)


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


def _split_camel(name: str) -> str:
    """Insert separators at camelCase and PascalCase boundaries."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)


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
    else:
        # Identifier-aware match. Requiring the whole underscored symbol as a
        # literal substring means a query can name a function in prose and
        # still miss it: "chunk oversized source files into parts" never
        # contains "chunk_oversized_source", so the block that answers the
        # question scored 0.5 while unrelated blocks reached FULL. Measured on
        # entroly/auto_index.py, that is exactly what happened.
        #
        # Splitting the symbol the way the retrieval tokenizer already splits
        # identifiers lets the two meet: {chunk, oversized, source} is fully
        # contained in the query's words, so this is a complete name match, not
        # a partial one. Scored by the fraction matched so a symbol sharing one
        # common word does not get the same credit as one fully named.
        name_parts = {
            part
            for part in re.split(r"[^a-z0-9]+", _split_camel(block.name).lower())
            if len(part) > 2
        }
        q_words = {w.strip(".,()[]{}:;\"'") for w in q_lower.split()}
        if name_parts:
            matched = len(name_parts & q_words) / len(name_parts)
            if matched >= 1.0:
                name_bonus = 1.0
            elif matched > 0.0:
                name_bonus = max(0.5, matched)
        if name_bonus == 0.0 and any(
            w in block.name.lower() for w in q_lower.split()
        ):
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

    Modified blocks (``block.is_modified``) are eligible for the DIFF
    level: it captures the change-driven signal without paying for the
    unchanged portion of the block. For high-relevance blocks we still
    prefer FULL — the agent wants the entire function when it's the
    one being edited. DIFF is most valuable for the *contextual*
    blocks around the change.
    """
    has_diff = block.is_modified

    if relevance > 0.5:
        return Resolution.FULL
    if relevance > 0.25:
        if budget_pressure < 0.5:
            return Resolution.FULL
        if has_diff and budget_pressure < 0.7:
            return Resolution.DIFF
        return Resolution.MEDIUM
    if relevance > 0.10:
        if has_diff:
            return Resolution.DIFF
        if budget_pressure < 0.3:
            return Resolution.MEDIUM
        return Resolution.LOW
    if relevance > 0.02:
        if has_diff:
            return Resolution.DIFF
        return Resolution.LOW
    return Resolution.SKIP


def _render_block(block: CodeBlock, resolution: str) -> str:
    """Render a block at the specified resolution level."""
    if resolution == Resolution.FULL:
        return block.source
    if resolution == Resolution.MEDIUM:
        return block.summary
    if resolution == Resolution.DIFF:
        diff_text = block.diff
        if not diff_text:
            # Diff unavailable (no previous_source attached, or unchanged).
            # Fall back to LOW so the block still contributes a stub.
            return block.stub
        # Prepend the signature so the LLM has a structural anchor for
        # the diff lines (which only carry +/- without context type).
        return f"{block.signature}\n{diff_text}"
    if resolution == Resolution.LOW:
        return block.stub
    return ""  # SKIP


def resolve(
    source: str,
    query: str = "",
    budget: int = 1000,
    file_path: str = "",
    previous_source: str | None = None,
    resolution: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
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
    previous_source : str
        Optional previous version of the file. When provided, SRP
        enables the DIFF resolution: modified blocks render as a
        compact unified diff (signature + n=1 unified hunks) instead
        of full source or signature-only. Ideal for change-driven
        flows (post-commit, post-edit, agent revision loops) where
        the agent must learn *what changed* without re-reading the
        unchanged portion. Unmodified blocks fall through to the
        standard FULL / MEDIUM / LOW / SKIP ladder.
    resolution : str | None
        Pin every block to one level, bypassing automatic assignment.
        One of ``Resolution.FULL`` / ``MEDIUM`` / ``DIFF`` / ``STRUCTURE`` /
        ``LOW``. STRUCTURE keeps declarations and signatures while removing
        implementation bodies when the native engine can do so profitably.

        Automatic assignment is the right default and stays the default, but
        it cannot be right for every question. Measured on this repository, a
        signature-level view answered 12/12 questions whose evidence lives in
        a signature and **0/20** whose evidence lives in a function body. No
        single automatic choice serves both, so a caller that knows which kind
        of question it is asking needs a way to say so.

        A pinned resolution is honoured exactly: it is **not** demoted to fit
        the budget, because silently truncating the level the caller asked for
        would defeat the point of asking. The result reports ``over_budget``
        instead, so the caller can see the cost rather than discover a
        quietly-degraded answer.
    line_start, line_end : int | None
        Inclusive, 1-indexed exact line range. Both values are required
        together and cannot be combined with ``resolution``. Newline
        characters inside the selected range are preserved.

    Returns
    -------
    SRPResult
        Mixed-resolution file representation with metadata.
    """
    if resolution is not None and resolution not in {
        Resolution.FULL, Resolution.MEDIUM, Resolution.DIFF,
        Resolution.STRUCTURE, Resolution.LOW,
    }:
        # SKIP is deliberately excluded: pinning every block to SKIP asks for
        # an empty document, which is never what a caller means.
        raise ValueError(
            "resolution must be one of full/medium/diff/structure/low, "
            f"got {resolution!r}"
        )

    has_line_start = line_start is not None
    has_line_end = line_end is not None
    if has_line_start != has_line_end:
        raise ValueError("line_start and line_end must be provided together")
    if has_line_start:
        if resolution is not None:
            raise ValueError("line ranges cannot be combined with resolution")
        assert line_start is not None and line_end is not None
        lines = source.splitlines(keepends=True)
        if line_start < 1 or line_end < line_start:
            raise ValueError("line range must satisfy 1 <= line_start <= line_end")
        if line_end > len(lines):
            raise ValueError(
                f"line_end {line_end} exceeds file length {len(lines)}"
            )
        exact_range = "".join(lines[line_start - 1:line_end])
        range_tokens = (
            max(1, int(len(exact_range) / _CHARS_PER_TOKEN) + 1)
            if exact_range else 0
        )
        return SRPResult(
            output=exact_range,
            file_path=file_path,
            total_blocks=1,
            resolution_counts={"lines": 1},
            total_tokens=range_tokens,
            budget=budget,
            over_budget=range_tokens > budget,
            line_range=(line_start, line_end),
        )

    # FULL is the caller's lossless escape hatch. Per-block FULL remains part
    # of automatic SRP, but a caller-forced FULL must return the original text
    # exactly: reconstructing extracted blocks drops imports, assignments,
    # comments, and inter-block whitespace.
    if resolution == Resolution.FULL:
        full_tokens = max(1, int(len(source) / _CHARS_PER_TOKEN) + 1) if source else 0
        return SRPResult(
            output=source,
            file_path=file_path,
            total_blocks=1,
            resolution_counts={Resolution.FULL: 1},
            total_tokens=full_tokens,
            budget=budget,
            forced_resolution=Resolution.FULL,
            over_budget=full_tokens > budget,
        )

    # A forced DIFF is a whole-file fidelity mode, not a per-block heuristic.
    # Requiring the baseline prevents the previous silent failure where DIFF
    # without ``previous_source`` emitted LOW stubs while reporting "diff".
    # Whole-file unified diff also preserves additions and deletions that do
    # not have a matching block in both versions.
    if resolution == Resolution.DIFF:
        if previous_source is None:
            raise ValueError("resolution='diff' requires previous_source")
        label = file_path or "<source>"
        diff_output = "\n".join(difflib.unified_diff(
            previous_source.splitlines(),
            source.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="",
            n=1,
        ))
        diff_tokens = (
            max(1, int(len(diff_output) / _CHARS_PER_TOKEN) + 1)
            if diff_output else 0
        )
        return SRPResult(
            output=diff_output,
            file_path=file_path,
            total_blocks=1,
            resolution_counts={Resolution.DIFF: 1},
            total_tokens=diff_tokens,
            budget=budget,
            forced_resolution=Resolution.DIFF,
            over_budget=diff_tokens > budget,
        )

    if resolution == Resolution.STRUCTURE:
        structure_output, structure_backend = _extract_structure(source, file_path)
        structure_tokens = (
            max(1, int(len(structure_output) / _CHARS_PER_TOKEN) + 1)
            if structure_output else 0
        )
        delivered_resolution = (
            Resolution.STRUCTURE
            if structure_backend == "native-skeleton"
            else Resolution.FULL
        )
        return SRPResult(
            output=structure_output,
            file_path=file_path,
            total_blocks=1,
            resolution_counts={delivered_resolution: 1},
            total_tokens=structure_tokens,
            budget=budget,
            forced_resolution=Resolution.STRUCTURE,
            over_budget=structure_tokens > budget,
            structure_backend=structure_backend,
        )

    blocks = extract_blocks(source, file_path)

    # Attach previous_source per-block for DIFF eligibility. Match by
    # (name, kind) — handles re-ordering, additions, and deletions
    # gracefully. Blocks that exist only in the new version stay
    # unmatched and render at standard resolutions (no diff to show).
    if previous_source is not None:
        prev_blocks = extract_blocks(previous_source, file_path)
        prev_by_key: dict[tuple[str, str], str] = {}
        for pb in prev_blocks:
            prev_by_key[(pb.name, pb.kind)] = pb.source
        for blk in blocks:
            prev = prev_by_key.get((blk.name, blk.kind))
            if prev is not None:
                blk.previous_source = prev

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
        assigned = (
            resolution
            if resolution is not None
            else _assign_resolution(block, relevance, budget_pressure)
        )
        output_text = _render_block(block, assigned)
        tokens = max(0, int(len(output_text) / _CHARS_PER_TOKEN) + 1) if output_text else 0

        resolved.append(ResolvedBlock(
            block=block,
            resolution=assigned,
            relevance=relevance,
            output=output_text,
            tokens=tokens,
        ))

    # ── Budget enforcement via greedy demotion ──
    # If total tokens exceed budget, demote lowest-relevance blocks.
    # Skipped entirely when the caller pinned a resolution: demoting it would
    # silently return a different level than the one requested.
    total_tokens = sum(r.tokens for r in resolved)

    if total_tokens > budget and resolution is None:
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

            has_diff = r.block.is_modified
            if r.resolution == Resolution.FULL:
                # Modified blocks prefer DIFF on the first demotion step:
                # captures the change-driven signal at a fraction of the
                # cost. Unmodified blocks demote to MEDIUM as before.
                new_res = Resolution.DIFF if has_diff else Resolution.MEDIUM
            elif r.resolution == Resolution.MEDIUM:
                new_res = Resolution.DIFF if has_diff else Resolution.LOW
            elif r.resolution == Resolution.DIFF:
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

    # ── Budget-fill (runs after demotion too) ──
    # A single oversized block (e.g. a 50K-token class) demoted to fit the
    # budget overshoots far below it, stranding room the most-relevant fitting
    # blocks (e.g. the actual request handler) should claim. This was an
    # ``elif`` on the demotion branch, so a post-demotion overshoot never
    # re-filled — smart_read under-resolved to 126/1500 on real files.
    #
    # Skipped when the caller pinned a resolution. Promotion is as much a
    # violation of a pin as demotion is: asking for LOW and receiving FULL
    # because there happened to be spare budget returns a different level than
    # the one requested. Caught by `test_low_returns_only_stubs`, which saw
    # `{'full'}` where it had pinned `low`.
    if total_tokens < budget and resolution is None:
        # ── Budget utilization via greedy promotion ──
        # Under budget → upgrade the most-relevant blocks toward FULL so the
        # spare budget actually surfaces query-relevant detail (the tool's
        # whole point). Without this, an under-budget read left every block at
        # LOW even when there was room to show the matching function in full.
        # Only LOW/MEDIUM promote — DIFF keeps its change-driven meaning and
        # SKIP keeps irrelevant blocks omitted. Highest-relevance block first,
        # re-scanning from the top after each upgrade so the most relevant
        # block reaches FULL before less relevant ones gain detail.
        # SKIP belongs in the ladder. Without it a block scored below the skip
        # threshold is unreachable no matter how much budget is free, so a file
        # whose blocks mostly score low can never spend what the caller asked
        # for. Measured: smart_read("entroly/provenance.py", budget=800) used 23
        # tokens -- 3% -- with 7 of 8 blocks skipped and the block the query
        # named left at a bare signature.
        #
        # This does not weaken SKIP as the default for irrelevant content: the
        # initial classification is unchanged, and upgrades only happen while
        # `total_tokens < budget`, strictly highest-relevance first, re-scanning
        # after each step. So the block that best matches the query still climbs
        # to FULL before anything less relevant gains a single line, and a read
        # that is already at budget behaves exactly as before.
        _upgrade = {
            Resolution.SKIP: Resolution.LOW,
            Resolution.LOW: Resolution.MEDIUM,
            Resolution.MEDIUM: Resolution.FULL,
        }
        by_rel_desc = sorted(range(len(resolved)), key=lambda i: -resolved[i].relevance)
        improved = True
        while improved and total_tokens < budget:
            improved = False
            for idx in by_rel_desc:
                r = resolved[idx]
                new_res = _upgrade.get(r.resolution)
                if new_res is None:
                    continue
                new_output = _render_block(r.block, new_res)
                new_tokens = max(0, int(len(new_output) / _CHARS_PER_TOKEN) + 1) if new_output else 0
                if total_tokens - r.tokens + new_tokens <= budget:
                    total_tokens += new_tokens - r.tokens
                    resolved[idx] = ResolvedBlock(
                        block=r.block,
                        resolution=new_res,
                        relevance=r.relevance,
                        output=new_output,
                        tokens=new_tokens,
                    )
                    improved = True
                    break  # re-scan from the top (highest relevance) after each upgrade

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
        forced_resolution=resolution,
        over_budget=final_tokens > budget,
    )
