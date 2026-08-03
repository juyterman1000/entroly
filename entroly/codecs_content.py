"""Content codecs for code, documents, conversation and API schemas.

Each follows the `entroly.codec` contract: offer representations, declare what
was protected, hand back a recovery reference for what was dropped, and never
claim the result is *sufficient* -- that decision needs the whole selection.

What each protects is chosen from what the content is read for:

* **code** -- imports and signatures, because a body without its imports and
  call surface cannot be reasoned about; error strings, because they are what a
  user greps for.
* **documents** -- the spans that answer the query plus their neighbours, and
  citations, because an unattributed claim is not evidence.
* **conversation** -- standing instructions and decisions, which are the part a
  later turn is still bound by, and the cache-hot prefix, which must not be
  rewritten or every subsequent request pays a cache miss.
* **schema** -- required fields, types, enum values and error shapes, which are
  the contract; prose descriptions and surplus examples are not.
"""

from __future__ import annotations

import re
from typing import Any

from .codec import (
    RecoveryStore,
    Representation,
    SupportDecision,
    content_digest,
    estimate_tokens,
)

# ── Code ────────────────────────────────────────────────────────────────────

_DEF_RE = re.compile(
    r"^\s*(?:@|def |class |async def |fn |pub fn |impl |struct |enum |"
    r"function |const |export |interface |type )",
)
_IMPORT_RE = re.compile(r"^\s*(?:import |from .+ import|use |require\(|#include)")
_ERROR_STRING_RE = re.compile(r"""["'][^"']*(?:error|fail|invalid|denied|expired)[^"']*["']""", re.I)


def _python_ast_skeleton(text: str) -> tuple[list[str], list[str], tuple[str, ...]] | None:
    """(kept, dropped, protected) using a real parse, or None if not Python.

    The line-regex skeleton could not tell a `def` from the word "define" in a
    string, kept decorators only when they started a line, and had no idea
    where a body ended -- so it dropped continuation lines of a signature and
    kept stray lines that merely looked declarative.

    A parse gives exact line ranges. Signature lines, decorators, docstrings,
    imports and module-level assignments are kept; statement bodies are
    dropped and returned for recovery. Anything that does not parse -- another
    language, a syntax error, a partial file -- returns None so the caller
    falls back rather than guessing.
    """
    import ast

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None

    lines = text.split("\n")
    keep: set[int] = set()          # 1-indexed
    protected: list[str] = []

    def signature_span(node) -> tuple[int, int]:
        """Decorators through the line the signature's colon closes on."""
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        body_starts_at = node.body[0].lineno if node.body else node.end_lineno
        # A docstring is part of the surface a caller reads.
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body_starts_at = first.end_lineno + 1
        return start, max(start, body_starts_at - 1)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.update(range(node.lineno, node.end_lineno + 1))
            protected.append(lines[node.lineno - 1].strip())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = signature_span(node)
            keep.update(range(start, end + 1))
            protected.append(lines[node.lineno - 1].strip())

    # Module-level constants are part of the surface; bodies of functions are not.
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            keep.update(range(node.lineno, node.end_lineno + 1))

    kept, dropped = [], []
    for i, line in enumerate(lines, start=1):
        if i in keep or not line.strip():
            kept.append(line)
        else:
            dropped.append(line)
    return kept, dropped, tuple(dict.fromkeys(protected))[:40]


class CodeCodec:
    """Source code: full, skeleton (imports + signatures), or reference only.

    The skeleton keeps every import and declaration line and drops bodies. That
    is the representation a caller wants when it needs to know a module's
    surface without paying for its implementation, and it is reversible because
    the dropped bodies are stored.
    """

    name = "code"
    version = "1"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type in {"code", "source"}:
            return SupportDecision(True, 1.0, "declared content type")
        lines = text.split("\n")
        if len(lines) < 3:
            return SupportDecision(False, 0.0, "too short to be a source file")
        decls = sum(1 for line in lines if _DEF_RE.match(line))
        imports = sum(1 for line in lines if _IMPORT_RE.match(line))
        if decls == 0:
            return SupportDecision(False, 0.0, "no declarations found")
        # Confidence must not scale with the DENSITY of declarations: a large,
        # well-factored module has a low ratio and was losing the routing race
        # to ShellCodec, which claimed it on comment lines. Imports plus
        # declarations are a positive signal of source code regardless of file
        # length, so count evidence rather than measure its dilution.
        evidence = min(decls, 10) + min(imports, 10)
        return SupportDecision(
            True,
            min(0.92, 0.55 + 0.02 * evidence),
            f"{decls} declarations, {imports} imports",
        )

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#code.full",
                source_id=source_id,
                content_type="code",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]

        # Prefer a real parse; fall back to line heuristics for other
        # languages or for input that does not parse.
        parsed = _python_ast_skeleton(text)
        if parsed is not None:
            kept, dropped, ast_protected = parsed
            method = "ast"
            # Error strings are what a user greps for, so pull them back in.
            errs = {ln for ln in dropped if _ERROR_STRING_RE.search(ln)}
            if errs:
                dropped = [ln for ln in dropped if ln not in errs]
                keepset = set(kept) | errs
                kept = [
                    ln
                    for ln in text.split("\n")
                    if ln in keepset or not ln.strip()
                ]
        else:
            ast_protected = ()
            method = "lines"
            kept, dropped = [], []
            for line in text.split("\n"):
                if (
                    _IMPORT_RE.match(line)
                    or _DEF_RE.match(line)
                    or _ERROR_STRING_RE.search(line)
                    or not line.strip()
                ):
                    kept.append(line)
                else:
                    dropped.append(line)

        skeleton = "\n".join(kept)
        if not dropped or len(skeleton) >= len(text):
            return reps

        recovery = self.store.put(
            "\n".join(dropped),
            item_count=len(dropped),
            note=f"bodies elided from {source_id or 'source'}",
        )
        protected = ast_protected or tuple(
            dict.fromkeys(
                [ln.strip() for ln in kept if _IMPORT_RE.match(ln)][:20]
                + [ln.strip() for ln in kept if _DEF_RE.match(ln)][:20]
            )
        )
        # Only claim what the emitted skeleton actually contains.
        protected = tuple(p for p in protected if p in skeleton)
        reps.append(
            Representation(
                representation_id=f"{source_id}#code.skeleton.{method}",
                source_id=source_id,
                content_type="code",
                text=skeleton,
                token_cost=estimate_tokens(skeleton),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                protected_evidence=protected,
                distortion_risk=1.0 - (len(skeleton) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


# ── Documents / RAG ─────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[\d+\]|\([A-Z][a-z]+(?: et al\.?)?,? \d{4}\)|https?://\S+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class DocumentCodec:
    """Prose documents, conditioned on a query when one is supplied.

    A topically relevant span is not automatically an answer-bearing span, so
    selection keeps the neighbours of every matched sentence: the sentence that
    names the subject and the one that answers about it are frequently not the
    same sentence. Citations are protected because an unattributed claim is not
    evidence.

    With no query this declines rather than guessing which prose matters.
    """

    name = "document"
    version = "1"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type in {"document", "prose", "rag"}:
            return SupportDecision(True, 0.9, "declared content type")
        if len(text) < 400:
            return SupportDecision(False, 0.0, "too short to be worth extracting")
        sentences = _SENTENCE_SPLIT.split(text)
        if len(sentences) < 5:
            return SupportDecision(False, 0.0, "not sentence-structured")
        return SupportDecision(True, 0.4, "sentence-structured prose")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        query = str(options.get("query", "") or "")
        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#doc.full",
                source_id=source_id,
                content_type="document",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]
        if not query.strip():
            # Without a query there is no basis for calling one span more
            # answer-bearing than another. Offer the original only.
            return reps

        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        terms = {t for t in re.findall(r"[A-Za-z0-9_]{3,}", query.lower())}
        hit = {
            i
            for i, s in enumerate(sentences)
            if terms & set(re.findall(r"[A-Za-z0-9_]{3,}", s.lower()))
        }
        if not hit:
            return reps

        # Neighbour protection: a matched sentence keeps the one before and
        # after it, which is where the answer often actually sits.
        keep = set()
        for i in hit:
            keep.update({i - 1, i, i + 1})
        keep = {i for i in keep if 0 <= i < len(sentences)}
        # Citations are evidence regardless of whether they matched the query.
        keep.update(i for i, s in enumerate(sentences) if _CITATION_RE.search(s))

        selected = "\n".join(sentences[i] for i in sorted(keep))
        if len(selected) >= len(text):
            return reps

        dropped = [s for i, s in enumerate(sentences) if i not in keep]
        recovery = self.store.put(
            "\n".join(dropped),
            item_count=len(dropped),
            note=f"spans not selected from {source_id or 'document'}",
        )
        protected = tuple(
            dict.fromkeys(m.group(0) for s in sentences if s for m in _CITATION_RE.finditer(s))
        )[:20]
        reps.append(
            Representation(
                representation_id=f"{source_id}#doc.spans",
                source_id=source_id,
                content_type="document",
                text=selected,
                token_cost=estimate_tokens(selected),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                protected_evidence=protected,
                distortion_risk=1.0 - (len(selected) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


# ── Conversation / agent memory ─────────────────────────────────────────────

_ROLE_RE = re.compile(r"^\s*(system|user|assistant|tool)\s*:", re.IGNORECASE)
_DECISION_RE = re.compile(
    r"\b(?:decided|decision|we will|must|never|always|do not|don't|instead of|"
    r"agreed|chose|prefer)\b",
    re.IGNORECASE,
)


class ConversationCodec:
    """Conversation history, split by what a later turn is still bound by.

    Standing instructions and recorded decisions stay verbatim: those bind
    future turns and paraphrasing them changes what the agent is committed to.
    Narrative history is what gets dropped.

    The leading system block is emitted unchanged and first. Rewriting it would
    invalidate the cached prefix on every subsequent request, so any saving
    there is paid back immediately in cache misses.
    """

    name = "conversation"
    version = "1"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type in {"conversation", "chat", "history"}:
            return SupportDecision(True, 1.0, "declared content type")
        turns = [ln for ln in text.split("\n") if _ROLE_RE.match(ln)]
        if len(turns) < 3:
            return SupportDecision(False, 0.0, "fewer than three role-tagged turns")
        return SupportDecision(True, 0.75, f"{len(turns)} role-tagged turns")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#conv.full",
                source_id=source_id,
                content_type="conversation",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]

        lines = text.split("\n")
        kept, dropped = [], []
        in_prefix = True
        for line in lines:
            role = _ROLE_RE.match(line)
            if in_prefix and (not role or role.group(1).lower() == "system"):
                kept.append(line)          # cache-hot prefix, untouched
                continue
            in_prefix = False
            if role and role.group(1).lower() == "system":
                kept.append(line)
            elif _DECISION_RE.search(line):
                kept.append(line)
            elif not line.strip():
                kept.append(line)
            else:
                dropped.append(line)

        pruned = "\n".join(kept)
        if not dropped or len(pruned) >= len(text):
            return reps

        recovery = self.store.put(
            "\n".join(dropped),
            item_count=len(dropped),
            note=f"narrative turns pruned from {source_id or 'conversation'}",
        )
        protected = tuple(
            dict.fromkeys(
                ln.strip()
                for ln in kept
                if ln.strip() and (_DECISION_RE.search(ln) or _ROLE_RE.match(ln))
            )
        )[:20]
        reps.append(
            Representation(
                representation_id=f"{source_id}#conv.pruned",
                source_id=source_id,
                content_type="conversation",
                text=pruned,
                token_cost=estimate_tokens(pruned),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                protected_evidence=protected,
                distortion_risk=1.0 - (len(pruned) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


# ── API schemas ─────────────────────────────────────────────────────────────

_CONTRACT_KEYS = {
    "required", "type", "enum", "properties", "items", "format", "pattern",
    "minimum", "maximum", "minLength", "maxLength", "oneOf", "anyOf", "allOf",
    "$ref", "additionalProperties", "nullable", "default", "responses",
    "parameters", "schema", "errors", "error",
}
_PROSE_KEYS = {"description", "summary", "title", "example", "examples", "externalDocs"}


class SchemaCodec:
    """OpenAPI / JSON-Schema payloads.

    Compresses prose, never the contract. Required-field lists, types, enum
    values, constraints and error shapes are what a caller must satisfy;
    descriptions and surplus examples are commentary. Dropping a constraint
    changes the contract, which is why this codec removes prose only.
    """

    name = "schema"
    version = "1"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        import json

        if content_type in {"schema", "openapi", "jsonschema"}:
            return SupportDecision(True, 1.0, "declared content type")
        stripped = text.strip()
        if not stripped.startswith("{"):
            return SupportDecision(False, 0.0, "not a JSON object")
        try:
            data = json.loads(stripped)
        except ValueError:
            return SupportDecision(False, 0.0, "does not parse")
        if not isinstance(data, dict):
            return SupportDecision(False, 0.0, "not an object")
        markers = {"openapi", "swagger", "$schema", "properties", "definitions", "components"}
        # Beat JsonCodec (0.9) only when this really is a schema.
        if markers & set(data):
            return SupportDecision(True, 0.95, "schema markers present")
        return SupportDecision(False, 0.0, "JSON but not a schema")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        import json

        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#schema.full",
                source_id=source_id,
                content_type="schema",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]
        try:
            data = json.loads(text)
        except ValueError:
            return reps

        dropped: list[str] = []

        def strip_prose(node: Any) -> Any:
            if isinstance(node, dict):
                out = {}
                for key, value in node.items():
                    if key in _PROSE_KEYS and isinstance(value, str) and len(value) > 40:
                        dropped.append(f"{key}: {value}")
                        continue
                    out[key] = strip_prose(value)
                return out
            if isinstance(node, list):
                return [strip_prose(v) for v in node]
            return node

        lean = json.dumps(strip_prose(data), indent=2)
        if not dropped or len(lean) >= len(text):
            return reps

        recovery = self.store.put(
            "\n".join(dropped),
            item_count=len(dropped),
            note=f"prose removed from {source_id or 'schema'}",
        )
        protected = tuple(k for k in _CONTRACT_KEYS if f'"{k}"' in lean)[:20]
        reps.append(
            Representation(
                representation_id=f"{source_id}#schema.lean",
                source_id=source_id,
                content_type="schema",
                text=lean,
                token_cost=estimate_tokens(lean),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                protected_evidence=protected,
                distortion_risk=1.0 - (len(lean) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps
