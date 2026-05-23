"""
Multi-Modal Ingestion — Convert non-text content to structured text fragments.

Research grounding:
  - MMCode (ICLR 2025/2026): current SOTA models struggle with visually-aided
    code generation. The key finding: visual elements only add value when converted
    to *structured semantic descriptions*, not raw OCR text.
  - SWE-bench Multimodal (ICLR 2025): framed as GUI element extraction +
    structured text. Architecture diagrams are interaction graphs, not images.
  - UGround (ICLR 2025, Boyu Gou et al.): 10M GUI element grounding dataset.
    Key: spatial layout matters as much as textual content.

Design principles:
  1. Every converter outputs a `ModalContent` — a richly structured text blob
     that `remember_fragment` ingests normally. No special storage path.
  2. Zero hard dependencies — all converters degrade gracefully.
  3. Semantic extraction beats OCR: we extract structure, relationships,
     and intent — not just raw character sequences.
  4. Confidence scores: each converter reports how confident it is in the
     extraction quality. Low-confidence extractions are flagged.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Core Data Type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModalContent:
    """
    The normalized output of every multi-modal converter.

    Always ingested via remember_fragment(text, source=source, ...).
    All metadata is embedded in `text` in a structured format.
    """
    text: str
    source_type: str          # 'image', 'diagram', 'voice', 'diff', 'screenshot'
    source: str               # Original file path or identifier
    confidence: float         # [0.0, 1.0] — quality of extraction
    token_estimate: int       # Rough token count of `text`
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.token_estimate = max(1, len(self.text) // 4)
        self.confidence = max(0.0, min(1.0, self.confidence))


# ─────────────────────────────────────────────────────────────────────────────
# Image / Screenshot Ingestion
#
# Strategy (MMCode insight): raw OCR text loses spatial structure. We:
#   1. Attempt OCR via pytesseract if available (high confidence)
#   2. Fall back to base64-decoding and extracting embedded text from simple
#      PNG/JPEG metadata if OCR unavailable (medium confidence)
#   3. If content is a pre-described string (e.g., from an LLM vision call),
#      structure it into a canonical format (high confidence)
#
# The spatial layout annotation (top/center/bottom regions) is inspired by
# UGround's finding that element position is as important as label text.
# ─────────────────────────────────────────────────────────────────────────────


def ingest_image(
    image_data: str,
    source: str,
    description: str = "",
    region_hints: list[str] | None = None,
) -> ModalContent:
    """
    Convert an image (path, base64 string, or pre-written description) into
    a structured text fragment for context retrieval.

    Args:
        image_data: Either a file path, a base64-encoded image string, or a
                    plain-text description of the image content.
        source:     Human-readable identifier (e.g., 'screenshot_auth_flow.png').
        description: Optional pre-existing description to supplement extraction.
        region_hints: Optional list of UI regions present, e.g. ['navbar', 'form', 'modal'].

    Returns:
        ModalContent with a structured description suitable for semantic search.
    """
    extracted_text = ""
    confidence = 0.5
    method = "description"

    # Strategy 1: File path → try pytesseract OCR
    if os.path.isfile(image_data):
        extracted_text, confidence = _ocr_file(image_data)
        method = "ocr"

    # Strategy 2: Base64 string → decode → OCR
    elif _is_base64(image_data):
        try:
            img_bytes = base64.b64decode(image_data)
            extracted_text, confidence = _ocr_bytes(img_bytes)
            method = "ocr_base64"
        except Exception:
            pass

    # Strategy 3: Pre-written description (from LLM vision call, etc.)
    if image_data and not _is_base64(image_data) and not os.path.isfile(image_data):
        extracted_text = image_data
        confidence = 0.80
        method = "pre-described"

    # Merge with explicit description if provided
    if description:
        if extracted_text:
            extracted_text = f"{description}\n\nExtracted text:\n{extracted_text}"
        else:
            extracted_text = description
        confidence = max(confidence, 0.75)

    # Structure the output (MMCode insight: structure beats raw text)
    regions = region_hints or _infer_ui_regions(extracted_text)
    structured = _format_image_content(source, extracted_text, regions, method, confidence)

    return ModalContent(
        text=structured,
        source_type="image",
        source=source,
        confidence=confidence,
        token_estimate=len(structured) // 4,
        metadata={"method": method, "regions": regions},
    )


def _ocr_file(path: str) -> tuple[str, float]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        conf = 0.90 if text.strip() else 0.20
        return text.strip(), conf
    except ImportError:
        return "", 0.30
    except Exception:
        return "", 0.20


def _ocr_bytes(img_bytes: bytes) -> tuple[str, float]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        conf = 0.85 if text.strip() else 0.20
        return text.strip(), conf
    except ImportError:
        return "", 0.30
    except Exception:
        return "", 0.20


def _is_base64(s: str) -> bool:
    if len(s) < 64 or len(s) % 4 != 0:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def _infer_ui_regions(text: str) -> list[str]:
    """Infer UI regions from text content heuristics."""
    regions = []
    lower = text.lower()
    if any(w in lower for w in ["login", "sign in", "username", "password", "email"]):
        regions.append("authentication-form")
    if any(w in lower for w in ["nav", "menu", "header", "breadcrumb"]):
        regions.append("navigation")
    if any(w in lower for w in ["button", "submit", "click", "action"]):
        regions.append("interactive-controls")
    if any(w in lower for w in ["table", "list", "row", "column", "grid"]):
        regions.append("data-table")
    if any(w in lower for w in ["error", "warning", "alert", "modal", "dialog"]):
        regions.append("alert-dialog")
    if any(w in lower for w in ["chart", "graph", "visualization", "axis"]):
        regions.append("data-visualization")
    return regions or ["general-ui"]


def _format_image_content(
    source: str,
    text: str,
    regions: list[str],
    method: str,
    confidence: float,
) -> str:
    lines = [
        f"# Visual Content: {source}",
        f"## Extraction: {method} (confidence: {confidence:.0%})",
        "",
        "### UI Regions Detected",
        *[f"- {r}" for r in regions],
        "",
        "### Extracted Content",
        text or "(no text content extracted)",
        "",
        "### Semantic Tags",
        f"source_type=image regions={','.join(regions)}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Diagram Ingestion — Architecture / Flow / ER Diagrams
#
# Strategy: Extract semantic graph structure (nodes + edges + labels),
# NOT pixel content. Architecture diagrams are interaction graphs.
#
# Supported formats:
#   - Mermaid .mmd text (flowchart, sequenceDiagram, classDiagram, erDiagram)
#   - PlantUML .puml text
#   - DOT/Graphviz .dot text
#   - Informal text description (fallback)
# ─────────────────────────────────────────────────────────────────────────────


def ingest_diagram(
    diagram_text: str,
    source: str,
    diagram_type: str = "auto",
) -> ModalContent:
    """
    Convert a diagram (Mermaid, PlantUML, DOT, or descriptive text) into a
    structured semantic description that captures nodes, edges, and flow.

    Args:
        diagram_text: The raw diagram source text.
        source:       Identifier (e.g., 'arch_diagram.mmd').
        diagram_type: One of 'mermaid', 'plantuml', 'dot', 'text', or 'auto'.

    Returns:
        ModalContent with extracted graph structure as queryable text.
    """
    detected = diagram_type
    if detected == "auto":
        detected = _detect_diagram_type(diagram_text)

    if detected == "mermaid":
        nodes, edges, metadata = _parse_mermaid(diagram_text)
        confidence = 0.90
    elif detected == "plantuml":
        nodes, edges, metadata = _parse_plantuml(diagram_text)
        confidence = 0.85
    elif detected == "dot":
        nodes, edges, metadata = _parse_dot(diagram_text)
        confidence = 0.80
    else:
        nodes, edges, metadata = _parse_text_diagram(diagram_text)
        confidence = 0.55

    structured = _format_diagram_content(source, detected, nodes, edges, metadata)

    return ModalContent(
        text=structured,
        source_type="diagram",
        source=source,
        confidence=confidence,
        token_estimate=len(structured) // 4,
        metadata={
            "diagram_type": detected,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    )


def _detect_diagram_type(text: str) -> str:
    lower = text.lower().strip()
    if lower.startswith(("graph ", "flowchart", "sequencediagram", "classdiagram",
                          "erdiagram", "gantt", "pie", "gitgraph", "mindmap", "statediagram")):
        return "mermaid"
    if lower.startswith(("@startuml", "@startmindmap", "@startsalt")):
        return "plantuml"
    if lower.startswith("digraph") or lower.startswith("graph {"):
        return "dot"
    return "text"


def _parse_mermaid(text: str) -> tuple[list[str], list[tuple[str, str, str]], dict]:
    """
    Parse Mermaid diagram text into nodes + edges.

    Handles: flowchart, sequenceDiagram, classDiagram, erDiagram.
    Uses line-by-line pattern matching (no full parser — resilient to variations).
    """
    nodes: list[str] = []
    edges: list[tuple[str, str, str]] = []  # (from, to, label)
    metadata: dict[str, Any] = {}

    lines = text.strip().splitlines()
    diagram_kind = "unknown"

    # Arrow patterns for flowchart: --> , -.- , ==> , -- text --> , -->|label|
    arrow_re = re.compile(r'(.+?)\s*(?:-->|==>|-.->|--\|(.+?)\|)\s*(.+)')
    # Sequence diagram: Participant A -> B: message
    seq_re = re.compile(r'(\w[\w\s]*?)\s*(?:->+|->>|-->>|--x)\s*(\w[\w\s]*?)\s*:\s*(.+)')
    # Class diagram: ClassA --|> ClassB
    class_rel_re = re.compile(r'(\w+)\s*(?:\|>|<\|--|o--|--o|\*--|--\*|\.\.>|<\.\.|\.\.\|>|<\|\.\.)?\s*(\w+)')
    # Entity definitions
    node_label_re = re.compile(r'(\w+)\s*[\[\({"\'](.+?)[\]\)}"\']\s*$')

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue

        # Detect diagram type from first line
        lower = stripped.lower()
        if any(lower.startswith(k) for k in ["flowchart", "graph "]):
            diagram_kind = "flowchart"
            metadata["direction"] = stripped.split()[-1] if len(stripped.split()) > 1 else "LR"
            continue
        if lower.startswith("sequencediagram"):
            diagram_kind = "sequence"
            continue
        if lower.startswith("classdiagram"):
            diagram_kind = "class"
            continue
        if lower.startswith("erdiagram"):
            diagram_kind = "er"
            continue

        # Extract nodes with labels
        nm = node_label_re.match(stripped)
        if nm:
            node_name = nm.group(1)
            node_label = nm.group(2).strip()
            if node_name not in nodes:
                nodes.append(f"{node_name}: {node_label}")
            continue

        # Extract arrows (flowchart)
        am = arrow_re.match(stripped)
        if am:
            src = am.group(1).strip().split("[")[0].split("(")[0].strip('"\' ')
            dst = am.group(3).strip().split("[")[0].split("(")[0].strip('"\' ')
            label = am.group(2) or ""
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, label.strip()))
            continue

        # Sequence diagram messages
        sm = seq_re.match(stripped)
        if sm:
            src, dst, msg = sm.group(1).strip(), sm.group(2).strip(), sm.group(3).strip()
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, msg))
            continue

        # Class relationships
        if diagram_kind == "class":
            cr = class_rel_re.match(stripped)
            if cr:
                a, b = cr.group(1), cr.group(2)
                if a not in nodes:
                    nodes.append(a)
                if b and b not in nodes:
                    nodes.append(b)
                if b:
                    edges.append((a, b, ""))

    metadata["diagram_kind"] = diagram_kind
    return nodes, edges, metadata


def _parse_plantuml(text: str) -> tuple[list[str], list[tuple[str, str, str]], dict]:
    nodes: list[str] = []
    edges: list[tuple[str, str, str]] = []
    metadata: dict = {"format": "plantuml"}

    arrow_re = re.compile(r'(\w[\w\s]*?)\s*(?:->|-->|<-|<--|->>|<<--)\s*(\w[\w\s]*?)\s*(?::\s*(.+))?$')
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("'") or stripped.startswith("@"):
            continue
        m = arrow_re.match(stripped)
        if m:
            src, dst = m.group(1).strip(), m.group(2).strip()
            label = m.group(3) or ""
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, label.strip()))
        elif stripped.startswith(("actor", "participant", "component", "database",
                                   "cloud", "node", "usecase", "class", "interface",
                                   "enum", "entity")):
            parts = stripped.split(None, 2)
            name = parts[1].strip('"\'') if len(parts) > 1 else stripped
            if name not in nodes:
                nodes.append(name)

    return nodes, edges, metadata


def _parse_dot(text: str) -> tuple[list[str], list[tuple[str, str, str]], dict]:
    nodes: list[str] = []
    edges: list[tuple[str, str, str]] = []
    metadata: dict = {"format": "graphviz"}

    node_re = re.compile(r'^\s*"?(\w+)"?\s*\[.*?label\s*=\s*"([^"]*)"')
    edge_re = re.compile(r'^\s*"?(\w+)"?\s*-[->]\s*"?(\w+)"?(?:\s*\[.*?label\s*=\s*"([^"]*)")?')

    for line in text.splitlines():
        nm = node_re.match(line)
        if nm:
            label = f"{nm.group(1)}: {nm.group(2)}"
            if label not in nodes:
                nodes.append(label)
            continue
        em = edge_re.match(line)
        if em:
            src, dst = em.group(1), em.group(2)
            label = em.group(3) or ""
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, label))

    return nodes, edges, metadata


def _parse_text_diagram(text: str) -> tuple[list[str], list[tuple[str, str, str]], dict]:
    """
    Heuristic extraction from informal diagram descriptions.
    Looks for: arrow-like patterns (→, ->, =>), component names in boxes/brackets.
    """
    nodes: list[str] = []
    edges: list[tuple[str, str, str]] = []

    arrow_re = re.compile(r'(\w[\w\s\-_]*?)\s*(?:→|->|=>|⟶|──>)\s*(\w[\w\s\-_]*?)(?:\s*:\s*(.+))?$')
    box_re = re.compile(r'[\[\(<%]\s*(.+?)\s*[\]\)>%]')

    for line in text.splitlines():
        m = arrow_re.search(line)
        if m:
            src, dst = m.group(1).strip(), m.group(2).strip()
            label = m.group(3) or ""
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)
            edges.append((src, dst, label.strip()))
            continue
        for match in box_re.finditer(line):
            label = match.group(1).strip()
            if label and label not in nodes and len(label) < 60:
                nodes.append(label)

    return nodes, edges, {"format": "text"}


def _format_diagram_content(
    source: str,
    diagram_type: str,
    nodes: list[str],
    edges: list[tuple[str, str, str]],
    metadata: dict,
) -> str:
    lines = [
        f"# Architecture Diagram: {source}",
        f"## Type: {diagram_type} | Nodes: {len(nodes)} | Edges: {len(edges)}",
        "",
        "### Components / Entities",
    ]
    for n in nodes[:50]:
        lines.append(f"  - {n}")
    if len(nodes) > 50:
        lines.append(f"  ... and {len(nodes) - 50} more")

    lines += ["", "### Relationships / Data Flow"]
    for src, dst, label in edges[:80]:
        rel = f"  - {src} → {dst}"
        if label:
            rel += f"  [{label}]"
        lines.append(rel)
    if len(edges) > 80:
        lines.append(f"  ... and {len(edges) - 80} more relationships")

    if metadata.get("diagram_kind"):
        lines += ["", f"### Diagram Kind: {metadata['diagram_kind']}"]

    lines += [
        "",
        "### Semantic Tags",
        f"source_type=diagram diagram_format={diagram_type} nodes={len(nodes)} edges={len(edges)}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Voice / Transcript Ingestion
#
# Takes pre-transcribed text (from Whisper, AssemblyAI, etc.) and converts it
# into a structured fragment. The NLP-light structuring extracts:
#   - Key decisions made ("we should", "let's", "I think we need to")
#   - Technical terms (CamelCase words, file extensions, known tech names)
#   - Action items ("TODO", "we need to", "action item:")
#   - Architectural mentions (service names, API endpoints, schema references)
# ─────────────────────────────────────────────────────────────────────────────

# Common tech terms that indicate software discussion
_TECH_TERMS = frozenset([
    "api", "endpoint", "service", "database", "schema", "migration", "cache",
    "queue", "event", "webhook", "auth", "oauth", "jwt", "token", "session",
    "redis", "postgres", "mongodb", "kafka", "rabbitmq", "docker", "kubernetes",
    "lambda", "function", "class", "module", "interface", "repository", "controller",
    "model", "view", "router", "middleware", "pipeline", "context", "fragment",
    "latency", "throughput", "timeout", "retry", "circuit-breaker", "idempotent",
    "async", "await", "promise", "callback", "stream", "subscribe", "publish",
    "graphql", "rest", "grpc", "websocket", "sse", "cors", "csrf", "xss", "sql",
])

def ingest_voice(
    transcript: str,
    source: str,
    speaker_labels: dict[str, str] | None = None,
) -> ModalContent:
    """
    Convert a voice transcript (pre-transcribed text) into a structured
    fragment capturing decisions, action items, and technical context.

    Args:
        transcript:     The full transcript text.
        source:         Identifier (e.g., 'design_meeting_2026-03-07.txt').
        speaker_labels: Optional dict mapping speaker IDs to names/roles
                        (e.g., {'SPEAKER_00': 'Alice (PM)', 'SPEAKER_01': 'Bob (Eng)'}).

    Returns:
        ModalContent with extracted decisions, actions, and technical vocabulary.
    """
    # Normalize speaker labels if provided
    text = transcript
    if speaker_labels:
        for sid, name in speaker_labels.items():
            text = text.replace(sid, name)

    sentences = _split_sentences(text)
    decisions, actions, tech_mentions, questions = _extract_speech_elements(sentences)
    tech_vocab = _extract_tech_vocabulary(text)

    # Compute confidence: longer transcripts with more structure → higher confidence
    has_structure = bool(decisions or actions)
    sentence_count = len(sentences)
    confidence = 0.90 if (sentence_count > 5 and has_structure) else 0.65

    structured = _format_voice_content(source, decisions, actions, tech_mentions, questions, tech_vocab, text)

    return ModalContent(
        text=structured,
        source_type="voice",
        source=source,
        confidence=confidence,
        token_estimate=len(structured) // 4,
        metadata={
            "sentence_count": sentence_count,
            "decisions": len(decisions),
            "actions": len(actions),
            "tech_terms": len(tech_vocab),
        },
    )


def _split_sentences(text: str) -> list[str]:
    """Split transcript into sentences, handling common transcript patterns."""
    # Split on sentence boundaries and newlines
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 3]


# Patterns indicating a decision
_DECISION_PATTERNS = re.compile(
    r'\b(we (should|will|decided|agreed|are going to)|'
    r"let'?s (use|go with|implement|adopt|move|switch)|"
    r'(the plan|our approach) (is|will be)|'
    r'i think we (need|should|must)|'
    r'decision:|agreed:|conclusion:)\b',
    re.IGNORECASE
)

# Patterns indicating an action item
_ACTION_PATTERNS = re.compile(
    r'\b(action item|todo|to-do|to do|follow.?up|'
    r'needs? to|will (implement|fix|add|create|update|remove|refactor)|'
    r'(you|i|we|someone) (should|must|need to)|'
    r'open question:|next step:)\b',
    re.IGNORECASE
)

# Questions (often unresolved concerns)
_QUESTION_PATTERNS = re.compile(
    r'\b(how (do|should|will|can)|what (if|about|happens)|'
    r'should we|do we need|is it (safe|ok|possible)|'
    r'have we (considered|thought about))\b',
    re.IGNORECASE
)


def _extract_speech_elements(
    sentences: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    decisions, actions, tech_mentions, questions = [], [], [], []

    for s in sentences:
        if _DECISION_PATTERNS.search(s):
            decisions.append(s)
        elif _ACTION_PATTERNS.search(s):
            actions.append(s)
        elif _QUESTION_PATTERNS.search(s):
            questions.append(s)
        # Check for technical discussion (e.g., specific tech mentioned)
        lower = s.lower()
        if any(t in lower for t in _TECH_TERMS):
            tech_mentions.append(s)

    return decisions[:20], actions[:20], tech_mentions[:15], questions[:10]


def _extract_tech_vocabulary(text: str) -> list[str]:
    """
    Extract technical vocabulary: CamelCase identifiers, known tech terms,
    file extensions, API paths, version numbers.
    """
    vocab: list[str] = []

    # CamelCase identifiers (likely class/component names)
    camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    vocab.extend(camel[:20])

    # Known tech terms
    words = re.findall(r'\b[a-z][a-z0-9_-]{2,}\b', text.lower())
    for w in words:
        if w in _TECH_TERMS:
            vocab.append(w)

    # API paths
    api_paths = re.findall(r'/api/[a-z0-9/_-]+', text.lower())
    vocab.extend(api_paths[:10])

    # File extensions mentioned
    file_refs = re.findall(r'\b[\w]+\.(py|rs|ts|js|go|java|sql|yaml|json|toml)\b', text.lower())
    vocab.extend([f[0] + "." + f[1] for f in file_refs[:10]])

    return list(dict.fromkeys(vocab))[:40]  # deduplicate, cap at 40


def _format_voice_content(
    source: str,
    decisions: list[str],
    actions: list[str],
    tech_mentions: list[str],
    questions: list[str],
    tech_vocab: list[str],
    full_text: str,
) -> str:
    lines = [f"# Voice/Meeting Transcript: {source}", ""]

    if decisions:
        lines += ["## Decisions Made", *[f"- {d}" for d in decisions], ""]

    if actions:
        lines += ["## Action Items", *[f"- [ ] {a}" for a in actions], ""]

    if questions:
        lines += ["## Open Questions", *[f"- ? {q}" for q in questions], ""]

    if tech_vocab:
        lines += ["## Technical Vocabulary Referenced", " ".join(tech_vocab), ""]

    if tech_mentions:
        lines += ["## Technical Discussion Excerpts",
                  *[f"  > {t}" for t in tech_mentions[:8]], ""]

    # Always include a condensed version of the full transcript
    preview = full_text[:600] + ("..." if len(full_text) > 600 else "")
    lines += ["## Full Transcript (excerpt)", preview, "",
              f"source_type=voice decisions={len(decisions)} actions={len(actions)}"]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Diff / Patch Ingestion
#
# Unified-diff parser: converts code changes into a structured description
# of what changed, why it matters, and which symbols were affected.
# Inspired by DebtGuardian (arXiv 2025): batch-level technical debt detection
# from source code changes. Key insight: diffs encode *intent*, not just content.
# ─────────────────────────────────────────────────────────────────────────────


def ingest_diff(
    diff_text: str,
    source: str,
    commit_message: str = "",
) -> ModalContent:
    """
    Convert a unified diff (git diff output) into a structured change summary.

    Extracts: files changed, symbols added/removed, net line delta,
    intent analysis (bug fix, feature, refactor, test), and structured
    description for semantic retrieval.

    Args:
        diff_text:       The raw unified diff text.
        source:          Identifier (e.g., 'pr_42_auth_refactor.diff').
        commit_message:  Optional commit message for intent classification.

    Returns:
        ModalContent ready for context retrieval.
    """
    hunks = _parse_unified_diff(diff_text)
    intent = _classify_diff_intent(diff_text, commit_message)
    symbols = _extract_diff_symbols(diff_text)
    net_adds, net_removes = _count_diff_lines(diff_text)

    confidence = 0.92 if hunks else 0.50

    structured = _format_diff_content(
        source, hunks, intent, symbols, net_adds, net_removes, commit_message
    )

    return ModalContent(
        text=structured,
        source_type="diff",
        source=source,
        confidence=confidence,
        token_estimate=len(structured) // 4,
        metadata={
            "files_changed": len(hunks),
            "added_lines": net_adds,
            "removed_lines": net_removes,
            "intent": intent,
            "symbols_changed": symbols,
        },
    )


@dataclass
class DiffHunk:
    path: str
    added: list[str]
    removed: list[str]
    context_lines: list[str]


def _parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current_path = ""
    current_added: list[str] = []
    current_removed: list[str] = []
    current_ctx: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("--- ") or line.startswith("diff --git"):
            if current_path:
                hunks.append(DiffHunk(current_path, current_added[:], current_removed[:], current_ctx[:]))
                current_added, current_removed, current_ctx = [], [], []
            continue
        if line.startswith("+++ "):
            path = line[4:].strip().lstrip("b/")
            current_path = path
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current_added.append(line[1:].rstrip())
        elif line.startswith("-") and not line.startswith("---"):
            current_removed.append(line[1:].rstrip())
        elif line.startswith(" ") and current_path:
            current_ctx.append(line[1:].rstrip())

    if current_path:
        hunks.append(DiffHunk(current_path, current_added, current_removed, current_ctx))

    return hunks


def _classify_diff_intent(diff_text: str, commit_msg: str) -> str:
    text = (diff_text + " " + commit_msg).lower()
    if any(w in text for w in ["fix", "bug", "error", "broken", "crash", "fail", "patch"]):
        return "bug-fix"
    if any(w in text for w in ["test", "spec", "assert", "mock", "stub"]):
        return "test"
    if any(w in text for w in ["refactor", "clean", "rename", "move", "extract", "reorganize"]):
        return "refactor"
    if any(w in text for w in ["feat", "feature", "add", "implement", "new", "create"]):
        return "feature"
    if any(w in text for w in ["doc", "readme", "comment", "docstring"]):
        return "docs"
    if any(w in text for w in ["perf", "optim", "speed", "fast", "slow", "latency", "benchmark"]):
        return "performance"
    if any(w in text for w in ["security", "vuln", "cve", "auth", "xss", "injection"]):
        return "security"
    return "other"


def _extract_diff_symbols(diff_text: str) -> list[str]:
    """Extract function/class names from added/removed lines."""
    symbols: list[str] = []
    fn_re = re.compile(r'(?:def|fn|func|function|class|impl|pub fn|async fn)\s+(\w+)')
    for line in diff_text.splitlines():
        if line.startswith("+") or line.startswith("-"):
            for m in fn_re.finditer(line):
                sym = m.group(1)
                if sym not in symbols:
                    symbols.append(sym)
    return symbols[:30]


def _count_diff_lines(diff_text: str) -> tuple[int, int]:
    adds = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removes = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return adds, removes


def _format_diff_content(
    source: str,
    hunks: list[DiffHunk],
    intent: str,
    symbols: list[str],
    adds: int,
    removes: int,
    commit_msg: str,
) -> str:
    lines = [
        f"# Code Change: {source}",
        f"## Intent: {intent.upper()} | +{adds} lines / -{removes} lines | {len(hunks)} files",
        "",
    ]
    if commit_msg:
        lines += ["## Commit Message", f"> {commit_msg}", ""]

    if symbols:
        lines += ["## Symbols Changed", " ".join(symbols), ""]

    lines += ["## Files Modified"]
    for hunk in hunks[:20]:
        lines.append(f"### {hunk.path} (+{len(hunk.added)} / -{len(hunk.removed)})")
        if hunk.removed:
            lines.append("  **Removed:**")
            for r in hunk.removed[:5]:
                lines.append(f"  - {r.strip()}")
        if hunk.added:
            lines.append("  **Added:**")
            for a in hunk.added[:5]:
                lines.append(f"  + {a.strip()}")
        lines.append("")

    if len(hunks) > 20:
        lines.append(f"... and {len(hunks) - 20} more files")

    lines += [f"source_type=diff intent={intent} symbols={','.join(symbols[:10])}"]
    return "\n".join(lines)
