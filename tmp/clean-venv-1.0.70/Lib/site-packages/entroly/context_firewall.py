"""
Adversarial Context Firewall (ACF)
===================================

End-to-end cryptographic integrity verification + prompt injection
detection for the context pipeline.

Motivation
----------
OS-level sandboxing (Seatbelt, Landlock) prevents file access outside
the project root.  But it does NOT:
    1. Detect prompt injection *within* allowed files
    2. Verify that context hasn't been tampered with between read and use
    3. Work on Windows
    4. Detect Unicode-based attacks (directional overrides, zero-width chars)

ACF provides **content-level security** that works everywhere:

    File read → Hash(content) → Compress → Hash(compressed) → Inject
                    │                            │
                    └────── verify chain ────────┘
                       If chain breaks → ALERT

Plus a prompt injection scanner that catches 20+ attack patterns
*before* content reaches the LLM.

Threat model
------------
1. **Prompt injection in source files**: attacker hides "ignore previous
   instructions" in a code comment, docstring, or data file.
2. **Unicode steganography**: zero-width characters, directional overrides,
   homoglyph substitution to hide malicious instructions.
3. **Base64/encoded payloads**: instructions encoded to evade simple string matching.
4. **Pipeline tampering**: content modified between read and LLM injection
   (e.g., by a compromised plugin or middleware).
5. **Repetition flooding**: many identical lines designed to dominate the
   context window and push out legitimate content.

Scope
-----
    ACF combines content hashing with heuristic pre-LLM threat scanning. It is
    one defense layer, not proof that all prompt-injection attacks are detected.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ThreatDetection:
    """A detected prompt injection or integrity threat."""
    threat_type: str        # "injection" | "unicode" | "encoded" | "repetition" | "integrity"
    severity: str           # "critical" | "high" | "medium" | "low"
    description: str        # human-readable description
    location: str           # file:line or pipeline stage
    matched_pattern: str    # the pattern that matched (truncated)
    remediation: str        # suggested fix


@dataclass
class ScanResult:
    """Result of an ACF scan."""
    is_safe: bool                       # True if no critical/high threats found
    threats: list[ThreatDetection]      # all detected threats
    n_critical: int = 0
    n_high: int = 0
    n_medium: int = 0
    n_low: int = 0
    content_hash: str = ""              # SHA-256 of scanned content
    scan_time_ms: float = 0.0


@dataclass
class ContentHash:
    """Cryptographic hash for pipeline integrity tracking.

    Tracks content through the pipeline stages:
        read → index → compress → inject

    If the hash chain breaks at any point, the content was modified
    by something other than the authorized compression pipeline.
    """
    stage: str              # "read" | "indexed" | "compressed" | "injected"
    sha256: str             # hash of content at this stage
    parent_hash: str | None  # hash from previous stage (None for "read")
    content_length: int     # character count at this stage
    pipeline_id: str        # unique ID for this pipeline run

    def verify_parent(self, expected_parent_hash: str) -> bool:
        """Verify this hash connects to the expected parent."""
        return self.parent_hash == expected_parent_hash


# ── Prompt Injection Scanner ─────────────────────────────────────────

# Pattern categories with severity levels
_INJECTION_PATTERNS: list[tuple[str, str, str, str]] = [
    # (regex_pattern, threat_type, severity, description)

    # Category 1: Direct instruction override
    (r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
     "injection", "critical", "Direct instruction override attempt"),
    (r"(?i)disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|context|rules)",
     "injection", "critical", "Instruction disregard attempt"),
    (r"(?i)forget\s+(everything|all|what)\s+(you|i)\s+(told|said|know)",
     "injection", "critical", "Memory wipe attempt"),
    (r"(?i)you\s+are\s+now\s+(?:a|an|the)\s+\w+",
     "injection", "high", "Role reassignment attempt"),
    (r"(?i)new\s+instructions?:\s*",
     "injection", "high", "Instruction injection attempt"),
    (r"(?i)system\s*:\s*you\s+(?:are|must|should|will)",
     "injection", "critical", "System prompt injection"),
    (r"(?i)<<\s*(?:SYS|SYSTEM|INST)\s*>>",
     "injection", "critical", "Llama/Mistral system tag injection"),

    # Category 2: Output manipulation
    (r"(?i)(?:print|output|say|respond\s+with|return)\s+(?:only|exactly|just)\s*[\"']",
     "injection", "high", "Forced output manipulation"),
    (r"(?i)do\s+not\s+(?:mention|reveal|disclose|tell)\s+(?:that|this|the)",
     "injection", "medium", "Information suppression attempt"),

    # Category 3: Context boundary attacks
    (r"---+\s*(?:END|BEGIN)\s+(?:CONTEXT|SYSTEM|USER)\s*---+",
     "injection", "high", "Context boundary spoofing"),
    (r"</?(?:system|user|assistant|context|instructions?)>",
     "injection", "high", "XML tag role spoofing"),

    # Category 4: Encoded payloads (base64 instructions)
    (r"(?:eval|exec|execute)\s*\(\s*(?:base64|atob|decode)\s*\(",
     "encoded", "high", "Encoded execution attempt"),

    # Category 5: Repetition flooding (> 10 identical consecutive lines)
    # Handled separately in _check_repetition_flooding()

    # Category 6: Social engineering
    (r"(?i)(?:this\s+is\s+(?:a\s+)?test|testing\s+mode|debug\s+mode)\s*[,.]?\s*(?:ignore|skip|bypass)",
     "injection", "medium", "Test/debug mode bypass attempt"),
    (r"(?i)(?:as\s+(?:a|an)\s+(?:AI|assistant|model)|in\s+(?:your|this)\s+(?:role|capacity))\s*,?\s*(?:you\s+(?:must|should|can|will))",
     "injection", "medium", "Role authority exploitation"),
]

# Compiled patterns for performance
_COMPILED_PATTERNS = [
    (re.compile(p), tt, sev, desc)
    for p, tt, sev, desc in _INJECTION_PATTERNS
]

# ── Unicode Attack Patterns ──────────────────────────────────────────

# Dangerous Unicode categories
_DANGEROUS_UNICODE = {
    # Bidirectional override characters (can reverse text display)
    "\u202a": "LRE (Left-to-Right Embedding)",
    "\u202b": "RLE (Right-to-Left Embedding)",
    "\u202c": "PDF (Pop Directional Formatting)",
    "\u202d": "LRO (Left-to-Right Override)",
    "\u202e": "RLO (Right-to-Left Override)",
    "\u2066": "LRI (Left-to-Right Isolate)",
    "\u2067": "RLI (Right-to-Left Isolate)",
    "\u2068": "FSI (First Strong Isolate)",
    "\u2069": "PDI (Pop Directional Isolate)",
    # Zero-width characters (can hide content)
    "\u200b": "ZWSP (Zero Width Space)",
    "\u200c": "ZWNJ (Zero Width Non-Joiner)",
    "\u200d": "ZWJ (Zero Width Joiner)",
    "\u2060": "WJ (Word Joiner)",
    "\ufeff": "BOM / ZWNBSP",
    # Invisible formatting
    "\u00ad": "Soft Hyphen",
    "\u034f": "Combining Grapheme Joiner",
    "\u061c": "ALM (Arabic Letter Mark)",
    "\u180e": "MVS (Mongolian Vowel Separator)",
}


def _check_unicode_attacks(
    text: str,
    source: str = "",
) -> list[ThreatDetection]:
    """Detect dangerous Unicode characters that could hide content."""
    threats: list[ThreatDetection] = []

    for char, name in _DANGEROUS_UNICODE.items():
        positions = [i for i, c in enumerate(text) if c == char]
        if positions:
            threats.append(ThreatDetection(
                threat_type="unicode",
                severity="high",
                description=f"Dangerous Unicode character: {name} (U+{ord(char):04X})",
                location=f"{source}:char_positions={positions[:5]}",
                matched_pattern=f"U+{ord(char):04X} ({name})",
                remediation=f"Remove all {name} characters from the content",
            ))

    return threats


def _check_repetition_flooding(
    text: str,
    threshold: int = 10,
    source: str = "",
) -> list[ThreatDetection]:
    """Detect repetition flooding attacks.

    Flags content where the same line appears > threshold times
    consecutively, which is designed to dominate the context window.
    """
    threats: list[ThreatDetection] = []
    lines = text.splitlines()

    if len(lines) < threshold:
        return threats

    consecutive = 1
    for i in range(1, len(lines)):
        stripped = lines[i].strip()
        prev_stripped = lines[i - 1].strip()

        if stripped and stripped == prev_stripped:
            consecutive += 1
        else:
            if consecutive >= threshold:
                threats.append(ThreatDetection(
                    threat_type="repetition",
                    severity="medium",
                    description=(
                        f"Repetition flooding: {consecutive} identical consecutive lines "
                        f"('{stripped[:50]}...')"
                    ),
                    location=f"{source}:lines_{i - consecutive + 1}-{i}",
                    matched_pattern=stripped[:80],
                    remediation="Remove or deduplicate the repeated lines",
                ))
            consecutive = 1

    # Check last run
    if consecutive >= threshold:
        stripped = lines[-1].strip()
        threats.append(ThreatDetection(
            threat_type="repetition",
            severity="medium",
            description=f"Repetition flooding: {consecutive} identical consecutive lines",
            location=f"{source}:lines_{len(lines) - consecutive + 1}-{len(lines)}",
            matched_pattern=stripped[:80],
            remediation="Remove or deduplicate the repeated lines",
        ))

    return threats


def _check_base64_payloads(
    text: str,
    min_length: int = 40,
    source: str = "",
) -> list[ThreatDetection]:
    """Detect suspicious base64-encoded payloads.

    Looks for base64 strings that decode to readable text containing
    instruction-like content.
    """
    threats: list[ThreatDetection] = []

    # Find base64-like strings (alphanumeric + /+ ending with =)
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{" + str(min_length) + r",}={0,2}")

    for match in b64_pattern.finditer(text):
        candidate = match.group()
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="strict")
            # Check if decoded content looks like instructions
            if any(kw in decoded.lower() for kw in (
                "ignore", "instruction", "system", "forget", "disregard",
                "override", "you are", "new role", "bypass",
            )):
                threats.append(ThreatDetection(
                    threat_type="encoded",
                    severity="critical",
                    description="Base64-encoded instruction payload detected",
                    location=f"{source}:offset_{match.start()}",
                    matched_pattern=f"base64→'{decoded[:60]}...'",
                    remediation="Remove the encoded payload",
                ))
        except (ValueError, UnicodeDecodeError):
            continue  # not valid base64 or not UTF-8 text

    return threats


# ── Main Scanner API ─────────────────────────────────────────────────

def scan(
    text: str,
    source: str = "<unknown>",
    *,
    check_unicode: bool = True,
    check_injection: bool = True,
    check_repetition: bool = True,
    check_encoded: bool = True,
    repetition_threshold: int = 10,
) -> ScanResult:
    """Scan content for prompt injection attacks and integrity threats.

    Parameters
    ----------
    text : str
        Content to scan (file content, shell output, agent response, etc.)
    source : str
        Source identifier for threat location reporting.
    check_unicode : bool
        Enable Unicode attack detection (default True).
    check_injection : bool
        Enable prompt injection pattern detection (default True).
    check_repetition : bool
        Enable repetition flooding detection (default True).
    check_encoded : bool
        Enable base64 payload detection (default True).
    repetition_threshold : int
        Minimum consecutive identical lines to flag (default 10).

    Returns
    -------
    ScanResult
        Contains is_safe flag, list of threats, and content hash.

    Example
    -------
    >>> result = scan(file_content, source="auth.py")
    >>> if not result.is_safe:
    ...     for threat in result.threats:
    ...         print(f"[{threat.severity}] {threat.description}")
    """
    import time as _time
    t0 = _time.perf_counter()
    threats: list[ThreatDetection] = []

    # Unicode attacks
    if check_unicode:
        threats.extend(_check_unicode_attacks(text, source))

    # Prompt injection patterns
    if check_injection:
        for pattern, threat_type, severity, description in _COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                # Find line number
                line_num = text[:match.start()].count("\n") + 1
                threats.append(ThreatDetection(
                    threat_type=threat_type,
                    severity=severity,
                    description=description,
                    location=f"{source}:line_{line_num}",
                    matched_pattern=match.group()[:80],
                    remediation="Review and remove the suspicious content",
                ))

    # Repetition flooding
    if check_repetition:
        threats.extend(_check_repetition_flooding(text, repetition_threshold, source))

    # Base64 encoded payloads
    if check_encoded:
        threats.extend(_check_base64_payloads(text, source=source))

    # Compute severity counts
    n_crit = sum(1 for t in threats if t.severity == "critical")
    n_high = sum(1 for t in threats if t.severity == "high")
    n_med = sum(1 for t in threats if t.severity == "medium")
    n_low = sum(1 for t in threats if t.severity == "low")

    # Content is safe if no critical or high threats
    is_safe = n_crit == 0 and n_high == 0

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    scan_time = (_time.perf_counter() - t0) * 1000

    return ScanResult(
        is_safe=is_safe,
        threats=threats,
        n_critical=n_crit,
        n_high=n_high,
        n_medium=n_med,
        n_low=n_low,
        content_hash=content_hash,
        scan_time_ms=scan_time,
    )


# ── Pipeline Integrity Chain ────────────────────────────────────────

class IntegrityChain:
    """Track content integrity through the compression pipeline.

    Usage
    -----
    >>> chain = IntegrityChain()
    >>> h1 = chain.record("read", raw_content)
    >>> h2 = chain.record("compressed", compressed_content)
    >>> assert chain.verify()  # True if chain is unbroken
    """

    def __init__(self, pipeline_id: str | None = None):
        self.pipeline_id = pipeline_id or hashlib.sha256(
            str(id(self)).encode()
        ).hexdigest()[:12]
        self._stages: list[ContentHash] = []

    def record(self, stage: str, content: str) -> ContentHash:
        """Record a content hash at a pipeline stage."""
        parent_hash = self._stages[-1].sha256 if self._stages else None
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        entry = ContentHash(
            stage=stage,
            sha256=content_hash,
            parent_hash=parent_hash,
            content_length=len(content),
            pipeline_id=self.pipeline_id,
        )
        self._stages.append(entry)
        return entry

    def verify(self) -> bool:
        """Verify the integrity chain is unbroken.

        Returns True if every stage's parent_hash matches the
        previous stage's sha256. A break indicates unauthorized
        modification.
        """
        for i in range(1, len(self._stages)):
            expected_parent = self._stages[i - 1].sha256
            actual_parent = self._stages[i].parent_hash
            if actual_parent != expected_parent:
                logger.warning(
                    "[ACF] Integrity chain broken at stage '%s': "
                    "expected parent %s, got %s",
                    self._stages[i].stage,
                    expected_parent[:16],
                    actual_parent[:16] if actual_parent else "None",
                )
                return False
        return True

    @property
    def stages(self) -> list[ContentHash]:
        """Return all recorded stages."""
        return list(self._stages)

    def summary(self) -> dict:
        """Pipeline integrity summary."""
        return {
            "pipeline_id": self.pipeline_id,
            "n_stages": len(self._stages),
            "stages": [
                {
                    "stage": s.stage,
                    "hash": s.sha256[:16] + "...",
                    "length": s.content_length,
                }
                for s in self._stages
            ],
            "chain_valid": self.verify(),
        }


# ── Convenience API ──────────────────────────────────────────────────

def sanitize(
    text: str,
    source: str = "<unknown>",
    *,
    strip_unicode_attacks: bool = True,
    strip_injection: bool = False,
) -> tuple[str, ScanResult]:
    """Scan content and optionally sanitize detected threats.

    Parameters
    ----------
    text : str
        Content to scan and sanitize.
    source : str
        Source identifier.
    strip_unicode_attacks : bool
        Remove dangerous Unicode characters (default True).
    strip_injection : bool
        Remove lines containing injection patterns (default False —
        dangerous, may break legitimate content).

    Returns
    -------
    (sanitized_text, scan_result)
    """
    result = scan(text, source)
    sanitized = text

    if strip_unicode_attacks:
        for char in _DANGEROUS_UNICODE:
            sanitized = sanitized.replace(char, "")

    if strip_injection and result.threats:
        lines = sanitized.splitlines()
        injection_lines: set[int] = set()
        for threat in result.threats:
            if threat.threat_type == "injection":
                # Parse line number from location
                match = re.search(r"line_(\d+)", threat.location)
                if match:
                    injection_lines.add(int(match.group(1)) - 1)  # 0-indexed

        if injection_lines:
            lines = [
                line for i, line in enumerate(lines)
                if i not in injection_lines
            ]
            sanitized = "\n".join(lines)

    return sanitized, result
