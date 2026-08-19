#!/usr/bin/env python3
"""Align SAST privacy regressions with the stronger full-line redaction contract.

SEC-003 intentionally requires a quoted `sk-` literal, so the adversarial test
varies position only within valid detected syntax. Legacy tests that expected
the older partial `[REDACTED]`/key-name representation are upgraded to require
exact full-line redaction instead. Product redaction is never weakened here.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "entroly-engine/src/sast.rs"
MARKER = "[REDACTED — secret-bearing line]"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")

    old_cases = '''        let cases = [
            format!("{secret} trailing diagnostic text"),
            format!("prefix text {secret} trailing text"),
            format!("prefix text ending with {secret}"),
            format!("{secret} = placeholder"),
        ];
'''
    new_cases = '''        let cases = [
            format!("\\\"{secret}\\\" trailing diagnostic text"),
            format!("prefix text \\\"{secret}\\\" trailing text"),
            format!("prefix text ending with \\\"{secret}\\\""),
            format!("\\\"{secret}\\\" = placeholder"),
        ];
'''
    if old_cases in text:
        text = replace_once(text, old_cases, new_cases, "SEC-003 adversarial cases")
    elif new_cases not in text:
        raise SystemExit("SAST adversarial privacy fixture is neither old nor expected new form")

    old_password_assert = '''        assert!(
            finding.line_content.contains("[REDACTED]"),
            "Secret-category finding must redact line_content, got: {}",
            finding.line_content
        );
'''
    new_password_assert = '''        assert_eq!(
            finding.line_content,
            "[REDACTED — secret-bearing line]",
            "Secret-category finding must fully redact source bytes"
        );
'''
    if old_password_assert in text:
        text = replace_once(
            text,
            old_password_assert,
            new_password_assert,
            "legacy password redaction assertion",
        )
    elif new_password_assert not in text:
        raise SystemExit("legacy password redaction assertion not found")

    old_openai_assert = '''        assert!(
            finding.line_content.contains("[REDACTED]"),
            "Finding must contain [REDACTED]: {}",
            finding.line_content
        );
'''
    new_openai_assert = '''        assert_eq!(
            finding.line_content,
            "[REDACTED — secret-bearing line]",
            "API-key finding must fully redact source bytes"
        );
'''
    if old_openai_assert in text:
        text = replace_once(
            text,
            old_openai_assert,
            new_openai_assert,
            "legacy OpenAI-key redaction assertion",
        )
    elif new_openai_assert not in text:
        raise SystemExit("legacy OpenAI-key redaction assertion not found")

    if MARKER not in text:
        raise SystemExit("strong SAST redaction marker missing after fixture alignment")
    PATH.write_text(text, encoding="utf-8")
    print("aligned all SAST privacy regressions with full-line redaction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
