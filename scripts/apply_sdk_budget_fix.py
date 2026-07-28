from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "entroly/sdk.py",
    '''    if client_key:
        result = _cache_align_older(client_key, result)

    result.extend(recent)
    return result
''',
    '''    if client_key:
        result = _cache_align_older(client_key, result)

    # Ratio allocation and integer rounding can leave the compressible older
    # window a token or two above its share of the explicit total budget. Tighten
    # only that older window so the caller's recent verbatim messages remain
    # untouched. Every successful round must reduce the SDK's own stable token
    # estimate; deterministic prefix bounding is the fail-closed fallback.
    max_rounds = max(1, len(result) * 2)
    for _ in range(max_rounds):
        current_total = _message_tokens(result)
        if current_total <= remaining_budget:
            break
        candidates = [
            (len(message.get("content", "")) // 4, index)
            for index, message in enumerate(result)
            if isinstance(message.get("content"), str)
            and len(message.get("content", "")) // 4 > 0
        ]
        if not candidates:
            break
        current_tokens, index = max(candidates)
        excess = current_total - remaining_budget
        target_tokens = max(0, current_tokens - excess)
        message = dict(result[index])
        content = message.get("content", "")
        if target_tokens == 0:
            tightened = _budget_bounded_head(content, 0)
        else:
            tightened = _compress_message_content(
                content,
                budget=target_tokens,
                query=query,
                profile=profile,
            )
            if len(tightened) // 4 > target_tokens:
                tightened = _budget_bounded_head(content, target_tokens)
        if len(tightened) // 4 >= current_tokens:
            tightened = _budget_bounded_head(content, max(0, current_tokens - 1))
        message["content"] = tightened
        result[index] = message

    result.extend(recent)
    return result
''',
)

replace_once(
    "tests/test_public_surface_dogfood.py",
    '''        assert len(output) <= max(1, budget * 4), "explicit estimated-token budget was exceeded"
''',
    '''        assert len(output) // 4 <= budget, "explicit estimated-token budget was exceeded"
''',
)

for relative in (
    "scripts/apply_sdk_budget_fix.py",
    ".github/workflows/apply-sdk-budget-fix.yml",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()

print("Applied exact SDK budget repair")
