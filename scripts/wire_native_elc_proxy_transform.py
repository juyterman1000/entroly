#!/usr/bin/env python3
"""Wire native ELC into Rust core and live proxy transform.

This script runs in a real git checkout, so it can safely apply small text
patches to large files without replacing them through the GitHub contents API.
It is idempotent and fails if a target needle has drifted.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "entroly-core" / "src" / "lib.rs"
TRANSFORM = ROOT / "entroly" / "proxy_transform.py"


def patch_once(text: str, needle: str, replacement: str, label: str) -> tuple[str, bool]:
    if replacement in text:
        print(f"already patched: {label}")
        return text, False
    if needle not in text:
        raise SystemExit(f"needle not found: {label}")
    print(f"patched: {label}")
    return text.replace(needle, replacement, 1), True


def patch_lib_rs() -> bool:
    text = LIB.read_text(encoding="utf-8")
    changed = False

    text, did = patch_once(
        text,
        "mod entropy;\n",
        "mod entropy;\nmod elc_native;\n",
        "lib.rs module declaration",
    )
    changed |= did

    py_fn = r'''
#[pyfunction]
fn elc_compress(text: &str, query: &str, budget_tokens: usize) -> PyResult<String> {
    let out = elc_native::compress_elc_native(text, query, budget_tokens);
    let omitted_spans: Vec<serde_json::Value> = out
        .omitted_spans
        .iter()
        .map(|(start, end)| {
            serde_json::json!({
                "start_line": *start,
                "end_line": *end,
                "line_count": end.saturating_sub(*start) + 1,
                "reason": "budget"
            })
        })
        .collect();
    let payload = serde_json::json!({
        "compressed": out.compressed,
        "changed": out.changed,
        "receipt": {
            "original_tokens": out.original_tokens,
            "compressed_tokens": out.compressed_tokens,
            "savings_ratio": out.savings_ratio,
            "compression_level": 3,
            "content_type": "native_elc",
            "anchors_preserved": {"anchor": out.anchor_count},
            "omitted_spans": omitted_spans,
            "recoverable": true
        }
    });
    serde_json::to_string(&payload)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
}
'''
    text, did = patch_once(
        text,
        "// ═══════════════════════════════════════════════════════════════════\n// Module definition\n",
        py_fn + "\n// ═══════════════════════════════════════════════════════════════════\n// Module definition\n",
        "lib.rs elc_compress export function",
    )
    changed |= did

    text, did = patch_once(
        text,
        "    m.add_function(wrap_pyfunction!(py_compress_block, m)?)?;\n",
        "    m.add_function(wrap_pyfunction!(py_compress_block, m)?)?;\n    m.add_function(wrap_pyfunction!(elc_compress, m)?)?;\n",
        "lib.rs pymodule function registration",
    )
    changed |= did

    if changed:
        LIB.write_text(text, encoding="utf-8")
    return changed


def patch_proxy_transform() -> bool:
    text = TRANSFORM.read_text(encoding="utf-8")
    changed = False

    needle = '''    if not messages:
        return messages, 0

    total_saved = 0
'''
    replacement = '''    if not messages:
        return messages, 0

    if _os.environ.get("ENTROLY_COMPRESSION_PROXY_MODE", "").strip().lower() == "elc":
        try:
            from .compression_proxy import compress_proxy_payload
            from .compression_retrieval_store import CompressionRetrievalStore
            from pathlib import Path

            store_path = _os.environ.get("ENTROLY_COMPRESSION_STORE")
            store = CompressionRetrievalStore(Path(store_path)) if store_path else None
            result = compress_proxy_payload(
                {"messages": messages},
                provider="openai",
                query="",
                budget_tokens=int(_os.environ.get("ENTROLY_ELC_BUDGET_TOKENS", "1200")),
                mode="elc",
                retrieval_store=store,
                compress_user_messages=False,
            )
            if result.changed:
                return result.body.get("messages", messages), result.receipt.tokens_saved
        except Exception:
            pass

    total_saved = 0
'''
    text, did = patch_once(
        text,
        needle,
        replacement,
        "proxy_transform compress_tool_messages ELC branch",
    )
    changed |= did

    if changed:
        TRANSFORM.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    changed_lib = patch_lib_rs()
    changed_transform = patch_proxy_transform()
    print(f"changed_lib={changed_lib}")
    print(f"changed_transform={changed_transform}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
