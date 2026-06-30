#!/usr/bin/env python3
"""Apply the final native ELC + direct proxy wiring patch locally.

Why this exists:
- GitHub's contents API replaces whole files.
- `entroly-core/src/lib.rs` and `entroly/proxy.py` are large, high-risk files.
- This script applies small deterministic string patches with validation.

Run from repository root:

    python scripts/apply_elc_native_and_proxy_patch.py
    cargo test -p entroly-core compress::tests::native_elc_keeps_signal_and_drops_repetition
    pytest tests/test_compression_proxy_direct.py -v

Then review `git diff` and commit normally.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "entroly-core" / "src" / "lib.rs"
PROXY = ROOT / "entroly" / "proxy.py"


def patch_once(text: str, needle: str, replacement: str, label: str) -> tuple[str, bool]:
    if replacement in text:
        return text, False
    if needle not in text:
        raise SystemExit(f"Cannot patch {label}: needle not found")
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
fn elc_compress(py: Python<'_>, text: &str, query: &str, budget_tokens: usize) -> PyResult<PyObject> {
    let out = elc_native::compress_elc_native(text, query, budget_tokens);
    let receipt = PyDict::new(py);
    receipt.set_item("original_tokens", out.original_tokens)?;
    receipt.set_item("compressed_tokens", out.compressed_tokens)?;
    receipt.set_item("savings_ratio", out.savings_ratio)?;
    receipt.set_item("compression_level", 3)?;
    receipt.set_item("content_type", "native_elc")?;
    receipt.set_item("anchors_preserved", [("anchor", out.anchor_count)].into_py_dict_bound(py))?;
    receipt.set_item("recoverable", true)?;
    let spans = out
        .omitted_spans
        .iter()
        .map(|(start, end)| {
            let d = PyDict::new(py);
            d.set_item("start_line", *start)?;
            d.set_item("end_line", *end)?;
            d.set_item("line_count", end.saturating_sub(*start) + 1)?;
            d.set_item("reason", "budget")?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    receipt.set_item("omitted_spans", spans)?;

    let result = PyDict::new(py);
    result.set_item("compressed", out.compressed)?;
    result.set_item("receipt", receipt)?;
    result.set_item("changed", out.changed)?;
    Ok(result.into())
}
'''
    text, did = patch_once(
        text,
        "// ═══════════════════════════════════════════════════════════════════\n// Module definition\n",
        py_fn + "\n// ═══════════════════════════════════════════════════════════════════\n// Module definition\n",
        "lib.rs elc_compress pyfunction",
    )
    changed |= did

    text, did = patch_once(
        text,
        "    m.add_function(wrap_pyfunction!(py_compress_block, m)?)?;\n",
        "    m.add_function(wrap_pyfunction!(py_compress_block, m)?)?;\n    m.add_function(wrap_pyfunction!(elc_compress, m)?)?;\n",
        "lib.rs pymodule export",
    )
    changed |= did

    if changed:
        LIB.write_text(text, encoding="utf-8")
    return changed


def patch_proxy_py() -> bool:
    text = PROXY.read_text(encoding="utf-8")
    changed = False

    import_block = "from .value_tracker import get_tracker\n"
    replacement_import = (
        "from .value_tracker import get_tracker\n"
        "from .compression_proxy_direct import apply_elc_to_proxy_body\n"
    )
    text, did = patch_once(text, import_block, replacement_import, "proxy.py import")
    changed |= did

    old = '''            # Stage 2: content-aware compression (test output, diffs, JSON, …)
            body["messages"], tool_tokens_saved = compress_tool_messages(
                body["messages"],
                policy=getattr(self.config, "tool_result_policy", "auto"),
                excluded_tools=getattr(self.config, "tool_result_excluded_tools", ""),
            )
            if tool_tokens_saved > 0:
                logger.info(f"Tool output compression: {tool_tokens_saved} tokens saved")
'''
    new = '''            # Stage 2: content-aware compression (test output, diffs, JSON, …)
            if os.environ.get("ENTROLY_COMPRESSION_PROXY_MODE", "").strip().lower() == "elc":
                elc_result = apply_elc_to_proxy_body(body, provider=provider)
                body = elc_result.body
                tool_tokens_saved = elc_result.receipt.tokens_saved
                if tool_tokens_saved > 0:
                    logger.info(f"ELC tool output compression: {tool_tokens_saved} tokens saved")
                    control_headers.update(elc_result.headers())
            else:
                body["messages"], tool_tokens_saved = compress_tool_messages(
                    body["messages"],
                    policy=getattr(self.config, "tool_result_policy", "auto"),
                    excluded_tools=getattr(self.config, "tool_result_excluded_tools", ""),
                )
                if tool_tokens_saved > 0:
                    logger.info(f"Tool output compression: {tool_tokens_saved} tokens saved")
'''
    text, did = patch_once(text, old, new, "proxy.py Stage 2 compression")
    changed |= did

    if changed:
        PROXY.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    changed_lib = patch_lib_rs()
    changed_proxy = patch_proxy_py()
    print(f"lib.rs patched: {changed_lib}")
    print(f"proxy.py patched: {changed_proxy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
