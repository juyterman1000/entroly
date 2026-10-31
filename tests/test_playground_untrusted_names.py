"""The public playground must render file names as text, never as markup.

The playground loads files from any GitHub repository a visitor names, and Git
allows `<`, `>` and quotes in a file name. Those names are written into the page
with innerHTML in three places: the receipt's fragment list, the omission
explorer and the custom-file tabs. Before escaping, a repository containing
`<img src=x onerror=...>.py` ran its script on the playground's origin.

Drives the real page in Chromium.
"""

from __future__ import annotations

from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

PAGE = Path(__file__).resolve().parent.parent / "playground" / "index.html"
HOSTILE = "<img src=x onerror=\"window.__pwned=(window.__pwned||0)+1\">.py"


@pytest.fixture
def page():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment without a browser
            pytest.skip(f"Chromium unavailable: {exc}")
        try:
            tab = browser.new_page()
            tab.goto(PAGE.as_uri())
            tab.wait_for_function("typeof runOptimization === 'function'")
            yield tab
        finally:
            browser.close()


def _injected(tab) -> dict:
    return tab.evaluate(
        """() => ({
            ran: window.__pwned || 0,
            images: document.querySelectorAll('img[src="x"]').length,
        })"""
    )


def test_repository_file_names_render_as_text_in_receipt_and_omissions(page) -> None:
    page.evaluate(
        """async (name) => {
            const big = 'def handler(x):\\n' + '    return x + 1\\n'.repeat(400);
            currentCorpus = [
                {path: name, content: 'def tiny():\\n    return 1\\n'},
                {path: 'z' + name, content: big},
            ];
            document.getElementById('budgetSlider').value = 200;
            await runOptimization();
        }""",
        HOSTILE,
    )
    page.wait_for_timeout(300)

    assert _injected(page) == {"ran": 0, "images": 0}
    fragments = page.inner_text("#receiptFragmentList") + page.inner_text("#omissionList")
    assert "<img src=x" in fragments  # shown to the reader, as text


def test_custom_file_tab_names_render_as_text(page) -> None:
    page.evaluate(
        """(name) => {
            customFiles = [{path: name, content: 'x = 1\\n'}];
            activeCustomIndex = 0;
            renderCustomFileTabs();
        }""",
        HOSTILE,
    )
    page.wait_for_timeout(300)

    assert _injected(page) == {"ran": 0, "images": 0}
    assert "<img src=x" in page.inner_text("#customFileTabs")
