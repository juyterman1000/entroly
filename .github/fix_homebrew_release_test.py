from __future__ import annotations

from pathlib import Path

PATH = Path("tests/test_release_surface.py")
text = PATH.read_text(encoding="utf-8")

old_constants = '''HOMEBREW_FORMULA_VERSION = "1.0.64"
HOMEBREW_FORMULA_SHA256 = "c259fe1e25311679f54ef356f14047f3f6c1e1a6943e82c27bc00966fcea1a3f"
'''
new_constants = '''HOMEBREW_FORMULA_VERSION = "1.0.66"
HOMEBREW_FORMULA_URL = (
    "https://files.pythonhosted.org/packages/31/fe/"
    "3338271b75ccb26b13ddb597488c54636e61678474e48bb1f177842bf5e3/"
    "entroly-1.0.66.tar.gz"
)
HOMEBREW_FORMULA_SHA256 = "26eb4bf302f7c1caf1846a30e9da3f9eb5d7e5d12f5ea4f245875431ade37e46"
'''
old_test = '''    assert f"entroly-{HOMEBREW_FORMULA_VERSION}.tar.gz" in text
    assert "packages/source/e/entroly/" in text
    assert HOMEBREW_FORMULA_SHA256 in text
'''
new_test = '''    assert f'url "{HOMEBREW_FORMULA_URL}"' in text
    assert f"entroly-{HOMEBREW_FORMULA_VERSION}.tar.gz" in HOMEBREW_FORMULA_URL
    assert f'sha256 "{HOMEBREW_FORMULA_SHA256}"' in text
'''

for old, label in ((old_constants, "Homebrew constants"), (old_test, "Homebrew test")):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")

text = text.replace(old_constants, new_constants, 1)
text = text.replace(old_test, new_test, 1)

for marker in (
    'HOMEBREW_FORMULA_VERSION = "1.0.66"',
    'entroly-1.0.66.tar.gz',
    '26eb4bf302f7c1caf1846a30e9da3f9eb5d7e5d12f5ea4f245875431ade37e46',
):
    if marker not in text:
        raise RuntimeError(f"missing repaired marker: {marker}")

PATH.write_text(text, encoding="utf-8")
