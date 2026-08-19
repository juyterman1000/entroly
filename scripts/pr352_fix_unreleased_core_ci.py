#!/usr/bin/env python3
"""Converge PR/source workflows on exact-head native bootstrap.

Temporary PR352 migration helper. Every replacement is exact-count guarded so a
workflow edit by another agent aborts this transform rather than being guessed over.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    actual = text.count(old)
    if actual != count:
        raise SystemExit(f"{path}: expected {count} exact bootstrap anchor(s), found {actual}")
    target.write_text(text.replace(old, new), encoding="utf-8")
    print(f"updated {path}: {count} bootstrap site(s)")


def main() -> int:
    replace(
        ".github/workflows/agent-integration-contracts.yml",
        '      - name: Install package and test dependencies\n        run: python -m pip install -e ".[test]"\n',
        '      - name: Install exact-head package and test dependencies\n        run: python scripts/ci_install_exact_head.py --extras test\n',
    )

    replace(
        ".github/workflows/qccr-signature-completion-gate.yml",
        """      - name: Create test environment
        run: |
          python -m venv .venv
          .venv/bin/pip install --upgrade pip
          .venv/bin/pip install maturin ruff --timeout 120 --retries 5
          .venv/bin/pip install -e '.[test]' --timeout 120 --retries 5
""",
        """      - name: Create exact-head test environment
        run: |
          python -m venv .venv
          .venv/bin/python scripts/ci_install_exact_head.py \\
            --python .venv/bin/python --extras test --extra-package ruff
""",
    )

    replace(
        ".github/workflows/code-intelligence.yml",
        """      - name: Install optional parser surface
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[code-intelligence,test]"
""",
        """      - name: Install exact-head optional parser surface
        run: python scripts/ci_install_exact_head.py --extras code-intelligence,test
""",
    )

    replace(
        ".github/workflows/public-trust.yml",
        '      - name: Install package and test dependencies\n        run: python -m pip install -e ".[test]"\n',
        '      - name: Install exact-head package and test dependencies\n        run: python scripts/ci_install_exact_head.py --extras test\n',
        count=2,
    )

    replace(
        ".github/workflows/onboarding-self-dogfood.yml",
        """      - name: Install as a fresh source-package user
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,proxy]"
""",
        """      - name: Install as a fresh exact-head source-package user
        run: python scripts/ci_install_exact_head.py --extras test,proxy
""",
    )

    replace(
        ".github/workflows/benchmark.yml",
        '      - name: Install dependencies\n        run: pip install -e ".[test]" "tiktoken==${EVIDENCE_TIKTOKEN_VERSION}"\n',
        '      - name: Install exact-head dependencies\n        run: python scripts/ci_install_exact_head.py --extras test --extra-package "tiktoken==${EVIDENCE_TIKTOKEN_VERSION}"\n',
    )

    replace(
        ".github/workflows/deep-dogfood.yml",
        """      - name: Install exactly as a source-package user
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,proxy]" pytest-timeout psutil
""",
        """      - name: Install exact-head source package
        run: |
          python scripts/ci_install_exact_head.py --extras test,proxy \\
            --extra-package pytest-timeout --extra-package psutil
""",
    )
    replace(
        ".github/workflows/deep-dogfood.yml",
        """      - name: Install source package and feature-test dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,proxy]" pytest-timeout psutil
""",
        """      - name: Install exact-head package and feature-test dependencies
        run: |
          python scripts/ci_install_exact_head.py --extras test,proxy \\
            --extra-package pytest-timeout --extra-package psutil
""",
    )
    replace(
        ".github/workflows/deep-dogfood.yml",
        """            python -m pip install --upgrade build
            python -m build --wheel --outdir "$RUNNER_TEMP/dist"
            mapfile -t wheels < <(find "$RUNNER_TEMP/dist" -maxdepth 1 -name 'entroly-*.whl' -print)
            test "${#wheels[@]}" -eq 1
            "$BIN/python" -m pip install --no-cache-dir "${wheels[0]}"
""",
        """            python scripts/ci_install_exact_head.py \\
              --python "$BIN/python" --mode wheel
""",
    )
    replace(
        ".github/workflows/deep-dogfood.yml",
        """      - name: Install Python bridge and Rust toolchain
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test]"
          rustup toolchain install stable --profile minimal
          cargo install wasm-pack --locked
""",
        """      - name: Install exact-head Python bridge and Rust toolchain
        run: |
          python scripts/ci_install_exact_head.py --extras test
          cargo install wasm-pack --locked
""",
    )
    replace(
        ".github/workflows/deep-dogfood.yml",
        """      - name: Install scan tools and project
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,proxy]" ruff bandit pip-audit
""",
        """      - name: Install scan tools and exact-head project
        run: |
          python scripts/ci_install_exact_head.py --extras test,proxy \\
            --extra-package ruff --extra-package bandit --extra-package pip-audit
""",
    )

    replace(
        ".github/workflows/publish-openclaw-clawhub.yml",
        """      - name: Test Python bridge contract
        run: |
          python -m pip install --upgrade pip
          pip install -e . pytest
          pytest tests/test_openclaw_bridge.py -q
""",
        """      - name: Test exact-head Python bridge contract
        run: |
          python scripts/ci_install_exact_head.py --extras test
          pytest tests/test_openclaw_bridge.py -q
""",
    )

    replace(
        ".github/workflows/user-journey-trust.yml",
        """      - name: Install source journey surface
        shell: bash
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test,proxy]" pytest-timeout psutil
          python -m pip check
""",
        """      - name: Install exact-head journey surface
        shell: bash
        run: |
          python scripts/ci_install_exact_head.py --extras test,proxy \\
            --extra-package pytest-timeout --extra-package psutil
          python -m pip check
""",
    )
    replace(
        ".github/workflows/user-journey-trust.yml",
        """          python -m pip install --upgrade pip build
          python -m build --wheel
          python -m venv "$RUNNER_TEMP/entroly-user"
          "$RUNNER_TEMP/entroly-user/bin/python" -m pip install --upgrade pip
          "$RUNNER_TEMP/entroly-user/bin/python" -m pip install dist/*.whl
""",
        """          python -m venv "$RUNNER_TEMP/entroly-user"
          python scripts/ci_install_exact_head.py \\
            --python "$RUNNER_TEMP/entroly-user/bin/python" --mode wheel
""",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
