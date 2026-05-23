"""
FORGE Live Test -- Repair Loop Against GPT-4o-mini
====================================================

Tests FORGE on the two cases where GPT-4o-mini FAILED the stress test:
  Test 5: Invented payment schemas (IPD=0.747)
  Test 6: Used uninstalled sqlmodel (IPD=1.000)

Shows whether the repair loop can suppress the hallucinations
by feeding rejection reasons back as retrieval queries.
"""

from __future__ import annotations

import os
import textwrap

import pytest
pytest.importorskip("openai", reason="openai required for live FORGE test")
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

from openai import OpenAI

from entroly.verifiers.repair_loop import (
    forge_loop,
    SimpleContextStore,
)
from entroly.verifiers.symbol_resolution import SymbolManifest

client = OpenAI()
MODEL = "gpt-4o-mini"


def llm_generate(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


def test_forge_missing_context():
    """Test 5: LLM invents payment schemas without evidence."""
    print("\n" + "=" * 60)
    print(" FORGE TEST -- Missing Context (Payment Schemas)")
    print("=" * 60)

    # The REAL codebase -- only settle_payment exists
    codebase = {
        "payment_service.py": textwrap.dedent("""\
            def settle_payment(payment_id):
                \"\"\"Settle a single payment by ID.\"\"\"
                pass
        """),
        "payment_models.py": textwrap.dedent("""\
            class Payment:
                def __init__(self, payment_id, amount, currency, status):
                    self.payment_id = payment_id
                    self.amount = amount
                    self.currency = currency
                    self.status = status  # "pending", "settled", "failed"

            class PaymentResult:
                def __init__(self, success, message):
                    self.success = success
                    self.message = message
        """),
    }

    initial_context = codebase["payment_service.py"]
    store = SimpleContextStore(codebase)

    manifest = SymbolManifest()
    for sym in ["settle_payment", "Payment", "PaymentResult",
                "payment_id", "amount", "currency", "status"]:
        manifest.repo.add(sym)

    prompt = "Extend settlement logic to support split payments. Write only Python code."

    print(f"  Context: {len(initial_context)} bytes (just settle_payment)")
    print(f"  Store: {len(codebase)} files available for retrieval")
    print(f"  Calling FORGE with max_iters=3...\n")

    result = forge_loop(
        prompt=prompt,
        initial_context=initial_context,
        generate_fn=llm_generate,
        context_store=store,
        manifest=manifest,
        max_iters=3,
        ipd_threshold=0.25,
    )

    print(result.explain())
    print(f"\n  Final output:\n{textwrap.indent(result.final_output[:500], '    ')}")
    return result


def test_forge_package():
    """Test 6: LLM uses sqlmodel which is not installed."""
    print("\n" + "=" * 60)
    print(" FORGE TEST -- Package Verification (sqlmodel)")
    print("=" * 60)

    # The REAL codebase -- only these packages are available
    codebase = {
        "requirements.txt": "numpy\npandas\nrequests\nfastapi\n",
        "models.py": textwrap.dedent("""\
            from fastapi import FastAPI
            from pydantic import BaseModel

            app = FastAPI()

            class User(BaseModel):
                id: int
                username: str
                email: str

            class Item(BaseModel):
                id: int
                name: str
                owner_id: int
        """),
    }

    initial_context = codebase["requirements.txt"]
    store = SimpleContextStore(codebase)

    manifest = SymbolManifest()
    for pkg in ["numpy", "pandas", "requests", "fastapi",
                "FastAPI", "BaseModel", "User", "Item"]:
        manifest.installed.add(pkg) if pkg.islower() else manifest.repo.add(pkg)

    prompt = "Build an ORM model using the available packages. Write only Python code."

    print(f"  Context: requirements.txt only")
    print(f"  Store: {len(codebase)} files available for retrieval")
    print(f"  Calling FORGE with max_iters=3...\n")

    result = forge_loop(
        prompt=prompt,
        initial_context=initial_context,
        generate_fn=llm_generate,
        context_store=store,
        manifest=manifest,
        max_iters=3,
        ipd_threshold=0.25,
    )

    print(result.explain())
    print(f"\n  Final output:\n{textwrap.indent(result.final_output[:500], '    ')}")
    return result


if __name__ == "__main__":
    r1 = test_forge_missing_context()
    r2 = test_forge_package()

    print("\n" + "=" * 60)
    print(" FORGE SUMMARY")
    print("=" * 60)
    print(f"  Test 5 (Payment): IPD {r1.original_ipd:.3f} -> {r1.final_ipd:.3f}  "
          f"iters={r1.total_iterations}  converged={r1.converged}")
    print(f"  Test 6 (Package): IPD {r2.original_ipd:.3f} -> {r2.final_ipd:.3f}  "
          f"iters={r2.total_iterations}  converged={r2.converged}")
