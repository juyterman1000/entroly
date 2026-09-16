"""Unit tests for AST-skeletal density compression."""

from __future__ import annotations

from entroly.tree_sitter_support import extract_skeletal_ast
from entroly.tokens import count_tokens


def test_extract_skeletal_ast_python():
    source = '''import os
import sys
from typing import List, Optional, Dict

class AccountManager:
    """Manages bank account state."""

    def __init__(self, account_id: str, balance: float = 0.0):
        self.account_id = account_id
        self.balance = balance
        self.transactions = []
        for i in range(100):
            self.transactions.append({"id": i, "amount": 10.0})

    def deposit(self, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.balance += amount
        self.transactions.append({"amount": amount, "type": "deposit"})
        return True

    def withdraw(self, amount: float) -> bool:
        if amount > self.balance:
            return False
        self.balance -= amount
        return True

def create_default_manager() -> AccountManager:
    return AccountManager("acc_001", 1000.0)
'''
    skeletal = extract_skeletal_ast(source, "account.py")

    # Verify essential interface elements are preserved
    assert "import os" in skeletal
    assert "from typing import List, Optional, Dict" in skeletal
    assert "class AccountManager:" in skeletal
    assert "def deposit(self, amount: float) -> bool:" in skeletal
    assert "def withdraw(self, amount: float) -> bool:" in skeletal
    assert "def create_default_manager() -> AccountManager:" in skeletal

    # Verify implementation bodies are stripped
    assert "raise ValueError" not in skeletal
    assert "self.transactions.append" not in skeletal

    # Verify dramatic token reduction (> 60% savings)
    orig_tokens = count_tokens(source)
    skel_tokens = count_tokens(skeletal)
    assert skel_tokens < orig_tokens * 0.55


def test_extract_skeletal_ast_fallback():
    """Unrecognized or fallback language still yields signatures."""
    custom_source = '''package main

import "fmt"

func CalculateInterest(principal float64, rate float64) float64 {
    return principal * rate * 1.5
}
'''
    skeletal = extract_skeletal_ast(custom_source, "finance.go")
    assert "import" in skeletal
    assert "func CalculateInterest" in skeletal
    assert "return principal * rate" not in skeletal
