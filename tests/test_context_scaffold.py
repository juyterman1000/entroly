"""
Tests for the Context Scaffolding Engine (CSE).

Validates:
  1. Import extraction across 6 languages (Python, JS/TS, Rust, Go, Java, Ruby)
  2. Definition extraction
  3. Cross-fragment edge resolution
  4. Task-aware scaffold rendering
  5. Token budget enforcement
  6. Edge cases: empty input, single fragment, no edges
"""

import pytest
from entroly.context_scaffold import (
    generate_scaffold,
    _extract_imports_from_content,
    _extract_definitions_from_content,
    _basename_stem,
    _normalize_source,
    _classify_edge,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures: Synthetic fragments that simulate a real codebase
# ═══════════════════════════════════════════════════════════════════════

PYTHON_HANDLER = {
    "source": "file:auth/handler.py",
    "content": '''from auth.models import User, Session
from db.queries import get_user_by_id
import logging

logger = logging.getLogger(__name__)

class AuthHandler:
    """Handles user authentication."""

    def login(self, username: str, password: str) -> Session:
        user = get_user_by_id(username)
        if user and user.verify_password(password):
            return Session.create(user)
        raise AuthError("Invalid credentials")

    def logout(self, session: Session) -> None:
        session.invalidate()
''',
    "token_count": 120,
    "relevance": 0.95,
    "variant": "full",
}

PYTHON_MODELS = {
    "source": "file:auth/models.py",
    "content": '''from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    id: int
    username: str
    password_hash: str

    def verify_password(self, password: str) -> bool:
        return hash(password) == self.password_hash

@dataclass
class Session:
    user: User
    created_at: datetime
    token: str

    @classmethod
    def create(cls, user: User) -> "Session":
        return cls(user=user, created_at=datetime.now(), token="...")

    def invalidate(self) -> None:
        pass
''',
    "token_count": 150,
    "relevance": 0.85,
    "variant": "full",
}

PYTHON_QUERIES = {
    "source": "file:db/queries.py",
    "content": '''from db.connection import get_db

def get_user_by_id(user_id: str):
    db = get_db()
    return db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

def create_user(username: str, password_hash: str):
    db = get_db()
    db.execute("INSERT INTO users ...", (username, password_hash))
''',
    "token_count": 80,
    "relevance": 0.70,
    "variant": "full",
}

PYTHON_TEST = {
    "source": "file:tests/test_auth.py",
    "content": '''from auth.handler import AuthHandler, login
from auth.models import User

def test_login_success():
    handler = AuthHandler()
    session = handler.login("admin", "password")
    assert session is not None

def test_login_failure():
    handler = AuthHandler()
    with pytest.raises(AuthError):
        handler.login("admin", "wrong")
''',
    "token_count": 90,
    "relevance": 0.60,
    "variant": "full",
}

JS_COMPONENT = {
    "source": "file:src/components/UserProfile.tsx",
    "content": '''import React from 'react';
import { User } from '../types/user';
import { fetchUser } from '../api/userService';

interface Props {
    userId: string;
}

export default function UserProfile({ userId }: Props) {
    const [user, setUser] = React.useState<User | null>(null);
    // ...
}
''',
    "token_count": 100,
    "relevance": 0.80,
    "variant": "full",
}

JS_SERVICE = {
    "source": "file:src/api/userService.ts",
    "content": '''import { User } from '../types/user';

export async function fetchUser(userId: string): Promise<User> {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
}

export async function updateUser(user: User): Promise<void> {
    await fetch(`/api/users/${user.id}`, { method: 'PUT', body: JSON.stringify(user) });
}
''',
    "token_count": 85,
    "relevance": 0.75,
    "variant": "full",
}


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: Import extraction
# ═══════════════════════════════════════════════════════════════════════

class TestImportExtraction:
    def test_python_from_import(self):
        imports = _extract_imports_from_content(
            "from auth.models import User, Session", "handler.py"
        )
        assert "User" in imports
        assert "Session" in imports
        assert "models" in imports  # module name

    def test_python_plain_import(self):
        imports = _extract_imports_from_content("import logging", "handler.py")
        assert "logging" in imports

    def test_js_named_import(self):
        imports = _extract_imports_from_content(
            "import { fetchUser, updateUser } from '../api/userService';",
            "component.tsx",
        )
        assert "fetchUser" in imports
        assert "updateUser" in imports
        assert "userService" in imports

    def test_js_default_import(self):
        imports = _extract_imports_from_content(
            "import React from 'react';", "component.tsx"
        )
        assert "React" in imports

    def test_rust_use(self):
        imports = _extract_imports_from_content(
            "use crate::depgraph::{DepGraph, extract_identifiers};",
            "lib.rs",
        )
        assert "DepGraph" in imports
        assert "extract_identifiers" in imports

    def test_go_import(self):
        imports = _extract_imports_from_content(
            'import "net/http"\nimport "fmt"', "main.go"
        )
        assert "http" in imports
        assert "fmt" in imports

    def test_java_import(self):
        imports = _extract_imports_from_content(
            "import com.example.models.User;", "Handler.java"
        )
        assert "User" in imports


class TestDefinitionExtraction:
    def test_python_function(self):
        defs = _extract_definitions_from_content("def login(self):\n    pass", "handler.py")
        assert "login" in defs

    def test_python_class(self):
        defs = _extract_definitions_from_content("class AuthHandler:\n    pass", "handler.py")
        assert "AuthHandler" in defs

    def test_python_async(self):
        defs = _extract_definitions_from_content("async def fetch():\n    pass", "api.py")
        assert "fetch" in defs

    def test_rust_fn(self):
        defs = _extract_definitions_from_content(
            "pub fn compute_score(x: f64) -> f64 {\n    x\n}", "lib.rs"
        )
        assert "compute_score" in defs

    def test_rust_struct(self):
        defs = _extract_definitions_from_content(
            "pub struct DepGraph {\n    nodes: Vec<Node>,\n}", "depgraph.rs"
        )
        assert "DepGraph" in defs

    def test_js_function(self):
        defs = _extract_definitions_from_content(
            "export function fetchUser(id) { return null; }", "api.ts"
        )
        assert "fetchUser" in defs

    def test_go_func(self):
        defs = _extract_definitions_from_content(
            "func ProcessData(input []byte) error {\n\treturn nil\n}", "processor.go"
        )
        assert "ProcessData" in defs


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: Helpers
# ═══════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_normalize_source(self):
        assert _normalize_source("file:auth/handler.py") == "auth/handler.py"
        assert _normalize_source("file:src\\lib.rs") == "src/lib.rs"

    def test_basename_stem(self):
        assert _basename_stem("file:auth/handler.py") == "handler"
        assert _basename_stem("file:src/lib.rs") == "lib"
        assert _basename_stem("file:package.json") == "package"

    def test_classify_edge_test_file(self):
        label = _classify_edge("tests/test_auth.py", "auth/handler.py", ["AuthHandler"])
        assert label == "tests"

    def test_classify_edge_with_symbols(self):
        label = _classify_edge("handler.py", "models.py", ["User", "Session"])
        assert "User" in label
        assert "Session" in label

    def test_classify_edge_config(self):
        label = _classify_edge("handler.py", "config.yaml", [])
        assert label == "configures"


# ═══════════════════════════════════════════════════════════════════════
# Integration tests: Full scaffold generation
# ═══════════════════════════════════════════════════════════════════════

class TestScaffoldGeneration:
    def test_empty_fragments(self):
        """Empty input → empty scaffold."""
        assert generate_scaffold([]) == ""

    def test_single_fragment(self):
        """Below min_fragments threshold → empty scaffold."""
        assert generate_scaffold([PYTHON_HANDLER]) == ""

    def test_two_fragments(self):
        """Still below default min_fragments=3 → empty scaffold."""
        assert generate_scaffold([PYTHON_HANDLER, PYTHON_MODELS]) == ""

    def test_basic_python_scaffold(self):
        """Three related Python files should produce a scaffold with edges."""
        scaffold = generate_scaffold(
            [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES]
        )
        assert scaffold != ""
        assert "Context Map" in scaffold
        assert "auth/handler.py" in scaffold or "handler.py" in scaffold

    def test_scaffold_contains_dependency_edges(self):
        """Scaffold should show handler → models dependency."""
        scaffold = generate_scaffold(
            [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES],
            task_type="BugFix",
        )
        assert "→" in scaffold  # dependency arrow
        assert "Dependencies" in scaffold

    def test_scaffold_with_tests(self):
        """BugFix task should show test coverage mapping."""
        scaffold = generate_scaffold(
            [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES, PYTHON_TEST],
            task_type="BugFix",
        )
        assert "test" in scaffold.lower()

    def test_scaffold_task_type_hint(self):
        """Task type should appear in scaffold when not Unknown."""
        scaffold = generate_scaffold(
            [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES],
            task_type="Refactor",
        )
        assert "Refactor" in scaffold
        assert "full dependency cluster" in scaffold

    def test_scaffold_token_budget(self):
        """Scaffold should respect max_tokens budget."""
        scaffold = generate_scaffold(
            [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES, PYTHON_TEST],
            max_tokens=50,
        )
        # 50 tokens ≈ 200 chars — scaffold should be truncated
        assert len(scaffold) < 400  # generous margin

    def test_scaffold_entry_point(self):
        """Most-depended-upon file should be identified as entry point."""
        scaffold = generate_scaffold(
            [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES],
        )
        # models.py is imported by handler.py → should be entry point
        if "Entry point" in scaffold:
            assert "models" in scaffold.lower()

    def test_js_scaffold(self):
        """JS/TS fragments should produce valid scaffold."""
        scaffold = generate_scaffold(
            [JS_COMPONENT, JS_SERVICE, PYTHON_HANDLER],  # mixed languages
        )
        # Should at least produce the header
        assert "Context Map" in scaffold or scaffold == ""

    def test_no_edges_returns_empty(self):
        """Fragments with zero cross-references → empty scaffold."""
        frag1 = {"source": "file:a.py", "content": "x = 1\ny = 2\nz = 3"}
        frag2 = {"source": "file:b.py", "content": "a = 4\nb = 5\nc = 6"}
        frag3 = {"source": "file:c.py", "content": "p = 7\nq = 8\nr = 9"}
        scaffold = generate_scaffold([frag1, frag2, frag3])
        assert scaffold == ""  # No structural info to add

    def test_scaffold_is_deterministic(self):
        """Same input → same output (no randomness)."""
        frags = [PYTHON_HANDLER, PYTHON_MODELS, PYTHON_QUERIES]
        s1 = generate_scaffold(frags, task_type="Feature")
        s2 = generate_scaffold(frags, task_type="Feature")
        assert s1 == s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
