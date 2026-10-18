"""Tests for Phase 4 Deep Tool & Shell Codecs."""

from entroly.shell_codec import (
    compress_compiler_diagnostics,
    compress_git_output,
    compress_test_output,
    esc_compress,
)


def test_compress_test_output_suppresses_passing_tests():
    sample_pytest = """
============================= test session starts =============================
platform linux -- Python 3.11.0, pytest-7.4.0
rootdir: /home/user/project
collected 5 items

tests/test_core.py::test_alpha PASSED                                    [ 20%]
tests/test_core.py::test_beta PASSED                                     [ 40%]
tests/test_core.py::test_gamma FAILED                                    [ 60%]
tests/test_core.py::test_delta PASSED                                    [ 80%]
tests/test_core.py::test_epsilon PASSED                                  [100%]

=================================== FAILURES ===================================
__________________________________ test_gamma __________________________________

    def test_gamma():
        val = compute_answer()
>       assert val == 42
E       AssertionError: assert 41 == 42

tests/test_core.py:45: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_core.py::test_gamma - AssertionError: assert 41 == 42
========================= 1 failed, 4 passed in 0.25s =========================
"""
    compressed = compress_test_output(sample_pytest, budget=500)

    # Must suppress passing lines
    assert "test_alpha PASSED" not in compressed
    assert "test_beta PASSED" not in compressed
    assert "passing tests suppressed" in compressed

    # Must preserve failures and summary
    assert "test_gamma" in compressed
    assert "AssertionError: assert 41 == 42" in compressed
    assert "1 failed, 4 passed" in compressed


def test_compress_compiler_diagnostics_deduplicates_warnings():
    sample_cargo = """
   Compiling my_project v0.1.0 (/home/user/my_project)
warning: unused variable: `x`
  --> src/main.rs:10:9
   |
10 |     let x = 1;
   |         ^ help: if this is intentional, prefix it with an underscore: `_x`

warning: unused variable: `y`
  --> src/lib.rs:22:9
   |
22 |     let y = 2;
   |         ^ help: if this is intentional, prefix it with an underscore: `_y`

warning: unused variable: `z`
  --> src/utils.rs:5:9
   |
 5 |     let z = 3;
   |         ^ help: if this is intentional, prefix it with an underscore: `_z`

error[E0308]: mismatched types
  --> src/main.rs:35:18
   |
35 |     let res: u32 = "string";
   |              ---   ^^^^^^^^ expected `u32`, found `&str`
   |              |
   |              expected due to this

error: aborting due to 1 previous error; 3 warnings emitted
"""
    compressed = compress_compiler_diagnostics(sample_cargo, budget=500)

    # Must deduplicate warnings
    assert "warnings deduplicated" in compressed

    # Must isolate error
    assert "error[E0308]: mismatched types" in compressed
    assert "src/main.rs:35:18" in compressed
    assert "expected `u32`, found `&str`" in compressed


def test_compress_git_output_diff_and_status():
    sample_diff = """
diff --git a/entroly/memory.py b/entroly/memory.py
index abc1234..def5678 100644
--- a/entroly/memory.py
+++ b/entroly/memory.py
@@ -100,6 +100,8 @@ class MemoryOS:
     def __init__(self):
         self.entries = {}
+        self.graph = None
+        self.temporal_index = {}
         self.lock = threading.Lock()
     def recall(self, query):
         return self.entries
"""
    compressed_diff = compress_git_output(sample_diff, budget=500)
    assert "diff --git" in compressed_diff
    assert "+        self.graph = None" in compressed_diff
    assert "files modified" in compressed_diff

    sample_status = """
On branch feat/pac-kv
Your branch is up to date with 'origin/feat/pac-kv'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   entroly/memory.py
	modified:   entroly/cache_aligner.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	entroly/pac_kv.py
	entroly/memory_graph.py

no changes added to commit (use "git add" to track)
"""
    compressed_status = compress_git_output(sample_status, budget=500)
    assert "On branch feat/pac-kv" in compressed_status
    assert "modified:   entroly/memory.py" in compressed_status
    assert "git restore" not in compressed_status
    assert "use \"git add\"" not in compressed_status


def test_esc_compress_auto_dispatches_specialized_codec():
    sample_pytest = """
============================= test session starts =============================
test_one.py::test_pass1 PASSED [ 50%]
test_one.py::test_fail2 FAILED [100%]
=================================== FAILURES ===================================
___ test_fail2 ___
> assert False
E AssertionError
========================= 1 failed, 1 passed in 0.1s ==========================
"""
    result = esc_compress(sample_pytest, tool_name="pytest")
    assert "pytest" in result.profile
    assert result.compression_ratio > 0.1
    assert "assert False" in result.compressed
