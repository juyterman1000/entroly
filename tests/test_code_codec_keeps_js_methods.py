"""The code skeleton must keep JavaScript methods assigned to objects.

Express declares its whole API as `res.send = function send(body) {...}`.
The line skeleton recognised none of those lines as declarations, so
`compress()` and `entroly compress` turned express's lib/response.js into
178 tokens of blank lines and stray statements with no res.* method left,
while the codec promises to keep "every import and declaration line".
"""

from __future__ import annotations

import pytest

from entroly import compress
from entroly.codec import RecoveryStore
from entroly.codecs_builtin import default_registry
from entroly.codecs_content import _DEF_RE

DECLARATIONS = [
    "res.status = function status(code) {",
    "res.send = function send(body) {",
    "exports.render = function (name, options, callback) {",
    "module.exports = function createApplication() {",
    "app.handle = async (req, res, next) => {",
    "res.json = async function json(obj) {",
    "Router.prototype.route = function route(path) {",
    "proto.param = name => {",
]


def _express_style_module() -> str:
    blocks = ["'use strict';", "var http = require('node:http');", ""]
    for i, declaration in enumerate(DECLARATIONS):
        blocks.append(f"/**\n * Handler {i}.\n */\n{declaration}")
        blocks.extend(f"  var step{j} = this.compute({i}, {j});" for j in range(12))
        blocks.append(f"  return this.finish(step0, {i});\n}};\n")
    return "\n".join(blocks)


@pytest.mark.parametrize("line", DECLARATIONS)
def test_assigned_function_declarations_are_declarations(line):
    assert _DEF_RE.match(line)
    assert _DEF_RE.match("    " + line)


@pytest.mark.parametrize(
    "line",
    [
        "obj.count = count + 1;",
        "self.handler = make_handler(config)",
        "res.statusCode = 200;",
        'msg.text = "function call failed"',
        "value.function_name = other",
        "x.y = func(a, b)",
    ],
)
def test_ordinary_assignments_are_not_declarations(line):
    assert not _DEF_RE.match(line)


def test_skeleton_keeps_every_method_declaration(tmp_path):
    text = _express_style_module()
    reps = default_registry(RecoveryStore(tmp_path / "r.json")).representations(
        text, source_id="response.js"
    )
    skeleton = next(r for r in reps if "#code.skeleton." in r.representation_id)
    for declaration in DECLARATIONS:
        assert declaration in skeleton.text, f"{declaration!r} missing from the skeleton"
    assert "var step5" not in skeleton.text


def test_public_compress_keeps_the_module_surface():
    text = _express_style_module()
    out = compress(text, budget=len(text) // 8)
    for declaration in DECLARATIONS:
        assert declaration in out, f"{declaration!r} missing from compress() output"
