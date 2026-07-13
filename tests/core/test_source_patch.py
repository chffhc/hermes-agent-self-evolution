"""Tests for the AST-based string-constant patcher (evolution.core.source_patch).

The patcher only ever fires when exact-snippet replacement failed, so its own
failure modes must all be closed: no match, multiple matches, f-string parts,
unparseable source, and any rewrite that does not round-trip to the expected
AST must return an error rather than a guessed patch.
"""

from __future__ import annotations

import ast

from evolution.core.source_patch import replace_string_constant


def test_replaces_escaped_single_line_literal():
    # The decoded value contains a real newline; the literal spells it "\n",
    # so exact-substring matching can never find it.
    source = 'GUIDANCE = "line one\\nline two"\nOTHER = 1\n'
    baseline = "line one\nline two"
    evolved = "line one\nline two improved"

    patched, error = replace_string_constant(source, baseline, evolved)

    assert error is None
    assert "OTHER = 1" in patched
    module = {}
    exec(patched, module)
    assert module["GUIDANCE"] == evolved


def test_multiline_replacement_prefers_readable_triple_quotes():
    source = 'GUIDANCE = "a\\nb"\n'
    patched, error = replace_string_constant(source, "a\nb", "new a\nnew b")

    assert error is None
    assert '"""new a\nnew b"""' in patched


def test_strip_matched_constant_keeps_whitespace_framing():
    source = 'SECTION = "\\n  body text\\n"\n'
    patched, error = replace_string_constant(source, "body text", "evolved body")

    assert error is None
    module = {}
    exec(patched, module)
    assert module["SECTION"] == "\n  evolved body\n"


def test_implicit_concatenation_collapses_to_one_literal():
    source = 'SECTION = (\n    "part one. "\n    "part two."\n)\n'
    patched, error = replace_string_constant(source, "part one. part two.", "single part.")

    assert error is None
    module = {}
    exec(patched, module)
    assert module["SECTION"] == "single part."


def test_no_matching_constant_fails_closed():
    patched, error = replace_string_constant('X = "something"\n', "missing", "new")
    assert patched is None
    assert "no string constant" in error


def test_ambiguous_match_fails_closed():
    source = 'A = "dup\\n"\nB = "dup\\n"\n'
    patched, error = replace_string_constant(source, "dup\n", "new\n")
    assert patched is None
    assert "ambiguous" in error


def test_exact_and_strip_matches_both_count_toward_ambiguity():
    source = 'A = "text"\nB = "  text  "\n'
    patched, error = replace_string_constant(source, "text", "new")
    assert patched is None
    assert "2 string constants" in error


def test_fstring_part_fails_closed():
    source = 'X = f"hello {name}\\n"\n'
    patched, error = replace_string_constant(source, "hello ", "hi ")
    assert patched is None
    # Depending on the Python version the constant part is either invisible
    # (no match) or visible inside the JoinedStr (refused as an f-string
    # part); both are closed failures.
    assert "f-string" in error or "no string constant" in error


def test_unparseable_source_fails_closed():
    patched, error = replace_string_constant("def broken(:\n", "x", "y")
    assert patched is None
    assert "does not parse" in error


def test_evolved_text_with_backslashes_still_roundtrips():
    source = 'X = "old\\nvalue"\n'
    evolved = 'weird \\ backslash and "quotes"\nsecond line'
    patched, error = replace_string_constant(source, "old\nvalue", evolved)

    assert error is None
    module = {}
    exec(patched, module)
    assert module["X"] == evolved


def test_docstring_like_and_nested_constants_only_one_match(tmp_path):
    source = 'CONST = "target\\nvalue"\n' "def f():\n" '    return "unrelated"\n'
    patched, error = replace_string_constant(source, "target\nvalue", "patched\nvalue")

    assert error is None
    tree = ast.parse(patched)
    assert isinstance(tree.body[0], ast.Assign)
    assert tree.body[0].value.value == "patched\nvalue"


def test_non_ascii_source_uses_correct_byte_offsets():
    source = 'CAFÉ = "café"\nTARGET = "old\\ntext"  # café ☕\n'
    patched, error = replace_string_constant(source, "old\ntext", "new\ntext")

    assert error is None
    module = {}
    exec(patched, module)
    assert module["TARGET"] == "new\ntext"
    assert module["CAFÉ"] == "café"


def test_rest_of_module_never_changes():
    source = 'A = 1\nB = "old\\nvalue"\nC = [1, 2, 3]\n'
    patched, error = replace_string_constant(source, "old\nvalue", "new\nvalue")

    assert error is None
    old_tree = ast.parse(source)
    new_tree = ast.parse(patched)
    # Everything except the patched constant is structurally identical.
    old_tree.body[1].value.value = "new\nvalue"
    assert ast.dump(new_tree) == ast.dump(old_tree)
