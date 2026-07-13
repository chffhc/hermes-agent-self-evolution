"""AST-based string-constant patching for opt-in PR change building.

Exact-snippet replacement (evolution.core.pr_optin) fails whenever the
baseline text does not appear verbatim in the source — most commonly because
the literal uses escape sequences (``\\n``, ``\\"``) whose *decoded values*
were extracted for evolution. This module handles that case safely: it finds
the single string constant whose value equals the baseline text, rewrites
just that literal, and verifies by reparsing that nothing else in the module
changed. Every ambiguous or unprovable case fails closed with an error
instead of guessing: zero or multiple matching constants, f-string parts,
literals whose replacement does not round-trip to the expected AST.

Stdlib only.
"""

import ast


def replace_string_constant(
    source: str, baseline: str, evolved: str
) -> tuple[str | None, str | None]:
    """Replace the one string constant valued ``baseline`` with ``evolved``.

    A constant matches when its decoded value equals ``baseline`` exactly or
    after ``.strip()`` (extractors strip triple-quote framing whitespace; the
    original framing is preserved around ``evolved`` in that case).

    Returns ``(patched_source, None)`` on success, ``(None, error)`` on any
    refusal.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return None, f"source does not parse as Python: {e}"

    fstring_parts = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue))
        for child in ast.walk(node)
        if child is not node
    }

    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value == baseline or node.value.strip() == baseline)
    ]
    if not matches:
        return None, "no string constant matches the baseline text"
    if len(matches) > 1:
        return None, f"baseline text matches {len(matches)} string constants; ambiguous"

    node = matches[0]
    if id(node) in fstring_parts:
        return None, "baseline text is part of an f-string; refusing to patch"

    if node.value == baseline:
        new_value = evolved
    else:
        # Matched after strip(): keep the literal's whitespace framing so
        # e.g. '"""\\n...\\n"""' constants keep their newline padding.
        prefix_len = len(node.value) - len(node.value.lstrip())
        suffix_len = len(node.value) - len(node.value.rstrip())
        new_value = (
            node.value[:prefix_len] + evolved + (node.value[-suffix_len:] if suffix_len else "")
        )

    literal = _render_literal(new_value)
    if literal is None:
        return None, "evolved text is not renderable as a round-trippable literal"

    # ast columns are byte offsets into the UTF-8 encoded line.
    source_bytes = source.encode("utf-8")
    line_starts, pos = [], 0
    for line in source_bytes.splitlines(keepends=True):
        line_starts.append(pos)
        pos += len(line)
    start = line_starts[node.lineno - 1] + node.col_offset
    end = line_starts[node.end_lineno - 1] + node.end_col_offset
    patched_bytes = source_bytes[:start] + literal.encode("utf-8") + source_bytes[end:]
    try:
        patched = patched_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None, "patched source is not valid UTF-8"

    # Fail-closed proof: the patched module must parse to exactly the
    # original AST with only this constant's value changed.
    try:
        patched_tree = ast.parse(patched)
    except SyntaxError as e:
        return None, f"patched source does not parse: {e}"
    node.value = new_value
    if ast.dump(patched_tree) != ast.dump(tree):
        return None, "patched source does not round-trip to the expected AST"
    return patched, None


def _render_literal(value: str) -> str | None:
    """Render ``value`` as a literal that eval's back to it exactly.

    Prefers a plain triple-quoted form for multi-line text (readable PR
    diffs); falls back to repr(), which is always round-trippable.
    """
    candidates = []
    if "\n" in value:
        candidates.append(f'"""{value}"""')
    candidates.append(repr(value))
    for candidate in candidates:
        try:
            if ast.literal_eval(candidate) == value:
                return candidate
        except (ValueError, SyntaxError):
            continue
    return None
