"""AST-aware Phase 3 prompt-section extraction and round-trip tests.

Extraction must return the *exact* decoded value of the string constant
(including whitespace/newline framing) so that the same constant can later
be located and rewritten by build_source_replacement_changes /
evolution.core.source_patch, and re-extraction of the patched source yields
the evolved text byte-for-byte. Every ambiguous or computed form fails
closed with a reason instead of a lossy guess.
"""

from __future__ import annotations

from pathlib import Path

from evolution.core.pr_optin import build_source_replacement_changes
from evolution.prompts.evolve_prompt_section import (
    PromptSection,
    PromptSectionModule,
    _extract_constant,
    extract_prompt_sections,
)


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "prompt_builder.py"
    path.write_text(source, encoding="utf-8")
    return path


# ── Exact decoded extraction ────────────────────────────────────────────


def test_extracts_simple_string_constant(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = "Use memory to save facts."\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert error is None
    assert value == "Use memory to save facts."


def test_extracts_decoded_escape_sequences_exactly(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = "Line one.\\nLine \\"two\\".\\tEnd"\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert error is None
    assert value == 'Line one.\nLine "two".\tEnd'


def test_extracts_triple_quoted_with_framing_newlines(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = """\nUse memory.\nSave facts.\n"""\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert error is None
    # Framing newlines are part of the decoded value and must be preserved.
    assert value == "\nUse memory.\nSave facts.\n"


def test_extracts_adjacent_literal_concatenation(tmp_path):
    path = _write(
        tmp_path,
        'MEMORY_GUIDANCE = (\n    "Use memory.\\n"\n    "Save important facts."\n)\n',
    )
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert error is None
    # Implicit concatenation folds at parse time; no artificial separator.
    assert value == "Use memory.\nSave important facts."


def test_extracts_annotated_assignment(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE: str = "Annotated section text."\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert error is None
    assert value == "Annotated section text."


# ── Fail-closed refusals ────────────────────────────────────────────────


def test_refuses_fstring(tmp_path):
    path = _write(tmp_path, 'X = "v1"\nMEMORY_GUIDANCE = f"Use memory {X}."\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "f-string" in error


def test_refuses_computed_expression(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = "Use memory." + " Save facts."\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "not a plain string constant" in error


def test_refuses_non_string_constant(tmp_path):
    path = _write(tmp_path, "MEMORY_GUIDANCE = 42\n")
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "not a plain string constant" in error


def test_refuses_duplicate_assignments(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = "first"\nMEMORY_GUIDANCE = "second"\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "ambiguous" in error


def test_refuses_shadowing_binding_elsewhere(tmp_path):
    path = _write(
        tmp_path,
        'MEMORY_GUIDANCE = "top"\n\ndef f():\n    MEMORY_GUIDANCE = "local"\n',
    )
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "ambiguous" in error


def test_refuses_non_assignment_binding(tmp_path):
    path = _write(tmp_path, 'for MEMORY_GUIDANCE in ["a"]:\n    pass\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "simple assignment" in error


def test_refuses_missing_name(tmp_path):
    path = _write(tmp_path, 'OTHER = "text"\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "no assignment" in error


def test_refuses_syntax_error(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = "unterminated\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "does not parse" in error


def test_refuses_whitespace_only_value(tmp_path):
    path = _write(tmp_path, 'MEMORY_GUIDANCE = "   \\n  "\n')
    value, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert value is None
    assert "whitespace-only" in error


def test_extract_prompt_sections_skips_failed_sections(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(
        'MEMORY_GUIDANCE = "Use memory."\nSKILLS_GUIDANCE = f"{MEMORY_GUIDANCE}"\n',
        encoding="utf-8",
    )
    sections = extract_prompt_sections(tmp_path, ["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"])
    assert [s.name for s in sections] == ["MEMORY_GUIDANCE"]
    assert sections[0].content == "Use memory."


# ── Extraction + source_patch round-trip ────────────────────────────────


def _section(content: str) -> PromptSection:
    return PromptSection(
        name="MEMORY_GUIDANCE",
        content=content,
        file_path="agent/prompt_builder.py",
        description="How and when to use persistent memory",
        max_growth_pct=20,
        risk_level="medium",
    )


def _roundtrip(tmp_path: Path, source: str, evolved: str) -> str:
    """Extract → patch via build_source_replacement_changes → re-extract."""
    (tmp_path / "agent").mkdir(exist_ok=True)
    path = tmp_path / "agent" / "prompt_builder.py"
    path.write_text(source, encoding="utf-8")

    baseline, error = _extract_constant(path, "MEMORY_GUIDANCE")
    assert error is None, error

    changes, error = build_source_replacement_changes(
        tmp_path,
        {"agent/prompt_builder.py": [(baseline, evolved)]},
        "prompt_section",
    )
    assert error is None, error
    assert len(changes) == 1

    patched = tmp_path / "patched.py"
    patched.write_text(changes[0].evolved_content, encoding="utf-8")
    value, error = _extract_constant(patched, "MEMORY_GUIDANCE")
    assert error is None, error
    return value


def test_roundtrip_triple_quoted_preserves_framing(tmp_path):
    evolved = "\nUse memory carefully.\nSave only durable facts.\n"
    result = _roundtrip(
        tmp_path,
        'MEMORY_GUIDANCE = """\nUse memory.\nSave facts.\n"""\n',
        evolved,
    )
    assert result == evolved


def test_roundtrip_escaped_literal_via_ast_patcher(tmp_path):
    # Decoded value is not a verbatim substring of the source (the literal
    # spells newlines as \n), so this exercises the AST patcher fallback.
    evolved = 'Use memory better.\nSave "important" facts twice.'
    result = _roundtrip(
        tmp_path,
        'MEMORY_GUIDANCE = "Use memory.\\nSave \\"important\\" facts."\n',
        evolved,
    )
    assert result == evolved


def test_roundtrip_adjacent_concatenation_via_ast_patcher(tmp_path):
    evolved = "Use memory sparingly.\nReview before saving."
    result = _roundtrip(
        tmp_path,
        'MEMORY_GUIDANCE = (\n    "Use memory.\\n"\n    "Save important facts."\n)\n',
        evolved,
    )
    assert result == evolved


# ── Module embed round-trip (framing re-attachment) ─────────────────────


def test_unchanged_section_roundtrips_through_module_embed():
    # The sentinel embed adds its own newlines; an unchanged section must
    # come back byte-identical to the baseline (framing re-attached), or a
    # no-op run would propose whitespace-only PR changes.
    baseline = "\nUse memory.\nSave facts.\n"
    module = PromptSectionModule([_section(baseline)])
    evolved = module.get_evolved_sections()
    assert evolved[0].content == baseline


def test_full_extract_embed_patch_reextract_roundtrip(tmp_path):
    (tmp_path / "agent").mkdir()
    path = tmp_path / "agent" / "prompt_builder.py"
    path.write_text('MEMORY_GUIDANCE = """\nUse memory.\nSave facts.\n"""\n', encoding="utf-8")

    [section] = extract_prompt_sections(tmp_path, ["MEMORY_GUIDANCE"])
    module = PromptSectionModule([section])
    [roundtripped] = module.get_evolved_sections()

    assert roundtripped.content == section.content
    # And an identical section produces no PR change at all.
    assert roundtripped.content == "\nUse memory.\nSave facts.\n"
