"""Regression tests for GEPA budget configuration.

DSPy 3.x requires exactly one of auto, max_full_evals, or max_metric_calls to be
provided to dspy.GEPA at construction time. Passing budget only to compile()
causes a runtime error before optimization starts.
"""

import inspect


def _assert_gepa_has_constructor_budget(func):
    source = inspect.getsource(func)
    gepa_pos = source.index("dspy.GEPA(")
    compile_pos = source.index(".compile(", gepa_pos)
    gepa_block = source[gepa_pos:compile_pos]
    assert "max_metric_calls=" in gepa_block or "max_full_evals=" in gepa_block or "auto=" in gepa_block


def test_phase1_gepa_sets_budget_at_constructor():
    from evolution.skills.evolve_skill import evolve

    _assert_gepa_has_constructor_budget(evolve)


def test_phase2_gepa_sets_budget_at_constructor():
    from evolution.tools.evolve_tool_descriptions import evolve_tool_descriptions

    _assert_gepa_has_constructor_budget(evolve_tool_descriptions)


def test_phase3_gepa_sets_budget_at_constructor():
    from evolution.prompts.evolve_prompt_section import evolve_prompt_section

    _assert_gepa_has_constructor_budget(evolve_prompt_section)
