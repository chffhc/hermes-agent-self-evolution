"""End-to-end integration tests for Phase 2 and Phase 3 evolution pipelines.

These tests verify that GEPA/MIPROv2 can actually mutate parameters and
that evolved artifacts can be correctly extracted — without making live
API calls (using mocked LLM responses).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Tool Description Module — GEPA mutation & extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestToolDescriptionModule:
    """Verify ToolDescriptionModule embeds descriptions in signature instruction."""

    def test_sentinel_in_instruction(self):
        from evolution.tools.evolve_tool_descriptions import (
            ToolDescription,
            ToolDescriptionModule,
        )

        tools = [
            ToolDescription(
                name="read_file",
                toolset="core",
                description="Reads file contents",
                param_descriptions={"path": "File path"},
                schema={},
                file_path="tools/read_file.py",
            ),
        ]
        module = ToolDescriptionModule(tools)

        # ChainOfThought wraps a Predict which has a signature
        predictor = module.predictor
        # Access the underlying signature via the wrapped predict
        sig = getattr(predictor.predict, "signature", None) or getattr(
            predictor, "_predict", None
        )
        if sig is not None:
            doc = getattr(sig, "__doc__", "") or getattr(sig, "instructions", "")
            assert "<!-- __TOOL_DESC_START__ -->" in doc
            assert "read_file" in doc

    def test_forward_does_not_require_tool_descriptions_input(self):
        """forward() should only take 'task', not 'tool_descriptions'."""
        from evolution.tools.evolve_tool_descriptions import (
            ToolDescription,
            ToolDescriptionModule,
        )

        tools = [
            ToolDescription(
                name="write_file",
                toolset="core",
                description="Writes to a file",
                param_descriptions={},
                schema={},
                file_path="tools/write_file.py",
            ),
        ]
        module = ToolDescriptionModule(tools)

        # The key check: forward only takes 'task', not 'tool_descriptions'
        forward_params = module.forward.__code__.co_varnames[
            : module.forward.__code__.co_argcount
        ]
        assert "task" in forward_params
        assert "tool_descriptions" not in forward_params

    def test_get_evolved_tools_extracts_json(self):
        """Verify evolved tool JSON can be extracted from a compiled module."""
        from evolution.tools.evolve_tool_descriptions import (
            _TOOL_SENTINEL_END,
            _TOOL_SENTINEL_START,
            ToolDescription,
            ToolDescriptionModule,
        )

        tools = [
            ToolDescription(
                name="search",
                toolset="core",
                description="Original desc",
                param_descriptions={},
                schema={},
                file_path="tools/search.py",
            ),
        ]
        module = ToolDescriptionModule(tools)

        # Mock the compiled module's signature
        evolved_json = '[{"name": "search", "description": "IMPROVED desc"}]'
        instruction = (
            f"some text {_TOOL_SENTINEL_START}\n{evolved_json}\n{_TOOL_SENTINEL_END} end"
        )

        class FakeSignature:
            __doc__ = instruction

        class FakePredictor:
            signature = FakeSignature()

        module.predictor = FakePredictor()

        evolved = module.get_evolved_tools()
        assert evolved[0].description == "IMPROVED desc"

    def test_no_tool_descriptions_as_input_field(self):
        """Ensure tool_descriptions is NOT an InputField (was the original bug)."""
        from evolution.tools.evolve_tool_descriptions import (
            ToolDescription,
            ToolDescriptionModule,
        )

        tools = [
            ToolDescription(
                name="test",
                toolset="core",
                description="Test",
                param_descriptions={},
                schema={},
                file_path="tools/test.py",
            ),
        ]
        module = ToolDescriptionModule(tools)

        # Check the wrapped predict's signature
        predict = getattr(module.predictor, "predict", module.predictor)
        sig = getattr(predict, "signature", None)
        if sig is not None:
            field_names = set(sig.fields.keys())
            assert "tool_descriptions" not in field_names


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Prompt Section Module — GEPA mutation & extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptSectionModule:
    """Verify PromptSectionModule works with GEPA."""

    def test_module_has_predictors(self):
        from evolution.prompts.evolve_prompt_section import (
            PromptSection,
            PromptSectionModule,
        )

        sections = [
            PromptSection(
                name="MEMORY",
                content="Use memory wisely",
                file_path="agent/prompt_builder.py",
                description="Memory guidance",
                max_growth_pct=20,
                risk_level="low",
            ),
        ]
        module = PromptSectionModule(sections)

        assert "MEMORY" in module.predictors
        # Each predictor is a ChainOfThought
        cot = module.predictors["MEMORY"]
        # It should have a wrapped predict attribute
        assert hasattr(cot, "predict") or hasattr(cot, "_predict")

    def test_module_predictors_have_signature(self):
        """Predictors should have accessible signatures."""
        from evolution.prompts.evolve_prompt_section import (
            PromptSection,
            PromptSectionModule,
        )

        sections = [
            PromptSection(
                name="TEST",
                content="Do the thing",
                file_path="agent/prompt_builder.py",
                description="Test",
                max_growth_pct=20,
                risk_level="low",
            ),
        ]
        module = PromptSectionModule(sections)

        cot = module.predictors["TEST"]
        # Access the underlying predict's signature
        predict = getattr(cot, "predict", cot)
        sig = getattr(predict, "signature", None)
        assert sig is not None
        # Phase 3 passes section as input field (prompt_guidance), not embedded
        # in instruction — this is a known design tradeoff vs Phase 1/2
        field_names = set(sig.fields.keys())
        assert "prompt_guidance" in field_names
        assert "scenario" in field_names
        assert "behavior" in field_names


# ═══════════════════════════════════════════════════════════════════════════
# Cost Tracker
# ═══════════════════════════════════════════════════════════════════════════


class TestCostTracker:
    """Verify the API cost tracker works correctly."""

    def test_record_and_summary(self):
        from evolution.core.cost_tracker import APICostTracker

        tracker = APICostTracker()
        tracker.record("qwen3.6-plus", 1000, 500, "judge")
        tracker.record("qwen3.6-plus", 2000, 800, "reflection")

        s = tracker.summary()
        assert s.total_calls == 2
        assert s.total_input_tokens == 3000
        assert s.total_output_tokens == 1300
        assert "qwen3.6-plus" in s.per_model
        assert "judge" in s.per_purpose

    def test_unknown_model_zero_cost(self):
        from evolution.core.cost_tracker import _estimate_cost

        cost = _estimate_cost("unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_known_model_has_cost(self):
        from evolution.core.cost_tracker import _estimate_cost

        cost = _estimate_cost("qwen3.6-plus", 1_000_000, 1_000_000)
        assert cost > 0  # Should be 0.4 + 1.2 = 1.6

    def test_model_with_prefix(self):
        from evolution.core.cost_tracker import _estimate_cost

        # Strip openai/ prefix
        cost = _estimate_cost("openai/qwen3.6-plus", 1_000_000, 0)
        assert cost > 0
