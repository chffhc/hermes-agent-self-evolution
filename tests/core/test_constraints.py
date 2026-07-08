"""Tests for constraint validators."""

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintResult, ConstraintValidator


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 60_000, "skill")
        assert not result.passed
        assert "exceeded" in result.message

    def test_tool_description_under_limit(self, validator):
        result = validator._check_size("Search files by content", "tool_description")
        assert result.passed

    def test_tool_description_over_limit(self, validator):
        result = validator._check_size("x" * 600, "tool_description")
        assert not result.passed


class TestGrowthConstraints:
    def test_acceptable_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # 10% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_excessive_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1300  # 30% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert not result.passed

    def test_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # 20% smaller
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed


class TestNonEmpty:
    def test_non_empty_passes(self, validator):
        result = validator._check_non_empty("some content")
        assert result.passed

    def test_empty_fails(self, validator):
        result = validator._check_non_empty("")
        assert not result.passed

    def test_whitespace_only_fails(self, validator):
        result = validator._check_non_empty("   \n  ")
        assert not result.passed


class TestSkillStructure:
    def test_valid_skill(self, validator):
        skill = "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\nContent here"
        result = validator._check_skill_structure(skill)
        assert result.passed

    def test_missing_frontmatter(self, validator):
        skill = "# Test\nContent without frontmatter"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_name(self, validator):
        skill = "---\ndescription: A test skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_description(self, validator):
        skill = "---\nname: test-skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed


class TestRunTestSuite:
    def test_repo_sanity_mode_does_not_claim_artifact_gate(self, validator, tmp_path, monkeypatch):
        def fake_run_pytest(repo, constraint_name):
            assert repo == tmp_path
            return ConstraintResult(True, constraint_name, "All tests passed")

        monkeypatch.setattr(validator, "_run_pytest", fake_run_pytest)

        result = validator.run_test_suite(tmp_path)

        assert result.passed
        assert result.constraint_name == "repo_sanity_test_suite"
        assert "evolved artifact was NOT applied" in result.message

    def test_artifact_gate_applies_artifact_in_temp_workspace(
        self, validator, tmp_path, monkeypatch
    ):
        skill_path = tmp_path / "skills" / "demo" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("baseline", encoding="utf-8")
        (tmp_path / "tests").mkdir()

        def fake_run_pytest(repo, constraint_name):
            assert repo != tmp_path
            assert constraint_name == "artifact_test_suite"
            assert (repo / "skills" / "demo" / "SKILL.md").read_text(encoding="utf-8") == "evolved"
            return ConstraintResult(True, constraint_name, "All tests passed")

        monkeypatch.setattr(validator, "_run_pytest", fake_run_pytest)

        result = validator.run_test_suite(
            tmp_path,
            artifact_relpath="skills/demo/SKILL.md",
            artifact_text="evolved",
        )

        assert result.passed
        assert result.constraint_name == "artifact_test_suite"
        assert "evolved artifact applied" in result.message
        assert skill_path.read_text(encoding="utf-8") == "baseline"

    def test_artifact_gate_rejects_partial_artifact_arguments(self, validator, tmp_path):
        result = validator.run_test_suite(tmp_path, artifact_relpath="skills/demo/SKILL.md")

        assert not result.passed
        assert result.constraint_name == "artifact_test_suite"
        assert "Both artifact_relpath and artifact_text are required" in result.message


class TestValidateAll:
    def test_valid_skill_passes_all(self, validator):
        skill = "---\nname: test\ndescription: Test skill\n---\n\n# Procedure\n1. Do thing"
        results = validator.validate_all(skill, "skill")
        assert all(r.passed for r in results)

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0
