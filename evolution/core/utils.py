"""Shared utility functions for the self-evolution pipeline.

Pulled here to eliminate duplication across Phase 2 and Phase 3 modules.
"""

import ast
import json
import re
from typing import Any


def parse_json_array(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array from LLM output, handling common formatting issues.

    Tolerates markdown code fences, stray text before/after the array,
    trailing commas, single-quoted Python dicts, and ast.literal_eval
    as a last-resort fallback.
    """
    text = text.strip()

    # Strategy 1: Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Strip markdown code fences
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        try:
            result = json.loads(stripped)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Extract JSON array from surrounding text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidate = match.group()

        # 3a. Try direct parse of extracted array
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # 3b. Fix trailing commas before closing brackets
        fixed = re.sub(r",(\s*[\]}])", r"\1", candidate)
        try:
            result = json.loads(fixed)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # 3c. Replace single quotes with double quotes for Python-style dicts
        fixed2 = re.sub(r"(?P<key>[{,])\s*'([^']+)'\s*:", r'\1 "\2":', fixed)
        fixed2 = re.sub(
            r":\s*'([^']*)'", lambda m: ": " + json.dumps(m.group(1)), fixed2
        )
        try:
            result = json.loads(fixed2)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # 3d. ast.literal_eval for Python-style list of dicts
        try:
            result = ast.literal_eval(fixed)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass

        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass

    return []
