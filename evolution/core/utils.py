"""Shared utility functions for the self-evolution pipeline.

Pulled here to eliminate duplication across Phase 2 and Phase 3 modules.
"""

import json
import re
from typing import Any


def parse_json_array(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array from LLM output, handling common formatting issues.

    Tolerates markdown code fences, stray text before/after the array,
    and line-level extraction as a second-chance fallback.
    """
    text = text.strip()

    # Extract JSON array from markdown code block if present
    if "```" in text:
        # First-chance: extract between ```json and ```
        match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
        if match:
            text = match.group(1)
        else:
            # Second-chance: find a line that starts with '[' and use it
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("["):
                    text = line
                    break

    # Find the JSON array bounds
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []
