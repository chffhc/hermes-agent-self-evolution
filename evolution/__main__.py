"""Package entry point for self-evolution CLI.

Usage:
    python -m evolution --skill systematic-debugging --iterations 50
    python -m evolution --help
"""

import sys
from pathlib import Path

# Ensure the project root is importable
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evolution.skills.evolve_skill import main as evolve_main  # noqa: E402

if __name__ == "__main__":
    evolve_main()
