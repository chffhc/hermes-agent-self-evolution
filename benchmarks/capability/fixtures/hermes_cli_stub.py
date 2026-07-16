#!/usr/bin/env python3
"""Local emulator of the current-Hermes single-query CLI contract.

This script is NOT Hermes and never produces capability evidence. It
reproduces the externally observable contract of
``python cli.py --query <prompt> --quiet --skills <name>`` in Hermes 0.18.2
so the adapter's command construction, HERMES_HOME isolation, skill
consumption proof, state.db usage attribution, timeout handling, and
evidence labeling can be tested end-to-end without Hermes or a model:

- loads ``$HERMES_HOME/skills/<name>/SKILL.md`` and embeds its body into the
  session system prompt (``agent/skill_commands.py:_build_skill_message``);
  a missing skill prints ``Error: Unknown skill(s): …`` and exits 1
  (``cli.py:main`` via ``hermes_cli/main.py:cmd_chat``);
- records a ``sessions`` + ``messages`` row in ``$HERMES_HOME/state.db``
  with the columns the adapter consumes (``hermes_state.py`` schema subset);
- prints the final response to stdout and ``session_id: <id>`` to stderr.

Failure flags exist solely so tests can prove the adapter fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    model TEXT,
    system_prompt TEXT,
    cwd TEXT,
    started_at REAL NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL
);
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skills", required=True)
    parser.add_argument("--toolsets", default="terminal")
    parser.add_argument("--model", default="stub/model")
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--solutions", help="directory copied over the workspace cwd")
    parser.add_argument("--sleep", type=float, default=0.0, help="stall to trigger timeouts")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--cost-usd", type=float, default=0.0)
    parser.add_argument("--cost-status", default="estimated")
    parser.add_argument("--cost-source", default="stub-pricing")
    parser.add_argument("--input-tokens", type=int, default=120)
    parser.add_argument("--output-tokens", type=int, default=80)
    parser.add_argument(
        "--omit-session-line", action="store_true", help="suppress the stderr session_id line"
    )
    parser.add_argument(
        "--skip-skill-load",
        action="store_true",
        help="build the system prompt WITHOUT the skill body (copied but not consumed)",
    )
    parser.add_argument("--omit-cost", action="store_true", help="store NULL estimated_cost_usd")
    parser.add_argument(
        "--text-cost", action="store_true", help="store a non-numeric estimated_cost_usd"
    )
    parser.add_argument(
        "--wrong-cwd", action="store_true", help="record a cwd outside the task workspace"
    )
    parser.add_argument(
        "--report-model", help="record this model in the session row instead of --model"
    )
    args = parser.parse_args()

    if args.sleep:
        time.sleep(args.sleep)

    hermes_home = Path(os.environ["HERMES_HOME"])
    # Names only: lets tests prove the parent's secrets were scrubbed.
    (hermes_home / "env_keys.json").write_text(
        json.dumps(sorted(os.environ), indent=2) + "\n", encoding="utf-8"
    )

    system_prompt = "You are Hermes, an AI assistant. (stub system prompt)"
    skill_md = hermes_home / "skills" / args.skills / "SKILL.md"
    if not args.skip_skill_load:
        if not skill_md.is_file():
            print(f"Error: Unknown skill(s): {args.skills}")
            return 1
        text = skill_md.read_text(encoding="utf-8")
        end = text.find("\n---\n", 4)
        body = text[end + 5 :] if text.startswith("---\n") and end >= 0 else text
        system_prompt += (
            f'\n\n[IMPORTANT: The user launched this CLI session with the "{args.skills}" '
            "skill preloaded.]\n\n" + body.strip()
        )

    if args.solutions:
        solutions = Path(args.solutions)
        if not solutions.is_dir():
            print(f"hermes-stub: solutions directory missing: {solutions}", file=sys.stderr)
            return 3
        shutil.copytree(solutions, Path.cwd(), dirs_exist_ok=True)

    session_id = f"stub-{uuid.uuid4().hex[:8]}"
    cost: object = args.cost_usd
    if args.omit_cost:
        cost = None
    elif args.text_cost:
        cost = "not-a-number"
    cwd = "/tmp/somewhere-else" if args.wrong_cwd else str(Path.cwd())
    conn = sqlite3.connect(hermes_home / "state.db")
    try:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO sessions (id, source, model, system_prompt, cwd, started_at,"
            " input_tokens, output_tokens, estimated_cost_usd, cost_status, cost_source)"
            " VALUES (?, 'cli', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                args.report_model or args.model,
                system_prompt,
                cwd,
                time.time(),
                args.input_tokens,
                args.output_tokens,
                cost,
                args.cost_status,
                args.cost_source,
            ),
        )
        response = "Task completed. (stub response, not capability evidence)"
        for role, content in (("user", args.query), ("assistant", response)):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
        conn.commit()
    finally:
        conn.close()

    print(response)
    if not args.omit_session_line:
        print(f"\nsession_id: {session_id}", file=sys.stderr)
    return args.exit_code


if __name__ == "__main__":
    sys.exit(main())
