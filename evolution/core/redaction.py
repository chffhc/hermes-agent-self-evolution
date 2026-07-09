"""Secret detection and redaction shared across the evolution pipeline.

Kept dependency-free (stdlib only) so light modules like PRBuilder can use it
without pulling in dspy or other heavy imports.
"""

import re

# Patterns that indicate secrets — NEVER include these in datasets, PR bodies,
# or commit messages. Each pattern is intentionally anchored to known key
# formats to minimize false positives on normal prose.
SECRET_PATTERNS = re.compile(
    r"("
    r"sk-ant-api\S+"  # Anthropic API keys
    r"|sk-or-v1-\S+"  # OpenRouter API keys
    r"|sk-\S{20,}"  # Generic OpenAI-style keys (20+ chars after sk-)
    r"|ghp_\S+"  # GitHub personal access tokens
    r"|ghu_\S+"  # GitHub user tokens
    r"|xoxb-\S+"  # Slack bot tokens
    r"|xapp-\S+"  # Slack app tokens
    r"|ntn_\S+"  # Notion integration tokens
    r"|AKIA[0-9A-Z]{16}"  # AWS access key IDs
    r"|Bearer\s+\S{20,}"  # Bearer auth headers (20+ char tokens)
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"  # JWT values
    r"|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----"  # PEM private keys
    r"|ANTHROPIC_API_KEY"  # Known env var names (exact match)
    r"|OPENAI_API_KEY"
    r"|OPENROUTER_API_KEY"
    r"|DASHSCOPE_API_KEY"
    r"|DASHSCOPE_BASE_URL"
    r"|ALIBABA_CLOUD_ACCESS_KEY_ID"
    r"|ALIBABA_CLOUD_ACCESS_KEY_SECRET"
    r"|SLACK_BOT_TOKEN"
    r"|GITHUB_TOKEN"
    r"|AWS_SECRET_ACCESS_KEY"
    r"|DATABASE_URL"
    r"|\bpassword\s*[=:]\s*\S+"  # password assignments (password=xxx, password: xxx)
    r"|\bsecret\s*[=:]\s*\S+"  # secret assignments (secret=xxx, secret: xxx)
    r"|\btoken\s*[=:]\s*\S{10,}"  # token assignments with 10+ char values
    r"|\bjwt\s*[=:]\s*\S{20,}"  # jwt=xxx / jwt: xxx assignments
    r"|\bapi[_-]?key\s*[=:]\s*\S{10,}"  # api_key=xxx assignments
    r")",
    re.IGNORECASE,
)


def contains_secret(text: str) -> bool:
    """Check if text contains potential API keys or tokens."""
    return bool(SECRET_PATTERNS.search(text))


def redact_secrets(text: str) -> str:
    """Replace secret-looking substrings before any text is persisted or published."""
    redacted = SECRET_PATTERNS.sub("[REDACTED]", text or "")
    # If both an env-var name and its value matched, collapse
    # ``[REDACTED]=[REDACTED]`` to a single marker.
    return re.sub(r"\[REDACTED\](?:\s*[=:]\s*\[REDACTED\])+", "[REDACTED]", redacted)
