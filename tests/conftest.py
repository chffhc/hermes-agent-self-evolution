"""Shared test configuration.

The suite must not depend on real credentials in ~/.hermes/.env: unit tests
mock LLM calls, but constructing an LM still calls get_api_key(). Without
this, results depend on the machine (does ~/.hermes/.env exist?) and on test
order (config._load_hermes_env is lru_cached, so a test that fakes HOME can
cache an empty env load for the rest of the session). Provide a deterministic
dummy key so make_lm() always constructs; no test performs real API calls.
"""

import os

os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-dummy-key-for-unit-tests")
