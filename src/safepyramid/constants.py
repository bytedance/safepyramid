"""Centralized constants for SafePyramid.

    from safepyramid.constants import DATASET_REPO_ID, EVAL_API_TIMEOUT
"""

# ---------------------------------------------------------------------------
# Benchmark dataset
# ---------------------------------------------------------------------------

# Hugging Face dataset repo id. Set HF_TOKEN if the repo is private.
DATASET_REPO_ID = "ByteDance/SafePyramid"
DATASET_FILENAME = "benchmark.json"

# ---------------------------------------------------------------------------
# API call settings
# ---------------------------------------------------------------------------

EVAL_API_TIMEOUT = 7200.0        # 2 hr — long-reasoning models (xhigh effort)
                                 # legitimately push past 1h on long L2
                                 # policies. httpx keepalive_expiry=30 still
                                 # kills truly dead connections.

# ---------------------------------------------------------------------------
# API key environment variables per backend
# ---------------------------------------------------------------------------

# Each API backend reads its key from an environment variable — never from a
# config file and never from a CLI argument (both leak into history / VCS).
# Users can point a model at a different variable with `api_key_env`.
PROVIDER_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "openai_compatible": ["OPENAI_COMPATIBLE_API_KEY", "OPENAI_API_KEY"],
}
