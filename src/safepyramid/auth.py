"""API-key resolution and leak prevention.

Design rules:
  * Keys are read from environment variables only — never from config
    files (which get committed) and never from CLI arguments (which leak
    into shell history and `ps` output).
  * As a convenience for interactive use, a missing key triggers a hidden
    `getpass` prompt; the value lives only in process memory.
  * Every key that passes through here is registered so that `redact()`
    can scrub it from error messages and logs before they are printed or
    written to disk.
"""

import os
import sys
import getpass as _getpass

from safepyramid.constants import PROVIDER_ENV_VARS

_KNOWN_SECRETS: set[str] = set()


class MissingAPIKeyError(RuntimeError):
    pass


def register_secret(value: str) -> None:
    """Remember a secret so redact() can scrub it from any output."""
    if value and len(value) >= 8:
        _KNOWN_SECRETS.add(value)


def redact(text: str) -> str:
    """Replace any registered secret occurring in *text* with a placeholder.

    Apply this to exception messages and any string that is printed or
    persisted — API SDK errors occasionally echo request headers.
    """
    if not text:
        return text
    for secret in _KNOWN_SECRETS:
        if secret in text:
            text = text.replace(secret, "***REDACTED***")
    return text


def resolve_api_key(
    backend: str,
    api_key_env: str | None = None,
    interactive: bool = True,
) -> str:
    """Resolve the API key for *backend*.

    Resolution order:
      1. The env var named by *api_key_env* (per-model `api_key_env` config).
      2. The backend's default env var(s) (see constants.PROVIDER_ENV_VARS).
      3. An interactive hidden prompt, when stdin is a TTY.

    Raises MissingAPIKeyError with setup guidance when nothing is found.
    """
    candidates = []
    if api_key_env:
        candidates.append(api_key_env)
    candidates.extend(PROVIDER_ENV_VARS.get(backend, []))

    for var in candidates:
        value = os.environ.get(var, "").strip()
        if value:
            register_secret(value)
            return value

    if interactive and sys.stdin.isatty():
        value = _getpass.getpass(
            f"API key for backend '{backend}' (input hidden, "
            f"kept in memory only): "
        ).strip()
        if value:
            register_secret(value)
            return value

    if candidates:
        hint = (f"Set one of these environment variables: "
                f"{', '.join(candidates)} (e.g. `export {candidates[0]}=...`)")
    else:
        hint = (f"Backend '{backend}' has no default key variable — pass "
                f"`api_key_env: YOUR_VAR` in the model config")
    raise MissingAPIKeyError(
        f"No API key found for backend '{backend}'. {hint}, or pass "
        f"`api_key_env: YOUR_VAR` to read a custom variable. "
        f"Keys are never read from config files or CLI flags."
    )
