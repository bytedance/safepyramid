"""Guard model registry and the plug-in interface.

SafePyramid ships two ready-to-use guards:

  - ``api``      — APIGuard: evaluate any API model (OpenAI / Anthropic /
                   Gemini / xAI / OpenRouter / OpenAI-compatible). Keys come
                   from environment variables only.
  - ``generic``  — GenericGuard: evaluate any local chat guardrail (a HF
                   repo id or a local path) served via vLLM, using the
                   shared structured-output prompt + parser.

To evaluate a model these two don't cover, implement the
:class:`BaseGuardModel` contract and either pass an instance straight to
``safepyramid.evaluate(...)`` or register it as a CLI type with
:func:`register_guard`. See ``examples/custom_guard.py``.

Loaders are imported lazily so an API-only install (no torch/vLLM) can run
API evaluations without the local-inference stack.
"""

import gc
import importlib
import inspect

from safepyramid.models.base import (  # noqa: F401 — re-exported plug-in API
    BaseGuardModel,
    SafetyResult,
    parse_structured_output,
)

# type key → (module, loader attribute, guard class). The guard class
# __init__ signature is the kwargs contract; build_loader_kwargs filters
# config keys against it (loaders themselves take **kwargs passthrough).
_LOADER_SPECS = {
    "api": ("safepyramid.models.api_guard", "load_api_guard", "APIGuard"),
    "generic": ("safepyramid.models.generic_guard", "load_generic_guard", "GenericGuard"),
}

# Custom guards registered at runtime via register_guard():
#   type_key -> {"loader": callable, "cls": class}
_CUSTOM_GUARDS: dict = {}

# Auto-detect the `api` type from common API model-name substrings, so
# `--model gpt-5.2` works without `--type`. Any other model (e.g. a local
# guardrail) must set `type:` explicitly — there is no way to tell a HF
# guardrail id apart from an API id by name alone.
_NAME_PATTERNS = [
    ("gpt-", "api"),
    ("o1", "api"),
    ("o3", "api"),
    ("gemini", "api"),
    ("claude", "api"),
    ("grok", "api"),
]


def register_guard(type_key: str, loader, guard_class=None) -> None:
    """Register a custom guard type so it can be used via config / CLI.

    Args:
        type_key:    The string used as ``type:`` in a model config.
        loader:      A callable returning a ready-to-use guard
                     (a :class:`BaseGuardModel`). Receives keyword args
                     filtered against ``guard_class.__init__`` (or, if not
                     given, the loader's own signature).
        guard_class: Optional class used to filter config kwargs. If
                     omitted, the loader's signature is used.

    For one-off use you do not need this — just pass a guard *instance*
    directly to ``safepyramid.evaluate(my_guard, cases)``.
    """
    _CUSTOM_GUARDS[type_key] = {"loader": loader, "cls": guard_class}


def _resolve_type_key(model_cfg: dict) -> str:
    """Resolve a model config to a registered guard type key."""
    model_type = model_cfg.get("type")
    if model_type:
        if model_type not in _LOADER_SPECS and model_type not in _CUSTOM_GUARDS:
            available = list(_LOADER_SPECS) + list(_CUSTOM_GUARDS)
            raise ValueError(
                f"Unknown model type '{model_type}'. Available: {available}. "
                f"For a fully custom guard, implement BaseGuardModel and pass "
                f"an instance to evaluate(), or register it with register_guard()."
            )
        return model_type

    name = model_cfg.get("name", "").lower()
    for pattern, loader_key in _NAME_PATTERNS:
        if pattern in name:
            return loader_key

    raise ValueError(
        f"Cannot auto-detect a guard type for model '{model_cfg.get('name')}'. "
        f"Set `type:` explicitly — `api` for an API model, or `generic` for "
        f"your own local guardrail (HF id or local path)."
    )


def get_loader(type_key: str):
    """Import and return the loader callable for a guard type key."""
    if type_key in _CUSTOM_GUARDS:
        return _CUSTOM_GUARDS[type_key]["loader"]
    spec = _LOADER_SPECS.get(type_key)
    if spec is None:
        raise ValueError(
            f"Unknown model type '{type_key}'. Available: "
            f"{list(_LOADER_SPECS) + list(_CUSTOM_GUARDS)}")
    module_name, attr, _cls = spec
    return getattr(importlib.import_module(module_name), attr)


def _get_kwargs_signature(type_key: str):
    """Return the parameters used to filter config kwargs for a type."""
    if type_key in _CUSTOM_GUARDS:
        entry = _CUSTOM_GUARDS[type_key]
        target = entry["cls"] or entry["loader"]
        sig_obj = target.__init__ if inspect.isclass(target) else target
        return inspect.signature(sig_obj).parameters
    module_name, _attr, cls_name = _LOADER_SPECS[type_key]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return inspect.signature(cls.__init__).parameters


def resolve_loader(model_cfg: dict):
    """Return the appropriate loader callable for a model config."""
    return get_loader(_resolve_type_key(model_cfg))


def build_loader_kwargs(model_cfg: dict) -> dict:
    """Translate a config model entry into loader kwargs.

    Config is permissive — tuning knobs can be listed uniformly — but each
    guard accepts a different subset, so candidates are filtered against the
    guard constructor's signature. Dropped knobs are reported once.
    """
    type_key = _resolve_type_key(model_cfg)
    gen_cfg = model_cfg.get("generation", {})
    vllm_cfg = model_cfg.get("vllm", {})
    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("Model config needs a `name`.")
    device = model_cfg.get("device", "auto")
    if device == "auto":
        device = None

    if type_key == "api":
        kwargs: dict = {
            "model_name": model_name,
            "backend": model_cfg.get("backend", "openai"),
        }
        for k in ("api_key_env", "base_url", "reasoning_effort",
                  "max_completion_tokens", "max_retries"):
            if k in model_cfg:
                kwargs[k] = model_cfg[k]
        # Generation knobs map differently for API backends: max_new_tokens
        # caps the completion; temperature is not sent (providers run at
        # their defaults).
        if "max_new_tokens" in gen_cfg and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = gen_cfg["max_new_tokens"]
        if "temperature" in gen_cfg:
            print(f"  [config] note: `generation.temperature` is ignored for "
                  f"API backends ({model_name})")
        return kwargs

    # Local / custom guard candidate kwargs (config is permissive).
    candidate: dict = {
        "model_name": model_name,
        "device": device,
        "cache_dir": model_cfg.get("cache_dir", "./cache"),
        "max_new_tokens": gen_cfg.get("max_new_tokens", 4096),
        "temperature": gen_cfg.get("temperature", 0.0),
        "top_p": gen_cfg.get("top_p", 1.0),
    }
    if "repetition_penalty" in gen_cfg:
        candidate["repetition_penalty"] = gen_cfg["repetition_penalty"]
    if "enable_thinking" in model_cfg:
        candidate["enable_thinking"] = model_cfg["enable_thinking"]
    for k in ("max_model_len", "gpu_memory_utilization",
              "tensor_parallel_size", "max_num_seqs"):
        if k in vllm_cfg:
            candidate[k] = vllm_cfg[k]
    # Pass any extra scalar config keys through too (custom guards may want
    # them); they get filtered against the constructor below.
    for k, v in model_cfg.items():
        if k not in ("name", "type", "device", "cache_dir", "generation",
                     "vllm", "backend") and k not in candidate:
            if isinstance(v, (str, int, float, bool)):
                candidate[k] = v

    try:
        params = _get_kwargs_signature(type_key)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kw:
            return candidate
        kept = {k: v for k, v in candidate.items() if k in params}
        dropped = sorted(set(candidate) - set(kept))
        if dropped:
            print(f"  [config] note: {type_key} guard does not accept "
                  f"{dropped} — ignored for {model_name}")
        return kept
    except (ValueError, TypeError):
        return candidate


def cleanup_cuda() -> None:
    """Best-effort GPU memory cleanup between model loads."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
