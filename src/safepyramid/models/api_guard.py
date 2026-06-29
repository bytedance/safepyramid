"""API-based guard model (OpenAI, Anthropic, Gemini, xAI, OpenRouter, ...).

Uses external LLM APIs as policy-grounded guardrails, providing an
upper-bound reference for the benchmark.

Credentials are resolved from environment variables only (see
safepyramid.auth) and scrubbed from every error message before it is
printed. Prompt construction, response parsing, refusal detection, and
retry semantics are ported verbatim from the reference implementation —
they directly affect RMR / RMR@1.0 / RDR and must not drift.

Supported backends:
    openai             OpenAI API (or any base URL the OpenAI SDK accepts
                       via OPENAI_BASE_URL).
    anthropic          Anthropic API (alias: "claude").
    gemini             Google Gemini via Google's OpenAI-compatible
                       endpoint (aliases: "google", "gemini_direct").
    xai                xAI Grok via xai_sdk.
    openrouter         OpenRouter unified gateway (OpenAI-compatible).
    openai_compatible  Any OpenAI-compatible endpoint (vLLM serve, Azure
                       gateways, BytePlus ARK, ...); requires `base_url`.
"""

import json
import re
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from safepyramid.constants import EVAL_API_TIMEOUT
from safepyramid.auth import redact, resolve_api_key
from safepyramid.models.base import (
    BaseGuardModel,
    SafetyResult,
    _parse_verdict_output,
    STRUCTURED_POLICY_DEVELOPER_PROMPT,
    STRUCTURED_POLICY_USER_PROMPT,
    STRUCTURED_SAFETY_PROMPT_NO_POLICY,
    VERDICT_DEVELOPER_PROMPT,
    VERDICT_DEVELOPER_PROMPT_EXCEPTION,
)

_BACKEND_ALIASES = {
    "claude": "anthropic",
    "google": "gemini",
    "gemini_direct": "gemini",
    "ark": "openai_compatible",
}

_KNOWN_BACKENDS = (
    "openai", "anthropic", "gemini", "xai", "openrouter", "openai_compatible",
)


def _build_openai_client(api_key: str, base_url: Optional[str] = None):
    """Standard OpenAI-SDK client with the tuned httpx transport.

    Explicit httpx client config: short keepalive_expiry forces stale
    connections to be closed (long-running batches saw the default
    infinite keepalive cause multi-hour stalls where the SDK kept
    waiting on dead sockets). max_retries=0 disables the SDK's internal
    retry — we handle retry ourselves in _call_api, so stacking two
    retry loops at long timeouts can mean 30min+ wait per case before
    our backoff even kicks in.
    """
    import httpx
    from openai import OpenAI

    kwargs: dict = {
        "api_key": api_key,
        "timeout": EVAL_API_TIMEOUT,
        "max_retries": 0,
        "http_client": httpx.Client(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=60,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(
                EVAL_API_TIMEOUT, connect=10.0, read=EVAL_API_TIMEOUT
            ),
        ),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class APIGuard(BaseGuardModel):
    """API-based guard model using external LLM APIs."""

    def __init__(
        self,
        model_name: str,
        backend: str = "openai",
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,
        api_key_env: Optional[str] = None,
        base_url: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.backend = _BACKEND_ALIASES.get(backend, backend)
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort  # "low"/"medium"/"high"/"xhigh"
        self.api_key_env = api_key_env            # custom env var for the key
        self.base_url = base_url
        self.max_completion_tokens = max_completion_tokens
        self._client = None
        self._claude_client = None
        self._grok_api_key = None

    def load_model(self) -> None:
        """Initialize API client. The key is resolved from the environment."""
        if self.backend not in _KNOWN_BACKENDS:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Supported: {', '.join(_KNOWN_BACKENDS)}"
            )

        api_key = resolve_api_key(self.backend, api_key_env=self.api_key_env)

        if self.backend == "openai":
            self._client = _build_openai_client(api_key, self.base_url)
        elif self.backend == "anthropic":
            import anthropic
            self._claude_client = anthropic.Anthropic(api_key=api_key)
        elif self.backend == "gemini":
            # Google's OpenAI-compatible endpoint — no extra SDK needed.
            self._client = _build_openai_client(
                api_key,
                self.base_url
                or "https://generativelanguage.googleapis.com/v1beta/openai/")
        elif self.backend == "xai":
            self._grok_api_key = api_key
        elif self.backend == "openrouter":
            self._client = _build_openai_client(
                api_key, self.base_url or "https://openrouter.ai/api/v1")
        elif self.backend == "openai_compatible":
            import os
            base_url = self.base_url or os.environ.get(
                "OPENAI_COMPATIBLE_BASE_URL")
            if not base_url:
                raise ValueError(
                    "backend 'openai_compatible' needs a base_url — set it "
                    "in the model config or via OPENAI_COMPATIBLE_BASE_URL."
                )
            self._client = _build_openai_client(api_key, base_url)

        print(f"[APIGuard] Initialized {self.backend} backend, model={self.model_name}")

    def evaluate(
        self,
        text: str,
        policy: Optional[str] = None,
        case_id: Optional[str] = None,
        **kwargs,
    ) -> SafetyResult:
        """Evaluate text using the API.

        `case_id` is propagated to error logs so failures can be located
        post-hoc. Optional for backward compat.
        """
        if policy:
            system_content = STRUCTURED_POLICY_DEVELOPER_PROMPT.format(policy=policy)
            user_content = STRUCTURED_POLICY_USER_PROMPT.format(text=text)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [{"role": "user",
                         "content": STRUCTURED_SAFETY_PROMPT_NO_POLICY.format(text=text)}]
        raw, usage = self._call_api(messages, case_id=case_id)

        if raw is None:
            return SafetyResult(
                is_safe=False, confidence=0.0,
                label="refused", analysis="", final="API call failed",
                refused=True, usage=usage or None,
            )

        result = self._parse_response(raw)
        result.usage = usage or None
        return result

    def evaluate_batch(
        self,
        texts: list[str],
        policies: list[Optional[str]] | None = None,
        batch_size: int = 20,
        case_ids: list[str] | None = None,
        **kwargs,
    ) -> list[SafetyResult]:
        """Evaluate batch with concurrent API calls.

        Prints a progress line every 50 completions or every 30 seconds,
        whichever comes first. If `case_ids` is supplied (parallel list
        to `texts`), every per-call retry / final-error log line is
        tagged with `[case_id=X]` so failures can be traced back to
        specific benchmark cases. The summary at end also lists which
        case_ids were refused.
        """
        if policies is None:
            policies = [None] * len(texts)
        if case_ids is None:
            case_ids = [f"idx_{i}" for i in range(len(texts))]

        results = [None] * len(texts)
        total = len(texts)
        max_workers = min(batch_size, total)

        # Progress reporting tunables
        PRINT_EVERY = 50
        PRINT_INTERVAL_SEC = 30.0

        n_done = 0
        n_refused = 0
        n_error = 0
        refused_case_ids: list[str] = []
        error_case_ids: list[str] = []
        t_start = time.time()
        t_last_print = t_start

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i, (text, policy, cid) in enumerate(zip(texts, policies, case_ids)):
                fut = pool.submit(self.evaluate, text, policy, case_id=cid)
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                cid = case_ids[idx]
                try:
                    r = fut.result()
                    results[idx] = r
                    if getattr(r, "refused", False):
                        n_refused += 1
                        refused_case_ids.append(cid)
                except Exception as e:
                    results[idx] = SafetyResult(
                        is_safe=False, confidence=0.0,
                        label="unsafe", analysis="", final=redact(str(e)),
                    )
                    n_error += 1
                    error_case_ids.append(cid)

                n_done += 1
                now = time.time()
                if (n_done == 1
                        or n_done % PRINT_EVERY == 0
                        or n_done == total
                        or now - t_last_print >= PRINT_INTERVAL_SEC):
                    elapsed = now - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0.0
                    eta = (total - n_done) / rate if rate > 0 else 0.0
                    print(f"  [APIGuard] progress: {n_done}/{total} "
                          f"({n_done/total*100:.1f}%) "
                          f"| refused={n_refused} | errors={n_error} "
                          f"| {rate:.2f}/s | elapsed={elapsed/60:.1f}m "
                          f"| ETA={eta/60:.1f}m",
                          flush=True)
                    t_last_print = now

        # End-of-batch summary: dump which case_ids failed so they're easy
        # to grep / locate after the run completes.
        if refused_case_ids or error_case_ids:
            print(f"  [APIGuard] === failed cases summary ===")
            if refused_case_ids:
                print(f"  [APIGuard] refused ({len(refused_case_ids)}): "
                      f"{', '.join(refused_case_ids)}")
            if error_case_ids:
                print(f"  [APIGuard] errored ({len(error_case_ids)}): "
                      f"{', '.join(error_case_ids)}")

        return results

    # ------------------------------------------------------------------
    # Per-rule mode (DECOUPLED from all-rule path above).
    #
    # System message uses VERDICT_DEVELOPER_PROMPT (or its exception
    # variant) — NOT STRUCTURED_POLICY_DEVELOPER_PROMPT. The model is
    # asked for a binary verdict on a single Target rule, output as
    # {"verdict": "violated" | "not_violated", "reason": "..."}.
    # ------------------------------------------------------------------

    def evaluate_per_rule(self, task: dict) -> SafetyResult:
        """Evaluate one per-rule task (target rule + bundle context).

        Propagates a `<cid>/r<rule>` tag into `_call_api` so retry / 429 /
        error log lines can be traced back to the specific (case, rule)
        pair — same idea as the case_id tagging in `evaluate_batch`.
        """
        # Lazy import — both modules reference each other; importing at
        # module top would create a cycle.
        from safepyramid.per_rule import build_per_rule_user_text

        rtype = task.get("rule_type", "decisive")
        system_content = (
            VERDICT_DEVELOPER_PROMPT_EXCEPTION
            if rtype == "exception"
            else VERDICT_DEVELOPER_PROMPT
        )
        user_content = build_per_rule_user_text(task)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        tag_cid = f"{task.get('cid','?')}/r{task.get('rule','?')}"
        raw, usage = self._call_api(messages, case_id=tag_cid)
        if raw is None:
            return SafetyResult(
                is_safe=False, confidence=0.0,
                label="refused", analysis="", final="API call failed",
                refused=True, usage=usage or None,
            )
        result = _parse_verdict_output(raw)
        result.usage = usage or None
        return result

    def evaluate_per_rule_batch(
        self,
        tasks: list[dict],
        batch_size: int = 20,
        **kwargs,
    ) -> list[SafetyResult]:
        """Per-rule batch — one task per call, concurrent dispatch.

        Mirrors `evaluate_batch`'s progress logging: every 50 completions
        or every 30 seconds, prints `done/total | refused | parse_fail |
        errors | rate | elapsed | ETA`. At end, dumps the `<cid>/r<rule>`
        tags of any refused / errored / parse-failed tasks so they're
        easy to locate post-hoc — important when running 60k+ judgments.
        """
        results: list[SafetyResult | None] = [None] * len(tasks)
        if not tasks:
            return []
        total = len(tasks)
        max_workers = min(batch_size, total)

        PRINT_EVERY = 50
        PRINT_INTERVAL_SEC = 30.0

        n_done = 0
        n_refused = 0
        n_parse_fail = 0
        n_error = 0
        refused_tags: list[str] = []
        parse_fail_tags: list[str] = []
        error_tags: list[str] = []
        t_start = time.time()
        t_last_print = t_start

        def _tag(task: dict) -> str:
            return f"{task.get('cid','?')}/r{task.get('rule','?')}"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i, task in enumerate(tasks):
                fut = pool.submit(self.evaluate_per_rule, task)
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                tag = _tag(tasks[idx])
                try:
                    r = fut.result()
                    results[idx] = r
                    if getattr(r, "refused", False):
                        n_refused += 1
                        refused_tags.append(tag)
                    elif getattr(r, "parse_failed", False):
                        n_parse_fail += 1
                        parse_fail_tags.append(tag)
                except Exception as e:
                    results[idx] = SafetyResult(
                        is_safe=False, confidence=0.0,
                        label="refused", analysis="", final=redact(str(e)),
                        refused=True,
                    )
                    n_error += 1
                    error_tags.append(tag)

                n_done += 1
                now = time.time()
                if (n_done == 1
                        or n_done % PRINT_EVERY == 0
                        or n_done == total
                        or now - t_last_print >= PRINT_INTERVAL_SEC):
                    elapsed = now - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0.0
                    eta = (total - n_done) / rate if rate > 0 else 0.0
                    print(f"  [APIGuard per-rule] progress: {n_done}/{total} "
                          f"({n_done/total*100:.1f}%) "
                          f"| refused={n_refused} | parse_fail={n_parse_fail} "
                          f"| errors={n_error} | {rate:.2f}/s "
                          f"| elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m",
                          flush=True)
                    t_last_print = now

        if refused_tags or parse_fail_tags or error_tags:
            print(f"  [APIGuard per-rule] === failed tasks summary ===")
            if refused_tags:
                print(f"  [APIGuard per-rule] refused ({len(refused_tags)}): "
                      f"{', '.join(refused_tags[:50])}"
                      + (f" ...+{len(refused_tags)-50} more" if len(refused_tags) > 50 else ""))
            if parse_fail_tags:
                print(f"  [APIGuard per-rule] parse_failed ({len(parse_fail_tags)}): "
                      f"{', '.join(parse_fail_tags[:50])}"
                      + (f" ...+{len(parse_fail_tags)-50} more" if len(parse_fail_tags) > 50 else ""))
            if error_tags:
                print(f"  [APIGuard per-rule] errored ({len(error_tags)}): "
                      f"{', '.join(error_tags[:50])}"
                      + (f" ...+{len(error_tags)-50} more" if len(error_tags) > 50 else ""))

        return results  # type: ignore[return-value]

    def _call_api(
        self,
        messages: list[dict],
        case_id: Optional[str] = None,
    ) -> tuple[Optional[str], dict]:
        """Call the API with retries.

        Returns `(content_or_None, usage_dict)`. `usage_dict` carries
        token counts for cost/quality analysis post-hoc; it may be empty
        `{}` for backends/SDKs that don't expose usage. Keys normalised
        across backends: `prompt_tokens`, `completion_tokens`,
        `total_tokens`, optional `reasoning_tokens` for thinking models.

        `case_id` (optional) is included in all error/retry log lines so
        failures can be traced back to specific benchmark cases post-hoc.
        """
        # Build a short tag for log lines
        tag = f"[case_id={case_id}]" if case_id else ""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.backend == "openai":
                    api_kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        # Generous output budget: xhigh reasoning on long L2
                        # policies has been observed using up to ~29k
                        # reasoning tokens; 65k is 2x that, well above the
                        # observed tail.
                        "max_completion_tokens":
                            self.max_completion_tokens or 65536,
                    }
                    if self.reasoning_effort:
                        api_kwargs["reasoning_effort"] = self.reasoning_effort
                    resp = self._client.chat.completions.create(**api_kwargs)
                    content = resp.choices[0].message.content
                    usage = {}
                    u = getattr(resp, "usage", None)
                    if u:
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            v = getattr(u, k, None)
                            if v is not None: usage[k] = v
                        details = getattr(u, "completion_tokens_details", None)
                        if details:
                            rt = getattr(details, "reasoning_tokens", None)
                            if rt is not None: usage["reasoning_tokens"] = rt
                    return (content.strip() if content else None, usage)

                elif self.backend == "xai":
                    from xai_sdk import Client
                    from xai_sdk.chat import user as xai_user, system as xai_system
                    client = Client(api_key=self._grok_api_key)
                    chat = client.chat.create(model=self.model_name)
                    # Preserve system/user role split
                    for m in messages:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if role == "system":
                            chat.append(xai_system(content))
                        else:
                            chat.append(xai_user(content))
                    response = chat.sample()
                    usage = {}
                    u = getattr(response, "usage", None)
                    if u:
                        for src, dst in (
                            ("prompt_tokens", "prompt_tokens"),
                            ("completion_tokens", "completion_tokens"),
                            ("total_tokens", "total_tokens"),
                            ("reasoning_tokens", "reasoning_tokens"),
                        ):
                            v = getattr(u, src, None)
                            if v is not None: usage[dst] = v
                    return (response.content if response.content else None, usage)

                elif self.backend == "anthropic":
                    # Split system message from user/assistant messages
                    system_text = None
                    claude_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system_text = msg["content"]
                        else:
                            claude_messages.append(msg)
                    if not claude_messages:
                        claude_messages = messages

                    api_kwargs = {
                        "model": self.model_name,
                        # 32768 total budget for thinking + response. Lower
                        # values cause truncation on complex policy reasoning.
                        "max_tokens": self.max_completion_tokens or 32768,
                        "messages": claude_messages,
                    }
                    if system_text:
                        api_kwargs["system"] = system_text
                    # Adaptive thinking: Claude decides thinking budget per
                    # task. Don't force enabled+budget — adaptive handles
                    # complex policy reasoning well without wasted cost on
                    # simple cases.
                    api_kwargs["thinking"] = {
                        "type": "adaptive",
                    }

                    resp = self._claude_client.messages.create(**api_kwargs)
                    # Collect thinking and answer separately
                    thinking_parts = []
                    answer_parts = []
                    for block in resp.content:
                        if block.type == "thinking":
                            thinking_parts.append(block.thinking)
                        elif block.type == "text":
                            answer_parts.append(block.text)
                    # Pack both into return: <think>...</think> + answer
                    thinking_text = "\n".join(thinking_parts)
                    answer_text = "\n".join(answer_parts).strip()
                    # Anthropic usage. Note: input_tokens / output_tokens map
                    # to prompt / completion; thinking tokens are NOT exposed
                    # as a separate field — they're part of output_tokens.
                    usage = {}
                    u = getattr(resp, "usage", None)
                    if u:
                        pt = getattr(u, "input_tokens", None)
                        ct = getattr(u, "output_tokens", None)
                        if pt is not None: usage["prompt_tokens"] = pt
                        if ct is not None: usage["completion_tokens"] = ct
                        if pt is not None and ct is not None:
                            usage["total_tokens"] = pt + ct
                    if thinking_text:
                        return (f"<think>\n{thinking_text}\n</think>\n{answer_text}", usage)
                    return (answer_text or None, usage)

                elif self.backend == "openrouter":
                    # OpenAI-compatible call via OpenRouter's unified
                    # gateway. Effort is forwarded via the platform-unified
                    # `reasoning` field (extra_body) — OpenRouter translates
                    # it to each provider's native parameter on the back
                    # end. max_tokens=32768 caps total output (reasoning +
                    # answer); without this cap, long-thinking models can
                    # produce 70k+ token responses that get truncated in
                    # OpenRouter's streaming layer and fail JSON parsing.
                    api_kwargs: dict = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": self.max_completion_tokens or 32768,
                        "extra_headers": {
                            "X-OpenRouter-Title": "SafePyramid Evaluation",
                        },
                    }
                    if self.reasoning_effort:
                        api_kwargs["extra_body"] = {
                            "reasoning": {"effort": self.reasoning_effort},
                        }
                    resp = self._client.chat.completions.create(**api_kwargs)
                    content = resp.choices[0].message.content
                    usage = {}
                    u = getattr(resp, "usage", None)
                    if u:
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            v = getattr(u, k, None)
                            if v is not None: usage[k] = v
                        details = getattr(u, "completion_tokens_details", None)
                        if details:
                            rt = getattr(details, "reasoning_tokens", None)
                            if rt is not None: usage["reasoning_tokens"] = rt
                    return (content.strip() if content else None, usage)

                elif self.backend in ("openai_compatible", "gemini"):
                    # Generic OpenAI-compatible endpoint (also Gemini, via
                    # Google's OpenAI-compatible API). Reasoning effort via
                    # top-level `reasoning_effort` (OpenAI-standard); output
                    # cap only when explicitly configured — some gateways
                    # reject unknown caps.
                    api_kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                    }
                    if self.max_completion_tokens:
                        api_kwargs["max_completion_tokens"] = self.max_completion_tokens
                    if self.reasoning_effort:
                        api_kwargs["reasoning_effort"] = self.reasoning_effort
                    resp = self._client.chat.completions.create(**api_kwargs)
                    content = resp.choices[0].message.content
                    usage = {}
                    u = getattr(resp, "usage", None)
                    if u:
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            v = getattr(u, k, None)
                            if v is not None: usage[k] = v
                        details = getattr(u, "completion_tokens_details", None)
                        if details:
                            rt = getattr(details, "reasoning_tokens", None)
                            if rt is not None: usage["reasoning_tokens"] = rt
                    return (content.strip() if content else None, usage)

            except Exception as e:
                # Retry-eligible classes:
                #   1. 429 rate limit — gateway throttle. Backoff 10/20/30s.
                #   2. Connection error — transient network/socket. Retry.
                # NOT retried:
                #   - Timeout: when a request times out it likely means the
                #     model is still reasoning past our budget. Retrying just
                #     burns another full budget worth of tokens for the same
                #     case. Mark as refused and move on.
                #   - 400 / auth: permanent.
                err_str = str(e)
                err_lower = err_str.lower()
                is_rate_limit = "429" in err_str or "rate" in err_lower
                is_timeout = (
                    "timeout" in err_lower
                    or "timed out" in err_lower
                    or e.__class__.__name__ in ("APITimeoutError", "Timeout")
                )
                is_connection = (
                    "connection" in err_lower
                    or e.__class__.__name__ in ("APIConnectionError",
                                                "ConnectionError")
                )

                if attempt < self.max_retries:
                    if is_rate_limit:
                        wait = 10 * attempt
                        print(f"  [APIGuard]{tag} Rate limited, waiting {wait}s "
                              f"(attempt {attempt}/{self.max_retries})...")
                        time.sleep(wait)
                        continue
                    if is_connection and not is_timeout:
                        # Connection error only — retry quickly. Timeout
                        # is explicitly NOT retried (see comment above).
                        wait = 3 * attempt
                        print(f"  [APIGuard]{tag} connection error, retrying in {wait}s "
                              f"(attempt {attempt}/{self.max_retries})...")
                        time.sleep(wait)
                        continue
                    if is_timeout:
                        print(f"  [APIGuard]{tag} timeout — NOT retrying, marking case as refused")
                        # fall through to return (None, {}) below

                # Non-retryable error (e.g. 400, 401 auth) OR retries
                # exhausted. We MUST return a tuple here to match the
                # signature: bare `None` would crash the caller's unpack
                # (`raw, usage = self._call_api(...)`) and route to the
                # exception path in evaluate_batch — which counts as `error`
                # rather than `refused`, deflating reported metrics. Return
                # (None, {}) so callers can treat it uniformly as a refusal
                # (no model signal received).
                print(f"  [APIGuard]{tag} API error: {redact(str(e))}")
                return (None, {})
        return (None, {})

    _REFUSAL_PATTERNS = [
        "i cannot", "i can't", "i'm not able to", "i am not able to",
        "i'm unable to", "i am unable to", "i must decline",
        "i cannot evaluate", "i can't evaluate", "i cannot assess",
        "i can't assess", "i cannot analyze", "i can't analyze",
        "not appropriate for me to", "i must refrain",
        "against my guidelines", "violates my policy",
    ]

    def _is_refusal(self, text: str) -> bool:
        """Detect if the model refused to evaluate the conversation.

        A refusal is distinguished from a substantive answer by the
        ABSENCE of structured output markers. If the output carries any
        attempt at the JSON schema we asked for ("analysis",
        "violated_rules", or legacy "label"), we treat it as an answer
        even if the analysis text happens to contain a phrase like
        "I cannot confirm whether...". That's reasoning hedging, not a
        refusal to engage.
        """
        lower = text.lower()
        has_refusal = any(p in lower for p in self._REFUSAL_PATTERNS)
        has_structured = (
            '"analysis"' in lower
            or '"violated_rules"' in lower
            or ('"label"' in lower and ('"safe"' in lower or '"unsafe"' in lower))
        )
        return has_refusal and not has_structured

    _THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

    def _parse_response(self, raw: str) -> SafetyResult:
        """Parse JSON response from API."""
        # Extract thinking content if present (Claude adaptive thinking)
        thinking_content = ""
        cleaned_raw = raw
        think_match = self._THINK_RE.search(raw)
        if think_match:
            thinking_content = think_match.group(1)
            cleaned_raw = raw[think_match.end():].strip()

        # Refusal → treat as "unsafe" (model perceives content as unsafe)
        if self._is_refusal(cleaned_raw or raw):
            return SafetyResult(
                is_safe=False, confidence=0.0,
                label="unsafe", analysis=thinking_content, final=raw[:500],
                refused=True,
            )

        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned_raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            # Label derivation:
            # The current STRUCTURED_POLICY_DEVELOPER_PROMPT schema does
            # NOT ask the model to emit a top-level `label` field — only
            # `analysis` + `violated_rules`. So `data.get("label", ...)`
            # almost always falls through to the default. A conservative
            # `"unsafe"` default would make EVERY parsed response carry
            # label='unsafe' regardless of whether the rule list was
            # empty, which breaks per-rule eval's label fallback.
            #
            # Correct behavior: when the schema-required `violated_rules`
            # field is empty, the response is implicitly safe; non-empty
            # → unsafe. Only honor an explicit `label` when the model
            # supplies one (legacy schemas).
            raw_violated = data.get("violated_rules", [])
            if "label" in data:
                label = str(data["label"]).lower().strip()
                if label not in ("safe", "unsafe"):
                    label = "safe" if not raw_violated else "unsafe"
            else:
                label = "safe" if not raw_violated else "unsafe"
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            analysis = data.get("analysis", "")

            # Extract structured violation/exception data
            violated_rules = []
            rule_reasoning = {}
            for v in data.get("violated_rules", []):
                if isinstance(v, dict):
                    rn = v.get("rule")
                    if rn is not None:
                        rn = int(rn)
                        violated_rules.append(rn)
                        rule_reasoning[rn] = v.get("explanation", "")
                elif isinstance(v, (int, float)):
                    violated_rules.append(int(v))

            applicable_exceptions = []
            for e in data.get("applicable_exceptions", []):
                if isinstance(e, dict):
                    rn = e.get("rule")
                    if rn is not None:
                        applicable_exceptions.append(int(rn))
                elif isinstance(e, (int, float)):
                    applicable_exceptions.append(int(e))

            # Merge thinking content with JSON analysis
            if thinking_content:
                full_analysis = f"[Thinking]\n{thinking_content}\n\n[Analysis]\n{analysis}" if analysis else thinking_content
            else:
                full_analysis = analysis

            return SafetyResult(
                is_safe=(label == "safe"),
                confidence=confidence,
                label=label,
                analysis=full_analysis,
                final=f"{label} (confidence: {confidence})",
                violated_rules=violated_rules,
                applicable_exceptions=applicable_exceptions,
                rule_reasoning=rule_reasoning,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback: JSON was malformed or truncated (e.g., hit max_tokens).
            # Preserve the raw output in `analysis` so partial violated_rules
            # can still be recovered post-hoc from whatever text the model
            # produced before truncation.
            lower = raw.lower()
            if "unsafe" in lower:
                label = "unsafe"
            elif "safe" in lower:
                label = "safe"
            else:
                label = "unsafe"
            fallback_analysis = (
                f"[Thinking]\n{thinking_content}\n\n[Raw]\n{raw}"
                if thinking_content else raw
            )
            return SafetyResult(
                is_safe=(label == "safe"),
                confidence=0.5,
                label=label,
                analysis=fallback_analysis,
                final=raw[:500],
                parse_failed=True,
            )

    def release(self) -> None:
        """No GPU resources to release."""
        self._client = None
        self._claude_client = None
        self._grok_api_key = None


def load_api_guard(
    model_name: str,
    backend: str = "openai",
    **kwargs,
) -> APIGuard:
    """Convenience loader: instantiate and initialize."""
    guard = APIGuard(model_name=model_name, backend=backend, **kwargs)
    guard.load_model()
    return guard
