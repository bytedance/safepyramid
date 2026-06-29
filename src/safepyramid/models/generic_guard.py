"""Generic local guardrail loader (vLLM, chat-template based).

Use this for your own fine-tuned guardrail when it follows standard
chat-model conventions (a chat template with system/user roles) and can
emit JSON when asked. The guard is driven through the SAME shared
prompt wrappers used for API models:

  Per-policy: policy → system message (STRUCTURED_POLICY_DEVELOPER_PROMPT,
  task + JSON schema), conversation → user message; output parsed by
  parse_structured_output (handles <think> blocks, fences, refusals).

  Per-rule: VERDICT_DEVELOPER_PROMPT(_EXCEPTION) → system, structured
  per-rule body → user; output parsed by _parse_verdict_output.

This means any chat model evaluated through GenericGuard is directly
comparable to the API-model rows of the benchmark.

Config example:
    - name: "your-org/your-guardrail"
      type: generic
      enable_thinking: true     # only for chat templates that accept it
      vllm: {max_model_len: 16384}
      generation: {max_new_tokens: 4096, temperature: 0.0, top_p: 1.0}
"""

from typing import Optional

from safepyramid.models.base import (
    BaseGuardModel, SafetyResult, parse_structured_output,
    _parse_verdict_output,
    STRUCTURED_POLICY_DEVELOPER_PROMPT, STRUCTURED_POLICY_USER_PROMPT,
    STRUCTURED_SAFETY_PROMPT_NO_POLICY,
    VERDICT_DEVELOPER_PROMPT, VERDICT_DEVELOPER_PROMPT_EXCEPTION,
)


class GenericGuard(BaseGuardModel):
    """Generic chat-model guardrail served via vLLM."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: str = "./cache",
        enable_thinking: Optional[bool] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # None → don't pass the flag to the chat template at all (templates
        # that don't define it would otherwise be handed a stray variable).
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = None
        self._engine = None

    def _chat_template_kwargs(self, kwargs: dict) -> Optional[dict]:
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        if enable_thinking is None:
            return None
        return {"enable_thinking": enable_thinking}

    def load_model(self) -> None:
        """Load model and tokenizer."""
        print(f"Loading generic guardrail: {self.model_name}")
        print(f"Device: {self.device}  Thinking: {self.enable_thinking}")

        from safepyramid.models.vllm_engine import VLLMEngine
        self._engine = VLLMEngine(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
        )
        self._engine.load()
        self.tokenizer = self._engine.tokenizer
        print("Model loaded successfully")

    def evaluate(
        self,
        text: str,
        policy: Optional[str] = None,
        **kwargs,
    ) -> SafetyResult:
        """Evaluate whether *text* is safe."""
        return self.evaluate_batch([text], [policy], **kwargs)[0]

    def evaluate_batch(
        self,
        texts: list[str],
        policies: list[Optional[str]] | None = None,
        **kwargs,
    ) -> list[SafetyResult]:
        """Evaluate multiple texts via the shared STRUCTURED_POLICY wrapper.

        Same policy→system, conversation→user split that API guards use.
        """
        if policies is None:
            policies = [None] * len(texts)

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        messages_list: list[list[dict]] = []
        for text, policy in zip(texts, policies):
            if policy:
                system_content = STRUCTURED_POLICY_DEVELOPER_PROMPT.format(policy=policy)
                user_content = STRUCTURED_POLICY_USER_PROMPT.format(text=text)
                messages_list.append([
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ])
            else:
                messages_list.append([
                    {"role": "user",
                     "content": STRUCTURED_SAFETY_PROMPT_NO_POLICY.format(text=text)},
                ])

        if self._engine is None:
            raise RuntimeError("Engine not loaded. Call load_model() first.")
        raw_texts = self._engine.generate_from_messages(
            messages_list,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            add_generation_prompt=True,
            chat_template_kwargs=self._chat_template_kwargs(kwargs),
        )

        return [parse_structured_output(raw) for raw in raw_texts]

    def evaluate_per_rule_batch(
        self,
        tasks: list[dict],
        **kwargs,
    ) -> list[SafetyResult]:
        if not tasks:
            return []
        if self._engine is None:
            raise RuntimeError("Engine not loaded. Call load_model() first.")

        from safepyramid.per_rule import build_per_rule_user_text

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)

        messages_list: list[list[dict]] = []
        for task in tasks:
            rtype = task.get("rule_type", "decisive")
            system_content = (
                VERDICT_DEVELOPER_PROMPT_EXCEPTION
                if rtype == "exception"
                else VERDICT_DEVELOPER_PROMPT
            )
            user_content = build_per_rule_user_text(task)
            messages_list.append([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ])

        raw_texts = self._engine.generate_from_messages(
            messages_list,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            add_generation_prompt=True,
            chat_template_kwargs=self._chat_template_kwargs(kwargs),
        )

        return [_parse_verdict_output(raw) for raw in raw_texts]


def load_generic_guard(
    model_name: str,
    device: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs,
) -> GenericGuard:
    """Convenience function: instantiate and load the model in one call."""
    guard = GenericGuard(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        **kwargs,
    )
    guard.load_model()
    return guard
