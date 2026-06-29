"""Shared vLLM inference engine for guard models.

Provides a thin wrapper around vLLM's LLM class that all guard models
can use for accelerated inference. Supports both chat-template-based
prompting (messages → tokenized prompt) and raw string prompting.

Usage:
    engine = VLLMEngine(model_name, ...)
    engine.load()
    outputs = engine.generate(prompts, sampling_params)
    outputs = engine.generate_from_messages(messages_list, sampling_params)
"""

import os
import warnings
from typing import Optional

# Suppress noisy warnings from CUDA bindings, HF tokenizer, and vLLM internals
warnings.filterwarnings("ignore", message=".*trust_remote_code.*Auto classes.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cuda\\.cudart.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cuda\\.nvrtc.*")
warnings.filterwarnings("ignore", message=".*Using a slow tokenizer.*")

# Suppress NCCL verbose logs (default is INFO which floods the output)
os.environ.setdefault("NCCL_DEBUG", "WARN")
# Suppress vLLM internal logging noise
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams


class VLLMEngine:
    """Shared vLLM inference engine."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "./cache",
        dtype: str = "bfloat16",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
        trust_remote_code: bool = True,
        max_num_seqs: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        # Auto-detect: use all available GPUs
        if tensor_parallel_size is None:
            import torch
            tensor_parallel_size = torch.cuda.device_count() or 1
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code

        self.llm: Optional[LLM] = None
        self.tokenizer = None

    def load(self) -> None:
        """Initialize vLLM engine and load model.

        HF_TOKEN is read automatically from the environment by HuggingFace
        libraries, so we don't pass it explicitly (avoids V0/V1 API differences).
        """
        kwargs = {
            "model": self.model_name,
            "download_dir": self.cache_dir,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.max_num_seqs is not None:
            kwargs["max_num_seqs"] = self.max_num_seqs

        print(f"[vLLM] Loading {self.model_name} ...")
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[vLLM] Model loaded successfully")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        skip_special_tokens: bool = True,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        """Generate from raw string prompts. Returns list of output texts."""
        if self.llm is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            skip_special_tokens=skip_special_tokens,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def generate_from_messages(
        self,
        messages_list: list[list[dict]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[dict] = None,
        skip_special_tokens: bool = True,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        """Generate from chat messages. Applies chat template, then generates."""
        if self.llm is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        tk_kwargs = chat_template_kwargs or {}

        prompts = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **tk_kwargs,
            )
            prompts.append(prompt)

        return self.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            skip_special_tokens=skip_special_tokens,
            repetition_penalty=repetition_penalty,
        )

    def release(self) -> None:
        """Release GPU memory."""
        import gc
        import torch
        import torch.distributed as dist

        if self.llm is not None:
            del self.llm
            self.llm = None
        self.tokenizer = None

        # Destroy NCCL process group to avoid shutdown warning
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
