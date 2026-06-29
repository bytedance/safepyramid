"""SafePyramid command-line interface.

    safepyramid eval --model gpt-5.2 --backend openai
    safepyramid eval --model your-org/your-guardrail --type generic
    safepyramid eval --config configs/models.yaml --models 1,2
    safepyramid per-rule --model gpt-5.2 --backend openai --level L0
    safepyramid score output/results/results_*.jsonl
    safepyramid list-models --config configs/models.yaml
"""

import argparse
import sys

import yaml


# ---------------------------------------------------------------------------
# Config / model selection
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> list[dict]:
    """Load a YAML config and return the list of model configs.

    Supports both the ``models:`` list format and a single ``model:``
    entry (auto-wrapped into a one-element list).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "models" in cfg:
        return cfg["models"]
    if "model" in cfg:
        return [cfg["model"]]
    raise ValueError(f"Config file {config_path} must contain 'models' or 'model' key")


def _filter_models(model_cfgs: list[dict], selection: str | None) -> list[dict]:
    """Filter model configs by user selection.

    *selection* is a comma-separated string of 1-based indices or substrings
    of model names. Examples: "1", "1,2", "gpt-5", "claude".
    Returns the full list when *selection* is None.
    """
    if selection is None:
        return model_cfgs

    chosen = []
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        # Try as 1-based index first
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(model_cfgs):
                if model_cfgs[idx] not in chosen:
                    chosen.append(model_cfgs[idx])
            else:
                print(f"Warning: model index {token} out of range (1-{len(model_cfgs)}), skipped")
            continue
        # Otherwise match as substring of model name
        matched = [c for c in model_cfgs if token in c.get("name", "")]
        if matched:
            for m in matched:
                if m not in chosen:
                    chosen.append(m)
        else:
            print(f"Warning: no model matching '{token}', skipped")

    if not chosen:
        raise ValueError(f"No models matched selection '{selection}'. Available: "
                         + ", ".join(f"[{i+1}] {c.get('name')}" for i, c in enumerate(model_cfgs)))
    return chosen


def _model_cfgs_from_args(args) -> list[dict]:
    """Build the model config list from --config/--models or --model flags."""
    if args.config:
        return _filter_models(_load_config(args.config), args.models)

    if not args.model:
        raise SystemExit(
            "Specify a model: either --model <name> (with --backend/--type) "
            "or --config <yaml> [--models <selection>]."
        )

    cfg: dict = {"name": args.model}
    if args.type:
        cfg["type"] = args.type
    if args.backend:
        cfg.setdefault("type", "api")
        cfg["backend"] = args.backend
    if args.reasoning_effort:
        cfg["reasoning_effort"] = args.reasoning_effort
    if args.api_key_env:
        cfg["api_key_env"] = args.api_key_env
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.enable_thinking is not None:
        cfg["enable_thinking"] = args.enable_thinking

    vllm: dict = {}
    if args.max_model_len is not None:
        vllm["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        vllm["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        vllm["max_num_seqs"] = args.max_num_seqs
    if vllm:
        cfg["vllm"] = vllm

    gen: dict = {}
    if args.max_new_tokens is not None:
        gen["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        gen["temperature"] = args.temperature
    if gen:
        cfg["generation"] = gen

    return [cfg]


def _apply_vllm_overrides(model_cfgs: list[dict], args) -> list[dict]:
    """Apply CLI tuning flags as overrides to all configs (config-file mode)."""
    for cfg in model_cfgs:
        vllm_cfg = cfg.setdefault("vllm", {})
        if getattr(args, "max_model_len", None) is not None:
            vllm_cfg["max_model_len"] = args.max_model_len
        if getattr(args, "gpu_memory_utilization", None) is not None:
            vllm_cfg["gpu_memory_utilization"] = args.gpu_memory_utilization
        if getattr(args, "max_num_seqs", None) is not None:
            vllm_cfg["max_num_seqs"] = args.max_num_seqs
        gen_cfg = cfg.setdefault("generation", {})
        if getattr(args, "max_new_tokens", None) is not None:
            gen_cfg["max_new_tokens"] = args.max_new_tokens
        if getattr(args, "temperature", None) is not None:
            gen_cfg["temperature"] = args.temperature
    return model_cfgs


# ---------------------------------------------------------------------------
# Shared flags
# ---------------------------------------------------------------------------

def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model selection")
    g.add_argument("--model", help="Model name: API model id, HF repo id, or local path")
    g.add_argument("--type", choices=["api", "generic"],
                   help="Guard type: `api` (API model) or `generic` (your own "
                        "local chat guardrail). Auto-detected as `api` for known "
                        "API model names. For a fully custom guard, use the "
                        "Python API (see examples/custom_guard.py).")
    g.add_argument("--backend",
                   choices=["openai", "anthropic", "claude", "gemini", "xai",
                            "openrouter", "openai_compatible"],
                   help="API backend (implies --type api)")
    g.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"],
                   help="Reasoning effort for API models")
    g.add_argument("--api-key-env", metavar="VAR",
                   help="Name of the environment variable holding the API key "
                        "(defaults to the backend's standard variable, e.g. "
                        "OPENAI_API_KEY). Keys are never accepted as CLI values.")
    g.add_argument("--base-url", help="Override API base URL "
                                      "(required for openai_compatible)")
    g.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                   default=None, help="Chat-template thinking flag for local guards")
    g.add_argument("--max-model-len", type=int, help="vLLM context length")
    g.add_argument("--gpu-memory-utilization", type=float, help="vLLM GPU mem fraction")
    g.add_argument("--max-num-seqs", type=int, help="vLLM max concurrent sequences")
    g.add_argument("--max-new-tokens", type=int, help="Generation token budget")
    g.add_argument("--temperature", type=float, help="Generation temperature")
    g.add_argument("--config", help="YAML config with a models: list "
                                    "(see configs/models.yaml)")
    g.add_argument("--models", help="Selection within --config: comma-separated "
                                    "1-based indices or name substrings")


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", default=None,
                   help="Local path to benchmark.json or a Hugging Face dataset "
                        "repo id (default: the official SafePyramid dataset)")
    g.add_argument("--level", choices=["L0", "L1", "L2"],
                   help="Evaluate only this level (default: all)")
    g.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    g.add_argument("--start-idx", type=int, default=0,
                   help="Skip the first N cases (data-parallel sharding)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_eval(args) -> None:
    from safepyramid.data import load_benchmark
    from safepyramid import runner

    model_cfgs = _model_cfgs_from_args(args)
    if args.config:
        model_cfgs = _apply_vllm_overrides(model_cfgs, args)

    metadata, cases = load_benchmark(
        dataset=args.dataset, level=args.level,
        limit=args.limit, start_idx=args.start_idx,
    )
    print("=" * 60)
    print("SafePyramid: Per-Policy Evaluation")
    print("=" * 60)
    print(f"  Cases:  {len(cases)}" + (f" (level={args.level})" if args.level else ""))
    print(f"  Models: {len(model_cfgs)}")
    for i, cfg in enumerate(model_cfgs):
        print(f"    [{i+1}] {cfg.get('name', '(unknown)')}")

    runner.run(
        model_cfgs, cases,
        guard_batch_size=args.batch_size,
        output_dir=args.output_dir,
        resume=args.resume,
        verbose=args.verbose,
    )


def _cmd_per_rule(args) -> None:
    from safepyramid.data import load_benchmark
    from safepyramid import per_rule

    model_cfgs = _model_cfgs_from_args(args)
    if args.config:
        model_cfgs = _apply_vllm_overrides(model_cfgs, args)

    metadata, cases = load_benchmark(
        dataset=args.dataset, level=args.level,
        limit=args.limit, start_idx=args.start_idx,
    )
    print("=" * 60)
    print("SafePyramid: Per-Rule Evaluation")
    print("=" * 60)

    per_rule.run(
        model_cfgs, cases,
        guard_batch_size=args.batch_size,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


def _cmd_score(args) -> None:
    from safepyramid.data import load_benchmark
    from safepyramid.scoring import aggregate_case_records, format_metrics_table

    metadata, cases = load_benchmark(dataset=args.dataset, level=args.level)

    if args.per_rule:
        import json
        from safepyramid.per_rule import aggregate_judgments_to_case_metrics
        judgments = []
        for p in args.results:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") == "judgment":
                        judgments.append(r)
        metrics = aggregate_judgments_to_case_metrics(judgments, cases)
        print(format_metrics_table(
            metrics, title="Per-rule case-level metrics", show_tau_breakdown=True))
        prr = metrics.get("ALL", {}).get("per_rule_refusal_rate", 0.0)
        print(f"  Per-rule judgment refusal rate: {prr:.1f}%")
    else:
        from safepyramid.runner import load_records
        records = load_records(args.results)
        metrics = aggregate_case_records(records, cases, cap_aware=args.cap_aware)
        print(format_metrics_table(
            metrics, title="Per-policy metrics", show_tau_breakdown=True))


def _cmd_list_models(args) -> None:
    model_cfgs = _load_config(args.config)
    for i, cfg in enumerate(model_cfgs):
        kind = cfg.get("type") or cfg.get("backend") or "auto"
        print(f"  [{i+1}] {cfg.get('name', '(unknown)')}  ({kind})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="safepyramid",
        description="SafePyramid: benchmark LLMs and guardrail models on "
                    "policy-grounded guardrailing (RMR / RMR@1.0 / RDR).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Per-policy (all-rule) evaluation")
    _add_model_args(p_eval)
    _add_dataset_args(p_eval)
    p_eval.add_argument("--batch-size", type=int, default=10,
                        help="Concurrent API requests / client-side batch for "
                             "local guards (default: 10)")
    p_eval.add_argument("--output-dir", default="output/results")
    p_eval.add_argument("--resume", action="store_true",
                        help="Continue the most recent results file for this "
                             "model (skips cases already recorded). Set the "
                             "RUN_ID env var to pin an exact file / shard.")
    p_eval.add_argument("--verbose", "-v", action="store_true")
    p_eval.set_defaults(func=_cmd_eval)

    p_pr = sub.add_parser("per-rule", help="Per-rule evaluation (one verdict "
                                           "per (case, rule); aggregated back "
                                           "to case-level RMR/RDR)")
    _add_model_args(p_pr)
    _add_dataset_args(p_pr)
    p_pr.add_argument("--batch-size", type=int, default=50,
                      help="Concurrent API requests / client-side batch "
                           "(default: 50)")
    p_pr.add_argument("--output-dir", default="output/results")
    p_pr.add_argument("--verbose", "-v", action="store_true")
    p_pr.set_defaults(func=_cmd_per_rule)

    p_score = sub.add_parser("score", help="Recompute metrics from result files")
    p_score.add_argument("results", nargs="+",
                         help="Result JSONL file(s); later files override "
                              "earlier on case-id collision (retry semantics)")
    p_score.add_argument("--dataset", default=None)
    p_score.add_argument("--level", choices=["L0", "L1", "L2"])
    p_score.add_argument("--per-rule", action="store_true",
                         help="Treat inputs as per-rule judgment files")
    p_score.add_argument("--cap-aware", action="store_true",
                         help="Exclude completions that hit the provider's "
                              "output cap (truncated reasoning)")
    p_score.set_defaults(func=_cmd_score)

    p_list = sub.add_parser("list-models", help="List models in a config file")
    p_list.add_argument("--config", required=True)
    p_list.set_defaults(func=_cmd_list_models)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nTerminated by user")
    except Exception as e:
        from safepyramid.auth import redact
        import traceback
        print(f"Error: {redact(str(e))}")
        print(redact(traceback.format_exc()))
        sys.exit(1)


if __name__ == "__main__":
    main()
