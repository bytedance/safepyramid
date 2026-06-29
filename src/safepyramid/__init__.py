"""SafePyramid: benchmark LLMs and guardrail models on policy-grounded
guardrailing.

Python API quick start:

    from safepyramid import load_benchmark, evaluate

    metadata, cases = load_benchmark(level="L0", limit=10)
    summaries = evaluate(
        {"name": "gpt-5.2", "type": "api", "backend": "openai"},
        cases,
    )

CLI equivalent: ``safepyramid eval --model gpt-5.2 --backend openai``.
"""

__version__ = "0.1.0"

from safepyramid.data import load_benchmark  # noqa: F401


def evaluate(model, cases: list[dict], **kwargs) -> list[dict]:
    """Run the per-policy evaluation for one or more models.

    Args:
        model: One of, or a list mixing:
            * a config dict (``{"name": ..., "type": "api"|"generic", ...}``,
              same schema as entries under ``models:`` in configs/models.yaml), or
            * a ready-to-use ``BaseGuardModel`` instance — your own guard
              (see ``examples/custom_guard.py``).
        cases: Benchmark cases from load_benchmark().
        **kwargs: Forwarded to safepyramid.runner.run
            (guard_batch_size, output_dir, resume, verbose, ...).

    Returns:
        A list of per-model summaries with the headline metrics
        (RMR@1.0 / RMR / RDR / refusal per level).
    """
    from safepyramid import runner
    return runner.run(model, cases, **kwargs)


def evaluate_per_rule(model, cases: list[dict], **kwargs) -> list[dict]:
    """Run the per-rule evaluation for one or more models.

    *model* accepts the same forms as :func:`evaluate`. Returns a list of
    per-model results carrying the binary judgment diagnostic and the
    case-level RMR/RDR aggregation.
    """
    from safepyramid import per_rule
    return per_rule.run(model, cases, **kwargs)
