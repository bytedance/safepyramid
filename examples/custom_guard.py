"""Plug your own guard model into SafePyramid.

SafePyramid ships two ready-to-use guards — `api` (any API model) and
`generic` (any local chat guardrail via vLLM). If neither fits your model,
implement the small `BaseGuardModel` contract and pass an instance straight
to `evaluate()` / `evaluate_per_rule()`.

The contract:

  * `load_model(self)`            — set up clients / weights (may be a no-op).
  * `evaluate(self, text, policy) -> SafetyResult`
                                   — judge ONE conversation against the policy
                                     (used by the per-policy protocol).
  * `evaluate_batch(...)`         — provided by BaseGuardModel (sequential over
                                     `evaluate`); override for real batching.
  * `evaluate_per_rule_batch(...)` — OPTIONAL; implement to support the
                                     per-rule protocol (one binary verdict per
                                     (case, rule) task).

Your job is just: build a prompt, call your model, return a `SafetyResult`
whose `violated_rules` is the set of rule numbers you predict. The shared
`STRUCTURED_POLICY_PROMPT` + `parse_structured_output` helpers make the
common "ask for JSON, parse it back" path one line each — but you are free
to build the prompt and the `SafetyResult` however you like.

Run:  python examples/custom_guard.py
"""

from safepyramid import load_benchmark, evaluate
from safepyramid.models import BaseGuardModel, SafetyResult, parse_structured_output
from safepyramid.models.base import STRUCTURED_POLICY_PROMPT


def my_model_call(prompt: str) -> str:
    """Replace this with your model's inference.

    It must return text containing a JSON object of the form
        {"violated_rules": [{"rule": <n>, "explanation": "..."}, ...]}
    (an empty list means "no violation"). Here we just fake one output.
    """
    return '{"analysis": "demo", "violated_rules": [{"rule": 1, "explanation": "demo"}]}'


class MyGuard(BaseGuardModel):
    model_name = "my-custom-guard"

    def load_model(self) -> None:
        # Set up your client / load weights here. No-op for this demo.
        pass

    def evaluate(self, text: str, policy: str | None = None, **kwargs) -> SafetyResult:
        # 1. Build the prompt. The shared template carries the JSON output
        #    contract the benchmark expects; `policy` already contains the
        #    full in-context policy (task instruction + framework + rules).
        prompt = STRUCTURED_POLICY_PROMPT.format(policy=policy or "", text=text)

        # 2. Call your model.
        raw = my_model_call(prompt)

        # 3. Parse the output into a SafetyResult (or construct one yourself —
        #    set `violated_rules` to your predicted rule numbers, or
        #    `refused=True` / `parse_failed=True` to exclude the case).
        return parse_structured_output(raw)


if __name__ == "__main__":
    metadata, cases = load_benchmark(level="L0", limit=10)

    guard = MyGuard()
    guard.load_model()

    summaries = evaluate(guard, cases)            # pass the instance directly
    m = summaries[0]["metrics"]["ALL"]
    print(f"\nRMR@1.0 {m['rmr_exact']}%   RMR {m['rmr_avg']}%   "
          f"RDR {m['rdr']}%   excluded {m['refusal']}%")
