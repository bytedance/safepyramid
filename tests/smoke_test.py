"""End-to-end smoke test on a synthetic fixture — no GPU, no API key.

Run from the repo root:
    python3 tests/smoke_test.py

Covers: dataset loading, the per-policy runner with a mock guard,
RMR@τ / RMR / RDR math against hand-computed values, refusal exclusion,
per-rule task collection + verdict extraction + case-level aggregation,
and the `safepyramid score` path.
"""

import json
import os
import sys
import tempfile

# Import the package from src/ so the test runs without `pip install` too.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "mini_benchmark.json")

PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        raise SystemExit(f"smoke test failed at: {name} {detail}")
    PASSED += 1


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

from safepyramid.data import load_benchmark

metadata, cases = load_benchmark(dataset=FIXTURE)
check("load_benchmark loads fixture", len(cases) == 5)
meta_l0, cases_l0 = load_benchmark(dataset=FIXTURE, level="L0")
check("level filter", len(cases_l0) == 3 and all(c["level"] == "L0" for c in cases_l0))
_, cases_lim = load_benchmark(dataset=FIXTURE, limit=2)
check("limit", len(cases_lim) == 2)
_, cases_shard = load_benchmark(dataset=FIXTURE, start_idx=3)
check("start_idx sharding", len(cases_shard) == 2 and cases_shard[0]["id"] == "test-004-l2")


# ---------------------------------------------------------------------------
# 2. Scoring math (hand-computed expectations)
# ---------------------------------------------------------------------------

from safepyramid.scoring import (
    aggregate_case_records,
    rmr_match,
    score_structured_output,
)

# RMR@τ budget math: |GT|=5, FN=1, FP=0
check("rmr_match exact fails on FN=1", not rmr_match(0, 1, 5, 0))
check("rmr_match @0.9 fails on FN=1 (k=(10*5)//100=0)", not rmr_match(0, 1, 5, 10))
check("rmr_match @0.8 passes on FN=1 (k=1)", rmr_match(0, 1, 5, 20))
check("rmr_match @0.7 passes on FN=1 (k=1)", rmr_match(0, 1, 5, 30))
# |GT|=10 → @0.9 allows 1 per side
check("rmr_match @0.9 passes FN=1 at |GT|=10", rmr_match(1, 1, 10, 10))

s = score_structured_output(
    violated_rules=[1, 2, 3],
    applicable_exceptions=[],
    predicted_label="unsafe",
    rubric=cases[1]["rubric"],
    expected_label="",
    ground_truth_violations=[1, 2, 5],
    distractor_rules=[3, 4],
    layer="L0",
)
check("score: rule_match False on partial", s["rule_match"] is False)
check("score: FP/FN sets", s["false_positives"] == [3] and s["false_negatives"] == [5])
check("score: label_correct None when expected empty", s["label_correct"] is None)
check("score: per-case RDR = 2/4", abs(s["rule_disagreement_rate"] - 0.5) < 1e-9)
check("score: FM-V1 + FM-D1 flagged",
      "FM-V1" in s["failure_modes"] and "FM-D1" in s["failure_modes"])

# Aggregation: 5 cases —
#   test-001: exact match            (GT {1,2},   pred {1,2})
#   test-002: FN=1                   (GT {1,2,5}, pred {1,2})
#   test-003: exact match            (GT {2,3},   pred {2,3})
#   test-004: refused                → excluded
#   test-005: FP=1                   (GT {4},     pred {4,5})
records = {
    "test-001-l0": {"id": "test-001-l0", "violated_rules": [1, 2]},
    "test-002-l0": {"id": "test-002-l0", "violated_rules": [1, 2]},
    "test-003-l1": {"id": "test-003-l1", "violated_rules": [2, 3]},
    "test-004-l2": {"id": "test-004-l2", "refused": True},
    "test-005-l0": {"id": "test-005-l0", "violated_rules": [4, 5]},
}
m = aggregate_case_records(records, cases)
all_cell = m["ALL"]
check("agg: total/excluded/n_eval", (all_cell["total"], all_cell["excluded"],
                                     all_cell["n_eval"]) == (5, 1, 4))
# RMR@1.0: 2/4 = 50.0
check("agg: RMR@1.0 = 50.0", abs(all_cell["rmr_exact"] - 50.0) < 1e-9,
      f"got {all_cell['rmr_exact']}")
# Per-τ: test-002 (|GT|=3, FN=1): k=0/0/0/0 for ε=0,10,20; ε=30→k=0 → never matches
#        test-005 (|GT|=1, FP=1): k=0 at every ε≤30 → never matches
# So RMR@τ = 50.0 at every τ in the schedule → RMR = 50.0
check("agg: RMR = 50.0", abs(all_cell["rmr_avg"] - 50.0) < 1e-9,
      f"got {all_cell['rmr_avg']}")
# RDR: Σdis = 0+1+0+1 = 2; Σuni = 2+3+2+2 = 9 → 22.22
check("agg: RDR = 22.22", abs(all_cell["rdr"] - 22.22) < 0.01,
      f"got {all_cell['rdr']}")
check("agg: refusal = 20.0", abs(all_cell["refusal"] - 20.0) < 1e-9)
check("agg: per-level buckets exist", all(k in m for k in ("L0", "L1", "L2")))
check("agg: L1 exact", m["L1"]["rmr_exact"] == 100.0)
check("agg: L2 fully refused → n_eval 0", m["L2"]["n_eval"] == 0)


# ---------------------------------------------------------------------------
# 3. Per-policy runner with a mock guard (policy prefix + records + resume)
# ---------------------------------------------------------------------------

from pathlib import Path

from safepyramid.models.base import SafetyResult
from safepyramid import runner

PREDS = {
    "test-001-l0": [1, 2],
    "test-002-l0": [1, 2],
    "test-003-l1": [2, 3],
    "test-005-l0": [4, 5],
}


class MockGuard:
    model_name = "mock-guard"
    seen_policies: list = []

    def evaluate_batch(self, texts, policies=None, batch_size=10, case_ids=None, **kw):
        type(self).seen_policies = list(policies)
        out = []
        for cid in case_ids:
            if cid == "test-004-l2":
                out.append(SafetyResult(is_safe=False, confidence=0.0,
                                        label="refused", final="API call failed",
                                        refused=True))
            else:
                vr = PREDS[cid]
                out.append(SafetyResult(is_safe=False, confidence=1.0,
                                        label="unsafe", analysis="mock analysis",
                                        final="unsafe", violated_rules=list(vr)))
        return out

    def release(self):
        pass


tmpdir = tempfile.mkdtemp(prefix="safepyramid_smoke_")
out_path = Path(tmpdir) / "results_20990101_000000_mock-guard.jsonl"
guard = MockGuard()
records_by_id, scores = runner.evaluate_model(
    guard, cases, model_name="mock-guard", output_path=out_path,
)
check("runner: one record per case", len(records_by_id) == 5)
check("runner: eval-note prefix prepended",
      all(p.startswith("[Evaluation Note]\nDefinition: A rule is \"violated\"")
          for p in MockGuard.seen_policies),
      "policy prefix mismatch")
check("runner: policy body preserved after prefix",
      MockGuard.seen_policies[0].endswith(cases[0]["policy"]))
check("runner: refused record has no violated_rules",
      "violated_rules" not in records_by_id["test-004-l2"])
check("runner: structured_scores attached",
      records_by_id["test-001-l0"]["structured_scores"]["rule_match"] is True)

m2 = aggregate_case_records(records_by_id, cases)
check("runner records reproduce hand-computed metrics",
      m2["ALL"]["rmr_exact"] == 50.0 and abs(m2["ALL"]["rdr"] - 22.22) < 0.01
      and m2["ALL"]["rmr_avg"] == 50.0)

done = set(runner.load_records([str(out_path)]))
check("runner: resume sees all done ids", done == set(records_by_id))
check("runner: resume path locates latest file for the model",
      runner._results_path(tmpdir, "mock-guard", resume=True).name == out_path.name)
check("runner: record index uses original position",
      records_by_id["test-003-l1"]["index"] == 3)

# Resume safety: the glob must not match other models or custom run ids,
# and an explicit RUN_ID pin is never overridden by the fallback.
(Path(tmpdir) / "results_20990101_000000_qwen3_mock-guard.jsonl").touch()
check("runner: resume never matches a different model's file",
      runner._results_path(tmpdir, "guard", resume=True).name
      != "results_20990101_000000_qwen3_mock-guard.jsonl")
os.environ["RUN_ID"] = "shard0"
check("runner: RUN_ID pin wins over existing files on resume",
      runner._results_path(tmpdir, "mock-guard", resume=True).name
      == "results_shard0_mock-guard.jsonl")
del os.environ["RUN_ID"]

# Loader kwargs are filtered against the guard class __init__ — unknown
# knobs are dropped with a note instead of crashing the constructor.
from safepyramid.models import build_loader_kwargs
kw = build_loader_kwargs({"name": "x", "type": "generic", "mode": "cot",
                          "vllm": {"max_model_len": 4096}})
check("loader kwargs: unsupported knob dropped, supported kept",
      "mode" not in kw and kw["max_model_len"] == 4096)
kw = build_loader_kwargs({"name": "gpt-5.2", "type": "api", "backend": "openai",
                          "generation": {"max_new_tokens": 1000}})
check("loader kwargs: api maps max_new_tokens → max_completion_tokens",
      kw.get("max_completion_tokens") == 1000)

# n_eval=0 buckets report None (undefined), not a fake 0.0 score.
from safepyramid.scoring import aggregate_case_records as _agg
empty_m = _agg({}, cases)
check("agg: all-excluded bucket reports None metrics",
      empty_m["ALL"]["rmr_exact"] is None and empty_m["ALL"]["rdr"] is None)
check("agg: rmr_10 compat alias present",
      m["ALL"]["rmr_10"] == m["ALL"]["rmr_exact"])


# ---------------------------------------------------------------------------
# 4. Per-rule: task collection, verdict extraction, case-level aggregation
# ---------------------------------------------------------------------------

from safepyramid import per_rule

tasks = per_rule.collect_tasks(cases)
# Conditionals are never targets: test-003 has rules {1,2,4} (decisive/distr)
# + exception 3 → 4 targets; rule 5 (conditional) excluded.
t3 = [t for t in tasks if t["cid"] == "test-003-l1"]
check("per-rule: conditional skipped as target",
      sorted(t["rule"] for t in t3) == [1, 2, 3, 4])
exc_task = next(t for t in t3 if t["rule"] == 3)
check("per-rule: exception bundles its paired base rule",
      [b["rule"] for b in exc_task["_bundle"]] == [1])
base_task = next(t for t in t3 if t["rule"] == 1)
check("per-rule: base rule bundles its modifiers",
      [b["rule"] for b in base_task["_bundle"]] == [3])
cond_mod_task = next(t for t in t3 if t["rule"] == 2)
check("per-rule: conditional appears as bundle member",
      [b["rule"] for b in cond_mod_task["_bundle"]] == [5])
check("per-rule: GT flags", exc_task["gt"] is True and base_task["gt"] is False)

body = per_rule.build_per_rule_user_text(base_task)
check("per-rule: user text sections",
      body.startswith("[Target — Rule 1 — JUDGE ONLY THIS ONE]")
      and "[Context — paired modifier rules; do NOT judge these]" in body
      and "\n[Conversation]\n" in body)
empty_bundle_task = next(t for t in tasks if t["cid"] == "test-005-l0" and t["rule"] == 4)
check("per-rule: empty [Context] filler always rendered",
      "(no paired modifier rules" in per_rule.build_per_rule_user_text(empty_bundle_task))
l2_task = next(t for t in tasks if t["cid"] == "test-004-l2")
check("per-rule: L2 framework block included",
      "\n[Framework]\n" in per_rule.build_per_rule_user_text(l2_task))

v = per_rule._extract_verdict(
    SafetyResult(is_safe=False, confidence=1.0, label="unsafe", final="violated"))
check("per-rule: unsafe → pred True", v["pred"] is True and not v["refused"])
v = per_rule._extract_verdict(
    SafetyResult(is_safe=True, confidence=1.0, label="safe", final="not_violated"))
check("per-rule: safe → pred False", v["pred"] is False)
v = per_rule._extract_verdict(
    SafetyResult(is_safe=False, confidence=0.0, label="",
                 final='{"verdict": "violated", "reason": "x"}', parse_failed=True))
check("per-rule: verdict regex fallback", v["pred"] is True and v["source"] == "verdict_regex")
v = per_rule._extract_verdict(None)
check("per-rule: missing result → refused", v["refused"] is True and v["pred"] is None)

# Case-level aggregation: per-rule judgments → RMR/RDR.
# Predict exactly GT for every judged rule except: one FN on test-002 rule 5,
# and ALL judgments refused for test-004 (case excluded).
judgments = []
for t in tasks:
    if t["cid"] == "test-004-l2":
        judgments.append({"cid": t["cid"], "rule": t["rule"], "pred": False, "refused": True})
    elif t["cid"] == "test-002-l0" and t["rule"] == 5:
        judgments.append({"cid": t["cid"], "rule": t["rule"], "pred": False, "refused": False})
    else:
        judgments.append({"cid": t["cid"], "rule": t["rule"], "pred": t["gt"], "refused": False})

pm = per_rule.aggregate_judgments_to_case_metrics(judgments, cases)
check("per-rule agg: all-refused case excluded",
      pm["ALL"]["excluded"] == 1 and pm["ALL"]["n_eval"] == 4)
# exact on 001/003/005, FN=1 on 002 → RMR@1.0 = 75
check("per-rule agg: RMR@1.0 = 75.0", pm["ALL"]["rmr_exact"] == 75.0,
      f"got {pm['ALL']['rmr_exact']}")
# RDR = 1 / (2+3+2+1) = 12.5
check("per-rule agg: RDR = 12.5", abs(pm["ALL"]["rdr"] - 12.5) < 1e-9)
check("per-rule agg: per-rule refusal rate > 0",
      pm["ALL"]["per_rule_refusal_rate"] > 0)

summary = per_rule.score_judgments(tasks, [
    {"pred": j["pred"], "refused": j["refused"], "raw_label": None, "source": "label"}
    for j in judgments
])
check("per-rule diagnostic: L0 accuracy reflects 1 miss",
      summary["L0"]["fn"] == 1 and summary["L0"]["accuracy"] < 1.0)


# ---------------------------------------------------------------------------
# 5. `safepyramid score` CLI path
# ---------------------------------------------------------------------------

from safepyramid import cli

print("\n--- score CLI output ---")
cli.main(["score", str(out_path), "--dataset", FIXTURE])

# Retry-merge semantics: a second file overriding the refused case.
retry_path = Path(tmpdir) / "results_retry.jsonl"
with retry_path.open("w") as f:
    f.write(json.dumps({"id": "test-004-l2", "violated_rules": [1, 2],
                        "model_name": "mock-guard"}) + "\n")
merged = runner.load_records([str(out_path), str(retry_path)])
m3 = aggregate_case_records(merged, cases)
check("score: retry file overrides refused case",
      m3["ALL"]["excluded"] == 0 and m3["ALL"]["rmr_exact"] == 60.0)

# Secrets: redact() scrubs registered keys from arbitrary text.
from safepyramid.auth import redact, register_secret
register_secret("sk-test-1234567890abcdef")
check("secrets: redact scrubs key",
      "sk-test" not in redact("Error: invalid key sk-test-1234567890abcdef used"))

print(f"\nAll {PASSED} smoke checks passed.")
