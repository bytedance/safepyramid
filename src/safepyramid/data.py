"""Benchmark dataset loading.

The benchmark ships as a single JSON artifact:
    {"metadata": {...}, "data": [<case>, ...]}

Each case (see README for full schema):
    id, domain, level ("L0"|"L1"|"L2"), conversation,
    framework ("" for L0/L1), policy, ground_truth_violations (list[int]),
    rubric {violations, non_violations}, failure_mode_metadata

`load_benchmark` accepts either a local file path or a Hugging Face
dataset repo id; the default is the official SafePyramid dataset repo.
"""

import json
import os
from typing import Optional

from safepyramid.constants import DATASET_FILENAME, DATASET_REPO_ID


def _download_from_hub(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download

    from safepyramid.auth import register_secret
    token = os.getenv("HF_TOKEN")
    if token:
        register_secret(token)  # so redact() can scrub it from error text
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=DATASET_FILENAME,
            repo_type="dataset",
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not download '{DATASET_FILENAME}' from Hugging Face "
            f"dataset repo '{repo_id}': {e}\n"
            f"If the dataset is not public yet, pass a local file via "
            f"--dataset /path/to/benchmark.json."
        ) from e


def load_benchmark(
    dataset: Optional[str] = None,
    level: Optional[str] = None,
    limit: Optional[int] = None,
    start_idx: int = 0,
) -> tuple[dict, list[dict]]:
    """Load benchmark cases.

    Args:
        dataset: Local path to benchmark.json OR a Hugging Face dataset
            repo id. Defaults to the official dataset repo.
        level: Optional level filter ("L0" / "L1" / "L2").
        limit: Keep only the first N cases (after level filter and
            start_idx) — useful for smoke tests.
        start_idx: Skip the first N cases (for data-parallel sharding;
            each shard writes its own results file).

    Returns:
        (metadata, cases)
    """
    dataset = dataset or DATASET_REPO_ID
    if os.path.exists(dataset):
        path = dataset
    else:
        path = _download_from_hub(dataset)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # {"metadata": ..., "data": [...]} wrapper, or a bare list
    if isinstance(raw, dict) and "data" in raw:
        metadata = raw.get("metadata", {})
        cases = raw["data"]
    else:
        metadata = {}
        cases = raw

    # The whole pipeline (records, resume, scoring) is keyed by case id.
    # Custom datasets without ids get stable position-based ids assigned
    # BEFORE any filtering, so level/limit/shard runs stay consistent.
    for i, c in enumerate(cases):
        if not c.get("id"):
            c["id"] = f"case-{i}"

    if level:
        cases = [c for c in cases if c.get("level") == level]
    if start_idx > 0:
        cases = cases[start_idx:]
    if limit is not None and limit < len(cases):
        cases = cases[:limit]

    return metadata, cases
