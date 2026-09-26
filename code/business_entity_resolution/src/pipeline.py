"""
Shared cached loader for the expensive normalize + blocking step.

Every entry point (model.py, evaluate.py, threshold.py, main.py) calls
load_and_prepare() instead of duplicating normalize/blocking calls directly.
That step took 40+ minutes on the full dataset before this cache — there's
no reason to pay that cost more than once while iterating on model or
threshold code.

Delete the cache file manually (or pass use_cache=False) after changing
normalize.py or blocking.py logic — stale cached candidates would silently
carry old logic forward otherwise.
"""

import pickle
import time
from typing import Tuple

import pandas as pd

import config
import io_utils
import normalize
import blocking


def load_and_prepare(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (s1_normalized, s2_normalized, s3_normalized, candidates, ground_truth)."""
    cache_path = config.NORMALIZED_CANDIDATES_CACHE

    if use_cache and cache_path.exists():
        print(f"Loading cached normalized data + candidates from {cache_path} ...")
        with open(cache_path, "rb") as f:
            s1n, s2n, s3n, candidates, gt = pickle.load(f)
        print(f"Cache loaded: {len(candidates)} candidate pairs, {len(gt)} ground-truth entities")
        return s1n, s2n, s3n, candidates, gt

    s1, s2, s3 = io_utils.load_train_sources()
    gt = io_utils.load_ground_truth()

    start = time.time()
    s1n = normalize.normalize_dataframe(s1)
    s2n = normalize.normalize_dataframe(s2)
    s3n = normalize.normalize_dataframe(s3)
    print(f"Normalized all sources in {time.time() - start:.1f}s")

    candidates = blocking.generate_all_candidates(s1n, s2n, s3n)
    blocking.measure_blocking_recall(candidates, gt)

    with open(cache_path, "wb") as f:
        pickle.dump((s1n, s2n, s3n, candidates, gt), f)
    print(f"Cached normalized data + candidates to {cache_path}")

    return s1n, s2n, s3n, candidates, gt