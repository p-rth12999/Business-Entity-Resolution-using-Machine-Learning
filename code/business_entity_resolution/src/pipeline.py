"""
Shared cached loader for the expensive normalize + blocking step, and for
the expensive feature-extraction step.

Every entry point (model.py, evaluate.py, threshold.py, main.py) calls
load_and_prepare() or load_and_prepare_features() instead of duplicating
these calls directly. Together these are the ~15-30 minute cost of this
pipeline — there's no reason to pay it more than once per code change.

Caching is best-effort: a failure to WRITE a cache file (e.g. disk full)
is never allowed to throw away already-completed work. It's logged as a
warning and the in-memory result is returned/used normally either way.
"""

import gc
import pickle
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

import config
import io_utils
import normalize
import blocking


def _safe_cache_write(obj, cache_path: Path) -> None:
    """Write a cache file without ever crashing the caller. Caching is an
    optimization, not a correctness requirement — a disk-full or permission
    error just means the next run pays the recompute cost again, nothing more."""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Cached data to {cache_path}")
    except OSError as e:
        print(f"WARNING: could not write cache to {cache_path} ({e}). "
              f"Continuing without caching — free up disk space to enable caching next time.")
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass


def load_and_prepare(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (s1_normalized, s2_normalized, s3_normalized, candidates, ground_truth)."""
    cache_path = config.NORMALIZED_CANDIDATES_CACHE

    if use_cache and cache_path.exists():
        print(f"Loading cached normalized data + candidates from {cache_path} ...")
        with open(cache_path, "rb") as f:
            s1n, s2n, s3n, candidates, gt = pickle.load(f)
        print(f"Cache loaded: {len(candidates)} candidate pairs, {len(gt)} ground-truth entities")
        return s1n, s2n, s3n, candidates, gt

    s1 = io_utils.load_source(config.TRAIN_SOURCE1_FILE, "train_source1")
    gt = io_utils.load_ground_truth()

    start = time.time()
    s1n = normalize.normalize_dataframe(s1)
    del s1
    gc.collect()

    s2 = io_utils.load_source(config.TRAIN_SOURCE2_FILE, "train_source2")
    s2n = normalize.normalize_dataframe(s2)
    del s2
    gc.collect()

    s3 = io_utils.load_source(config.TRAIN_SOURCE3_FILE, "train_source3")
    s3n = normalize.normalize_dataframe(s3)
    del s3
    gc.collect()

    print(f"Normalized all sources in {time.time() - start:.1f}s")

    candidates = blocking.generate_all_candidates(s1n, s2n, s3n)
    blocking.measure_blocking_recall(candidates, gt)

    _safe_cache_write((s1n, s2n, s3n, candidates, gt), cache_path)

    return s1n, s2n, s3n, candidates, gt


def load_and_prepare_features(use_cache: bool = True):
    """Returns (featured, ground_truth). Caches the labeled+featured
    dataframe SEPARATELY from the normalize+block cache — this is the
    ~15 minute step (feature extraction), and every script (model.py,
    threshold.py, evaluate.py) was independently redoing it from scratch."""
    cache_path = config.FEATURED_CACHE

    if use_cache and cache_path.exists():
        print(f"Loading cached featured data from {cache_path} ...")
        with open(cache_path, "rb") as f:
            featured, gt = pickle.load(f)
        print(f"Cache loaded: {len(featured)} feature rows, {len(gt)} ground-truth entities")
        return featured, gt

    import labels
    import features as feat

    s1n, s2n, s3n, candidates, gt = load_and_prepare(use_cache=use_cache)
    labeled, missed = labels.build_pairwise_labels(candidates, gt)
    labels.summarize_labels(labeled, missed)
    featured = feat.extract_features(labeled, s1n, s2n, s3n)
    del s1n, s2n, s3n, candidates, labeled
    gc.collect()

    _safe_cache_write((featured, gt), cache_path)

    return featured, gt