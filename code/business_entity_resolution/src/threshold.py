"""
Threshold selection (Step 7).

Takes the sweep evaluate.py can already compute and turns it into a single,
persisted threshold that predict.py actually uses — instead of the
hardcoded config.DEFAULT_THRESHOLD placeholder.

Decided: single global threshold (not per-source) unless validation shows a
real, consistent gap between S2 and S3 score distributions.
"""

import json
from pathlib import Path
from typing import Set

import pandas as pd

import config
import evaluate


def sweep_thresholds(scored: pd.DataFrame, ground_truth: pd.DataFrame, entity_ids: Set[str],
                      candidates=None) -> pd.DataFrame:
    candidates = candidates or config.THRESHOLD_SWEEP
    rows = [evaluate.evaluate_at_threshold(scored, ground_truth, entity_ids, t) for t in candidates]
    return pd.DataFrame(rows).sort_values("macro_f0.5", ascending=False).reset_index(drop=True)


def refine_best_threshold(scored: pd.DataFrame, ground_truth: pd.DataFrame, entity_ids: Set[str]) -> dict:
    """Coarse sweep (config.THRESHOLD_SWEEP, 0.05 steps) then a finer sweep
    around the coarse winner (0.01 steps), so the persisted threshold isn't
    stuck on a coarse grid."""
    coarse = sweep_thresholds(scored, ground_truth, entity_ids)
    best_coarse = coarse.iloc[0]["threshold"]

    fine_candidates = [round(best_coarse - 0.04 + 0.01 * i, 2) for i in range(9)]
    fine_candidates = [t for t in fine_candidates if 0.0 < t < 1.0]
    fine = sweep_thresholds(scored, ground_truth, entity_ids, candidates=fine_candidates)

    best = pd.concat([coarse, fine]).sort_values("macro_f0.5", ascending=False).iloc[0]
    return {"threshold": float(best["threshold"]), "macro_f0.5": float(best["macro_f0.5"])}


def save_threshold(threshold: float, path: Path = None) -> None:
    path = path or config.THRESHOLD_FILE
    path.write_text(json.dumps({"threshold": threshold}, indent=2))
    print(f"Saved tuned threshold ({threshold}) to {path}")


def load_threshold(path: Path = None) -> float:
    path = path or config.THRESHOLD_FILE
    if not path.exists():
        print(f"No tuned threshold found at {path} — falling back to "
              f"config.DEFAULT_THRESHOLD ({config.DEFAULT_THRESHOLD}). "
              f"Run threshold.py to tune and save one.")
        return config.DEFAULT_THRESHOLD
    return json.loads(path.read_text())["threshold"]


if __name__ == "__main__":
    import io_utils
    import normalize
    import blocking
    import labels
    import features as feat
    import model

    s1, s2, s3 = io_utils.load_train_sources()
    gt = io_utils.load_ground_truth()

    s1n = normalize.normalize_dataframe(s1)
    s2n = normalize.normalize_dataframe(s2)
    s3n = normalize.normalize_dataframe(s3)

    candidates = blocking.generate_all_candidates(s1n, s2n, s3n)
    labeled, missed = labels.build_pairwise_labels(candidates, gt)
    featured = feat.extract_features(labeled, s1n, s2n, s3n)

    train_ids, valid_ids = model.split_entity_ids(gt)
    train_df = model.filter_by_entity_ids(featured, train_ids)
    valid_df = model.filter_by_entity_ids(featured, valid_ids)

    trained_model = model.train_lightgbm(train_df, valid_df)
    valid_df = valid_df.copy()
    valid_df["match_probability"] = model.predict_proba(trained_model, valid_df)

    print("\n--- Coarse + fine threshold sweep ---")
    best = refine_best_threshold(valid_df, gt, valid_ids)
    print(f"Best threshold: {best['threshold']:.3f}  macro_F0.5: {best['macro_f0.5']:.4f}")

    save_threshold(best["threshold"])