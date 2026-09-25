"""
LightGBM matcher (Step 6).

Decided: LightGBM. Validation split is by Source-1 entity (GroupKFold on
s1_id) — never split by pair, or related pairs leak across train/valid.
"""

from pathlib import Path
from typing import Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config
import features as feat


def split_entity_ids(ground_truth: pd.DataFrame, n_splits: int = None) -> Tuple[Set[str], Set[str]]:
    """Splits ALL Source-1 entity IDs — not just ones that survived blocking —
    into train/valid. This matters: an entity with zero candidates still has
    to be evaluated (as an empty prediction), or macro-F0.5 silently ignores
    it and looks better than the real submission would score."""
    n_splits = n_splits or config.N_VALIDATION_FOLDS
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    ids = ground_truth[config.COL_S1_ID].to_numpy()
    train_idx, valid_idx = next(kf.split(ids))
    return set(ids[train_idx]), set(ids[valid_idx])


def filter_by_entity_ids(featured: pd.DataFrame, entity_ids: Set[str]) -> pd.DataFrame:
    return featured[featured["s1_id"].isin(entity_ids)].reset_index(drop=True)


def train_lightgbm(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> lgb.Booster:
    X_train, y_train = train_df[feat.FEATURE_COLUMNS], train_df["label"]
    X_valid, y_valid = valid_df[feat.FEATURE_COLUMNS], valid_df["label"]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": config.RANDOM_SEED,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
    )
    return model


def predict_proba(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[feat.FEATURE_COLUMNS], num_iteration=model.best_iteration)


def save_model(model: lgb.Booster, path: Path = None) -> None:
    path = path or config.MODEL_FILE
    model.save_model(str(path))
    print(f"Saved model to {path}")


def load_model(path: Path = None) -> lgb.Booster:
    path = path or config.MODEL_FILE
    return lgb.Booster(model_file=str(path))


if __name__ == "__main__":
    import io_utils
    import normalize
    import blocking
    import labels

    s1, s2, s3 = io_utils.load_train_sources()
    gt = io_utils.load_ground_truth()

    s1n = normalize.normalize_dataframe(s1)
    s2n = normalize.normalize_dataframe(s2)
    s3n = normalize.normalize_dataframe(s3)

    candidates = blocking.generate_all_candidates(s1n, s2n, s3n)
    labeled, missed = labels.build_pairwise_labels(candidates, gt)
    labels.summarize_labels(labeled, missed)

    featured = feat.extract_features(labeled, s1n, s2n, s3n)
    train_ids, valid_ids = split_entity_ids(gt)
    train_df = filter_by_entity_ids(featured, train_ids)
    valid_df = filter_by_entity_ids(featured, valid_ids)
    print(f"Train: {len(train_ids)} entities / {len(train_df)} pairs, "
          f"Valid: {len(valid_ids)} entities / {len(valid_df)} pairs")

    model = train_lightgbm(train_df, valid_df)
    save_model(model)

    valid_df = valid_df.copy()
    valid_df["match_probability"] = predict_proba(model, valid_df)
    print("\nSample scored validation pairs:")
    print(valid_df[["s1_id", "candidate_id", "label", "match_probability"]].sample(min(10, len(valid_df))))