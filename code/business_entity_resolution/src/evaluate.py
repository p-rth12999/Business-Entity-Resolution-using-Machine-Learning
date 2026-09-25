"""
Evaluation (Step 8).

Computes macro F0.5 per Source-1 entity, then averages across entities —
including entities with zero candidates, which must count as a (correct or
incorrect) empty prediction rather than being silently skipped.
"""

from typing import Set

import pandas as pd

import config


def _f_beta(true_set: set, pred_set: set, beta: float = 0.5) -> float:
    if not true_set and not pred_set:
        return 1.0  # singleton correctly identified
    if not true_set and pred_set:
        return 0.0  # false merge on a true singleton
    if true_set and not pred_set:
        return 0.0  # missed every true match

    intersection = true_set & pred_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(true_set)
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def build_predictions(scored: pd.DataFrame, threshold: float, entity_ids: Set[str]) -> dict:
    """entity_ids: the FULL set of Source-1 entities being evaluated. An
    entity with no candidates above threshold (or no candidates at all)
    still gets an entry, with an empty predicted set."""
    predictions = {eid: set() for eid in entity_ids}
    above = scored[scored["match_probability"] >= threshold]
    for s1_id, group in above.groupby("s1_id"):
        if s1_id in predictions:
            predictions[s1_id] = set(group["candidate_id"])
    return predictions


def evaluate_at_threshold(scored: pd.DataFrame, ground_truth: pd.DataFrame, entity_ids: Set[str], threshold: float) -> dict:
    predictions = build_predictions(scored, threshold, entity_ids)
    gt_lookup = ground_truth.set_index(config.COL_S1_ID)["matched_ids_list"].to_dict()

    scores = [
        _f_beta(set(gt_lookup.get(eid, [])), predictions.get(eid, set()))
        for eid in entity_ids
    ]
    macro_f_beta = sum(scores) / len(scores) if scores else 0.0
    return {"threshold": threshold, "macro_f0.5": macro_f_beta, "n_entities": len(scores)}


def get_error_examples(scored: pd.DataFrame, ground_truth: pd.DataFrame, entity_ids: Set[str], threshold: float, n: int = 10):
    """Returns (false_positives, false_negatives) sample DataFrames for the
    error-analysis section of the write-up."""
    predictions = build_predictions(scored, threshold, entity_ids)
    gt_lookup = ground_truth.set_index(config.COL_S1_ID)["matched_ids_list"].to_dict()

    fp_rows, fn_rows = [], []
    for eid in entity_ids:
        true_set = set(gt_lookup.get(eid, []))
        pred_set = predictions.get(eid, set())
        fp_rows += [{"s1_id": eid, "candidate_id": c} for c in pred_set - true_set]
        fn_rows += [{"s1_id": eid, "candidate_id": c} for c in true_set - pred_set]

    return pd.DataFrame(fp_rows).head(n), pd.DataFrame(fn_rows).head(n)


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
    labels.summarize_labels(labeled, missed)
    featured = feat.extract_features(labeled, s1n, s2n, s3n)

    train_ids, valid_ids = model.split_entity_ids(gt)
    train_df = model.filter_by_entity_ids(featured, train_ids)
    valid_df = model.filter_by_entity_ids(featured, valid_ids)

    trained_model = model.train_lightgbm(train_df, valid_df)
    model.save_model(trained_model)

    valid_df = valid_df.copy()
    valid_df["match_probability"] = model.predict_proba(trained_model, valid_df)

    print("\n--- Threshold sweep (macro F0.5) ---")
    for t in config.THRESHOLD_SWEEP:
        result = evaluate_at_threshold(valid_df, gt, valid_ids, t)
        print(f"threshold={t:.2f}  macro_F0.5={result['macro_f0.5']:.4f}  n_entities={result['n_entities']}")

    fp, fn = get_error_examples(valid_df, gt, valid_ids, config.DEFAULT_THRESHOLD)
    print("\nSample false positives:\n", fp)
    print("\nSample false negatives:\n", fn)