"""
Pairwise training labels (Step 3).

Decided: hard negatives only — a candidate that survived blocking but isn't
in the true match list becomes label=0. We do NOT sample random pairs from
outside the candidate set; that's not the distribution the model sees at
test time.
"""

import pandas as pd

import config


def build_pairwise_labels(candidates: pd.DataFrame, ground_truth: pd.DataFrame):
    """Returns (labeled_pairs, missed_positives).

    labeled_pairs: candidates + label column (1 = true match, 0 = hard negative)
    missed_positives: true matches that never survived blocking — can't be
    used as training positives, and count directly against blocking recall.
    """
    gt_pairs = ground_truth[[config.COL_S1_ID, "matched_ids_list"]].explode("matched_ids_list")
    gt_pairs = gt_pairs.dropna(subset=["matched_ids_list"]).rename(
        columns={config.COL_S1_ID: "s1_id", "matched_ids_list": "candidate_id"}
    )
    gt_pairs = gt_pairs.drop_duplicates()
    gt_pairs["label"] = 1

    labeled = candidates.merge(gt_pairs, on=["s1_id", "candidate_id"], how="left")
    labeled["label"] = labeled["label"].fillna(0).astype(int)

    candidate_keys = set(zip(candidates["s1_id"], candidates["candidate_id"]))
    missed_mask = ~gt_pairs.apply(lambda r: (r["s1_id"], r["candidate_id"]) in candidate_keys, axis=1)
    missed_positives = gt_pairs[missed_mask].drop(columns=["label"])

    return labeled, missed_positives


def summarize_labels(labeled: pd.DataFrame, missed_positives: pd.DataFrame) -> None:
    n_pos = int((labeled["label"] == 1).sum())
    n_neg = int((labeled["label"] == 0).sum())
    total = n_pos + n_neg
    rate = n_pos / total if total else 0
    print(f"Training pairs: {n_pos} positive, {n_neg} hard-negative ({rate:.2%} positive rate)")
    if len(missed_positives):
        print(f"WARNING: {len(missed_positives)} true matches were not in the "
              f"candidate set (lost to blocking) — can't be used as positives.")


if __name__ == "__main__":
    import io_utils
    import normalize
    import blocking

    s1, s2, s3 = io_utils.load_train_sources()
    gt = io_utils.load_ground_truth()

    s1n = normalize.normalize_dataframe(s1)
    s2n = normalize.normalize_dataframe(s2)
    s3n = normalize.normalize_dataframe(s3)

    candidates = blocking.generate_all_candidates(s1n, s2n, s3n)
    labeled, missed = build_pairwise_labels(candidates, gt)
    summarize_labels(labeled, missed)