"""
Test-time inference (Step 10).

Runs normalize -> block -> feature-extract -> score -> threshold on the test
data and writes both required submission files. candidate_pairs.tsv holds
the FINAL candidate set actually scored by the model — not raw/intermediate
blocking output.
"""

import pandas as pd

import config
import io_utils
import normalize
import blocking
import features as feat
import model
import threshold as threshold_module


def run_test_pipeline(threshold: float = None) -> pd.DataFrame:
    threshold = threshold if threshold is not None else threshold_module.load_threshold()

    s1, s2, s3 = io_utils.load_test_sources()
    s1n = normalize.normalize_dataframe(s1)
    s2n = normalize.normalize_dataframe(s2)
    s3n = normalize.normalize_dataframe(s3)

    candidates = blocking.generate_all_candidates(s1n, s2n, s3n)
    featured = feat.extract_features(candidates, s1n, s2n, s3n)

    trained_model = model.load_model()
    featured["match_probability"] = model.predict_proba(trained_model, featured)

    # candidate_pairs.tsv = the exact final set the model scored
    candidate_output = featured[["s1_id", "candidate_id", "candidate_source", "match_probability"]].rename(
        columns={"s1_id": config.COL_S1_ID}
    )
    candidate_output.to_csv(config.CANDIDATE_PAIRS_FILE, sep="\t", index=False)
    print(f"Saved {len(candidate_output)} final candidate pairs to {config.CANDIDATE_PAIRS_FILE}")

    # matching_results.tsv — every Source-1 test entity gets a row, singleton or not
    above_threshold = featured[featured["match_probability"] >= threshold]
    matches_by_entity = above_threshold.groupby("s1_id")["candidate_id"].apply(list).to_dict()

    results = [
        {config.COL_S1_ID: s1_id, config.COL_MATCHED_IDS: ",".join(matches_by_entity.get(s1_id, []))}
        for s1_id in s1[config.COL_ENTITY_ID]
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.MATCHING_RESULTS_FILE, sep="\t", index=False)

    n_matched = (results_df[config.COL_MATCHED_IDS] != "").sum()
    print(f"Saved matching results for {len(results_df)} entities to {config.MATCHING_RESULTS_FILE}")
    print(f"{n_matched} matched, {len(results_df) - n_matched} predicted singleton")

    return results_df


if __name__ == "__main__":
    run_test_pipeline()