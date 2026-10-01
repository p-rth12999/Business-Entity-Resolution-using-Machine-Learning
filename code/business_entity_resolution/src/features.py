"""
Pairwise feature extraction (Step 5).

Processes candidate pairs in chunks (config.FEATURE_CHUNK_SIZE at a time)
and keeps only the compact numeric output between chunks — joining ALL
rows' string columns (name/address, both sides of every pair) at once is
what exhausted RAM on a 16GB machine at 34M+ rows. Each chunk's string data
is discarded immediately after its features are computed.
"""

import gc
import time

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import config

BLOCKING_RULES = [
    "exact_name", "postal_name_token", "house_number_address_overlap",
    "country_name_prefix", "sorted_neighborhood_name", "sorted_neighborhood_address",
]

_JOIN_COLS = [
    config.COL_ENTITY_ID, "name_normalized", "name_stripped", "address_normalized",
    "postal_code", "house_number", "country_normalized",
]

FEATURE_COLUMNS = [
    "name_exact", "name_stripped_exact", "name_jaccard", "name_levenshtein",
    "name_jaro_winkler",
    "address_exact", "address_jaccard", "address_levenshtein",
    "postal_match", "house_number_match", "country_match",
    "name_missing_s1", "name_missing_cand", "address_missing_s1", "address_missing_cand",
    "source_is_s2",
] + [f"rule_{r}" for r in BLOCKING_RULES] + [
    "name_address_product", "strong_name_and_postal", "strong_address_and_country",
]


def _token_jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _join_chunk(chunk_pairs: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """Builds the join sides fresh for THIS chunk only, so they're freed
    before the next iteration — not held for the whole run. Recomputing this
    ~18 times costs some CPU time; keeping a second full-size copy of s1/s2/s3
    alive the entire run is what actually exhausted memory last time."""
    s1_side = s1[_JOIN_COLS].add_suffix("_s1").rename(columns={f"{config.COL_ENTITY_ID}_s1": "s1_id"})
    joined = chunk_pairs.merge(s1_side, on="s1_id", how="left")
    del s1_side

    parts = []
    for source_label, cand_df in (("S2", s2), ("S3", s3)):
        subset = joined[joined["candidate_source"] == source_label]
        cand_side = cand_df[_JOIN_COLS].add_suffix("_cand").rename(columns={f"{config.COL_ENTITY_ID}_cand": "candidate_id"})
        parts.append(subset.merge(cand_side, on="candidate_id", how="left"))
        del cand_side

    return pd.concat(parts, ignore_index=True)


def _compute_chunk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes all features on an already-joined chunk, then returns ONLY
    the compact numeric output (ids + features + label if present) — the
    wide string columns from the join are dropped here, not carried forward."""
    df["name_exact"] = (df["name_normalized_s1"] == df["name_normalized_cand"]).astype(int)
    df["name_stripped_exact"] = (df["name_stripped_s1"] == df["name_stripped_cand"]).astype(int)
    df["name_jaccard"] = [_token_jaccard(a, b) for a, b in zip(df["name_normalized_s1"], df["name_normalized_cand"])]
    df["name_levenshtein"] = [fuzz.ratio(a, b) / 100 for a, b in zip(df["name_normalized_s1"], df["name_normalized_cand"])]
    df["name_jaro_winkler"] = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(df["name_normalized_s1"], df["name_normalized_cand"])]

    df["address_exact"] = (df["address_normalized_s1"] == df["address_normalized_cand"]).astype(int)
    df["address_jaccard"] = [
        fuzz.token_set_ratio(a, b) / 100
        for a, b in zip(df["address_normalized_s1"], df["address_normalized_cand"])
    ]
    df["address_levenshtein"] = [fuzz.ratio(a, b) / 100 for a, b in zip(df["address_normalized_s1"], df["address_normalized_cand"])]

    df["postal_match"] = ((df["postal_code_s1"] != "") & (df["postal_code_s1"] == df["postal_code_cand"])).astype(int)
    df["house_number_match"] = ((df["house_number_s1"] != "") & (df["house_number_s1"] == df["house_number_cand"])).astype(int)
    df["country_match"] = (df["country_normalized_s1"] == df["country_normalized_cand"]).astype(int)

    df["name_missing_s1"] = (df["name_normalized_s1"] == "").astype(int)
    df["name_missing_cand"] = (df["name_normalized_cand"] == "").astype(int)
    df["address_missing_s1"] = (df["address_normalized_s1"] == "").astype(int)
    df["address_missing_cand"] = (df["address_normalized_cand"] == "").astype(int)

    df["source_is_s2"] = (df["candidate_source"] == "S2").astype(int)
    for rule in BLOCKING_RULES:
        df[f"rule_{rule}"] = df["blocking_rules"].str.contains(rule, regex=False).astype(int)

    df["name_address_product"] = df["name_levenshtein"] * df["address_levenshtein"]
    df["strong_name_and_postal"] = ((df["name_levenshtein"] > 0.8) & (df["postal_match"] == 1)).astype(int)
    df["strong_address_and_country"] = ((df["address_levenshtein"] > 0.8) & (df["country_match"] == 1)).astype(int)

    keep_cols = ["s1_id", "candidate_id", "candidate_source"] + FEATURE_COLUMNS
    if "label" in df.columns:
        keep_cols.append("label")
    return df[keep_cols].copy()


def extract_features(pairs: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                      chunk_size: int = None) -> pd.DataFrame:
    chunk_size = chunk_size or config.FEATURE_CHUNK_SIZE
    n = len(pairs)
    start = time.time()

    results = []
    for chunk_start in range(0, n, chunk_size):
        chunk_pairs = pairs.iloc[chunk_start: chunk_start + chunk_size]
        joined = _join_chunk(chunk_pairs, s1, s2, s3)
        chunk_result = _compute_chunk_features(joined)
        results.append(chunk_result)
        del joined, chunk_result
        gc.collect()

        done = min(chunk_start + chunk_size, n)
        print(f"  Feature chunk done: {done}/{n} pairs ({time.time() - start:.1f}s elapsed)")

    result = pd.concat(results, ignore_index=True)
    del results
    gc.collect()

    # Compact ID storage: category dtype stores each unique string once —
    # rows just hold small integer codes instead of full string objects.
    result["s1_id"] = result["s1_id"].astype("category")
    result["candidate_id"] = result["candidate_id"].astype("category")

    print(f"  Feature extraction done ({time.time() - start:.1f}s total, {len(result)} rows)")
    return result


if __name__ == "__main__":
    import labels
    import pipeline

    s1n, s2n, s3n, candidates, gt = pipeline.load_and_prepare()
    labeled, missed = labels.build_pairwise_labels(candidates, gt)
    labels.summarize_labels(labeled, missed)

    featured = extract_features(labeled, s1n, s2n, s3n)
    print(f"\nFeature matrix: {len(featured)} rows, {len(FEATURE_COLUMNS)} features")
    print(featured[FEATURE_COLUMNS + ["label"]].head())