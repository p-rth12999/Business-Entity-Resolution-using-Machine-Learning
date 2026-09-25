"""
Pairwise feature extraction (Step 5).

Joins labeled candidate pairs back to their normalized S1/S2/S3 records and
computes name/address/structural/interaction similarity features.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

import config

BLOCKING_RULES = [
    "exact_name", "postal_name_token", "house_number_address_overlap",
    "country_name_prefix", "sorted_neighborhood_name", "sorted_neighborhood_address",
]

_JOIN_COLS = [
    config.COL_ENTITY_ID, "name_normalized", "name_stripped", "address_normalized",
    "address_tokens", "postal_code", "house_number", "country_normalized",
]


def _token_jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _set_jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _pairwise_tfidf_cosine(strings_a: pd.Series, strings_b: pd.Series) -> np.ndarray:
    """Cosine similarity for each (a[i], b[i]) pair — NOT the full cross
    product — using one TF-IDF vectorizer fit on the batch's vocabulary."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vectorizer.fit(pd.concat([strings_a, strings_b]).fillna(""))

    mat_a = vectorizer.transform(strings_a.fillna(""))
    mat_b = vectorizer.transform(strings_b.fillna(""))

    numerator = np.asarray(mat_a.multiply(mat_b).sum(axis=1)).ravel()
    norm_a = np.sqrt(np.asarray(mat_a.multiply(mat_a).sum(axis=1)).ravel())
    norm_b = np.sqrt(np.asarray(mat_b.multiply(mat_b).sum(axis=1)).ravel())
    denom = norm_a * norm_b
    denom[denom == 0] = 1e-9
    return numerator / denom


def _join_side(pairs: pd.DataFrame, s1: pd.DataFrame, cand_by_source: dict) -> pd.DataFrame:
    s1_side = s1[_JOIN_COLS].add_suffix("_s1").rename(columns={f"{config.COL_ENTITY_ID}_s1": "s1_id"})
    joined = pairs.merge(s1_side, on="s1_id", how="left")

    parts = []
    for source_label, cand_df in cand_by_source.items():
        subset = joined[joined["candidate_source"] == source_label]
        cand_side = cand_df[_JOIN_COLS].add_suffix("_cand").rename(
            columns={f"{config.COL_ENTITY_ID}_cand": "candidate_id"}
        )
        parts.append(subset.merge(cand_side, on="candidate_id", how="left"))

    return pd.concat(parts, ignore_index=True)


def extract_features(pairs: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    import time
    start = time.time()
    df = _join_side(pairs, s1, {"S2": s2, "S3": s3})
    print(f"  Joined {len(df)} candidate pairs to their records ({time.time() - start:.1f}s)")

    # Name features — zip-based list comprehensions instead of apply(axis=1),
    # which is one of the slowest patterns in pandas at millions of rows.
    df["name_exact"] = (df["name_normalized_s1"] == df["name_normalized_cand"]).astype(int)
    df["name_stripped_exact"] = (df["name_stripped_s1"] == df["name_stripped_cand"]).astype(int)
    df["name_jaccard"] = [_token_jaccard(a, b) for a, b in zip(df["name_normalized_s1"], df["name_normalized_cand"])]
    df["name_levenshtein"] = [fuzz.ratio(a, b) / 100 for a, b in zip(df["name_normalized_s1"], df["name_normalized_cand"])]
    df["name_jaro_winkler"] = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(df["name_normalized_s1"], df["name_normalized_cand"])]
    df["name_tfidf_cosine"] = _pairwise_tfidf_cosine(df["name_normalized_s1"], df["name_normalized_cand"])

    # Address features
    df["address_exact"] = (df["address_normalized_s1"] == df["address_normalized_cand"]).astype(int)
    df["address_jaccard"] = [_set_jaccard(a, b) for a, b in zip(df["address_tokens_s1"], df["address_tokens_cand"])]
    df["address_levenshtein"] = [fuzz.ratio(a, b) / 100 for a, b in zip(df["address_normalized_s1"], df["address_normalized_cand"])]

    df["postal_match"] = ((df["postal_code_s1"] != "") & (df["postal_code_s1"] == df["postal_code_cand"])).astype(int)
    df["house_number_match"] = ((df["house_number_s1"] != "") & (df["house_number_s1"] == df["house_number_cand"])).astype(int)
    df["country_match"] = (df["country_normalized_s1"] == df["country_normalized_cand"]).astype(int)

    # Missing-field flags
    df["name_missing_s1"] = (df["name_normalized_s1"] == "").astype(int)
    df["name_missing_cand"] = (df["name_normalized_cand"] == "").astype(int)
    df["address_missing_s1"] = (df["address_normalized_s1"] == "").astype(int)
    df["address_missing_cand"] = (df["address_normalized_cand"] == "").astype(int)

    # Source + blocking-rule indicators
    df["source_is_s2"] = (df["candidate_source"] == "S2").astype(int)
    for rule in BLOCKING_RULES:
        df[f"rule_{rule}"] = df["blocking_rules"].str.contains(rule, regex=False).astype(int)

    # Interaction features
    df["name_address_product"] = df["name_levenshtein"] * df["address_levenshtein"]
    df["strong_name_and_postal"] = ((df["name_levenshtein"] > 0.8) & (df["postal_match"] == 1)).astype(int)
    df["strong_address_and_country"] = ((df["address_levenshtein"] > 0.8) & (df["country_match"] == 1)).astype(int)

    print(f"  Feature extraction done ({time.time() - start:.1f}s total)")
    return df


FEATURE_COLUMNS = [
    "name_exact", "name_stripped_exact", "name_jaccard", "name_levenshtein",
    "name_jaro_winkler", "name_tfidf_cosine",
    "address_exact", "address_jaccard", "address_levenshtein",
    "postal_match", "house_number_match", "country_match",
    "name_missing_s1", "name_missing_cand", "address_missing_s1", "address_missing_cand",
    "source_is_s2",
] + [f"rule_{r}" for r in BLOCKING_RULES] + [
    "name_address_product", "strong_name_and_postal", "strong_address_and_country",
]


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

    featured = extract_features(labeled, s1n, s2n, s3n)
    print(f"\nFeature matrix: {len(featured)} rows, {len(FEATURE_COLUMNS)} features")
    print(featured[FEATURE_COLUMNS + ["label"]].head())