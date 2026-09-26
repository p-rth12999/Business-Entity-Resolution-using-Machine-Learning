"""
Blocking / candidate generation (Step 4).

Decided: 5 rule-based blocks now, including the TF-IDF fallback. A
sentence-embedding (MiniLM) block is deferred unless recall shows a real gap.
Candidates are the UNION of all blocks — blocking's only job is recall;
the classifier makes the precision call later.
"""

from typing import List
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import config

REQUIRED_BLOCK_COLUMNS = [
    config.COL_ENTITY_ID, "name_normalized", "name_first_token", "name_prefix",
    "postal_code", "house_number", "address_normalized", "country_normalized",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_BLOCK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"blocking.py expects normalized columns {missing} — "
            f"run normalize.normalize_dataframe() first."
        )
    return df


def _drop_overly_common_keys(df: pd.DataFrame, keys: List[str], max_freq: int) -> pd.DataFrame:
    """Drop rows whose key combination is shared by more than max_freq rows
    in this dataframe. A generic key value would otherwise blow the join up
    combinatorially — this is what makes blocking safe at millions of rows."""
    if df.empty:
        return df
    counts = df.groupby(keys)[keys[0]].transform("size")
    return df[counts <= max_freq]


def _merge_block(s1: pd.DataFrame, cand: pd.DataFrame, keys: List[str], rule: str) -> pd.DataFrame:
    left = s1[[config.COL_ENTITY_ID] + keys].copy()
    right = cand[[config.COL_ENTITY_ID] + keys].copy()
    for k in keys:
        left = left[left[k] != ""]
        right = right[right[k] != ""]

    left = _drop_overly_common_keys(left, keys, config.MAX_BLOCK_KEY_FREQUENCY)
    right = _drop_overly_common_keys(right, keys, config.MAX_BLOCK_KEY_FREQUENCY)
    if left.empty or right.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    merged = left.merge(right, on=keys, suffixes=("_s1", "_cand"))
    if merged.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    pairs = merged[[f"{config.COL_ENTITY_ID}_s1", f"{config.COL_ENTITY_ID}_cand"]].copy()
    pairs.columns = ["s1_id", "candidate_id"]
    pairs["rule"] = rule
    return pairs.drop_duplicates()


def block_exact_name(s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    return _merge_block(s1, cand, ["name_normalized"], "exact_name")


def block_postal_name_token(s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    if not config.POSTAL_CODE_BLOCK_ENABLED:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])
    return _merge_block(s1, cand, ["postal_code", "name_first_token"], "postal_name_token")


def block_house_number_address_overlap(s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Merge on house_number, then require real address similarity beyond
    just the shared number — a bare house-number match ('12') alone is too
    common to trust. Uses rapidfuzz's token_set_ratio directly on the
    normalized address strings (fast, C-level, no stored token objects)
    instead of Python set intersection, which was 200-400s/source at scale."""
    if not config.HOUSE_NUMBER_BLOCK_ENABLED:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    left = s1[[config.COL_ENTITY_ID, "house_number", "address_normalized"]]
    left = left[left["house_number"] != ""]
    right = cand[[config.COL_ENTITY_ID, "house_number", "address_normalized"]]
    right = right[right["house_number"] != ""]

    left = _drop_overly_common_keys(left, ["house_number"], config.MAX_BLOCK_KEY_FREQUENCY)
    right = _drop_overly_common_keys(right, ["house_number"], config.MAX_BLOCK_KEY_FREQUENCY)
    if left.empty or right.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    merged = left.merge(right, on="house_number", suffixes=("_s1", "_cand"))
    if merged.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    scores = [
        fuzz.token_set_ratio(a, b)
        for a, b in zip(merged["address_normalized_s1"], merged["address_normalized_cand"])
    ]
    merged = merged[pd.Series(scores, index=merged.index) >= 50]
    if merged.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    pairs = merged[[f"{config.COL_ENTITY_ID}_s1", f"{config.COL_ENTITY_ID}_cand"]].copy()
    pairs.columns = ["s1_id", "candidate_id"]
    pairs["rule"] = "house_number_address_overlap"
    return pairs.drop_duplicates()


def block_country_name_prefix(s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    if not config.NAME_PREFIX_BLOCK_ENABLED:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])
    return _merge_block(s1, cand, ["country_normalized", "name_prefix"], "country_name_prefix")


def block_tfidf_nearest_neighbors(s1: pd.DataFrame, cand: pd.DataFrame, top_k: int = None) -> pd.DataFrame:
    """Fuzzy fallback: character n-gram TF-IDF + cosine nearest neighbors, for
    typos/reordering the exact-key blocks above miss."""
    if not config.TFIDF_FALLBACK_BLOCK_ENABLED or cand.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    top_k = top_k or config.TFIDF_FALLBACK_TOP_K
    s1_names = s1["name_normalized"].fillna("")
    cand_names = cand["name_normalized"].fillna("")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    cand_matrix = vectorizer.fit_transform(cand_names)
    s1_matrix = vectorizer.transform(s1_names)

    k = min(top_k, cand_matrix.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(cand_matrix)
    _, indices = nn.kneighbors(s1_matrix)

    s1_ids = s1[config.COL_ENTITY_ID].values
    cand_ids = cand[config.COL_ENTITY_ID].values

    rows = [
        (s1_ids[i], cand_ids[indices[i, j]])
        for i in range(len(s1_ids)) for j in range(k)
    ]
    pairs = pd.DataFrame(rows, columns=["s1_id", "candidate_id"])
    pairs["rule"] = "tfidf_nn"
    return pairs.drop_duplicates()


def block_sorted_neighborhood(s1: pd.DataFrame, cand: pd.DataFrame, sort_column: str, rule_name: str,
                               window: int = None) -> pd.DataFrame:
    """Scalable fuzzy fallback. Sort S1 + candidate records together by a
    normalized key, then only compare records within a sliding window of
    each other — O((n+m) log(n+m)) instead of brute-force O(n*m). This is
    what actually scales to millions of rows; nearest-neighbor search over
    the full TF-IDF matrix does not.

    Fully vectorized with numpy (no per-row Python loop): for each window
    offset d, shift the sorted array by d and compare in bulk.
    """
    if not config.SORTED_NEIGHBORHOOD_BLOCK_ENABLED:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])
    window = window or config.SORTED_NEIGHBORHOOD_WINDOW

    left = s1[[config.COL_ENTITY_ID, sort_column]].rename(columns={config.COL_ENTITY_ID: "id", sort_column: "key"})
    left["is_s1"] = True
    right = cand[[config.COL_ENTITY_ID, sort_column]].rename(columns={config.COL_ENTITY_ID: "id", sort_column: "key"})
    right["is_s1"] = False

    combined = pd.concat([left, right], ignore_index=True)
    combined = combined[combined["key"] != ""]  # an empty key isn't meaningfully sortable
    combined = combined.sort_values("key", kind="mergesort").reset_index(drop=True)

    ids = combined["id"].to_numpy()
    is_s1 = combined["is_s1"].to_numpy()
    n = len(combined)

    s1_hits, cand_hits = [], []
    for d in range(1, min(window, n - 1) + 1):
        a_is_s1, b_is_s1 = is_s1[:-d], is_s1[d:]
        a_ids, b_ids = ids[:-d], ids[d:]

        mask_ab = a_is_s1 & ~b_is_s1  # a is S1, b is candidate
        s1_hits.append(a_ids[mask_ab])
        cand_hits.append(b_ids[mask_ab])

        mask_ba = ~a_is_s1 & b_is_s1  # a is candidate, b is S1
        s1_hits.append(b_ids[mask_ba])
        cand_hits.append(a_ids[mask_ba])

    if not s1_hits:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    pairs_df = pd.DataFrame({
        "s1_id": np.concatenate(s1_hits),
        "candidate_id": np.concatenate(cand_hits),
    }).drop_duplicates()
    pairs_df["rule"] = rule_name
    return pairs_df


def block_sorted_neighborhood_name(s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    return block_sorted_neighborhood(s1, cand, "name_normalized", "sorted_neighborhood_name")


def block_sorted_neighborhood_address(s1: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    return block_sorted_neighborhood(s1, cand, "address_normalized", "sorted_neighborhood_address")


def generate_candidates_for_source(s1: pd.DataFrame, cand: pd.DataFrame, source_label: str) -> pd.DataFrame:
    s1, cand = _prep(s1), _prep(cand)

    block_fns = [
        ("exact_name", block_exact_name),
        ("postal_name_token", block_postal_name_token),
        ("house_number_address_overlap", block_house_number_address_overlap),
        ("country_name_prefix", block_country_name_prefix),
        ("sorted_neighborhood_name", block_sorted_neighborhood_name),
        ("sorted_neighborhood_address", block_sorted_neighborhood_address),
    ]

    blocks = []
    for name, fn in block_fns:
        start = time.time()
        result = fn(s1, cand)
        elapsed = time.time() - start
        print(f"  [{source_label}] {name}: {len(result)} pairs ({elapsed:.1f}s)")
        blocks.append(result)

    combined = pd.concat(blocks, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_rules"])

    grouped = (
        combined.groupby(["s1_id", "candidate_id"])["rule"]
        .apply(lambda rules: ",".join(sorted(set(rules))))
        .reset_index()
        .rename(columns={"rule": "blocking_rules"})
    )
    grouped["candidate_source"] = source_label
    return grouped


def generate_all_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    c2 = generate_candidates_for_source(s1, s2, "S2")
    c3 = generate_candidates_for_source(s1, s3, "S3")
    all_candidates = pd.concat([c2, c3], ignore_index=True)
    print(f"Generated {len(all_candidates)} candidate pairs "
          f"({len(c2)} from S2, {len(c3)} from S3)")
    return all_candidates


def measure_blocking_recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Of every true match in the ground truth, what % survived into the
    candidate set? Uses a pandas merge (hash join, vectorized) instead of
    building a Python set of tens of millions of ID-pair tuples — that
    approach is what actually stalls at this data scale, not blocking itself."""
    gt_pairs = ground_truth[[config.COL_S1_ID, "matched_ids_list"]].explode("matched_ids_list")
    gt_pairs = gt_pairs.dropna(subset=["matched_ids_list"]).rename(
        columns={config.COL_S1_ID: "s1_id", "matched_ids_list": "candidate_id"}
    )
    total_matches = len(gt_pairs)
    if total_matches == 0:
        print("Blocking recall: 0/0 = 1.0000 (no true matches in ground truth)")
        return 1.0

    found = gt_pairs.merge(
        candidates[["s1_id", "candidate_id"]].drop_duplicates(),
        on=["s1_id", "candidate_id"], how="inner",
    )
    found_matches = len(found)

    recall = found_matches / total_matches
    print(f"Blocking recall: {found_matches}/{total_matches} = {recall:.4f}")
    return recall


if __name__ == "__main__":
    import io_utils
    import normalize

    s1, s2, s3 = io_utils.load_train_sources()
    gt = io_utils.load_ground_truth()

    s1n = normalize.normalize_dataframe(s1)
    s2n = normalize.normalize_dataframe(s2)
    s3n = normalize.normalize_dataframe(s3)

    candidates = generate_all_candidates(s1n, s2n, s3n)
    measure_blocking_recall(candidates, gt)

    train_candidates_path = config.OUTPUT_DIR / "train_candidate_pairs.tsv"
    candidates.to_csv(train_candidates_path, sep="\t", index=False)
    print(f"Saved training-time candidate pairs to {train_candidates_path}")