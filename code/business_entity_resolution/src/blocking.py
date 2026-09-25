"""
Blocking / candidate generation (Step 4).

Decided: 5 rule-based blocks now, including the TF-IDF fallback. A
sentence-embedding (MiniLM) block is deferred unless recall shows a real gap.
Candidates are the UNION of all blocks — blocking's only job is recall;
the classifier makes the precision call later.
"""

from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import config

REQUIRED_BLOCK_COLUMNS = [
    config.COL_ENTITY_ID, "name_normalized", "name_first_token", "name_prefix",
    "postal_code", "house_number", "address_tokens", "country_normalized",
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_BLOCK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"blocking.py expects normalized columns {missing} — "
            f"run normalize.normalize_dataframe() first."
        )
    return df


def _merge_block(s1: pd.DataFrame, cand: pd.DataFrame, keys: List[str], rule: str) -> pd.DataFrame:
    left = s1[[config.COL_ENTITY_ID] + keys].copy()
    right = cand[[config.COL_ENTITY_ID] + keys].copy()
    for k in keys:
        left = left[left[k] != ""]
        right = right[right[k] != ""]

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
    """Merge on house_number, then require >=1 more shared address token —
    a bare house-number match ('12') alone is too common to trust."""
    if not config.HOUSE_NUMBER_BLOCK_ENABLED:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    left = s1[[config.COL_ENTITY_ID, "house_number", "address_tokens"]]
    left = left[left["house_number"] != ""]
    right = cand[[config.COL_ENTITY_ID, "house_number", "address_tokens"]]
    right = right[right["house_number"] != ""]

    merged = left.merge(right, on="house_number", suffixes=("_s1", "_cand"))
    if merged.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "rule"])

    overlap_ok = merged.apply(
        lambda r: len(r["address_tokens_s1"] & r["address_tokens_cand"]) >= 2, axis=1
    )
    merged = merged[overlap_ok]
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


def generate_candidates_for_source(s1: pd.DataFrame, cand: pd.DataFrame, source_label: str) -> pd.DataFrame:
    s1, cand = _prep(s1), _prep(cand)

    blocks = [
        block_exact_name(s1, cand),
        block_postal_name_token(s1, cand),
        block_house_number_address_overlap(s1, cand),
        block_country_name_prefix(s1, cand),
        block_tfidf_nearest_neighbors(s1, cand),
    ]
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
    candidate set? Most important number before any model gets trained."""
    candidate_keys = set(zip(candidates["s1_id"], candidates["candidate_id"]))

    total_matches, found_matches = 0, 0
    for _, row in ground_truth.iterrows():
        for match_id in row["matched_ids_list"]:
            total_matches += 1
            if (row[config.COL_S1_ID], match_id) in candidate_keys:
                found_matches += 1

    recall = found_matches / total_matches if total_matches else 1.0
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