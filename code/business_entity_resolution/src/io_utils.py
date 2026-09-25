"""
Data loading and schema validation (Step 1).

All source files are TSV, not CSV — sep="\\t" is required or the whole line
gets read as a single column.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_tsv(path: Path, dtype=str) -> pd.DataFrame:
    """Load a TSV file. dtype=str by default since entity IDs, names, addresses,
    and countries are all text — this avoids pandas guessing numeric types for
    IDs like 'S1-925783039'."""
    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=dtype, keep_default_na=True)
    logger.info("Loaded %s — %d rows, %d columns", path.name, len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame, required_columns: List[str], name: str) -> None:
    """Raise a clear error if required columns are missing, and log null rates
    for each required column so data-quality issues surface early."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required column(s) {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    for col in required_columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct > 0:
            logger.info("%s.%s — %.1f%% null", name, col, null_pct)


def load_source(path: Path, name: str) -> pd.DataFrame:
    df = load_tsv(path)
    validate_schema(df, config.REQUIRED_SOURCE_COLUMNS, name)
    return df


def load_train_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = load_source(config.TRAIN_SOURCE1_FILE, "train_source1")
    s2 = load_source(config.TRAIN_SOURCE2_FILE, "train_source2")
    s3 = load_source(config.TRAIN_SOURCE3_FILE, "train_source3")
    return s1, s2, s3


def load_test_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = load_source(config.TEST_SOURCE1_FILE, "test_source1")
    s2 = load_source(config.TEST_SOURCE2_FILE, "test_source2")
    s3 = load_source(config.TEST_SOURCE3_FILE, "test_source3")
    return s1, s2, s3


def parse_matched_ids(cell) -> List[str]:
    """Turn a 'S2-047,S3-812' style cell (or empty/NaN, for singletons) into a
    clean list of IDs."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]


def load_ground_truth() -> pd.DataFrame:
    df = load_tsv(config.TRAIN_GROUND_TRUTH_FILE)
    validate_schema(df, config.REQUIRED_GROUND_TRUTH_COLUMNS, "train_ground_truth")

    df["matched_ids_list"] = df[config.COL_MATCHED_IDS].apply(parse_matched_ids)
    df["n_matches"] = df["matched_ids_list"].apply(len)

    n_singletons = (df["n_matches"] == 0).sum()
    logger.info(
        "Ground truth: %d Source-1 entities, %d singletons (%.1f%%), avg %.2f matches each",
        len(df), n_singletons, n_singletons / len(df) * 100, df["n_matches"].mean(),
    )
    return df


if __name__ == "__main__":
    # Quick smoke test: python io_utils.py
    s1, s2, s3 = load_train_sources()
    gt = load_ground_truth()
    print("\nSample S1 row:\n", s1.iloc[0])
    print("\nSample ground truth row:\n", gt.iloc[0])