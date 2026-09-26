"""
Central configuration for the entity resolution pipeline.
Single source of truth for paths, file names, column names, and shared settings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at: student_resource/code/business_entity_resolution/src/config.py
# parents[0] = src, parents[1] = business_entity_resolution, parents[2] = code,
# parents[3] = student_resource (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # business_entity_resolution/

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PACKAGE_ROOT / "output"
MODELS_DIR = PACKAGE_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
TRAIN_SOURCE1_FILE = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_FILE = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_FILE = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_FILE = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_FILE = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_FILE = TEST_DIR / "test_source3.tsv"

CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"

MODEL_FILE = MODELS_DIR / "lightgbm_matcher.txt"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"

# ---------------------------------------------------------------------------
# Column names (source tables: entity_id, business_name, business_address, country)
# ---------------------------------------------------------------------------
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"

# Ground truth table
COL_S1_ID = "source1_entity_id"
COL_MATCHED_IDS = "matched_entity_ids"

REQUIRED_SOURCE_COLUMNS = [COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY]
REQUIRED_GROUND_TRUTH_COLUMNS = [COL_S1_ID, COL_MATCHED_IDS]

# Output table
COL_CANDIDATE_ID = "candidate_id"
COL_LABEL = "label"
COL_SOURCE = "candidate_source"  # "S2" or "S3"
COL_SCORE = "match_probability"

# ---------------------------------------------------------------------------
# Modeling settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_VALIDATION_FOLDS = 5  # GroupKFold on source1_entity_id

# Threshold sweep range for macro-F0.5 tuning (Step 7) — decided: single global threshold
THRESHOLD_SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 -> 0.95
DEFAULT_THRESHOLD = 0.5  # placeholder until threshold.py tunes it on validation

# Blocking settings (Step 4) — decided: rule-based only for now
POSTAL_CODE_BLOCK_ENABLED = True
HOUSE_NUMBER_BLOCK_ENABLED = True
NAME_PREFIX_BLOCK_ENABLED = True

# Disabled by default: at this data scale (millions of rows per source),
# brute-force TF-IDF nearest neighbors does not finish in reasonable time.
# Replaced below by sorted-neighborhood blocking, which scales to millions.
TFIDF_FALLBACK_BLOCK_ENABLED = False
TFIDF_FALLBACK_TOP_K = 5  # kept for reference / possible small-scale reuse

# Scalable fuzzy fallback: sort combined records by a normalized key, only
# compare records within a sliding window of each other (O(n log n), not
# O(n*m)). Run twice — sorted by name, then by address — so a typo early in
# one field can still be caught via the other.
# Window kept small on purpose: at 2-5M rows/source, window=10 alone produced
# ~54M candidate pairs. Widen this ONLY after confirming (via blocking recall)
# that it's actually needed — it's the single biggest volume driver.
# Scalable fuzzy fallback: sort combined records by a normalized key, only
# compare records within a sliding window of each other (O(n log n), not
# O(n*m)). Run twice — sorted by name, then by address — so a typo early in
# one field can still be caught via the other.
# TEMPORARILY DISABLED: with 30 hours to a deadline, priority is one fast,
# complete run using just the 4 exact-key blocks. Flip this to True (window
# is already cut to 2) once you have a baseline recall number and time to
# spare for a second, slower run.
SORTED_NEIGHBORHOOD_BLOCK_ENABLED = False
SORTED_NEIGHBORHOOD_WINDOW = 2

# Safety cap: an exact-key block whose key is shared by more than this many
# records (on either side) gets dropped for that key. Tightened from 50 -> 15
# after 50 still let a handful of common keys balloon into millions of pairs.
MAX_BLOCK_KEY_FREQUENCY = 15

# Disk cache for the expensive normalize+block step (was ~40+ minutes at this
# data scale). Delete the cache file manually after changing normalize.py or
# blocking.py logic, or pass --no-cache to main.py.
CACHE_DIR = PACKAGE_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_CANDIDATES_CACHE = CACHE_DIR / "normalized_and_candidates.pkl"

# Safety cap: an exact-key block whose key is shared by more than this many
# records (on either side) gets dropped for that key. Without this, one
# generic key value (a common house number, an empty-ish name token) can
# turn into a many-to-many join of tens of millions of pairs by itself.
MAX_BLOCK_KEY_FREQUENCY = 50

# Negative sampling (Step 3) — decided: hard negatives only (from blocking output)
USE_RANDOM_NEGATIVES = False