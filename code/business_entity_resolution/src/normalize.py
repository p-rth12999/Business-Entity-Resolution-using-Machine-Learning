"""
Normalization (Step 2).

Decided: keep BOTH a normalized name and a legal-suffix-stripped name —
never suffix-stripped-only. Also derives the keys blocking.py needs:
postal_code, house_number, name_first_token, name_prefix.

Vectorized with pandas .str accessors instead of row-wise .apply(pd.Series) —
at millions of rows, that pattern is one of the slowest things in pandas.
This does the same normalization, just fast enough to actually finish.
"""

import re
import unicodedata

import pandas as pd

import config

LEGAL_SUFFIXES = [
    "pvt", "private", "ltd", "limited", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llp", "plc", "gmbh", "sa", "sarl",
]
# Matches one or more trailing suffix tokens in a single pass (handles "pvt ltd")
_SUFFIX_RE = re.compile(r"(?:\s+(?:" + "|".join(LEGAL_SUFFIXES) + r"))+$")

ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "bldg": "building",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _basic_clean_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    # Unicode fold (accented -> plain ascii) has no vectorized pandas
    # equivalent — but this is now the ONLY row-wise step, a single cheap
    # operation per row, not a whole dict of derived fields per row.
    s = s.map(lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii"))
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _expand_abbreviations_series(series: pd.Series) -> pd.Series:
    for abbr, full in ABBREVIATIONS.items():
        series = series.str.replace(rf"\b{abbr}\b", full, regex=True)
    return series


def normalize_names(raw_names: pd.Series) -> pd.DataFrame:
    normalized = _expand_abbreviations_series(_basic_clean_series(raw_names))
    stripped = normalized.str.replace(_SUFFIX_RE, "", regex=True).str.strip()
    stripped = stripped.where(stripped != "", normalized)  # don't over-strip to empty

    return pd.DataFrame({
        "name_normalized": normalized,
        "name_stripped": stripped,
        "name_first_token": normalized.str.split().str[0].fillna(""),
        "name_prefix": normalized.str.replace(" ", "", regex=False).str.slice(0, 4),
    })


def normalize_addresses(raw_addresses: pd.Series) -> pd.DataFrame:
    normalized = _expand_abbreviations_series(_basic_clean_series(raw_addresses))
    digit_groups = normalized.str.findall(r"\d+")

    house_number = digit_groups.map(lambda g: g[0] if isinstance(g, list) and g else "")
    postal_code = digit_groups.map(lambda g: max(g, key=len) if isinstance(g, list) and g else "")

    # NOTE: no address_tokens (Python set) column here anymore — at tens of
    # millions of rows, a stored Python object column carried through every
    # merge is what actually exhausts memory, not row count alone. Address
    # token overlap is computed on-the-fly from address_normalized strings
    # using rapidfuzz where needed (blocking.py, features.py) instead.
    return pd.DataFrame({
        "address_normalized": normalized,
        "house_number": house_number,
        "postal_code": postal_code,
    })


def normalize_country(raw_countries: pd.Series) -> pd.Series:
    return _basic_clean_series(raw_countries)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized columns to a COPY of df — raw fields are kept untouched
    for feature extraction later."""
    df = df.copy()
    df = pd.concat([
        df,
        normalize_names(df[config.COL_NAME]),
        normalize_addresses(df[config.COL_ADDRESS]),
    ], axis=1)
    df["country_normalized"] = normalize_country(df[config.COL_COUNTRY])
    return df


if __name__ == "__main__":
    import time
    import io_utils

    s1, _, _ = io_utils.load_train_sources()
    start = time.time()
    s1_norm = normalize_dataframe(s1)
    print(f"Normalized {len(s1_norm)} rows in {time.time() - start:.1f}s")
    print(s1_norm[[config.COL_NAME, "name_normalized", "name_stripped",
                   config.COL_ADDRESS, "house_number", "postal_code"]].head())