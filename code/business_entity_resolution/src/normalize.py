"""
Normalization (Step 2).

Decided: keep BOTH a normalized name and a legal-suffix-stripped name —
never suffix-stripped-only. Also derives the keys blocking.py needs:
postal_code, house_number, name_first_token, name_prefix.
"""

import re
import unicodedata
from typing import List

import pandas as pd

import config

# Legal-suffix tokens stripped for the *stripped* name representation only.
# Stripping repeats from the end so "pvt ltd" (two suffix tokens) is fully removed.
LEGAL_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llp", "plc", "gmbh", "sa", "sarl",
}

# Applied to both name and address tokens.
ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "bldg": "building",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def _basic_clean(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _expand_abbreviations(tokens: List[str]) -> List[str]:
    return [ABBREVIATIONS.get(tok, tok) for tok in tokens]


def normalize_name(raw_name) -> dict:
    cleaned = _basic_clean(raw_name)
    tokens = _expand_abbreviations(cleaned.split())
    normalized = " ".join(tokens)

    stripped_tokens = list(tokens)
    while stripped_tokens and stripped_tokens[-1] in LEGAL_SUFFIXES:
        stripped_tokens.pop()
    stripped = " ".join(stripped_tokens) if stripped_tokens else normalized

    return {
        "name_normalized": normalized,
        "name_stripped": stripped,
        "name_first_token": tokens[0] if tokens else "",
        "name_prefix": normalized.replace(" ", "")[:4],
    }


def normalize_address(raw_address) -> dict:
    cleaned = _basic_clean(raw_address)
    tokens = _expand_abbreviations(cleaned.split())
    normalized = " ".join(tokens)

    digit_groups = _DIGIT_RE.findall(cleaned)
    house_number = digit_groups[0] if digit_groups else ""
    # Longest digit group as postal-code heuristic (6-digit Indian PIN, 5-digit
    # US ZIP). Flag for the researcher: France uses 5-digit codes too — worth
    # a sanity check once test data is in hand.
    postal_code = max(digit_groups, key=len) if digit_groups else ""

    return {
        "address_normalized": normalized,
        "address_tokens": set(tokens),
        "house_number": house_number,
        "postal_code": postal_code,
    }


def normalize_country(raw_country) -> str:
    return _basic_clean(raw_country)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized columns to a COPY of df — raw fields are kept untouched
    for feature extraction later."""
    df = df.copy()

    name_parts = df[config.COL_NAME].apply(normalize_name).apply(pd.Series)
    address_parts = df[config.COL_ADDRESS].apply(normalize_address).apply(pd.Series)

    df = pd.concat([df, name_parts, address_parts], axis=1)
    df["country_normalized"] = df[config.COL_COUNTRY].apply(normalize_country)
    return df


if __name__ == "__main__":
    import io_utils

    s1, _, _ = io_utils.load_train_sources()
    s1_norm = normalize_dataframe(s1)
    print(s1_norm[[config.COL_NAME, "name_normalized", "name_stripped",
                   config.COL_ADDRESS, "house_number", "postal_code"]].head())