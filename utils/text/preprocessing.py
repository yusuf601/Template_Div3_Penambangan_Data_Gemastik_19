"""Stateless text preprocessing and normalization routines."""

import html
import json
from pathlib import Path
import re
import unicodedata
from typing import Optional, Union

import pandas as pd

DEFAULT_ID_SLANG_MAP: dict[str, str] = {
    "yg": "yang",
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "kalo": "kalau",
    "kl": "kalau",
    "udh": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "dr": "dari",
    "tp": "tetapi",
    "tapi": "tetapi",
    "aja": "saja",
    "jd": "jadi",
    "bkn": "bukan",
    "org": "orang",
    "sy": "saya",
    "aku": "saya",
    "gw": "saya",
    "gue": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "mantul": "mantap betul",
    "zonk": "buruk",
}

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_MENTION_PATTERN = re.compile(r"@\w+")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def load_slang_map(slang_dict_path: Optional[Union[str, Path]] = None) -> dict[str, str]:
    """Load slang mapping merged with DEFAULT_ID_SLANG_MAP."""
    mapping = dict(DEFAULT_ID_SLANG_MAP)
    if slang_dict_path is None:
        return mapping

    p = Path(slang_dict_path)
    if not p.exists():
        return mapping

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, v in data.items():
                mapping[str(k).lower()] = str(v)
    elif p.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        if df.shape[1] >= 2:
            for _, row in df.iterrows():
                mapping[str(row.iloc[0]).lower()] = str(row.iloc[1])

    return mapping


def _build_slang_replacer(slang_map: dict[str, str]):
    sorted_keys = sorted(slang_map.keys(), key=len, reverse=True)
    if not sorted_keys:
        return lambda t: t
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b", re.IGNORECASE)

    def _replace(match: re.Match) -> str:
        token = match.group(0).lower()
        return slang_map.get(token, match.group(0))

    return lambda t: pattern.sub(_replace, t)


def clean_text_series(
    s: pd.Series,
    mode: str = "minimal",
    lowercase: bool = True,
    slang_map: Optional[dict[str, str]] = None,
) -> pd.Series:
    """Clean and normalize a pandas Series of text according to mode."""
    # First ensure null values become empty strings
    s_clean = s.fillna("").astype(str)

    if mode == "none":
        if lowercase:
            return s_clean.str.lower()
        return s_clean

    active_slang = slang_map if slang_map is not None else DEFAULT_ID_SLANG_MAP
    slang_replacer = _build_slang_replacer(active_slang)

    def _clean_one(text: str) -> str:
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        if mode == "social_id":
            # Strip HTML
            text = html.unescape(text)
            text = _HTML_TAG_PATTERN.sub(" ", text)

            # Mask URLs, emails, mentions before lowercasing/slang
            text = _URL_PATTERN.sub("<url>", text)
            text = _EMAIL_PATTERN.sub("<email>", text)
            text = _MENTION_PATTERN.sub("<user>", text)

            # Collapse repeated characters (e.g., baguuusss -> baguus)
            text = _REPEAT_CHAR_PATTERN.sub(r"\1\1", text)

            if lowercase:
                text = text.lower()

            # Apply slang normalization
            text = slang_replacer(text)

            # Clean whitespace
            text = _WHITESPACE_PATTERN.sub(" ", text).strip()
            return text

        elif mode == "minimal":
            text = html.unescape(text)
            text = _HTML_TAG_PATTERN.sub(" ", text)
            if lowercase:
                text = text.lower()
            text = _WHITESPACE_PATTERN.sub(" ", text).strip()
            return text

        return text

    return s_clean.map(_clean_one)


__all__ = [
    "DEFAULT_ID_SLANG_MAP",
    "load_slang_map",
    "clean_text_series",
]
