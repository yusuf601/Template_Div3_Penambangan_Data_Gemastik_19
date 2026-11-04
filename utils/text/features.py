"""Stateless row-wise text and text-pair feature extraction."""

import re
import string
from typing import Optional

import numpy as np
import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_PUNCT_SET = set(string.punctuation)


def extract_text_stats(df: pd.DataFrame, text_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Compute row-wise text statistics on raw text columns without leakage."""
    out_df = df.copy()
    stat_cols: list[str] = []

    for col in text_cols:
        raw_s = df[col]
        is_na = raw_s.isna().astype(float)
        s = raw_s.fillna("").astype(str)

        char_lens = s.str.len().astype(float)
        word_counts = s.map(lambda t: float(len(t.split())))
        mean_word_lens = char_lens / np.maximum(word_counts, 1.0)

        upper_counts = s.map(lambda t: float(sum(1 for c in t if c.isupper())))
        upper_ratios = upper_counts / np.maximum(char_lens, 1.0)

        digit_counts = s.map(lambda t: float(sum(1 for c in t if c.isdigit())))
        digit_ratios = digit_counts / np.maximum(char_lens, 1.0)

        punct_counts = s.map(lambda t: float(sum(1 for c in t if c in _PUNCT_SET)))
        excl_counts = s.map(lambda t: float(t.count("!")))
        quest_counts = s.map(lambda t: float(t.count("?")))
        url_counts = s.map(lambda t: float(len(_URL_RE.findall(t))))
        mention_counts = s.map(lambda t: float(len(_MENTION_RE.findall(t))))
        hashtag_counts = s.map(lambda t: float(len(_HASHTAG_RE.findall(t))))

        col_stats = {
            f"{col}__is_na": is_na,
            f"{col}__char_len": char_lens,
            f"{col}__word_count": word_counts,
            f"{col}__mean_word_len": mean_word_lens,
            f"{col}__upper_ratio": upper_ratios,
            f"{col}__digit_ratio": digit_ratios,
            f"{col}__punct_count": punct_counts,
            f"{col}__excl_count": excl_counts,
            f"{col}__quest_count": quest_counts,
            f"{col}__url_count": url_counts,
            f"{col}__mention_count": mention_counts,
            f"{col}__hashtag_count": hashtag_counts,
        }

        for sc_name, sc_series in col_stats.items():
            out_df[sc_name] = sc_series
            stat_cols.append(sc_name)

    return out_df, stat_cols


def _char3_set(t: str) -> set[str]:
    if len(t) < 3:
        return {t} if t else set()
    return {t[i : i + 3] for i in range(len(t) - 2)}


def extract_pair_stats(
    df: pd.DataFrame,
    col_a_clean: str,
    col_b_clean: str,
    prefix: str = "pair",
) -> tuple[pd.DataFrame, list[str]]:
    """Compute row-wise text pair similarity and difference features."""
    out_df = df.copy()
    pair_cols: list[str] = []

    s_a = df[col_a_clean].fillna("").astype(str)
    s_b = df[col_b_clean].fillna("").astype(str)

    len_a = s_a.str.len().to_numpy(dtype=float)
    len_b = s_b.str.len().to_numpy(dtype=float)
    words_a = s_a.map(lambda t: float(len(t.split()))).to_numpy(dtype=float)
    words_b = s_b.map(lambda t: float(len(t.split()))).to_numpy(dtype=float)

    char_diff = np.abs(len_a - len_b)
    char_ratio = np.minimum(len_a, len_b) / np.maximum(np.maximum(len_a, len_b), 1.0)

    word_diff = np.abs(words_a - words_b)
    word_ratio = np.minimum(words_a, words_b) / np.maximum(np.maximum(words_a, words_b), 1.0)

    exact_match = (s_a == s_b).astype(float).to_numpy()

    # Jaccard word and char3
    n = len(df)
    word_jaccard = np.zeros(n, dtype=float)
    char3_jaccard = np.zeros(n, dtype=float)

    a_list = s_a.tolist()
    b_list = s_b.tolist()

    for i in range(n):
        ta = a_list[i]
        tb = b_list[i]

        toks_a = set(ta.split())
        toks_b = set(tb.split())
        w_union = toks_a | toks_b
        if not w_union:
            word_jaccard[i] = 1.0 if ta == tb else 0.0
        else:
            word_jaccard[i] = len(toks_a & toks_b) / len(w_union)

        c3_a = _char3_set(ta)
        c3_b = _char3_set(tb)
        c_union = c3_a | c3_b
        if not c_union:
            char3_jaccard[i] = 1.0 if ta == tb else 0.0
        else:
            char3_jaccard[i] = len(c3_a & c3_b) / len(c_union)

    new_feats = {
        f"{prefix}__char_len_diff": char_diff,
        f"{prefix}__char_len_ratio": char_ratio,
        f"{prefix}__word_len_diff": word_diff,
        f"{prefix}__word_len_ratio": word_ratio,
        f"{prefix}__exact_match": exact_match,
        f"{prefix}__word_jaccard": word_jaccard,
        f"{prefix}__char3_jaccard": char3_jaccard,
    }

    for name, arr in new_feats.items():
        out_df[name] = arr
        pair_cols.append(name)

    return out_df, pair_cols


__all__ = [
    "extract_text_stats",
    "extract_pair_stats",
]
