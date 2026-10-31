"""Vectorized canonical ID, hashing, and I/O helpers."""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd


def _canon_scalar(v: Any) -> str:
    """Canonicalize a single scalar value."""
    if v is None or v is pd.NA or v is pd.NaT:
        return "<NA>"
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "<NA>"
    try:
        if pd.isna(v):
            return "<NA>"
    except Exception:
        pass

    # Bool must precede int check as bool is a subclass of int in Python
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"

    if isinstance(v, (int, np.integer)):
        return str(int(v))

    if isinstance(v, (float, np.floating)):
        if np.isfinite(v) and (v == np.floor(v)) and abs(v) < 9007199254740992:
            return str(int(v))
        return f"{float(v):.6g}"

    if isinstance(v, pd.Timestamp) or hasattr(v, "isoformat"):
        return v.isoformat()

    if isinstance(v, str):
        return v

    return str(v)


def canon_id(series: pd.Series) -> pd.Series:
    """Vectorized canonical ID converter preserving index.

    Rules per dtype:
    - bool: mapped to '1'/'0', NA as '<NA>' (checked before integer)
    - integer: converted to str without 2**53 cap (preserving 64-bit int precision)
    - float: finite integers with |x| < 2**53 cast to int64 -> str,
             other finite/inf formatted as f'{x:.6g}', NaN as '<NA>'
    - datetime64: ISO 8601 string, NaT as '<NA>'
    - object / categorical / mixed: per-element fallback
    """
    s = series
    if pd.api.types.is_bool_dtype(s.dtype):
        res = pd.Series("<NA>", index=s.index, dtype=object)
        not_na = s.notna()
        if not_na.any():
            res.loc[not_na] = np.where(s[not_na].to_numpy(dtype=bool), "1", "0")
        return res

    if pd.api.types.is_integer_dtype(s.dtype):
        res = s.astype(str)
        if s.isna().any():
            res = res.mask(s.isna(), "<NA>")
        return res

    if pd.api.types.is_float_dtype(s.dtype):
        res = pd.Series("<NA>", index=s.index, dtype=object)
        not_na = s.notna()
        if not not_na.any():
            return res
        s_valid = s[not_na]
        finite = np.isfinite(s_valid)
        int_mask = finite & (s_valid == np.floor(s_valid)) & (s_valid.abs() < 9007199254740992)
        if int_mask.any():
            int_idx = s_valid[int_mask].index
            res.loc[int_idx] = s_valid[int_mask].astype(np.int64).astype(str)
        other_mask = ~int_mask
        if other_mask.any():
            other_idx = s_valid[other_mask].index
            res.loc[other_idx] = [f"{float(x):.6g}" for x in s_valid[other_mask]]
        return res

    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        res = pd.Series("<NA>", index=s.index, dtype=object)
        not_na = s.notna()
        if not_na.any():
            res.loc[not_na] = [ts.isoformat() for ts in s[not_na]]
        return res

    # Object / categorical / mixed fallback
    return pd.Series([_canon_scalar(v) for v in s], index=s.index, dtype=object)


def canon_hash(df: pd.DataFrame, cols: list[str], sort_by: str) -> str:
    """Deterministic hash across specified columns sorted by sort_by.

    Columns in each row joined by ASCII unit separator (0x1f), rows joined by newline (0x0a).
    """
    if df.empty:
        return hashlib.sha256(b"").hexdigest()

    all_cols = list(dict.fromkeys(cols + ([sort_by] if sort_by not in cols else [])))
    canon_dict = {c: canon_id(df[c]) for c in all_cols}
    canon_df = pd.DataFrame(canon_dict, index=df.index)

    sort_keys = [sort_by] + [c for c in cols if c != sort_by]
    sorted_df = canon_df.sort_values(by=sort_keys, kind="mergesort")

    if not cols:
        return hashlib.sha256(b"").hexdigest()

    joined_rows = sorted_df[cols[0]]
    for c in cols[1:]:
        joined_rows = joined_rows + "\x1f" + sorted_df[c]

    payload = "\n".join(joined_rows.tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fast_df_hash(df: pd.DataFrame) -> str:
    """Fast SHA-256 hash using pandas built-in hash_pandas_object.

    Returns 'n/a' if any column contains unhashable types (e.g., list, dict).
    """
    try:
        h = pd.util.hash_pandas_object(df, index=False).values.tobytes()
        return hashlib.sha256(h).hexdigest()
    except Exception:
        return "n/a"


def _json_default(obj: Any) -> Any:
    """JSON serializer for NumPy types, paths, datetimes, and sets."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)) or hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data to JSON file, automatically creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file from path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save DataFrame to Parquet file without index, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    return pd.read_parquet(path)


__all__ = [
    "canon_id",
    "canon_hash",
    "fast_df_hash",
    "save_json",
    "load_json",
    "save_parquet",
    "load_parquet",
]
