import datetime
from pathlib import Path
import random
import numpy as np
import pandas as pd
import pytest

from utils.core.errors import (
    TemplateError,
    ConfigError,
    DataError,
    FoldMismatchError,
    HookContractError,
    SubmissionError,
)
from utils.core.io import (
    canon_id,
    canon_hash,
    fast_df_hash,
    save_json,
    load_json,
    save_parquet,
    load_parquet,
)
from utils.core.seed import seed_everything


def test_template_error_format():
    err = FoldMismatchError("Set ID berbeda (3 ID hilang)", "Set cfg.folds_path=None untuk membuat fold baru")
    assert isinstance(err, TemplateError) and isinstance(err, ValueError)
    assert "[Masalah]" in str(err) and "[Perbaikan]" in str(err)
    assert err.problem == "Set ID berbeda (3 ID hilang)"
    assert err.fix == "Set cfg.folds_path=None untuk membuat fold baru"

    # Test all subclasses inherit from TemplateError and ValueError
    for cls in (ConfigError, DataError, FoldMismatchError, HookContractError, SubmissionError):
        instance = cls("problem description", "fix suggestion")
        assert isinstance(instance, TemplateError)
        assert isinstance(instance, ValueError)
        assert instance.problem == "problem description"
        assert instance.fix == "fix suggestion"
        assert "[Masalah]   problem description\n[Perbaikan] fix suggestion" in str(instance)


def test_canon_id_and_canon_hash_vectorized_equivalences():
    df_int = pd.DataFrame({"id": [7, 8], "y": [1, 0], "na": [1, np.nan]})
    df_flt = pd.DataFrame({"id": [7.0, 8.0], "y": [1.0, 0.0], "na": [1.0, None]})
    df_bool = pd.DataFrame({"id": ["7", "8"], "y": [True, False], "na": [True, np.nan]})
    h = canon_hash(df_int, ["id", "y", "na"], sort_by="id")
    assert h == canon_hash(df_flt, ["id", "y", "na"], sort_by="id")
    assert h == canon_hash(df_bool, ["id", "y", "na"], sort_by="id")


def test_canon_id_large_int64_and_row_order_invariance():
    big1 = np.int64(900719925474099300)
    big2 = np.int64(900719925474099301)
    s = canon_id(pd.Series([big1, big2]))
    assert s.iloc[0] != s.iloc[1]
    assert s.iloc[0] == str(int(big1))

    # Float series with np.inf and np.nan must not fail integer casting
    s_edge = canon_id(pd.Series([1.0, np.inf, -np.inf, np.nan, 2.5]))
    assert list(s_edge) == ["1", "inf", "-inf", "<NA>", "2.5"]

    df1 = pd.DataFrame({"id": [1, 2, 3], "y": np.array([0.1, 0.2, 0.3], dtype=np.float64)})
    df2 = pd.DataFrame({"id": ["3", "1", "2"], "y": np.array([0.3, 0.1, 0.2], dtype=np.float32)})
    assert canon_hash(df1, ["id", "y"], sort_by="id") == canon_hash(df2, ["id", "y"], sort_by="id")


def test_canon_id_dtypes():
    # Nullable integer with NA
    s_nullable_int = pd.Series([10, None, 20], dtype="Int64")
    assert list(canon_id(s_nullable_int)) == ["10", "<NA>", "20"]

    # Boolean dtype with NA
    s_bool = pd.Series([True, False, None], dtype="boolean")
    assert list(canon_id(s_bool)) == ["1", "0", "<NA>"]

    # Datetime64
    dt1 = pd.Timestamp("2026-10-31 12:00:00")
    s_dt = pd.Series([dt1, pd.NaT])
    res_dt = canon_id(s_dt)
    assert res_dt.iloc[0] == dt1.isoformat()
    assert res_dt.iloc[1] == "<NA>"

    # Categorical / object mixed fallback
    s_cat = pd.Series(pd.Categorical(["cat", "dog", None]))
    assert list(canon_id(s_cat)) == ["cat", "dog", "<NA>"]

    # Preserves index
    s_idx = pd.Series([100, 200], index=["row1", "row2"])
    res_idx = canon_id(s_idx)
    assert list(res_idx.index) == ["row1", "row2"]


def test_fast_df_hash_and_io_roundtrip(tmp_path):
    df = pd.DataFrame({"id": range(100), "a": np.linspace(0, 1, 100)})
    assert fast_df_hash(df) == fast_df_hash(df.copy())
    df_mod = df.copy()
    df_mod.loc[0, "a"] = 999.0
    assert fast_df_hash(df) != fast_df_hash(df_mod)

    # Unhashable column (e.g. list) falls back to "n/a" instead of raising TypeError
    df_list = pd.DataFrame({"id": [1, 2], "tags": [[1, 2], [3]]})
    assert fast_df_hash(df_list) == "n/a"

    # Nested directory json test
    nested_json = tmp_path / "nested" / "sub" / "x.json"
    save_json(
        {
            "n": np.int64(5),
            "flt": np.float32(1.5),
            "flag": np.bool_(True),
            "arr": np.array([1.0, 2.0]),
            "p": Path("/tmp/test"),
            "dt": pd.Timestamp("2026-10-31 00:00:00"),
            "st": {1, 2},
        },
        nested_json,
    )
    loaded = load_json(nested_json)
    assert loaded["n"] == 5
    assert loaded["flt"] == 1.5
    assert loaded["flag"] is True
    assert loaded["arr"] == [1.0, 2.0]
    assert loaded["p"] == "/tmp/test"
    assert loaded["dt"] == "2026-10-31T00:00:00"
    assert set(loaded["st"]) == {1, 2}

    # Nested directory parquet test
    nested_parquet = tmp_path / "nested" / "sub" / "x.parquet"
    save_parquet(df, nested_parquet)
    pd.testing.assert_frame_equal(load_parquet(nested_parquet), df)


def test_seed_everything():
    seed_everything(42)
    a = (random.random(), np.random.rand(3).tolist())
    seed_everything(42)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_core_exports():
    import utils.core

    expected = {
        "TemplateError",
        "ConfigError",
        "DataError",
        "FoldMismatchError",
        "HookContractError",
        "SubmissionError",
        "canon_id",
        "canon_hash",
        "fast_df_hash",
        "save_json",
        "load_json",
        "save_parquet",
        "load_parquet",
        "seed_everything",
    }
    assert expected.issubset(set(utils.core.__all__))

