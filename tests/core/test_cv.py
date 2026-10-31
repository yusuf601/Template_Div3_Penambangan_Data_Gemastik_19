import numpy as np
import pandas as pd
import pytest
from utils.core.config import BaseConfig
from utils.core.cv import (
    compute_folds_hash,
    compute_ids_target_hash,
    debug_subsample,
    get_folds,
    load_folds,
    make_folds,
    make_folds_file,
    save_folds,
    split_fold,
)
from utils.core.errors import DataError, FoldMismatchError


def _make_panel_df():
    rows = []
    idx = 0
    for t in pd.date_range("2026-01-01", periods=12, freq="D"):
        for prov in ["A", "B", "C"]:
            rows.append({"id": f"r{idx}", "dt": t, "prov": prov, "y": float(idx % 2)})
            idx += 1
    return pd.DataFrame(rows)


def test_panel_timeseries_and_split_fold_ids():
    df = _make_panel_df()
    cfg = BaseConfig(
        task="regression",
        target="y",
        metric="rmse",
        cv_strategy="timeseries",
        time_col="dt",
        n_folds=3,
        ts_gap=1,
    )
    cfg.resolve()
    cfg.validate()
    folds_df, _ = get_folds(df, cfg)
    merged = df.merge(folds_df, on="id")
    assert (merged.groupby("dt")["fold"].nunique() == 1).all()

    tr_ids, va_ids = split_fold(df, folds_df, 1, cfg)
    assert tr_ids.isdisjoint(va_ids)
    tr_max_dt = df[df["id"].isin(tr_ids)]["dt"].max()
    va_min_dt = df[df["id"].isin(va_ids)]["dt"].min()
    all_dates = pd.Series(df["dt"].unique()).sort_values().tolist()
    assert all_dates.index(va_min_dt) - all_dates.index(tr_max_dt) > cfg.ts_gap


def test_make_folds_file_and_reuse_across_shuffled_rows(tmp_path):
    df = pd.DataFrame({"id": range(40), "y": [0, 1] * 20, "f": range(40)})
    cfg = BaseConfig(task="binary", target="y", metric="auc", n_folds=4, seed=1, fold_seed=42)
    folds_csv = make_folds_file(df, cfg, out_dir=tmp_path / "folds")
    assert folds_csv.exists() and (tmp_path / "folds" / "folds_meta.json").exists()

    df_shuf = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    # Pass unresolved BaseConfig (cv_strategy=None) directly to get_folds
    cfg2 = BaseConfig(
        task="binary",
        target="y",
        metric="auc",
        n_folds=4,
        seed=999,
        fold_seed=123,
        folds_path=str(folds_csv),
    )
    f_loaded, meta = get_folds(df_shuf, cfg2)
    assert meta["source_run"] == "standalone"
    assert meta["cv_strategy"] == "stratified"

    # Also verify make_folds itself is row-order invariant
    f_direct, _ = make_folds(df_shuf, cfg)
    m1 = dict(zip(f_loaded["id"], f_loaded["fold"]))
    m2 = dict(zip(f_direct["id"], f_direct["fold"]))
    assert m1 == m2

    df_bad = df.iloc[1:].copy()
    with pytest.raises(FoldMismatchError, match=r"\[Perbaikan\]"):
        get_folds(df_bad, cfg2)


def test_debug_subsample_preserves_cv_validity():
    # Rare class binary
    df_rare = pd.DataFrame({"id": range(500), "y": [1, 1, 1] + [0] * 497})
    cfg_bin = BaseConfig(task="binary", target="y", metric="auc", debug=True)
    cfg_bin.resolve()
    sub_bin = debug_subsample(df_rare, cfg_bin, max_rows=50)
    assert len(sub_bin) <= 50
    assert (sub_bin["y"] == 1).sum() >= cfg_bin.n_folds
    get_folds(sub_bin, cfg_bin)  # must not raise DataError

    # Group CV keeps groups intact
    df_grp = pd.DataFrame({"id": range(100), "g": np.repeat(range(20), 5), "y": range(100)})
    cfg_grp = BaseConfig(task="regression", target="y", metric="rmse", group_col="g", debug=True)
    cfg_grp.resolve()
    sub_grp = debug_subsample(df_grp, cfg_grp, max_rows=30)
    assert len(sub_grp) <= 30
    assert set(sub_grp.groupby("g").size().unique()) == {5}
    get_folds(sub_grp, cfg_grp)

    # Stratified group CV keeps groups intact and classes represented
    df_sg = pd.DataFrame({
        "id": range(100),
        "g": np.repeat(range(20), 5),
        "y": np.repeat([0, 1] * 10, 5),
    })
    cfg_sg = BaseConfig(task="binary", target="y", metric="auc", group_col="g", debug=True)
    cfg_sg.resolve()
    sub_sg = debug_subsample(df_sg, cfg_sg, max_rows=30)
    assert len(sub_sg) <= 30
    assert set(sub_sg.groupby("g").size().unique()) == {5}
    get_folds(sub_sg, cfg_sg)

    # Timeseries takes contiguous tail of unique periods
    df_ts = _make_panel_df()
    cfg_ts = BaseConfig(
        task="regression",
        target="y",
        metric="rmse",
        cv_strategy="timeseries",
        time_col="dt",
        debug=True,
    )
    cfg_ts.resolve()
    sub_ts = debug_subsample(df_ts, cfg_ts, max_rows=15)
    assert sub_ts["dt"].max() == df_ts["dt"].max()
    assert (sub_ts.groupby("dt").size() == 3).all()
    get_folds(sub_ts, cfg_ts)


def test_cv_validation_errors():
    # NaN in id_col checked BEFORE duplicate check
    df_nan_id = pd.DataFrame({"id": [1.0, np.nan, np.nan, 4.0], "y": [0, 1, 0, 1]})
    cfg = BaseConfig(task="binary", target="y", metric="auc", n_folds=2)
    with pytest.raises(DataError, match=r"(?is)nan.*\[Perbaikan\]"):
        make_folds(df_nan_id, cfg)

    # Duplicate IDs
    df_dup = pd.DataFrame({"id": [1, 1, 2, 3], "y": [0, 1, 0, 1]})
    with pytest.raises(DataError, match=r"\[Perbaikan\]"):
        make_folds(df_dup, cfg)

    # Rare class < n_folds
    df_rare = pd.DataFrame({"id": range(10), "y": [1] + [0] * 9})
    cfg3 = BaseConfig(task="binary", target="y", metric="auc", n_folds=3)
    with pytest.raises(DataError, match=r"\[Perbaikan\]"):
        make_folds(df_rare, cfg3)

    # Insufficient groups < n_folds
    df_grp = pd.DataFrame({"id": range(6), "g": [0, 0, 0, 1, 1, 1], "y": range(6)})
    cfg_grp = BaseConfig(task="regression", target="y", metric="rmse", group_col="g", n_folds=3)
    with pytest.raises(DataError, match=r"\[Perbaikan\]"):
        make_folds(df_grp, cfg_grp)

    # Insufficient unique timestamps < n_folds + 1
    df_ts = pd.DataFrame({
        "id": range(6),
        "dt": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"]),
        "y": range(6),
    })
    cfg_ts = BaseConfig(task="regression", target="y", metric="rmse", cv_strategy="timeseries", time_col="dt", n_folds=3)
    with pytest.raises(DataError, match=r"\[Perbaikan\]"):
        make_folds(df_ts, cfg_ts)

    # Empty train in split_fold due to ts_gap too large
    df_panel = _make_panel_df()
    cfg_gap = BaseConfig(
        task="regression",
        target="y",
        metric="rmse",
        cv_strategy="timeseries",
        time_col="dt",
        n_folds=3,
        ts_gap=10,
    )
    folds_df, _ = get_folds(df_panel, cfg_gap)
    with pytest.raises(DataError, match=r"\[Perbaikan\]"):
        split_fold(df_panel, folds_df, 0, cfg_gap)
