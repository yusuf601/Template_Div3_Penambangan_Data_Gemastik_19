import numpy as np
import pandas as pd
import pytest
from utils.core.config import BaseConfig
from utils.core.cv import make_folds_file
from utils.core.ensemble import EnsembleData, load_runs, blend
from utils.core.errors import ConfigError, FoldMismatchError
from tests.core.helpers import (
    SklearnBinaryPipe,
    SklearnRegPipe,
    make_bin_df,
)


def _make_ts_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = np.repeat(np.arange(1, 7), 8)
    n = len(dates)
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.2, size=n)
    tr = pd.DataFrame({"id": range(n), "date": dates, "f1": x, "y": y})
    te = pd.DataFrame({"id": range(200, 208), "date": [7] * 8, "f1": rng.normal(size=8)})
    return tr, te


def test_blend_mean_and_weights_and_mismatch_error(tmp_path):
    tr, te = make_bin_df(60)
    folds_csv = make_folds_file(tr, BaseConfig(task="binary", target="y", metric="auc", n_folds=3), out_dir=tmp_path / "folds")

    r1 = SklearnBinaryPipe(BaseConfig(run_name="m1", task="binary", target="y", metric="auc",
                                      n_folds=3, seed=1, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr, te)
    r2 = SklearnBinaryPipe(BaseConfig(run_name="m2", task="binary", target="y", metric="auc",
                                      n_folds=3, seed=2, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr, te)

    ens = load_runs([r1.run_dir, r2.run_dir])
    b_mean = blend(ens, method="mean", runs_dir=tmp_path / "runs", run_name="ens_mean")
    b_w = blend(ens, method="weights", runs_dir=tmp_path / "runs", run_name="ens_weights")

    assert b_mean.scores["cv"] == pytest.approx(b_mean.scores["cv_raw"])
    assert b_mean.scores["cv_post"] == pytest.approx(b_mean.scores["cv_raw"])
    assert b_mean.scores["cv_honest"] == pytest.approx(b_mean.scores["cv_raw"])
    assert "weights" in b_w.cfg.extra
    assert pytest.approx(sum(b_w.cfg.extra["weights"])) == 1.0
    assert len(b_w.test_pred) == len(te)
    assert b_w.scores["data_hash"] == ens.folds_hash
    assert (b_w.run_dir / "summary.md").exists()

    # Mismatched folds_hash raises FoldMismatchError with [Perbaikan]
    r_bad = SklearnBinaryPipe(BaseConfig(run_name="m_bad", task="binary", target="y", metric="auc",
                                         fold_seed=999, n_folds=3, runs_dir=str(tmp_path / "runs"))).run(tr, te)
    with pytest.raises(FoldMismatchError, match=r"\[Perbaikan\]"):
        load_runs([r1.run_dir, r_bad.run_dir])


def test_blend_partial_runs_and_timeseries_nan(tmp_path):
    # Partial run handling
    tr, te = make_bin_df(60)
    folds_csv = make_folds_file(tr, BaseConfig(task="binary", target="y", metric="auc", n_folds=3), out_dir=tmp_path / "folds")

    r_ok = SklearnBinaryPipe(BaseConfig(run_name="m_ok", task="binary", target="y", metric="auc",
                                        n_folds=3, seed=1, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr, te)

    p_part = SklearnBinaryPipe(BaseConfig(run_name="m_part", task="binary", target="y", metric="auc",
                                          n_folds=3, seed=2, max_minutes=0.01, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs")))
    ticks = iter([0.0, 0.1, 1.0, 100.0, 200.0])
    p_part._clock = lambda: next(ticks, 200.0)
    r_part = p_part.run(tr, te)
    assert r_part.scores["status"] == "partial"

    with pytest.raises(ConfigError, match=r"allow_partial.*\[Perbaikan\]"):
        load_runs([r_ok.run_dir, r_part.run_dir])

    ens_part = load_runs([r_ok.run_dir, r_part.run_dir], allow_partial=True)
    assert ens_part.has_partial is True
    b_part = blend(ens_part, method="weights", runs_dir=tmp_path / "runs", run_name="ens_part")
    assert b_part.scores["status"] == "partial"
    assert (b_part.run_dir / "PARTIAL").exists()
    assert b_part.scores["n_rows_eval"] == r_part.scores["n_rows_eval"]

    # Timeseries fold -1 NaN and eval_folds=[1, 2] handling
    tr_ts, te_ts = _make_ts_df()
    r_ts1 = SklearnRegPipe(BaseConfig(run_name="ts1", task="regression", target="y", metric="rmse",
                                      cv_strategy="timeseries", time_col="date", n_folds=3, seed=1, runs_dir=str(tmp_path / "runs_ts"))).run(tr_ts, te_ts)
    r_ts2 = SklearnRegPipe(BaseConfig(run_name="ts2", task="regression", target="y", metric="rmse",
                                      cv_strategy="timeseries", time_col="date", n_folds=3, seed=2, runs_dir=str(tmp_path / "runs_ts"))).run(tr_ts, te_ts)
    ens_ts = load_runs([r_ts1.run_dir, r_ts2.run_dir])
    b_ts = blend(ens_ts, method="weights", runs_dir=tmp_path / "runs_ts", run_name="ens_ts")
    assert b_ts.scores["status"] == "ok"
    assert not np.isnan(b_ts.scores["cv"])
    assert b_ts.scores["eval_folds"] == [1, 2]


def test_blend_postprocess_and_validation_errors(tmp_path):
    tr, te = make_bin_df(60)
    folds_csv = make_folds_file(tr, BaseConfig(task="binary", target="y", metric="f1", n_folds=3), out_dir=tmp_path / "folds")

    r1 = SklearnBinaryPipe(BaseConfig(run_name="f1_1", task="binary", target="y", metric="f1",
                                      n_folds=3, seed=1, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr, te)
    # Shuffle train rows in r2 to verify canon_id alignment
    tr_shuf = tr.sample(frac=1.0, random_state=99).reset_index(drop=True)
    te_shuf = te.sample(frac=1.0, random_state=99).reset_index(drop=True)
    r2 = SklearnBinaryPipe(BaseConfig(run_name="f1_2", task="binary", target="y", metric="f1",
                                      n_folds=3, seed=2, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr_shuf, te_shuf)

    ens = load_runs([r1.run_dir, r2.run_dir])
    calls = []
    def pp(oof_df, test_df=None):
        calls.append(test_df is not None)
        return {"threshold": 0.45}

    b_pp = blend(ens, method="mean", postprocess=pp, runs_dir=tmp_path / "runs", run_name="ens_pp")
    assert calls[0] is True
    assert all(c is False for c in calls[1:])
    assert b_pp.params == {"threshold": 0.45}

    # Test ID mismatch raises FoldMismatchError
    te_diff = te.copy()
    te_diff["id"] = te_diff["id"] + 1000
    r_bad_te = SklearnBinaryPipe(BaseConfig(run_name="bad_te", task="binary", target="y", metric="f1",
                                            n_folds=3, seed=1, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr, te_diff)
    with pytest.raises(FoldMismatchError, match=r"\[Perbaikan\]"):
        load_runs([r1.run_dir, r_bad_te.run_dir])

    # Metric mismatch raises ConfigError
    r_auc = SklearnBinaryPipe(BaseConfig(run_name="auc_1", task="binary", target="y", metric="auc",
                                         n_folds=3, seed=1, folds_path=str(folds_csv), runs_dir=str(tmp_path / "runs"))).run(tr, te)
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        load_runs([r1.run_dir, r_auc.run_dir])
