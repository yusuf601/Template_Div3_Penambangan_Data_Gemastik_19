import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.core.config import BaseConfig
from utils.core.cv import get_folds, make_folds_file, split_fold
from utils.core.ensemble import blend, load_runs
from utils.core.errors import (
    ConfigError,
    DataError,
    FoldMismatchError,
    HookContractError,
    SubmissionError,
)
from utils.core.pipeline import BasePipeline
from utils.core.run import sync_leaderboard, update_lb_score
from tests.core.helpers import FakeClock, SklearnBinaryPipe, SklearnRegPipe, make_bin_df


def test_honest_scoring_source_folds_and_postprocess_none(tmp_path):
    tr, te = make_bin_df(45)
    calls_strat = []

    class RecordingPostPipe(SklearnBinaryPipe):
        def postprocess(self, oof_df, test_df=None):
            calls_strat.append((set(oof_df["fold"].dropna().astype(int).unique()), test_df is None))
            return {"threshold": 0.55}

    cfg_s = BaseConfig(task="binary", target="y", metric="f1", n_folds=3, runs_dir=str(tmp_path / "s"))
    RecordingPostPipe(cfg_s).run(tr, te)

    # Full OOF call has test_df is not None; 3 honest calls have test_df is None and source folds != k
    full_calls = [fset for fset, is_none in calls_strat if not is_none]
    honest_calls = [fset for fset, is_none in calls_strat if is_none]
    assert full_calls == [{0, 1, 2}]
    assert honest_calls == [{1, 2}, {0, 2}, {0, 1}]

    # Timeseries honest pass must only use past folds strictly < k (including fold 0)
    n = 36
    tr_ts = pd.DataFrame({
        "id": range(n),
        "dt": pd.date_range("2026-01-01", periods=n, freq="D"),
        "f1": np.linspace(-1, 1, n),
        "y": ([0, 1] * (n // 2)),
    })
    te_ts = pd.DataFrame({
        "id": range(100, 104),
        "dt": pd.date_range("2026-02-10", periods=4, freq="D"),
        "f1": [0.1, 0.2, 0.3, 0.4],
    })
    calls_ts = []

    class RecordingTSPostPipe(SklearnBinaryPipe):
        def postprocess(self, oof_df, test_df=None):
            calls_ts.append((set(oof_df["fold"].dropna().astype(int).unique()), test_df is None))
            return {"threshold": 0.45}

    cfg_ts = BaseConfig(task="binary", target="y", metric="f1", cv_strategy="timeseries",
                        time_col="dt", n_folds=3, test_mode="fold_avg", runs_dir=str(tmp_path / "ts"))
    RecordingTSPostPipe(cfg_ts).run(tr_ts, te_ts)
    full_ts = [fset for fset, is_none in calls_ts if not is_none]
    honest_ts = [fset for fset, is_none in calls_ts if is_none]
    assert full_ts == [{0, 1, 2}]
    assert honest_ts == [{0}, {0, 1}]  # k=1 uses {0}, k=2 uses {0, 1}

    # postprocess returning None is treated as {}
    class NonePostPipe(SklearnBinaryPipe):
        def postprocess(self, oof_df, test_df=None):
            return None

    res_none = NonePostPipe(BaseConfig(task="binary", target="y", metric="f1",
                                       n_folds=3, runs_dir=str(tmp_path / "none"))).run(tr, te)
    assert res_none.params == {}
    assert res_none.scores["cv"] == pytest.approx(res_none.scores["cv_raw"])
    assert res_none.scores["cv_post"] == pytest.approx(res_none.scores["cv_raw"])


def test_multiclass_string_labels_alignment(tmp_path):
    rng = np.random.default_rng(7)
    labels = ["cat", "dog", "bird"]
    y = np.array(labels * 20)
    f1 = np.where(y == "bird", -3.0, np.where(y == "cat", 0.0, 3.0)) + rng.normal(scale=0.2, size=60)
    tr = pd.DataFrame({"id": range(60), "f1": f1, "y": y})
    te = pd.DataFrame({"id": [201, 202, 203], "f1": [-3.0, 0.0, 3.0]})

    class MultiPipe(BasePipeline):
        model_name = "logreg_multi"
        def build_model(self, fold):
            return LogisticRegression(C=10.0)

    cfg = BaseConfig(task="multiclass", target="y", metric="f1_macro", n_folds=3, runs_dir=str(tmp_path))
    res = MultiPipe(cfg).run(tr, te)

    assert res.classes == ["bird", "cat", "dog"]
    assert list(res.oof.columns) == ["id", "fold", "y", "pred_0", "pred_1", "pred_2"]
    assert list(res.test_pred.columns) == ["id", "pred_0", "pred_1", "pred_2"]
    assert (res.run_dir / "plots" / "confusion_matrix.png").exists()

    # Label submission maps argmax back to string labels ["bird", "cat", "dog"]
    sub_lbl_tpl = pd.DataFrame({"id": [203, 201, 202], "y": ""})
    sub_lbl = res.make_submission(sub_lbl_tpl, tmp_path / "sub_lbl.csv")
    assert list(sub_lbl["id"]) == [203, 201, 202]
    assert list(sub_lbl["y"]) == ["dog", "bird", "cat"]

    # Probability submission with SHUFFLED class-named columns ("cat", "dog", "bird" != res.classes)
    # must map by column name, verified via argmax (idxmax) rather than brittle probability threshold
    sub_prob_tpl = pd.DataFrame({"id": [201, 202, 203], "cat": 0.0, "dog": 0.0, "bird": 0.0})
    sub_prob = res.make_submission(sub_prob_tpl, tmp_path / "sub_prob.csv")
    assert list(sub_prob.columns) == ["id", "cat", "dog", "bird"]
    assert sub_prob[["cat", "dog", "bird"]].idxmax(axis=1).tolist() == ["bird", "cat", "dog"]

    # Probability submission with non-matching K column names falls back to positional order with a warning
    sub_pos_tpl = pd.DataFrame({"id": [201, 202, 203], "p0": 0.0, "p1": 0.0, "p2": 0.0})
    with pytest.warns(UserWarning):
        sub_pos = res.make_submission(sub_pos_tpl, tmp_path / "sub_pos.csv")
    assert sub_pos[["p0", "p1", "p2"]].values.argmax(axis=1).tolist() == [0, 1, 2]


def test_refit_contract_empty_va_and_va_row_drop_error(tmp_path):
    tr, te = make_bin_df(45)
    seen_calls = []

    class ScalerRefitPipe(SklearnBinaryPipe):
        def fit_transform_fold(self, tr_df, va_df, te_df, fold):
            seen_calls.append((fold, len(va_df)))
            scaler = StandardScaler()
            tr_df = tr_df.copy(); va_df = va_df.copy(); te_df = te_df.copy()
            tr_df[["f1"]] = scaler.fit_transform(tr_df[["f1"]])
            if len(va_df) > 0:
                va_df[["f1"]] = scaler.transform(va_df[["f1"]])
            te_df[["f1"]] = scaler.transform(te_df[["f1"]])
            return tr_df, va_df, te_df

    cfg = BaseConfig(task="binary", target="y", metric="auc", n_folds=3,
                     test_mode="refit_full", runs_dir=str(tmp_path / "ok"))
    res = ScalerRefitPipe(cfg).run(tr, te)
    assert (-1, 0) in seen_calls
    assert res.scores["test_pred_source"] == "refit"

    class DropVaPipe(SklearnBinaryPipe):
        def fit_transform_fold(self, tr_df, va_df, te_df, fold):
            return tr_df, va_df.iloc[1:].copy(), te_df

    with pytest.raises(HookContractError, match=r"\[Perbaikan\]"):
        DropVaPipe(BaseConfig(task="binary", target="y", metric="auc", n_folds=3,
                              runs_dir=str(tmp_path / "bad"))).run(tr, te)


def test_plot_failure_emits_warning_without_failing_run(tmp_path, monkeypatch):
    import matplotlib.figure
    tr, te = make_bin_df(30)

    def _boom_savefig(self, *args, **kwargs):
        raise RuntimeError("simulated matplotlib savefig failure")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _boom_savefig)
    cfg = BaseConfig(task="binary", target="y", metric="auc", n_folds=2, runs_dir=str(tmp_path))
    with pytest.warns(UserWarning):
        res = SklearnBinaryPipe(cfg).run(tr, te)
    assert res.scores["status"] == "ok"
    assert (res.run_dir / "summary.md").exists()


def test_debug_run_and_failed_run_excluded_from_leaderboard(tmp_path):
    from utils.core.io import load_json
    tr, te = make_bin_df(60)
    runs_dir = tmp_path / "runs"

    # Debug run goes to runs/_debug/ and never creates/updates leaderboard.csv
    res_dbg = SklearnBinaryPipe(BaseConfig(run_name="dbg", task="binary", target="y",
                                           metric="auc", debug=True, runs_dir=str(runs_dir))).run(tr, te)
    assert res_dbg.run_dir.parent.name == "_debug"
    assert not (runs_dir / "leaderboard.csv").exists()

    # Failing run with spaces/dots in owner, modality, and run_name sanitizes folder name for Windows while keeping config.json intact
    class BoomPipe(SklearnBinaryPipe):
        def fit_predict_fold(self, model, tr_df, va_df, te_df, fold):
            raise RuntimeError("gpu out of memory simulation")

    with pytest.raises(RuntimeError, match="gpu out of memory simulation"):
        BoomPipe(BaseConfig(run_name=" crash ", owner="my laptop.", modality=" tabular ", task="binary", target="y",
                            metric="auc", n_folds=2, runs_dir=str(runs_dir))).run(tr, te)

    failed_files = list(runs_dir.glob("*/FAILED"))
    assert len(failed_files) == 1
    failed_dir = failed_files[0].parent
    assert failed_dir.name.endswith("_my-laptop_tabular_crash")
    assert " " not in failed_dir.name and "." not in failed_dir.name
    assert load_json(failed_dir / "config.json")["run_name"] == " crash "
    assert "gpu out of memory simulation" in failed_files[0].read_text()
    assert not (runs_dir / "leaderboard.csv").exists()


def test_cv_ts_gap_too_large_and_group_or_time_mutation_errors(tmp_path):
    df_ts = pd.DataFrame({
        "id": range(12),
        "dt": pd.date_range("2026-01-01", periods=12, freq="D"),
        "y": range(12),
    })
    cfg_ts = BaseConfig(task="regression", target="y", metric="rmse",
                        cv_strategy="timeseries", time_col="dt", n_folds=3, ts_gap=50)
    cfg_ts.resolve(); cfg_ts.validate()
    folds_ts, _ = get_folds(df_ts, cfg_ts)
    with pytest.raises(DataError, match=r"\[Perbaikan\]"):
        split_fold(df_ts, folds_ts, 1, cfg_ts)

    # Mutating group_col or time_col content must trigger FoldMismatchError on load_folds
    df_grp = pd.DataFrame({"id": range(20), "g": np.repeat(range(5), 4), "dt": range(20), "y": [0, 1] * 10})
    f_grp_path = make_folds_file(
        df_grp,
        BaseConfig(task="binary", target="y", metric="auc", cv_strategy="group", group_col="g", n_folds=3),
        out_dir=tmp_path / "f_grp",
    )
    df_grp_mod = df_grp.copy()
    df_grp_mod.loc[0, "g"] = 999
    with pytest.raises(FoldMismatchError, match=r"\[Perbaikan\]"):
        get_folds(df_grp_mod, BaseConfig(task="binary", target="y", metric="auc",
                                         cv_strategy="group", group_col="g", n_folds=3, folds_path=str(f_grp_path)))

    f_ts_path = make_folds_file(
        df_ts,
        BaseConfig(task="regression", target="y", metric="rmse", cv_strategy="timeseries", time_col="dt", n_folds=3),
        out_dir=tmp_path / "f_ts",
    )
    df_ts_mod = df_ts.copy()
    df_ts_mod.loc[0, "dt"] = pd.Timestamp("2030-01-01")
    with pytest.raises(FoldMismatchError, match=r"\[Perbaikan\]"):
        get_folds(df_ts_mod, BaseConfig(task="regression", target="y", metric="rmse",
                                        cv_strategy="timeseries", time_col="dt", n_folds=3, folds_path=str(f_ts_path)))


def test_resume_cross_machine_upserts_leaderboard_and_preserves_metadata(tmp_path):
    tr, te = make_bin_df(45)
    runs_dir = tmp_path / "runs"
    clock = FakeClock()

    class ClockedPipe(SklearnBinaryPipe):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._clock = clock
        def fit_predict_fold(self, model, tr_df, va_df, te_df, fold):
            clock.advance(60.0)
            return super().fit_predict_fold(model, tr_df, va_df, te_df, fold)

    r_part = ClockedPipe(BaseConfig(
        run_name="cross_pc", owner="laptop1", task="binary", target="y",
        metric="auc", n_folds=3, max_minutes=0.5, runs_dir=str(runs_dir),
    )).run(tr, te)

    lb_path = runs_dir / "leaderboard.csv"
    lb_part = pd.read_csv(lb_path)
    assert len(lb_part) == 1
    assert lb_part.loc[0, "status"] == "partial"
    assert (r_part.run_dir / "PARTIAL").exists()

    # Resume on "pcgpu": leaderboard updated in place (still 1 row), PARTIAL removed, original owner/timestamp/run_dir preserved
    r_full = ClockedPipe(BaseConfig(
        run_name="cross_pc", owner="pcgpu", task="binary", target="y",
        metric="auc", n_folds=3, max_minutes=10.0,
        resume_from=str(r_part.run_dir), runs_dir=str(runs_dir),
    )).run(tr, te)

    assert not (r_full.run_dir / "PARTIAL").exists()
    assert r_full.run_dir == r_part.run_dir
    assert r_full.scores["owner"] == "laptop1"
    assert r_full.scores["timestamp"] == r_part.scores["timestamp"]

    lb_full = pd.read_csv(lb_path)
    assert len(lb_full) == 1
    assert lb_full.loc[0, "status"] == "ok"
    assert lb_full.loc[0, "owner"] == "laptop1"
    assert lb_full.loc[0, "timestamp"] == r_part.scores["timestamp"]
    assert lb_full.loc[0, "run_dir"] == r_part.run_dir.name


def test_ensemble_leaderboard_row_and_sync_identity_and_missing_lb_error(tmp_path):
    tr, te = make_bin_df(60)
    runs_dir = tmp_path / "runs"
    folds_csv = make_folds_file(tr, BaseConfig(task="binary", target="y", metric="auc", n_folds=3),
                                out_dir=tmp_path / "folds")

    r1 = SklearnBinaryPipe(BaseConfig(run_name="m1", task="binary", target="y", metric="auc",
                                      n_folds=3, seed=1, folds_path=str(folds_csv), runs_dir=str(runs_dir))).run(tr, te)
    r2 = SklearnBinaryPipe(BaseConfig(run_name="m2", task="binary", target="y", metric="auc",
                                      n_folds=3, seed=2, folds_path=str(folds_csv), runs_dir=str(runs_dir))).run(tr, te)

    ens = load_runs([r1.run_dir, r2.run_dir])
    b_w = blend(ens, method="weights", runs_dir=runs_dir, run_name="ens_w")

    lb_path = runs_dir / "leaderboard.csv"
    lb_before = pd.read_csv(lb_path)
    ens_row = lb_before[lb_before["run_dir"] == b_w.run_dir.name].iloc[0]
    assert "blend" in str(ens_row["model"])
    assert ens_row["data_hash"] == ens_row["folds_hash"]

    # Deleting leaderboard.csv and running sync_leaderboard reproduces the exact table including ensemble run
    lb_path.unlink()
    sync_leaderboard(runs_dir=runs_dir)
    lb_after = pd.read_csv(lb_path)
    pd.testing.assert_frame_equal(lb_before, lb_after)

    # update_lb_score on non-existent run folder raises ConfigError with [Perbaikan]
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        update_lb_score("folder_tidak_ada", 0.91, runs_dir=runs_dir)


def test_id_nan_checks_in_train_test_and_sample_sub(tmp_path):
    tr, te = make_bin_df(30)
    cfg = BaseConfig(task="binary", target="y", metric="auc", n_folds=2, runs_dir=str(tmp_path))

    tr_nan = tr.copy()
    tr_nan["id"] = tr_nan["id"].astype(float)
    tr_nan.loc[0, "id"] = np.nan
    with pytest.raises(DataError, match=r"(?s)NaN.*\[Perbaikan\]"):
        SklearnBinaryPipe(cfg).run(tr_nan, te)

    te_nan = te.copy()
    te_nan["id"] = te_nan["id"].astype(float)
    te_nan.loc[0, "id"] = np.nan
    with pytest.raises(DataError, match=r"(?s)NaN.*\[Perbaikan\]"):
        SklearnBinaryPipe(cfg).run(tr, te_nan)

    res = SklearnBinaryPipe(cfg).run(tr, te)
    sub_nan = pd.DataFrame({"id": te["id"].astype(float), "y": 0.0})
    sub_nan.loc[0, "id"] = np.nan
    with pytest.raises(SubmissionError, match=r"(?s)NaN.*\[Perbaikan\]"):
        res.make_submission(sub_nan, tmp_path / "sub_nan.csv")


def test_non_numeric_and_missing_feature_columns_in_fit_predict_fold(tmp_path):
    tr, te = make_bin_df(30)

    # Boolean columns are valid numeric dtypes in default fit_predict_fold
    tr_bool = tr.copy(); te_bool = te.copy()
    tr_bool["flag"] = tr_bool["f1"] > 0
    te_bool["flag"] = te_bool["f1"] > 0
    res_bool = SklearnBinaryPipe(BaseConfig(task="binary", target="y", metric="auc",
                                            n_folds=2, runs_dir=str(tmp_path / "bool"))).run(tr_bool, te_bool)
    assert res_bool.scores["status"] == "ok"

    # Unencoded string feature column raises HookContractError naming the column
    tr_str = tr.copy(); te_str = te.copy()
    tr_str["kota"] = "jakarta"
    te_str["kota"] = "bandung"
    with pytest.raises(HookContractError, match=r"(?s)kota.*\[Perbaikan\]"):
        SklearnBinaryPipe(BaseConfig(task="binary", target="y", metric="auc",
                                     n_folds=2, runs_dir=str(tmp_path / "str"))).run(tr_str, te_str)

    # Dropping a feature column from te_df in fit_transform_fold raises HookContractError naming the column
    class DropTeColPipe(SklearnBinaryPipe):
        def fit_transform_fold(self, tr_df, va_df, te_df, fold):
            return tr_df, va_df, te_df.drop(columns=["f2"])

    with pytest.raises(HookContractError, match=r"(?s)f2.*\[Perbaikan\]"):
        DropTeColPipe(BaseConfig(task="binary", target="y", metric="auc",
                                 n_folds=2, runs_dir=str(tmp_path / "drop_te"))).run(
            tr.assign(f2=1.0), te.assign(f2=1.0)
        )


def test_blend_rejects_partial_by_default_and_marks_partial_when_allowed(tmp_path):
    tr, te = make_bin_df(45)
    runs_dir = tmp_path / "runs"
    folds_csv = make_folds_file(tr, BaseConfig(task="binary", target="y", metric="auc", n_folds=3),
                                out_dir=tmp_path / "folds")

    r_ok = SklearnBinaryPipe(BaseConfig(
        run_name="ok_run", task="binary", target="y", metric="auc",
        n_folds=3, folds_path=str(folds_csv), runs_dir=str(runs_dir),
    )).run(tr, te)

    clock = FakeClock()

    class SlowPipe(SklearnBinaryPipe):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._clock = clock
        def fit_predict_fold(self, model, tr_df, va_df, te_df, fold):
            clock.advance(60.0)
            return super().fit_predict_fold(model, tr_df, va_df, te_df, fold)

    r_part = SlowPipe(BaseConfig(
        run_name="part_run", task="binary", target="y", metric="auc",
        n_folds=3, max_minutes=0.5, folds_path=str(folds_csv), runs_dir=str(runs_dir),
    )).run(tr, te)
    assert r_part.scores["status"] == "partial"

    # Rejected by default
    with pytest.raises(ConfigError, match=r"(?s)allow_partial.*\[Perbaikan\]"):
        load_runs([r_ok.run_dir, r_part.run_dir])

    # Allowed when allow_partial=True, marking resulting ensemble as partial with intersection n_rows_eval
    ens_part = load_runs([r_ok.run_dir, r_part.run_dir], allow_partial=True)
    b_part = blend(ens_part, method="mean", runs_dir=runs_dir, run_name="ens_part")
    assert b_part.scores["status"] == "partial"
    assert (b_part.run_dir / "PARTIAL").exists()
    assert b_part.scores["n_rows_eval"] == r_part.scores["n_rows_eval"]
