"""Tests for BasePipeline, RunResult.make_submission, and compute_scores."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tests.core.helpers import FakeClock, SklearnBinaryPipe, SklearnRegPipe, make_bin_df
from utils.core.config import BaseConfig
from utils.core.errors import ConfigError, DataError, HookContractError, SubmissionError
from utils.core.report import compare_runs


def test_default_sklearn_fit_predict_and_submission_alignment(tmp_path):
    tr, te = make_bin_df()
    cfg = BaseConfig(task="binary", target="y", metric="f1", n_folds=3, runs_dir=str(tmp_path))
    res = SklearnBinaryPipe(cfg).run(tr, te)
    # Without postprocess override, cv == cv_honest == cv_post == cv_raw
    assert res.scores["cv"] == pytest.approx(res.scores["cv_raw"])
    assert res.scores["cv_post"] == pytest.approx(res.scores["cv_raw"])
    assert res.scores["status"] == "ok"
    assert res.scores["test_pred_source"] == "fold_avg"

    # Reversed sample_sub ID order must be preserved
    sample_sub = pd.DataFrame({"id": list(reversed(te["id"].tolist())), "y": 0})
    sub = res.make_submission(sample_sub, tmp_path / "sub.csv")
    assert list(sub["id"]) == list(sample_sub["id"])

    bad_sub = pd.DataFrame({"id": [99999], "y": [0]})
    with pytest.raises(SubmissionError, match=r"\[Perbaikan\]"):
        res.make_submission(bad_sub, tmp_path / "bad.csv")


def test_hook_contract_errors_and_target_mutation_warning(tmp_path):
    tr, te = make_bin_df(30)

    class DropRowPipe(SklearnBinaryPipe):
        def build_features(self, train, test):
            return train.iloc[1:].copy(), test

    with pytest.raises(HookContractError, match="build_features"):
        DropRowPipe(
            BaseConfig(task="binary", target="y", metric="auc", n_folds=3, runs_dir=str(tmp_path))
        ).run(tr, te)

    class MutateTargetPipe(SklearnBinaryPipe):
        def build_features(self, train, test):
            train = train.copy()
            train["y"] = 1 - train["y"]
            return train, test

    with pytest.warns(UserWarning, match="snapshot"):
        res_mut = MutateTargetPipe(
            BaseConfig(task="binary", target="y", metric="auc", n_folds=3, runs_dir=str(tmp_path))
        ).run(tr, te)
    # OOF target column always comes from pre-build_features snapshot
    assert list(res_mut.oof["y"]) == list(tr["y"])


def test_partial_run_and_resume_restores_fold_info_and_importances(tmp_path):
    tr, te = make_bin_df(40)
    clock = FakeClock()

    class ClockedIterPipe(SklearnBinaryPipe):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._clock = clock

        def fit_predict_fold(self, model, tr_df, va_df, te_df, fold):
            clock.advance(60.0)  # each fold takes 60s (1.0 min) on the fake clock
            model, va_p, te_p = super().fit_predict_fold(model, tr_df, va_df, te_df, fold)
            if fold >= 0:
                # fold_info is a defaultdict(dict), so direct key assignment never KeyErrors
                self.fold_info[fold]["best_iteration"] = 100 + fold * 20
                self.fold_info[fold]["importances"] = {"f1": float(fold + 1)}
            return model, va_p, te_p

    # max_minutes=0.5 (30s): fold 0 always runs (takes 60s), then before fold 1 elapsed=60s >= 30s -> stops!
    c_part = BaseConfig(
        run_name="res_exp",
        task="binary",
        target="y",
        metric="auc",
        n_folds=3,
        test_mode="refit_full",
        max_minutes=0.5,
        runs_dir=str(tmp_path),
    )
    r_part = ClockedIterPipe(c_part).run(tr, te)
    assert r_part.scores["status"] == "partial"
    assert r_part.scores["n_folds_done"] == 1
    assert r_part.scores["test_pred_source"] == "fold_avg_fallback"

    # Resume with max_minutes=10.0 -> loads fold 0, runs folds 1 & 2, and refits using restored fold_info
    c_res = BaseConfig(
        run_name="res_exp",
        task="binary",
        target="y",
        metric="auc",
        n_folds=3,
        test_mode="refit_full",
        max_minutes=10.0,
        resume_from=str(r_part.run_dir),
        runs_dir=str(tmp_path),
    )
    pipe_res = ClockedIterPipe(c_res)
    r_full = pipe_res.run(tr, te)
    assert r_full.scores["status"] == "ok"
    assert r_full.scores["n_folds_done"] == 3
    assert r_full.scores["test_pred_source"] == "refit"
    assert pipe_res.refit_iterations() == round(np.mean([100, 120, 140]) * 1.1)
    assert pipe_res.importances is not None and "f1" in pipe_res.importances.index
    assert r_full.scores["minutes"] == pytest.approx(4.0)  # 3 folds * 60s + 1 refit * 60s = 240s = 4.0m


def test_timeseries_skipping_refit_keeps_status_ok(tmp_path):
    n = 36
    tr = pd.DataFrame(
        {
            "id": range(n),
            "dt": pd.date_range("2026-01-01", periods=n, freq="D"),
            "f1": np.linspace(0, 1, n),
            "y": np.linspace(1, 2, n),
        }
    )
    te = pd.DataFrame(
        {
            "id": range(100, 104),
            "dt": pd.date_range("2026-02-10", periods=4, freq="D"),
            "f1": [1.1, 1.2, 1.3, 1.4],
        }
    )
    clock = FakeClock()

    class ClockedTSPipe(SklearnRegPipe):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._clock = clock

        def fit_predict_fold(self, model, tr_df, va_df, te_df, fold):
            # Folds 0, 1, 2 take 10s, 20s, 30s (total 60s = 1.0m); max(duration_sec) = 30s
            clock.advance(10.0 * (fold + 1) if fold >= 0 else 30.0)
            return super().fit_predict_fold(model, tr_df, va_df, te_df, fold)

    # max_minutes=1.5 (90s): all 3 folds finish at 60s (status="ok"), remaining=30s < 1.25 * 30s (37.5s) -> skip refit!
    cfg = BaseConfig(
        run_name="ts_norefit",
        task="regression",
        target="y",
        metric="rmse",
        cv_strategy="timeseries",
        time_col="dt",
        n_folds=3,
        max_minutes=1.5,
        runs_dir=str(tmp_path),
    )
    with pytest.warns(UserWarning, match="refit_full"):
        res = ClockedTSPipe(cfg).run(tr, te)

    assert res.scores["status"] == "ok"
    assert res.scores["test_pred_source"] == "fold_avg_fallback"
    assert res.scores["eval_folds"] == [1, 2]
    comp = compare_runs(tmp_path, include_partial=False)
    assert res.run_dir.name in comp["run_dir"].values


def test_multiclass_submission_and_refit_scaler(tmp_path):
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    labels = np.array(["cat", "dog", "bird"])[(np.abs(x1 * 3).astype(int)) % 3]
    tr = pd.DataFrame({"id": range(n), "f1": x1, "f2": x2, "y": labels})
    te = pd.DataFrame({"id": range(200, 210), "f1": rng.normal(size=10), "f2": rng.normal(size=10)})

    class ScaledMulticlassPipe(SklearnBinaryPipe):
        model_name = "mc_logreg"

        def fit_transform_fold(self, tr_df, va_df, te_df, fold):
            scaler = StandardScaler()
            cols = ["f1", "f2"]
            tr_df = tr_df.copy()
            te_df = te_df.copy()
            tr_df[cols] = scaler.fit_transform(tr_df[cols])
            te_df[cols] = scaler.transform(te_df[cols])
            if len(va_df) > 0:
                va_df = va_df.copy()
                va_df[cols] = scaler.transform(va_df[cols])
            return tr_df, va_df, te_df

    cfg = BaseConfig(
        task="multiclass",
        target="y",
        metric="f1_macro",
        n_folds=3,
        test_mode="refit_full",
        runs_dir=str(tmp_path),
    )
    res = ScaledMulticlassPipe(cfg).run(tr, te)
    assert res.classes == ["bird", "cat", "dog"]
    assert res.scores["test_pred_source"] == "refit"

    # Single target col submission -> maps argmax to class label
    sub_1col = pd.DataFrame({"id": te["id"], "y": ""})
    out_1 = res.make_submission(sub_1col, tmp_path / "sub_1col.csv")
    assert set(out_1["y"]).issubset({"bird", "cat", "dog"})

    # K-column matching class names in custom order
    sub_k_named = pd.DataFrame({"id": te["id"], "dog": 0.0, "bird": 0.0, "cat": 0.0})
    out_k_named = res.make_submission(sub_k_named, tmp_path / "sub_k_named.csv")
    assert np.allclose(out_k_named["bird"], res.test_pred["pred_0"])
    assert np.allclose(out_k_named["cat"], res.test_pred["pred_1"])
    assert np.allclose(out_k_named["dog"], res.test_pred["pred_2"])

    # K-column with arbitrary column names -> positional fill + UserWarning
    sub_k_pos = pd.DataFrame({"id": te["id"], "c0": 0.0, "c1": 0.0, "c2": 0.0})
    with pytest.warns(UserWarning, match="classes"):
        out_k_pos = res.make_submission(sub_k_pos, tmp_path / "sub_k_pos.csv")
    assert np.allclose(out_k_pos["c0"], res.test_pred["pred_0"])


def test_postprocess_honest_vs_post_and_feature_validation(tmp_path):
    tr, te = make_bin_df(60)
    calls = []

    class PostPipe(SklearnBinaryPipe):
        def postprocess(self, oof_df, test_df=None):
            calls.append(test_df is not None)
            return {"threshold": 0.35}

    cfg = BaseConfig(task="binary", target="y", metric="f1", n_folds=3, runs_dir=str(tmp_path))
    res = PostPipe(cfg).run(tr, te)
    assert calls[0] is True  # first full pass has test_df != None
    assert calls[1:] == [False, False, False]  # per-fold honest passes have test_df is None
    assert res.params == {"threshold": 0.35}
    assert res.scores["cv"] == pytest.approx(res.scores["cv_honest"])

    # Non-numeric feature column raises HookContractError
    class StringFeatPipe(SklearnBinaryPipe):
        def build_features(self, train, test):
            train = train.copy()
            test = test.copy()
            train["bad_str"] = "abc"
            test["bad_str"] = "abc"
            return train, test

    with pytest.raises(HookContractError, match="bad_str"):
        StringFeatPipe(cfg).run(tr, te)
    # Verify FAILED file written in run_dir
    failed_files = list(tmp_path.glob("*/FAILED"))
    assert len(failed_files) >= 1


def test_additional_pipeline_guards_debug_and_submission_nan(tmp_path):
    tr, te = make_bin_df(60)

    # 1. fit_transform_fold dropping va rows -> HookContractError
    class DropVaPipe(SklearnBinaryPipe):
        def fit_transform_fold(self, tr_df, va_df, te_df, fold):
            return tr_df, va_df.iloc[1:].copy(), te_df

    with pytest.raises(HookContractError, match="va"):
        DropVaPipe(
            BaseConfig(task="binary", target="y", metric="auc", n_folds=3, runs_dir=str(tmp_path))
        ).run(tr, te)

    # 2. make_submission checks NaN id_col before duplicate/set comparison
    res = SklearnBinaryPipe(
        BaseConfig(task="binary", target="y", metric="auc", n_folds=3, runs_dir=str(tmp_path))
    ).run(tr, te)
    nan_sub = pd.DataFrame({"id": [np.nan, np.nan] + te["id"].iloc[2:].tolist(), "y": 0.0})
    with pytest.raises(SubmissionError, match="NaN"):
        res.make_submission(nan_sub, tmp_path / "nan_sub.csv")

    # 3. resume_from rejects hyperparameter/config mismatch with ConfigError
    bad_resume_cfg = BaseConfig(
        task="binary",
        target="y",
        metric="auc",
        n_folds=3,
        seed=999,
        resume_from=str(res.run_dir),
        runs_dir=str(tmp_path),
    )
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        SklearnBinaryPipe(bad_resume_cfg).run(tr, te)

    # 4. debug=True runs in _debug folder and skips leaderboard
    dbg_cfg = BaseConfig(
        task="binary", target="y", metric="auc", debug=True, runs_dir=str(tmp_path / "dbg_runs")
    )
    dbg_res = SklearnBinaryPipe(dbg_cfg).run(tr, te)
    assert "_debug" in str(dbg_res.run_dir)
    assert not (tmp_path / "dbg_runs" / "leaderboard.csv").exists()

    # 5. regression QWK uses global clip_range
    tr_qwk = pd.DataFrame(
        {"id": range(30), "f1": np.linspace(0, 3, 30), "y": [0, 1, 2, 3, 4] * 6}
    )
    te_qwk = pd.DataFrame({"id": range(100, 105), "f1": [0.5, 1.0, 1.5, 2.0, 2.5]})
    qwk_res = SklearnRegPipe(
        BaseConfig(
            task="regression", target="y", metric="qwk", n_folds=3, runs_dir=str(tmp_path)
        )
    ).run(tr_qwk, te_qwk)
    assert qwk_res.scores["status"] == "ok"
