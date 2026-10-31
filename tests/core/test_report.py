import warnings
import numpy as np
import pandas as pd
import pytest
from utils.core.config import BaseConfig
from utils.core.report import compare_runs, generate_report


def test_generate_report_and_compare_runs_filters_partial(tmp_path):
    run_dir = tmp_path / "runs" / "20261031_1400_gpu_tabular_lgbm"
    run_dir.mkdir(parents=True)
    cfg = BaseConfig(task="binary", target="y", metric="f1", notes="baseline test")
    cfg.resolve()
    oof = pd.DataFrame(
        {"id": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "y": [0, 1, 0, 1], "pred": [0.1, 0.8, 0.2, 0.9]}
    )
    scores = {
        "model": "lgbm",
        "timestamp": "2026-10-31T14:00:00+07:00",
        "owner": "gpu",
        "minutes": 0.5,
        "metric": "f1",
        "direction": "maximize",
        "per_fold": {
            "0": {"raw": 1.0, "n_rows": 2, "duration_sec": 15.0},
            "1": {"raw": 1.0, "n_rows": 2, "duration_sec": 15.0},
        },
        "cv": 1.0,
        "cv_raw": 1.0,
        "cv_post": 1.0,
        "cv_honest": 1.0,
        "cv_std": 0.0,
        "status": "ok",
        "test_pred_source": "fold_avg",
        "n_folds_done": 2,
        "n_rows_eval": 4,
    }
    imp = pd.Series({"feat_a": 10.0, "feat_b": 5.0})
    generate_report(run_dir, cfg, scores, oof, {"threshold": 0.5}, importances=imp)
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "plots" / "fold_scores.png").exists()
    assert (run_dir / "plots" / "confusion_matrix.png").exists()
    assert (run_dir / "plots" / "importance.png").exists()

    lb_df = pd.DataFrame(
        [
            {"run_dir": "r_ok", "metric": "f1", "cv": 0.85, "status": "ok"},
            {"run_dir": "r_part", "metric": "f1", "cv": 0.99, "status": "partial"},
        ]
    )
    lb_df.to_csv(tmp_path / "runs" / "leaderboard.csv", index=False)
    comp = compare_runs(tmp_path / "runs", include_partial=False)
    assert list(comp["run_dir"]) == ["r_ok"]


def test_generate_report_regression_and_plot_failure_warning(tmp_path):
    run_dir = tmp_path / "runs" / "reg_run"
    run_dir.mkdir(parents=True)
    cfg = BaseConfig(task="regression", target="y", metric="rmse")
    cfg.resolve()
    oof = pd.DataFrame(
        {"id": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "y": [1.0, 2.0, 3.0, 4.0], "pred": [1.1, 1.9, 3.2, 3.8]}
    )
    scores = {
        "model": "ridge",
        "timestamp": "2026-10-31T14:05:00+07:00",
        "owner": "gpu",
        "minutes": 0.2,
        "metric": "rmse",
        "direction": "minimize",
        "per_fold": {"0": {"raw": 0.1, "n_rows": 2, "duration_sec": 5.0}},
        "cv": 0.15,
        "cv_raw": 0.15,
        "cv_post": 0.15,
        "cv_honest": 0.15,
        "cv_std": 0.05,
        "status": "ok",
        "test_pred_source": "fold_avg",
        "n_folds_done": 2,
        "n_rows_eval": 4,
    }
    generate_report(run_dir, cfg, scores, oof, {})
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "plots" / "residuals.png").exists()

    # Malformed OOF emits UserWarning instead of crashing
    bad_oof = pd.DataFrame({"id": [1], "fold": [0]})  # missing target and pred
    with pytest.warns(UserWarning):
        generate_report(run_dir, cfg, scores, bad_oof, {})
