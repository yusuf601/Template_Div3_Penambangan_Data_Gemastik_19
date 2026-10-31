import numpy as np
import pandas as pd
import pytest
from utils.core.config import BaseConfig
from utils.core.cv import get_folds
from utils.core.errors import ConfigError
from utils.core.io import load_json
from utils.core.run import (
    LEADERBOARD_COLS,
    RunManager,
    safe_name,
    sync_leaderboard,
    update_lb_score,
)


def _dummy_scores(rm, meta, dv, status="ok", cv=0.85, n_folds_done=2):
    return {
        "model": rm.model_name,
        "timestamp": rm.timestamp,
        "owner": rm.cfg.owner,
        "minutes": 1.25,
        "data_hash": dv["raw_train_hash"],
        "metric": rm.cfg.metric,
        "direction": "maximize",
        "per_fold": {"0": {"raw": cv, "n_rows": 2, "duration_sec": 30.0}},
        "refit_duration_sec": 0.0,
        "cv": cv,
        "cv_raw": cv,
        "cv_post": cv,
        "cv_honest": cv,
        "cv_std": 0.01,
        "cv_raw_all": cv,
        "eval_folds": [0, 1],
        "n_rows_eval": 4,
        "folds_hash": meta["folds_hash"],
        "test_ids_hash": dv["test_ids_hash"],
        "classes": None,
        "status": status,
        "test_pred_source": "fold_avg",
        "n_folds_done": n_folds_done,
    }


def test_upsert_leaderboard_lb_json_and_identical_sync(tmp_path):
    runs_dir = tmp_path / "runs"
    cfg = BaseConfig(
        run_name="base",
        owner="yusuf",
        modality="tabular",
        task="binary",
        target="y",
        metric="f1",
        n_folds=2,
        runs_dir=str(runs_dir),
    )
    cfg.resolve()
    cfg.validate()
    rm = RunManager(cfg, model_name="lgbm")
    assert "_yusuf_tabular_base" in rm.run_dir.name
    assert "+07:00" in rm.timestamp

    tr = pd.DataFrame({"id": [1, 2, 3, 4], "f": [0.1, 0.2, 0.3, 0.4], "y": [0, 1, 0, 1]})
    te = pd.DataFrame({"id": [10, 11], "f": [0.5, 0.6]})
    dv = rm.save_data_version(tr, te)
    folds_df, meta = get_folds(tr[["id", "y"]], cfg, source_run=rm.run_dir.name)
    rm.save_folds(folds_df, meta)

    # First write as partial
    s_part = _dummy_scores(rm, meta, dv, status="partial", cv=0.80, n_folds_done=1)
    rm.set_status("partial")
    rm.save_scores(s_part)
    lb_path = rm.append_leaderboard(s_part)
    assert (rm.run_dir / "PARTIAL").exists()

    # Upsert as ok (simulating resume completion)
    s_ok = _dummy_scores(rm, meta, dv, status="ok", cv=0.88, n_folds_done=2)
    rm.set_status("ok")
    rm.save_scores(s_ok)
    rm.append_leaderboard(s_ok)
    assert not (rm.run_dir / "PARTIAL").exists()

    update_lb_score(rm.run_dir.name, 0.865, note="sub 1", runs_dir=runs_dir)
    assert (rm.run_dir / "lb.json").exists()

    # Add a second run to verify deterministic row ordering across append and sync
    cfg2 = BaseConfig(
        run_name="second",
        owner="pcgpu",
        modality="tabular",
        task="binary",
        target="y",
        metric="f1",
        n_folds=2,
        runs_dir=str(runs_dir),
    )
    cfg2.resolve()
    cfg2.validate()
    rm2 = RunManager(cfg2, model_name="xgb")
    dv2 = rm2.save_data_version(tr, te)
    rm2.save_folds(folds_df, meta)
    s2 = _dummy_scores(rm2, meta, dv2, status="ok", cv=0.91, n_folds_done=2)
    rm2.set_status("ok")
    rm2.save_scores(s2)
    rm2.append_leaderboard(s2)

    lb_before = pd.read_csv(lb_path)
    assert list(lb_before.columns) == LEADERBOARD_COLS
    assert len(lb_before) == 2  # first run upserted (not duplicated) + second run added
    row1 = lb_before[lb_before["run_dir"] == rm.run_dir.name].iloc[0]
    assert row1["cv"] == pytest.approx(0.88)
    assert row1["lb_score"] == pytest.approx(0.865)

    # Delete leaderboard.csv, rebuild via sync_leaderboard, and compare CSV-on-disk to CSV-on-disk
    lb_path.unlink()
    sync_leaderboard(runs_dir=runs_dir)
    lb_after = pd.read_csv(lb_path)
    pd.testing.assert_frame_equal(lb_before, lb_after)


def test_resume_validates_full_config_except_allowed_keys(tmp_path):
    runs_dir = tmp_path / "runs"
    c1 = BaseConfig(
        run_name="r1",
        owner="laptop1",
        task="binary",
        target="y",
        metric="auc",
        seed=42,
        max_minutes=5,
        runs_dir=str(runs_dir),
    )
    c1.resolve()
    c1.validate()
    rm1 = RunManager(c1, model_name="lgbm")

    # Resuming with larger max_minutes and different owner (different PC) succeeds
    c_ok = BaseConfig(
        run_name="r1",
        owner="pcgpu",
        task="binary",
        target="y",
        metric="auc",
        seed=42,
        max_minutes=30,
        resume_from=str(rm1.run_dir),
        runs_dir=str(runs_dir),
    )
    c_ok.resolve()
    c_ok.validate()
    rm_resumed = RunManager(c_ok, model_name="lgbm")
    assert rm_resumed.is_resumed is True
    assert rm_resumed.run_dir == rm1.run_dir
    assert rm_resumed.cfg.owner == "laptop1"  # preserved from original run
    assert rm_resumed.timestamp == rm1.timestamp  # preserved from original run

    # Resuming with modified seed or extra hyperparam raises ConfigError
    c_bad = BaseConfig(
        run_name="r1",
        task="binary",
        target="y",
        metric="auc",
        seed=99,
        resume_from=str(rm1.run_dir),
        runs_dir=str(runs_dir),
    )
    c_bad.resolve()
    c_bad.validate()
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        RunManager(c_bad, model_name="lgbm")


def test_artifacts_checkpoints_debug_and_failed(tmp_path):
    runs_dir = tmp_path / "runs"
    cfg = BaseConfig(
        run_name=" crash ",
        owner="my laptop.",
        modality="tabular",
        task="binary",
        target="y",
        metric="f1",
        runs_dir=str(runs_dir),
    )
    rm = RunManager(cfg, model_name="rf")
    assert rm.run_dir.name.endswith("_my-laptop_tabular_crash")
    assert load_json(rm.run_dir / "config.json")["run_name"] == " crash "
    assert load_json(rm.run_dir / "config.json")["owner"] == "my laptop."

    # Checkpoints
    assert rm.load_fold_checkpoint(0) is None
    oof_f0 = pd.DataFrame({"id": [1, 2], "pred": [0.2, 0.8]})
    te_f0 = pd.DataFrame({"id": [10, 11], "pred": [0.4, 0.6]})
    info_f0 = {"best_iteration": 50, "duration_sec": 1.2, "importances": {"a": 1.0}}
    rm.save_fold_checkpoint(0, oof_f0, te_f0, info_f0)
    loaded = rm.load_fold_checkpoint(0)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded[0], oof_f0)
    pd.testing.assert_frame_equal(loaded[1], te_f0)
    assert loaded[2] == info_f0

    # Save OOF, test_pred, postprocess, mark_failed
    assert rm.save_oof(oof_f0).exists()
    assert rm.save_test_pred(te_f0).exists()
    assert rm.save_postprocess({"threshold": 0.45}).exists()
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        failed_path = rm.mark_failed(exc)
    assert failed_path.exists()
    assert "RuntimeError: boom" in failed_path.read_text(encoding="utf-8")

    # Debug run goes to _debug and skips leaderboard
    cfg_dbg = BaseConfig(
        run_name="dbg",
        task="binary",
        target="y",
        metric="f1",
        debug=True,
        runs_dir=str(runs_dir),
    )
    rm_dbg = RunManager(cfg_dbg, model_name="rf")
    assert rm_dbg.run_dir.parent.name == "_debug"
    assert rm_dbg.append_leaderboard({}) is None
    assert not (runs_dir / "leaderboard.csv").exists()

    # Missing run folder in update_lb_score raises ConfigError
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        update_lb_score("non_existent_run", 0.9, runs_dir=runs_dir)
