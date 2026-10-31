import pytest
from utils.core.config import BaseConfig
from utils.core.errors import ConfigError


def test_config_auto_resolve_and_hostname_owner():
    cfg = BaseConfig(task="binary", target="y", metric="f1")
    reasons = cfg.resolve()
    assert cfg.owner != "" and cfg.owner.isalnum()
    assert cfg.cv_strategy == "stratified"
    assert cfg.test_mode == "fold_avg"
    assert any("cv_strategy" in r for r in reasons)

    cfg_grp = BaseConfig(task="regression", target="y", metric="rmse", group_col="g")
    cfg_grp.resolve()
    assert cfg_grp.cv_strategy == "group"

    cfg_sgrp = BaseConfig(task="multiclass", target="y", metric="f1_macro", group_col="g")
    cfg_sgrp.resolve()
    assert cfg_sgrp.cv_strategy == "stratified_group"

    cfg_ts = BaseConfig(task="regression", target="y", metric="rmse", cv_strategy="timeseries", time_col="dt")
    cfg_ts.resolve()
    assert cfg_ts.test_mode == "refit_full"


def test_config_debug_and_warnings():
    cfg = BaseConfig(task="regression", target="y", metric="rmse", time_col="dt",
                     folds_path="folds/folds.csv", resume_from="runs/old", debug=True)
    with pytest.warns(UserWarning, match="timeseries"):
        cfg.resolve()
    assert cfg.n_folds == 2
    assert cfg.folds_path is None and cfg.resume_from is None


def test_config_validate_errors_have_remediation(tmp_path):
    cfg = BaseConfig(task="regression", target="y", metric="rmse", cv_strategy="stratified")
    cfg.resolve()
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        cfg.validate()

    # Invalid task
    c_task = BaseConfig(task="clustering", target="y", metric="rmse")
    c_task.resolve()
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        c_task.validate()

    # Group strategy without group_col
    c_grp = BaseConfig(task="binary", target="y", metric="auc", cv_strategy="group")
    c_grp.resolve()
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        c_grp.validate()

    # Timeseries without time_col
    c_ts = BaseConfig(task="regression", target="y", metric="rmse", cv_strategy="timeseries")
    c_ts.resolve()
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        c_ts.validate()

    # Save and load roundtrip
    valid_cfg = BaseConfig(task="binary", target="y", metric="auc", owner="tester", extra={"lr": 0.05})
    valid_cfg.resolve()
    valid_cfg.validate()
    p = tmp_path / "config.json"
    valid_cfg.save(p)
    loaded = BaseConfig.load(p)
    assert loaded.to_dict() == valid_cfg.to_dict()
