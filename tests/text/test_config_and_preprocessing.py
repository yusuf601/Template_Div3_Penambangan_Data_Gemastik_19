# tests/text/test_config_and_preprocessing.py
import numpy as np
import pandas as pd
import pytest

from utils.core.config import BaseConfig
from utils.core.errors import ConfigError, DataError
from utils.core.run import RunManager
from utils.text.config import TextConfig
from utils.text.features import extract_pair_stats, extract_text_stats
from utils.text.preprocessing import clean_text_series
from utils.text.base import TextPipelineBase
from tests.text.helpers import make_text_bin_df


def test_text_config_roundtrip_and_resume_compatibility(tmp_path):
    cfg = TextConfig(
        task="binary",
        target="y",
        metric="f1",
        text_cols=["judul", "isi"],
        word_ngram_range=(1, 3),
        extra={"foo": 1, "bar": ["a", "b"]},
        runs_dir=str(tmp_path / "runs"),
    )
    cfg.resolve()
    cfg.validate()
    assert cfg.select_metric == "logloss"

    cfg_path = tmp_path / "config.json"
    cfg.save(cfg_path)

    # 1. BaseConfig.load (used by load_runs / blend) must succeed without TypeError
    base_loaded = BaseConfig.load(cfg_path)
    assert base_loaded.task == "binary"
    assert base_loaded.extra["foo"] == 1
    assert "_text" in base_loaded.extra

    # 2. TextConfig.load must restore all TextConfig fields AND keep user extra clean (no "_text" key)
    text_loaded = TextConfig.load(cfg_path)
    text_loaded.resolve()
    assert text_loaded.text_cols == ["judul", "isi"]
    assert text_loaded.word_ngram_range == (1, 3)
    assert text_loaded.extra == {"foo": 1, "bar": ["a", "b"]}
    assert "_text" not in text_loaded.extra
    assert text_loaded.to_dict() == cfg.to_dict()

    # 3. RunManager resume_from comparison must pass without tuple vs list mismatch
    rm1 = RunManager(cfg, model_name="tfidf_logistic")
    cfg_resume = TextConfig(
        task="binary",
        target="y",
        metric="f1",
        text_cols=["judul", "isi"],
        word_ngram_range=(1, 3),
        extra={"foo": 1, "bar": ["a", "b"]},
        resume_from=str(rm1.run_dir),
        runs_dir=str(tmp_path / "runs"),
    )
    rm2 = RunManager(cfg_resume, model_name="tfidf_logistic")
    assert rm2.is_resumed and rm2.run_dir == rm1.run_dir


def test_text_config_validation_errors():
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        c = TextConfig(task="regression", target="y", metric="rmse", estimator="logistic")
        c.resolve(); c.validate()

    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        c = TextConfig(task="binary", target="y", metric="auc", precision="fp8")
        c.resolve(); c.validate()

    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        c = TextConfig(task="binary", target="y", metric="auc", text_cols=["y"])
        c.resolve(); c.validate()


def test_clean_text_modes_and_stateless_features():
    s = pd.Series([
        "<b>Baguuusss</b> bgt cek https://example.com @admin!",
        None,
        "Gk puas sama sekali??? 123 #kecewa",
    ])
    c_none = clean_text_series(s, mode="none", lowercase=False)
    assert c_none.iloc[1] == ""
    assert "<b>Baguuusss</b>" in c_none.iloc[0]

    c_soc = clean_text_series(s, mode="social_id", lowercase=True)
    assert "<b>" not in c_soc.iloc[0]
    assert "<url>" in c_soc.iloc[0]
    assert "<user>" in c_soc.iloc[0]
    assert "baguus" in c_soc.iloc[0]
    assert "banget" in c_soc.iloc[0]
    assert "tidak puas" in c_soc.iloc[2]

    df = pd.DataFrame({"id": [1, 2, 3], "t1": s, "t2": ["baguus banget", "kosong", "tidak puas"]})
    df_stats, stat_cols = extract_text_stats(df, ["t1"])
    assert "t1__is_na" in stat_cols and df_stats.loc[1, "t1__is_na"] == 1.0
    assert df_stats.loc[0, "t1__url_count"] == 1.0
    assert df_stats.loc[2, "t1__hashtag_count"] == 1.0

    df_stats["t1__clean"] = c_soc
    df_stats["t2__clean"] = clean_text_series(df_stats["t2"], mode="social_id", lowercase=True)
    df_pair, pair_cols = extract_pair_stats(df_stats, "t1__clean", "t2__clean")
    assert "pair__word_jaccard" in pair_cols
    assert "pair__char3_jaccard" in pair_cols
    assert not df_pair[pair_cols].isna().any().any()


def test_binary_string_target_rejected_and_postprocess_handles_nan_oof(tmp_path):
    tr, te = make_text_bin_df(30)
    tr_str_y = tr.copy()
    tr_str_y["y"] = np.where(tr["y"] == 1, "pos", "neg")

    class DummyTextPipe(TextPipelineBase):
        def build_model(self, fold):
            return None

    cfg = TextConfig(task="binary", target="y", metric="f1", runs_dir=str(tmp_path))
    with pytest.raises(ConfigError, match=r"(?s)0/1.*\[Perbaikan\]"):
        DummyTextPipe(cfg).run(tr_str_y, te)

    # Missing text_col raises DataError
    cfg_bad_col = TextConfig(task="binary", target="y", metric="f1", text_cols=["kolom_hilang"], runs_dir=str(tmp_path))
    with pytest.raises(DataError, match=r"(?s)kolom_hilang.*\[Perbaikan\]"):
        DummyTextPipe(cfg_bad_col).run(tr, te)

    # postprocess must ignore NaN predictions (e.g. from partial runs) and still find optimal threshold
    pipe = DummyTextPipe(cfg)
    pipe.cfg.resolve()
    oof_with_nan = pd.DataFrame({
        "id": range(10),
        "fold": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "y": [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        "pred": [0.1, 0.2, 0.35, 0.4, 0.42, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    params = pipe.postprocess(oof_with_nan, None)
    assert "threshold" in params
    assert 0.2 < params["threshold"] <= 0.35
