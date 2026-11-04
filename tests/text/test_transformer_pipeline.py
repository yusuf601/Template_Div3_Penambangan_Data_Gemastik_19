# tests/text/test_transformer_pipeline.py
import numpy as np
import pandas as pd
import pytest
import torch

from tests.core.helpers import FakeClock
from tests.text.helpers import create_tiny_hf_model_dir, make_text_bin_df
from utils.core.errors import ConfigError
from utils.text.config import TextConfig
from utils.text.transformer import TransformerPipeline


def test_transformer_fast_precheck_fails_on_missing_model_dir(tmp_path):
    tr, te = make_text_bin_df(20)
    cfg = TextConfig(
        task="binary",
        target="y",
        metric="auc",
        model_name_or_path=str(tmp_path / "folder_kosong"),
        local_files_only=True,
        n_folds=2,
        runs_dir=str(tmp_path / "runs"),
    )
    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        TransformerPipeline(cfg).run(tr, te)


def test_transformer_fresh_weights_deterministic_resume_and_time_guards(tmp_path):
    model_dir = create_tiny_hf_model_dir(tmp_path)
    tr, te = make_text_bin_df(32)

    # 1. Uninterrupted 2-fold run: verify fold 0 and fold 1 start from identical pretrained weights
    cfg_full = TextConfig(
        run_name="hf_full",
        task="binary",
        target="y",
        metric="f1",
        model_name_or_path=str(model_dir),
        max_length=32,
        epochs=1,
        train_batch_size=8,
        eval_batch_size=16,
        n_folds=2,
        runs_dir=str(tmp_path / "runs_full"),
    )
    pipe_full = TransformerPipeline(cfg_full)
    res_full = pipe_full.run(tr, te)
    assert res_full.scores["status"] == "ok"
    assert torch.equal(pipe_full.initial_fold_weights_[0], pipe_full.initial_fold_weights_[1])

    # 2. Pre-fold guard stops before fold 1 -> status="partial"; resume produces identical fold 1 predictions!
    clock = FakeClock(start=0.0)

    class ClockedHFPipe(TransformerPipeline):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._clock = clock

        def fit_predict_fold(self, model, tr_df, va_df, te_df, fold):
            out = super().fit_predict_fold(model, tr_df, va_df, te_df, fold)
            clock.advance(40.0)  # each fold takes 40s
            return out

    # max_minutes=1.0 (60s): fold 0 takes 40s -> rem=20s < 40s * 1.1 (44s) -> stops before fold 1
    cfg_part = TextConfig(
        run_name="hf_res",
        task="binary",
        target="y",
        metric="f1",
        model_name_or_path=str(model_dir),
        max_length=32,
        epochs=1,
        train_batch_size=8,
        eval_batch_size=16,
        n_folds=2,
        max_minutes=1.0,
        runs_dir=str(tmp_path / "runs_res"),
    )
    r_part = ClockedHFPipe(cfg_part).run(tr, te)
    assert r_part.scores["status"] == "partial"
    assert r_part.scores["n_folds_done"] == 1

    # Resume with max_minutes=10.0: fold 1 OOF predictions must match res_full fold 1 OOF predictions
    cfg_res = TextConfig(
        run_name="hf_res",
        task="binary",
        target="y",
        metric="f1",
        model_name_or_path=str(model_dir),
        max_length=32,
        epochs=1,
        train_batch_size=8,
        eval_batch_size=16,
        n_folds=2,
        max_minutes=10.0,
        resume_from=str(r_part.run_dir),
        runs_dir=str(tmp_path / "runs_res"),
    )
    r_res = ClockedHFPipe(cfg_res).run(tr, te)
    assert r_res.scores["status"] == "ok"
    mask_f1 = r_res.oof["fold"] == 1
    assert np.allclose(r_res.oof.loc[mask_f1, "pred"], res_full.oof.loc[mask_f1, "pred"], atol=1e-6)


def test_transformer_batch3_to_10_warning_and_emergency_truncation(tmp_path):
    model_dir = create_tiny_hf_model_dir(tmp_path)
    tr, te = make_text_bin_df(40)  # 20 rows per fold -> with batch_size=2, 10 batches per epoch
    clock = FakeClock(start=0.0)

    class SlowBatchHFPipe(TransformerPipeline):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._clock = clock

        def _on_train_batch_end(self, fold: int, epoch: int, batch_idx: int) -> None:
            clock.advance(5.0)  # each batch takes 5s -> 10 batches = 50s per epoch

    # max_minutes=0.6 (36s), epochs=2:
    # - Batches 2..9 average 5s -> est_epoch_sec = 50s > 36s -> emits UserWarning (not DataError!)
    # - Epoch 0 finishes at 50s (>= 36s) -> emergency cutoff triggers before Epoch 1 and sets truncated=True!
    cfg = TextConfig(
        run_name="hf_trunc",
        task="binary",
        target="y",
        metric="auc",
        model_name_or_path=str(model_dir),
        max_length=24,
        epochs=2,
        train_batch_size=2,
        eval_batch_size=16,
        n_folds=2,
        max_minutes=0.6,
        runs_dir=str(tmp_path / "runs_trunc"),
    )
    with pytest.warns(UserWarning):
        res = SlowBatchHFPipe(cfg).run(tr, te)
    assert res.scores["per_fold"]["0"].get("truncated") is True
    assert "0 (truncated)" in (res.run_dir / "summary.md").read_text(encoding="utf-8")


def test_transformer_multiclass_pair_and_regression_refit(tmp_path):
    model_dir = create_tiny_hf_model_dir(tmp_path)
    n = 30
    df_mc = pd.DataFrame({
        "id": range(n),
        "t1": ["produk bagus mantap"] * 10 + ["barang rusak parah"] * 10 + ["pengiriman cepat sekali"] * 10,
        "t2": ["puas sekali"] * 10 + ["kecewa berat"] * 10 + ["layanan ramah"] * 10,
        "y": ["positif"] * 10 + ["negatif"] * 10 + ["netral"] * 10,
    })
    te_mc = pd.DataFrame({
        "id": [101, 102, 103],
        "t1": ["produk bagus", "barang rusak", "pengiriman cepat"],
        "t2": ["puas", "kecewa", "ramah"],
    })
    cfg_mc = TextConfig(
        task="multiclass",
        target="y",
        metric="f1_macro",
        text_cols=["t1", "t2"],
        model_name_or_path=str(model_dir),
        max_length=24,
        epochs=1,
        train_batch_size=8,
        n_folds=2,
        runs_dir=str(tmp_path / "mc"),
    )
    res_mc = TransformerPipeline(cfg_mc).run(df_mc, te_mc)
    assert res_mc.classes == ["negatif", "netral", "positif"]
    assert list(res_mc.test_pred.columns) == ["id", "pred_0", "pred_1", "pred_2"]

    # Regression with target scaling (values around 1000.0) and refit_full
    df_reg = df_mc.copy()
    df_reg["y"] = np.linspace(1000.0, 2000.0, n)
    cfg_reg = TextConfig(
        task="regression",
        target="y",
        metric="rmse",
        text_cols=["t1"],
        model_name_or_path=str(model_dir),
        max_length=24,
        epochs=1,
        train_batch_size=8,
        n_folds=2,
        test_mode="refit_full",
        runs_dir=str(tmp_path / "reg"),
    )
    res_reg = TransformerPipeline(cfg_reg).run(df_reg, te_mc)
    assert res_reg.scores["test_pred_source"] == "refit"
    # Verify predictions were unscaled back to the ~1000..2000 range
    assert float(res_reg.test_pred["pred"].mean()) > 500.0
