# tests/text/test_tfidf_pipeline.py
import warnings
import numpy as np
import pandas as pd
import pytest

from tests.text.helpers import make_text_bin_df
from utils.text.config import TextConfig
from utils.text.tfidf import TfidfPipeline


@pytest.mark.parametrize("estimator", ["logistic", "linear_svc", "sgd", "ridge", "lgbm"])
def test_tfidf_binary_estimators_and_importances(tmp_path, estimator):
    tr, te = make_text_bin_df(48)
    cfg = TextConfig(
        run_name=f"bin_{estimator}",
        task="binary",
        target="y",
        metric="f1",
        estimator=estimator,
        num_cols=["num_meta"],
        cat_cols=["cat_meta"],
        n_folds=3,
        svd_components=8,
        top_k_importances=15,
        runs_dir=str(tmp_path),
    )
    res = TfidfPipeline(cfg).run(tr, te)
    assert res.scores["status"] == "ok"
    assert (res.run_dir / "plots" / "importance.png").exists()
    assert len(res.scores["per_fold"]) == 3
    assert "threshold" in res.params


def test_tfidf_pair_features_and_regression(tmp_path):
    rng = np.random.default_rng(7)
    n = 45
    q1 = [f"apa kabar produk nomor {i % 5} bagus" for i in range(n)]
    q2 = [f"apa kabar produk nomor {i % 5} bagus sekali" if i % 2 == 0 else f"keluhan pengiriman lambat {i}" for i in range(n)]
    y = np.array([1.5 if i % 2 == 0 else -0.5 for i in range(n)]) + rng.normal(scale=0.1, size=n)
    tr = pd.DataFrame({"id": range(n), "q1": q1, "q2": q2, "y": y})
    te = pd.DataFrame({
        "id": range(100, 108),
        "q1": [f"apa kabar produk nomor {i}" for i in range(8)],
        "q2": [f"apa kabar produk nomor {i}" for i in range(8)],
    })
    cfg = TextConfig(
        task="regression",
        target="y",
        metric="rmse",
        text_cols=["q1", "q2"],
        estimator="ridge",
        n_folds=3,
        test_mode="refit_full",
        runs_dir=str(tmp_path),
    )
    pipe = TfidfPipeline(cfg)
    res = pipe.run(tr, te)
    assert res.scores["status"] == "ok"
    assert res.scores["test_pred_source"] == "refit"
    assert any("pair__tfidf_word_cos" in k for k in pipe.fold_info[0]["importances"])


def test_tfidf_rare_class_fallback_missing_class_alignment_and_lgbm_refit(tmp_path):
    # 1. Multiclass where class "langka" has only 1 sample in train:
    #    - In folds where "langka" is in tr (count=1 < 2), linear_svc warns and falls back to LogisticRegression
    #    - In folds where "langka" is in va (0 samples in tr), predict_proba still aligns to K=3 columns!
    texts = ["berita olahraga bola menang"] * 15 + ["artikel ekonomi pasar saham"] * 15 + ["topik khusus langka"]
    labels = ["olahraga"] * 15 + ["ekonomi"] * 15 + ["langka"]
    tr = pd.DataFrame({"id": range(len(texts)), "text": texts, "y": labels})
    te = pd.DataFrame({"id": [101, 102, 103], "text": ["bola menang", "pasar saham", "khusus langka"]})

    cfg_svc = TextConfig(
        task="multiclass",
        target="y",
        metric="f1_macro",
        cv_strategy="kfold",
        estimator="linear_svc",
        n_folds=3,
        min_df=1,
        runs_dir=str(tmp_path / "svc"),
    )
    with pytest.warns(UserWarning, match="LogisticRegression"):
        res_svc = TfidfPipeline(cfg_svc).run(tr, te)
    assert res_svc.classes == ["ekonomi", "langka", "olahraga"]
    assert list(res_svc.oof.columns) == ["id", "fold", "y", "pred_0", "pred_1", "pred_2"]
    assert not res_svc.oof[["pred_0", "pred_1", "pred_2"]].isna().any().any()
    assert np.allclose(res_svc.test_pred[["pred_0", "pred_1", "pred_2"]].sum(axis=1), 1.0)

    # 2. LGBM refit_full (fold == -1) uses refit_iterations() without early stopping error
    tr_bin, te_bin = make_text_bin_df(45)
    cfg_lgb = TextConfig(
        task="binary",
        target="y",
        metric="auc",
        estimator="lgbm",
        n_folds=3,
        svd_components=6,
        test_mode="refit_full",
        runs_dir=str(tmp_path / "lgb"),
    )
    pipe_lgb = TfidfPipeline(cfg_lgb)
    res_lgb = pipe_lgb.run(tr_bin, te_bin)
    assert res_lgb.scores["test_pred_source"] == "refit"
    assert pipe_lgb.refit_iterations() is not None and pipe_lgb.refit_iterations() >= 1
