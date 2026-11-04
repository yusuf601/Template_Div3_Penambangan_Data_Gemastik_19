# tests/text/test_analysis_and_multilabel.py
import pandas as pd
import pytest

from tests.text.helpers import create_tiny_hf_model_dir, make_text_bin_df
from utils.core.ensemble import blend, load_runs
from utils.text.analysis import extract_top_errors, run_multilabel
from utils.text.config import TextConfig
from utils.text.tfidf import TfidfPipeline
from utils.text.transformer import TransformerPipeline


def test_extract_top_errors_creates_csv_and_sorts_descending(tmp_path):
    tr, te = make_text_bin_df(36)
    cfg = TextConfig(task="binary", target="y", metric="auc", estimator="ridge", n_folds=3, runs_dir=str(tmp_path))
    res = TfidfPipeline(cfg).run(tr, te)

    err_df = extract_top_errors(res, tr, n=10)
    assert len(err_df) == 10
    assert (res.run_dir / "error_analysis.csv").exists()
    assert "text" in err_df.columns and "error" in err_df.columns
    assert list(err_df["error"]) == sorted(err_df["error"], reverse=True)


def test_run_multilabel_two_labels_two_pipelines_and_blend(tmp_path):
    model_dir = create_tiny_hf_model_dir(tmp_path)
    tr, te = make_text_bin_df(36)
    tr["label_a"] = tr["y"]
    tr["label_b"] = 1 - tr["y"]
    folds_dir = tmp_path / "shared_folds"
    runs_dir = tmp_path / "runs"

    sample_sub = pd.DataFrame({"id": list(reversed(te["id"].tolist())), "label_a": 0.0, "label_b": 0.0})

    # 1. Run TfidfPipeline across both labels (creates folds/label_a/folds.csv and folds/label_b/folds.csv)
    cfg_tfidf = TextConfig(
        run_name="tfidf",
        task="binary",
        target="label_a",
        metric="auc",
        estimator="logistic",
        n_folds=2,
        runs_dir=str(runs_dir),
    )
    res_tfidf = run_multilabel(
        TfidfPipeline,
        cfg_tfidf,
        tr,
        te,
        target_cols=["label_a", "label_b"],
        folds_dir=folds_dir,
        sample_sub=sample_sub,
        sub_path=tmp_path / "sub_tfidf.csv",
    )
    assert (folds_dir / "label_a" / "folds.csv").exists()
    assert (folds_dir / "label_b" / "folds.csv").exists()
    sub_df = pd.read_csv(tmp_path / "sub_tfidf.csv")
    assert list(sub_df["id"]) == list(sample_sub["id"])
    assert list(sub_df.columns) == ["id", "label_a", "label_b"]

    # 2. Run TransformerPipeline across the same 2 labels reusing folds_dir -> folds_hash matches per label!
    cfg_hf = TextConfig(
        run_name="hf",
        task="binary",
        target="label_a",
        metric="auc",
        model_name_or_path=str(model_dir),
        max_length=24,
        epochs=1,
        train_batch_size=8,
        n_folds=2,
        runs_dir=str(runs_dir),
    )
    res_hf = run_multilabel(
        TransformerPipeline,
        cfg_hf,
        tr,
        te,
        target_cols=["label_a", "label_b"],
        folds_dir=folds_dir,
    )

    # 3. Blend TfidfPipeline + TransformerPipeline for each label without FoldMismatchError
    for lbl in ["label_a", "label_b"]:
        ens = load_runs([res_tfidf[lbl].run_dir, res_hf[lbl].run_dir])
        blended = blend(ens, method="weights", runs_dir=runs_dir, run_name=f"blend_{lbl}")
        assert blended.scores["status"] == "ok"
