"""
Quickstart Example: End-to-End Pipeline & Blending dengan utils.core
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils.core import (
    BaseConfig,
    BasePipeline,
    blend,
    load_runs,
    seed_everything,
)


# 1. Definisikan Pipeline Kustom Turunan BasePipeline
class DemoRidgePipeline(BasePipeline):
    model_name = "ridge_model"

    def build_model(self, fold: int):
        """Inisialisasi arsitektur/model untuk fold tertentu."""
        alpha = self.cfg.extra.get("alpha", 1.0)
        return Ridge(alpha=alpha, random_state=self.cfg.seed + fold)

    def fit_transform_fold(self, tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame, fold: int):
        """Preprocessing per-fold (fit hanya pada train, transform ke val dan test)."""
        features = [c for c in tr_df.columns if c.startswith("feat_")]
        means = tr_df[features].mean()
        tr_out = tr_df.copy()
        va_out = va_df.copy()
        te_out = te_df.copy()
        tr_out[features] = tr_out[features].fillna(means)
        va_out[features] = va_out[features].fillna(means)
        te_out[features] = te_out[features].fillna(means)
        return tr_out, va_out, te_out


def main():
    seed_everything(42)
    print("1. Menyiapkan data simulasi...")
    n_train, n_test, n_feats = 100, 30, 4
    np.random.seed(42)

    X_train = np.random.randn(n_train, n_feats)
    y_train = X_train[:, 0] * 2.5 - X_train[:, 1] * 1.2 + np.random.randn(n_train) * 0.1
    X_test = np.random.randn(n_test, n_feats)

    train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(n_feats)])
    train_df["id"] = [f"tr_{i}" for i in range(n_train)]
    train_df["target"] = y_train

    test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(n_feats)])
    test_df["id"] = [f"te_{i}" for i in range(n_test)]

    # 2. Inisialisasi Konfigurasi Run 1
    print("2. Inisialisasi BaseConfig & eksekusi Model 1...")
    cfg1 = BaseConfig(
        run_name="ridge_alpha_1",
        owner="gemastik",
        modality="tabular",
        task="regression",
        target="target",
        metric="rmse",
        n_folds=3,
        seed=42,
        notes="Model ridge alpha 1.0",
        extra={"alpha": 1.0},
    )

    pipe1 = DemoRidgePipeline(cfg1)
    res1 = pipe1.run(train_df, test_df)

    print(f"   [OK] Run 1 Selesai! Run Dir: {res1.run_dir}")
    print(f"   Scores: CV={res1.scores.get('cv', 0.0):.4f}, Raw={res1.scores.get('cv_raw', 0.0):.4f}")

    # 3. Buat submission file
    sample_sub = pd.DataFrame({"id": test_df["id"], "target": 0.0})
    sub_df = res1.make_submission(sample_sub, "runs/sample_submission_res1.csv")
    print(f"   Submission Model 1 disimpan ke runs/sample_submission_res1.csv (rows: {len(sub_df)})")

    # 3. Jalankan Model 2 dengan Hyperparameter Berbeda
    print("3. Inisialisasi BaseConfig & eksekusi Model 2...")
    cfg2 = BaseConfig(
        run_name="ridge_alpha_10",
        owner="gemastik",
        modality="tabular",
        task="regression",
        target="target",
        metric="rmse",
        n_folds=3,
        seed=42,
        notes="Model ridge alpha 10.0",
        extra={"alpha": 10.0},
    )
    pipe2 = DemoRidgePipeline(cfg2)
    res2 = pipe2.run(train_df, test_df)
    print(f"   [OK] Run 2 Selesai! CV={res2.scores.get('cv', 0.0):.4f}")

    # 4. Ensembling / Blending OOF
    print("4. Blending kedua model dengan SLSQP weights...")
    runs = load_runs([res1.run_dir, res2.run_dir])
    blend_res = blend(runs, method="weights", run_name="blend_ridge_models")
    print(f"   [OK] Ensemble Selesai! Score CV: {blend_res.scores.get('cv', 0.0):.4f}")
    print(f"   Bobot Model: {blend_res.scores.get('weights')}")

    # Buat submission file dari ensemble
    blend_sub = blend_res.make_submission(sample_sub, "runs/submission_ensemble.csv")
    print(f"   Submission Ensemble disimpan ke runs/submission_ensemble.csv (rows: {len(blend_sub)})")


if __name__ == "__main__":
    main()
