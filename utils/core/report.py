"""Slide-ready Markdown summary, diagnostic plots, and leaderboard comparison."""

from pathlib import Path
from typing import Any, Optional, Union
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from utils.core.config import BaseConfig
from utils.core.metrics import get_metric
from utils.core.run import LEADERBOARD_COLS


def _plot_fold_scores(
    plots_dir: Path,
    per_fold: dict[str, Any],
    cv_score: Optional[float],
    metric_name: str,
) -> None:
    if not per_fold:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        folds = list(per_fold.keys())
        vals = [float(per_fold[k].get("raw", np.nan)) for k in folds]
        ax.bar([f"Fold {k}" for k in folds], vals, color="#4C72B0", alpha=0.85)
        if cv_score is not None and not pd.isna(cv_score):
            ax.axhline(float(cv_score), color="#C44E52", linestyle="--", label=f"CV = {float(cv_score):.4f}")
            ax.legend()
        ax.set_ylabel(metric_name)
        ax.set_title(f"Per-Fold {metric_name} Scores")
        fig.tight_layout()
        fig.savefig(plots_dir / "fold_scores.png", dpi=120)
    finally:
        plt.close(fig)


def _plot_task_diagnostics(
    plots_dir: Path,
    cfg: BaseConfig,
    oof_df: pd.DataFrame,
    params: dict[str, Any],
    classes: Optional[list[Any]] = None,
) -> None:
    if cfg.target not in oof_df.columns:
        raise ValueError(f"Target column {cfg.target!r} missing from oof_df")

    if cfg.task == "binary":
        if "pred" not in oof_df.columns:
            raise ValueError("Column pred missing from binary oof_df")
        valid = oof_df.dropna(subset=[cfg.target, "pred"])
        if valid.empty:
            return
        y_true = valid[cfg.target].to_numpy()
        thr = float((params or {}).get("threshold", 0.5))
        y_pred = (valid["pred"].to_numpy(dtype=float) >= thr).astype(int)
        if classes is not None and len(classes) == 2:
            y_pred_labels = np.array([classes[i] for i in y_pred])
            cm = confusion_matrix(y_true, y_pred_labels, labels=classes)
            tick_labels = [str(c) for c in classes]
        else:
            cm = confusion_matrix(y_true, y_pred)
            tick_labels = [str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            im = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(tick_labels)))
            ax.set_yticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("OOF Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            fig.tight_layout()
            fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
        finally:
            plt.close(fig)

    elif cfg.task == "multiclass":
        pred_cols = [c for c in oof_df.columns if c.startswith("pred_")]
        if not pred_cols:
            raise ValueError("Multiclass prediction columns pred_0..pred_{K-1} missing from oof_df")
        pred_cols = sorted(
            pred_cols,
            key=lambda x: int(x.split("_")[1]) if len(x.split("_")) > 1 and x.split("_")[1].isdigit() else x,
        )
        valid = oof_df.dropna(subset=[cfg.target] + pred_cols)
        if valid.empty:
            return
        y_true = valid[cfg.target].to_numpy()
        probs = valid[pred_cols].to_numpy(dtype=float)
        idx = np.argmax(probs, axis=1)
        if classes is not None and len(classes) == len(pred_cols):
            y_pred = np.array([classes[i] for i in idx])
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            tick_labels = [str(c) for c in classes]
        else:
            unique_cls = sorted(pd.unique(y_true).tolist())
            if len(unique_cls) == len(pred_cols):
                y_pred = np.array([unique_cls[i] for i in idx])
                cm = confusion_matrix(y_true, y_pred, labels=unique_cls)
                tick_labels = [str(c) for c in unique_cls]
            else:
                cm = confusion_matrix(y_true, idx)
                tick_labels = [str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            im = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(tick_labels)))
            ax.set_yticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("OOF Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            fig.tight_layout()
            fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
        finally:
            plt.close(fig)

    elif cfg.task == "regression":
        if "pred" not in oof_df.columns:
            raise ValueError("Column pred missing from regression oof_df")
        valid = oof_df.dropna(subset=[cfg.target, "pred"])
        if valid.empty:
            return
        y_true = valid[cfg.target].to_numpy(dtype=float)
        y_pred = valid["pred"].to_numpy(dtype=float)
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ax.scatter(y_pred, residuals, alpha=0.6, color="#4C72B0", edgecolors="none")
            ax.axhline(0.0, color="#C44E52", linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual (Actual - Predicted)")
            ax.set_title("OOF Residuals vs Predicted")
            fig.tight_layout()
            fig.savefig(plots_dir / "residuals.png", dpi=120)
        finally:
            plt.close(fig)


def _plot_importances(plots_dir: Path, importances: pd.Series) -> None:
    top_imp = importances.dropna().sort_values(ascending=False).head(20)
    if top_imp.empty:
        return
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.3 * len(top_imp))))
    try:
        rev = top_imp.iloc[::-1]
        ax.barh([str(k) for k in rev.index], rev.to_numpy(dtype=float), color="#55A868")
        ax.set_xlabel("Importance")
        ax.set_title("Top-20 Feature Importances")
        fig.tight_layout()
        fig.savefig(plots_dir / "importance.png", dpi=120)
    finally:
        plt.close(fig)


def generate_report(
    run_dir: Union[str, Path],
    cfg: BaseConfig,
    scores: dict[str, Any],
    oof_df: pd.DataFrame,
    params: dict[str, Any],
    importances: Optional[pd.Series] = None,
    classes: Optional[list[Any]] = None,
) -> None:
    """Generate summary.md and plots/ directory for a finished run."""
    rdir = Path(run_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    plots_dir = rdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write summary.md
    model_val = scores.get("model", "base")
    owner_val = scores.get("owner", cfg.owner)
    dir_val = scores.get("direction", "")
    status_val = scores.get("status", "ok")
    n_folds_done_val = scores.get("n_folds_done", 0)
    tps_val = scores.get("test_pred_source", "fold_avg")
    mins_val = float(scores.get("minutes", 0.0))
    notes_val = cfg.notes if cfg.notes else "-"
    cv_val = scores.get("cv")
    cv_raw_val = scores.get("cv_raw")
    cv_post_val = scores.get("cv_post")
    cv_honest_val = scores.get("cv_honest")
    cv_std_val = scores.get("cv_std")
    n_rows_eval_val = scores.get("n_rows_eval", 0)

    lines: list[str] = [
        f"# Run Summary: `{cfg.run_name}`",
        "",
        "## Overview",
        f"- **Run Directory**: `{rdir.name}`",
        f"- **Model**: `{model_val}`",
        f"- **Owner**: `{owner_val}`",
        f"- **Modality**: `{cfg.modality}`",
        f"- **Task**: `{cfg.task}` (`target={cfg.target}`)",
        f"- **Metric**: `{cfg.metric}` (`direction={dir_val}`)",
        f"- **CV Strategy**: `{cfg.cv_strategy}` (`n_folds={cfg.n_folds}`, `seed={cfg.seed}`, `fold_seed={cfg.fold_seed}`)",
        f"- **Status**: `{status_val}` (`n_folds_done={n_folds_done_val}`)",
        f"- **Test Pred Source**: `{tps_val}`",
        f"- **Duration**: `{mins_val:.2f}` minutes",
        f"- **Notes**: {notes_val}",
        "",
        "## Cross-Validation Scores (`eval_folds`)",
        f"- **CV (Ranking / Honest)**: `{cv_val}`",
        f"- **CV Raw**: `{cv_raw_val}`",
        f"- **CV Post**: `{cv_post_val}`",
        f"- **CV Honest**: `{cv_honest_val}`",
        f"- **CV Std**: `{cv_std_val}`",
        f"- **Eval Rows (`n_rows_eval`)**: `{n_rows_eval_val}`",
        f"- **Postprocess Params**: `{params or {}}`",
        "",
        "## Per-Fold Breakdown",
        "| Fold | Raw Score | Rows | Duration (s) |",
        "|---|---|---|---|",
    ]

    per_fold = scores.get("per_fold", {}) or {}
    for k, f_info in per_fold.items():
        raw_v = f_info.get("raw")
        raw_str = f"{float(raw_v):.6f}" if raw_v is not None and not pd.isna(raw_v) else "NaN"
        n_r = f_info.get("n_rows", 0)
        dur = float(f_info.get("duration_sec", 0.0))
        fold_lbl = f"{k} (truncated)" if f_info.get("truncated") else f"{k}"
        lines.append(f"| {fold_lbl} | {raw_str} | {n_r} | {dur:.2f} |")

    if importances is not None and len(importances) > 0:
        top_imp = importances.dropna().sort_values(ascending=False).head(20)
        lines.extend(
            [
                "",
                "## Top-20 Feature Importances",
                "| Rank | Feature | Importance |",
                "|---|---|---|",
            ]
        )
        for rank, (feat, val) in enumerate(top_imp.items(), start=1):
            lines.append(f"| {rank} | `{feat}` | {float(val):.6f} |")

    (rdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. Generate plots (failures emit UserWarning and never crash the run)
    try:
        _plot_fold_scores(plots_dir, per_fold, scores.get("cv"), cfg.metric)
        _plot_task_diagnostics(plots_dir, cfg, oof_df, params, classes=classes)
        if importances is not None and len(importances) > 0:
            _plot_importances(plots_dir, importances)
    except Exception as exc:
        warnings.warn(
            f"Gagal membuat plot pada {rdir.name}: {exc}",
            UserWarning,
            stacklevel=2,
        )
    finally:
        plt.close("all")


def compare_runs(
    runs_dir: Union[str, Path] = "runs",
    top: int = 10,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Return top runs sorted by CV score direction, filtering out partial runs by default."""
    lb_path = Path(runs_dir) / "leaderboard.csv"
    if not lb_path.exists():
        return pd.DataFrame(columns=LEADERBOARD_COLS)

    df = pd.read_csv(lb_path)
    if not include_partial and "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if df.empty:
        return df.reset_index(drop=True)

    ascending = False
    if "metric" in df.columns:
        non_null_metrics = df["metric"].dropna()
        if not non_null_metrics.empty:
            first_metric = str(non_null_metrics.iloc[0])
            try:
                m = get_metric(first_metric)
                ascending = m.direction == "minimize"
            except Exception:
                ascending = False

    if "cv" in df.columns:
        df = df.sort_values(by="cv", ascending=ascending, na_position="last", kind="mergesort")

    return df.head(top).reset_index(drop=True)


__all__ = ["generate_report", "compare_runs"]
