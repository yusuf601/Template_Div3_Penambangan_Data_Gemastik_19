"""Error analysis and multi-label orchestration with per-label fold sharing."""

import copy
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from utils.core.cv import make_folds_file
from utils.core.errors import DataError, SubmissionError
from utils.core.io import canon_id
from utils.core.pipeline import BasePipeline, RunResult
from utils.core.run import safe_name
from utils.text.config import TextConfig


def extract_top_errors(
    result: RunResult,
    train_df: pd.DataFrame,
    n: int = 20,
    text_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract top N prediction errors from OOF predictions and write error_analysis.csv."""
    id_col = result.cfg.id_col
    target_col = result.cfg.target

    if text_cols is None:
        text_cols = getattr(result.cfg, "text_cols", ["text"])
    valid_text_cols = [c for c in text_cols if c in train_df.columns]

    pred_cols = [c for c in result.oof.columns if c.startswith("pred")]
    valid_oof = result.oof.dropna(subset=[target_col] + pred_cols).copy()
    if valid_oof.empty:
        return pd.DataFrame()

    oof_cid = canon_id(valid_oof[id_col])
    train_cid = canon_id(train_df[id_col])

    valid_oof["_cid"] = oof_cid
    train_copy = train_df.copy()
    train_copy["_cid"] = train_cid

    merged = pd.merge(
        valid_oof,
        train_copy[["_cid"] + [c for c in valid_text_cols if c != id_col]],
        on="_cid",
        how="inner",
    )

    if result.cfg.task == "binary":
        y_true = merged[target_col].to_numpy(dtype=float)
        p_val = merged["pred"].to_numpy(dtype=float)
        merged["error"] = np.abs(y_true - p_val)

    elif result.cfg.task == "multiclass":
        classes = result.classes
        errors = []
        for _, row in merged.iterrows():
            y_val = row[target_col]
            if classes is not None and y_val in classes:
                cls_idx = classes.index(y_val)
            else:
                try:
                    cls_idx = int(y_val)
                except Exception:
                    cls_idx = 0
            pred_col_name = f"pred_{cls_idx}"
            prob = float(row[pred_col_name]) if pred_col_name in row else 0.0
            errors.append(1.0 - prob)
        merged["error"] = errors

    else:
        # regression
        y_true = merged[target_col].to_numpy(dtype=float)
        p_val = merged["pred"].to_numpy(dtype=float)
        merged["error"] = np.abs(y_true - p_val)

    sorted_df = merged.sort_values(by="error", ascending=False).reset_index(drop=True)
    top_df = sorted_df.head(n)

    keep_cols = [id_col, "fold"] + valid_text_cols + [target_col] + pred_cols + ["error"]
    seen = set()
    ordered_cols = []
    for c in keep_cols:
        if c in top_df.columns and c not in seen:
            seen.add(c)
            ordered_cols.append(c)

    out_df = top_df[ordered_cols].copy()
    out_csv = result.run_dir / "error_analysis.csv"
    out_df.to_csv(out_csv, index=False)
    return out_df


def run_multilabel(
    pipeline_cls: type[BasePipeline],
    base_cfg: TextConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: list[str],
    folds_dir: Union[str, Path] = "folds",
    sample_sub: Optional[pd.DataFrame] = None,
    sub_path: Optional[Union[str, Path]] = None,
) -> dict[str, RunResult]:
    """Train independent binary pipelines across multiple target labels with shared per-label folds."""
    missing = [c for c in target_cols if c not in train_df.columns]
    if missing:
        raise DataError(
            f"Kolom target {missing!r} tidak ditemukan pada train_df.",
            "Pastikan semua target_cols tersedia di train_df sebelum menjalankan run_multilabel.",
        )

    results: dict[str, RunResult] = {}
    fdir = Path(folds_dir)

    for tcol in target_cols:
        label_slug = safe_name(tcol, "label")
        label_folds_dir = fdir / label_slug
        label_folds_csv = label_folds_dir / "folds.csv"

        cfg_t = copy.deepcopy(base_cfg)
        cfg_t.task = "binary"
        cfg_t.target = tcol
        cfg_t.run_name = f"{base_cfg.run_name}_{tcol}"

        if not cfg_t.debug:
            if not label_folds_csv.exists():
                make_folds_file(train_df, cfg_t, out_dir=label_folds_dir)
            cfg_t.folds_path = str(label_folds_csv)

        pipe = pipeline_cls(cfg_t)
        res = pipe.run(train_df, test_df)
        results[tcol] = res

    if sample_sub is not None and sub_path is not None:
        id_col = base_cfg.id_col
        if id_col not in sample_sub.columns:
            raise SubmissionError(
                f"Kolom id {id_col!r} tidak ditemukan pada sample_sub.",
                f"Pastikan sample_sub memuat kolom {id_col!r}.",
            )

        sub_cid = canon_id(sample_sub[id_col]).to_numpy()
        sub_out = sample_sub.copy()

        for tcol in target_cols:
            res_t = results[tcol]
            tp = res_t.test_pred
            tp_cid = canon_id(tp[id_col]).to_numpy()
            tp_map = pd.Series(tp["pred"].to_numpy(dtype=float), index=tp_cid)
            sub_out[tcol] = tp_map.loc[sub_cid].to_numpy()

        p = Path(sub_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sub_out.to_csv(p, index=False)

    return results


__all__ = [
    "extract_top_errors",
    "run_multilabel",
]
