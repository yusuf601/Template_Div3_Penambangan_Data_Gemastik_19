"""OOF blending (mean and SLSQP non-negative weights) with honest per-fold evaluation."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.core.config import BaseConfig
from utils.core.errors import ConfigError, FoldMismatchError
from utils.core.io import canon_id, load_json, load_parquet
from utils.core.metrics import get_metric, is_better, score
from utils.core.pipeline import RunResult
from utils.core.report import generate_report
from utils.core.run import RunManager


@dataclass
class EnsembleData:
    """Aligned artifacts and metadata from multiple runs ready for blending."""

    run_dirs: list[Path]
    configs: list[BaseConfig]
    scores_list: list[dict[str, Any]]
    oofs: list[pd.DataFrame]
    test_preds: list[pd.DataFrame]
    pred_cols: list[str]
    folds_hash: str
    test_ids_hash: str
    classes: Optional[list[Any]] = None
    has_partial: bool = False


def load_runs(
    run_dirs: Sequence[Union[str, Path]],
    allow_partial: bool = False,
) -> EnsembleData:
    """Load and validate multiple run directories for OOF blending, aligning rows via canon_id."""
    if not run_dirs or len(run_dirs) < 2:
        raise ConfigError(
            "Daftar run_dirs untuk load_runs membutuhkan minimal 2 run (lihat [Perbaikan]).",
            "Berikan minimal 2 folder run yang valid ke load_runs(run_dirs).",
        )

    paths: list[Path] = []
    configs: list[BaseConfig] = []
    scores_list: list[dict[str, Any]] = []
    raw_oofs: list[pd.DataFrame] = []
    raw_test_preds: list[pd.DataFrame] = []
    has_partial = False

    required_files = ("config.json", "scores.json", "oof.parquet", "test_pred.parquet")

    for rd_item in run_dirs:
        rd = Path(rd_item)
        if not rd.exists() or not rd.is_dir():
            raise ConfigError(
                f"Folder run {str(rd)!r} tidak ditemukan (lihat [Perbaikan]).",
                "Pastikan seluruh path di run_dirs menunjuk ke folder run yang valid.",
            )
        missing = [f for f in required_files if not (rd / f).exists()]
        if missing:
            raise ConfigError(
                f"Folder run {rd.name!r} kehilangan file artefak {missing!r} (lihat [Perbaikan]).",
                "Pastikan run telah selesai atau memiliki config.json, scores.json, oof.parquet, dan test_pred.parquet.",
            )

        cfg = BaseConfig.load(rd / "config.json")
        sc = load_json(rd / "scores.json")
        oof_df = load_parquet(rd / "oof.parquet")
        tp_df = load_parquet(rd / "test_pred.parquet")

        st = str(sc.get("status", "ok"))
        if st != "ok":
            if not allow_partial:
                raise ConfigError(
                    f"Run {rd.name!r} berstatus {st!r} (belum selesai, allow_partial=False; lihat [Perbaikan]).",
                    "Selesaikan run dengan resume_from, keluarkan dari daftar blend, atau set allow_partial=True.",
                )
            has_partial = True

        paths.append(rd)
        configs.append(cfg)
        scores_list.append(sc)
        raw_oofs.append(oof_df)
        raw_test_preds.append(tp_df)

    cfg0 = configs[0]
    sc0 = scores_list[0]
    folds_hash0 = str(sc0.get("folds_hash", ""))
    test_ids_hash0 = str(sc0.get("test_ids_hash", ""))
    classes0 = sc0.get("classes")
    id_col = cfg0.id_col

    for i in range(1, len(paths)):
        cfgi = configs[i]
        sci = scores_list[i]
        if str(sci.get("folds_hash", "")) != folds_hash0:
            raise FoldMismatchError(
                f"folds_hash pada run {paths[i].name!r} berbeda dengan {paths[0].name!r} (lihat [Perbaikan]).",
                "Jalankan ulang run dengan folds_path yang sama agar pembagian fold identik.",
            )
        if str(sci.get("test_ids_hash", "")) != test_ids_hash0:
            raise FoldMismatchError(
                f"test_ids_hash pada run {paths[i].name!r} berbeda dengan {paths[0].name!r} (lihat [Perbaikan]).",
                "Pastikan semua run dievaluasi pada dataset test dengan himpunan ID yang sama.",
            )
        if (
            cfgi.task != cfg0.task
            or cfgi.metric != cfg0.metric
            or sci.get("classes") != classes0
        ):
            raise ConfigError(
                f"Konfigurasi task/metric/classes pada run {paths[i].name!r} tidak cocok dengan {paths[0].name!r} (lihat [Perbaikan]).",
                "Hanya blend run yang memiliki task, metric, dan classes yang persis sama.",
            )
        if cfgi.id_col != id_col or cfgi.target != cfg0.target:
            raise ConfigError(
                f"Kolom id_col/target pada run {paths[i].name!r} berbeda dengan {paths[0].name!r} (lihat [Perbaikan]).",
                "Pastikan semua run memiliki id_col dan target yang sama.",
            )

    oof0 = raw_oofs[0].copy()
    tp0 = raw_test_preds[0].copy()
    oof0_cid = canon_id(oof0[id_col]).to_numpy()
    tp0_cid = canon_id(tp0[id_col]).to_numpy()

    pred_cols = [
        c for c in oof0.columns if c not in (id_col, "fold", cfg0.target)
    ]
    if not pred_cols:
        raise ConfigError(
            f"Tidak ditemukan kolom prediksi pada oof.parquet di {paths[0].name!r} (lihat [Perbaikan]).",
            "Pastikan oof.parquet memiliki kolom pred (atau pred_0..pred_{K-1}).",
        )

    aligned_oofs: list[pd.DataFrame] = [oof0]
    aligned_tps: list[pd.DataFrame] = [tp0]

    for i in range(1, len(paths)):
        oofi = raw_oofs[i]
        tpi = raw_test_preds[i]

        oofi_cid = canon_id(oofi[id_col]).to_numpy()
        if len(oofi_cid) != len(oof0_cid) or set(oofi_cid) != set(oof0_cid):
            raise FoldMismatchError(
                f"Himpunan ID OOF pada run {paths[i].name!r} berbeda dengan {paths[0].name!r} (lihat [Perbaikan]).",
                "Jalankan ulang run dengan folds_path dan data train yang sama.",
            )
        tpi_cid = canon_id(tpi[id_col]).to_numpy()
        if len(tpi_cid) != len(tp0_cid) or set(tpi_cid) != set(tp0_cid):
            raise FoldMismatchError(
                f"Himpunan ID test pada run {paths[i].name!r} berbeda dengan {paths[0].name!r} (lihat [Perbaikan]).",
                "Pastikan semua run menggunakan data test yang sama.",
            )

        if list(oofi_cid) != list(oof0_cid):
            oofi_aligned = (
                oofi.set_index(oofi_cid).loc[oof0_cid].reset_index(drop=True)
            )
            oofi_aligned[id_col] = oof0[id_col].to_numpy()
        else:
            oofi_aligned = oofi.copy()

        if list(tpi_cid) != list(tp0_cid):
            tpi_aligned = (
                tpi.set_index(tpi_cid).loc[tp0_cid].reset_index(drop=True)
            )
            tpi_aligned[id_col] = tp0[id_col].to_numpy()
        else:
            tpi_aligned = tpi.copy()

        aligned_oofs.append(oofi_aligned)
        aligned_tps.append(tpi_aligned)

    return EnsembleData(
        run_dirs=paths,
        configs=configs,
        scores_list=scores_list,
        oofs=aligned_oofs,
        test_preds=aligned_tps,
        pred_cols=pred_cols,
        folds_hash=folds_hash0,
        test_ids_hash=test_ids_hash0,
        classes=list(classes0) if classes0 is not None else None,
        has_partial=has_partial,
    )


def _extract_pred_array(df: pd.DataFrame, pred_cols: list[str]) -> np.ndarray:
    if len(pred_cols) == 1:
        return df[pred_cols[0]].to_numpy(dtype=float)
    return df[pred_cols].to_numpy(dtype=float)


def _weighted_sum(weights: np.ndarray, arrays: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(arrays[0], dtype=float)
    for w, arr in zip(weights, arrays):
        out = out + float(w) * arr
    return out


def blend(
    data: EnsembleData,
    method: Literal["mean", "weights"] = "weights",
    postprocess: Optional[Callable[..., Optional[dict[str, Any]]]] = None,
    runs_dir: Union[str, Path] = "runs",
    run_name: Optional[str] = None,
) -> RunResult:
    """Blend OOF and test predictions via uniform mean or SLSQP optimal weights and record run."""
    if method not in ("mean", "weights"):
        raise ConfigError(
            f"Method blend {method!r} tidak dikenali (lihat [Perbaikan]).",
            "Gunakan method=mean atau method=weights.",
        )

    cfg0 = data.configs[0]
    id_col = cfg0.id_col
    target_col = cfg0.target
    pred_cols = data.pred_cols
    m_runs = len(data.oofs)

    oof_arrs = [_extract_pred_array(df, pred_cols) for df in data.oofs]
    test_arrs = [_extract_pred_array(df, pred_cols) for df in data.test_preds]

    n_rows = len(data.oofs[0])
    valid_mask = np.ones(n_rows, dtype=bool)
    for arr in oof_arrs:
        if arr.ndim == 1:
            valid_mask &= ~np.isnan(arr)
        else:
            valid_mask &= ~np.isnan(arr).any(axis=1)

    folds = data.oofs[0]["fold"].to_numpy(dtype=int)
    y_all = data.oofs[0][target_col].to_numpy()

    eval_folds = (
        [k for k in range(1, cfg0.n_folds)]
        if cfg0.cv_strategy == "timeseries"
        else [k for k in range(cfg0.n_folds)]
    )
    active_mask = valid_mask & (folds >= 0)
    eval_mask = valid_mask & np.isin(folds, eval_folds)
    n_rows_eval = int(np.sum(eval_mask))

    score_mask = eval_mask if n_rows_eval > 0 else active_mask

    clip_range: Optional[tuple[int, int]] = None
    if cfg0.metric == "qwk" and cfg0.task == "regression":
        valid_targets = data.oofs[0][target_col].dropna()
        if len(valid_targets) > 0:
            clip_range = (int(np.min(valid_targets)), int(np.max(valid_targets)))

    m_desc = get_metric(cfg0.metric)
    w_uniform = np.full(m_runs, 1.0 / m_runs, dtype=float)

    def _eval_weights(w: np.ndarray, mask: np.ndarray) -> float:
        blended = _weighted_sum(w, [a[mask] for a in oof_arrs])
        return float(
            score(
                y_all[mask],
                blended,
                cfg0.metric,
                cfg0.task,
                threshold=0.5,
                classes=data.classes,
                clip_range=clip_range,
            )
        )

    def _optimize_weights(mask: np.ndarray) -> np.ndarray:
        if m_runs == 1 or int(np.sum(mask)) == 0:
            return w_uniform.copy()

        mask_arrs = [a[mask] for a in oof_arrs]
        y_m = y_all[mask]

        def obj(w: np.ndarray) -> float:
            w_norm = np.clip(w, 0.0, 1.0)
            s_w = float(np.sum(w_norm))
            if s_w <= 0.0:
                w_norm = w_uniform
            else:
                w_norm = w_norm / s_w
            blended = _weighted_sum(w_norm, mask_arrs)
            val = float(
                score(
                    y_m,
                    blended,
                    cfg0.metric,
                    cfg0.task,
                    threshold=0.5,
                    classes=data.classes,
                    clip_range=clip_range,
                )
            )
            return -val if m_desc.direction == "maximize" else val

        res = minimize(
            obj,
            w_uniform.copy(),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * m_runs,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
            options={"maxiter": 200, "ftol": 1e-7},
        )
        w_cand = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
        if float(np.sum(w_cand)) <= 0.0:
            w_cand = w_uniform.copy()
        else:
            w_cand = w_cand / float(np.sum(w_cand))

        best_w = w_uniform.copy()
        best_s = _eval_weights(best_w, mask)
        cand_s = _eval_weights(w_cand, mask)
        if is_better(cand_s, best_s, cfg0.metric):
            best_w = w_cand
            best_s = cand_s

        # Also check single-model corners in case SLSQP zero-gradient on step metrics missed a better corner
        for idx in range(m_runs):
            e_w = np.zeros(m_runs, dtype=float)
            e_w[idx] = 1.0
            e_s = _eval_weights(e_w, mask)
            if is_better(e_s, best_s, cfg0.metric):
                best_w = e_w
                best_s = e_s

        return best_w

    oof_uniform = _weighted_sum(w_uniform, oof_arrs)
    cv_raw = (
        float(
            score(
                y_all[score_mask],
                oof_uniform[score_mask],
                cfg0.metric,
                cfg0.task,
                threshold=0.5,
                classes=data.classes,
                clip_range=clip_range,
            )
        )
        if int(np.sum(score_mask)) > 0
        else float("nan")
    )
    cv_raw_all = (
        float(
            score(
                y_all[active_mask],
                oof_uniform[active_mask],
                cfg0.metric,
                cfg0.task,
                threshold=0.5,
                classes=data.classes,
                clip_range=clip_range,
            )
        )
        if int(np.sum(active_mask)) > 0
        else float("nan")
    )

    if method == "mean":
        w_full = w_uniform.copy()
    else:
        w_full = _optimize_weights(score_mask)

    oof_blend = _weighted_sum(w_full, oof_arrs)
    test_blend = _weighted_sum(w_full, test_arrs)

    oof_df = data.oofs[0][[id_col, "fold", target_col]].copy()
    test_pred_df = data.test_preds[0][[id_col]].copy()
    if len(pred_cols) == 1:
        oof_df[pred_cols[0]] = oof_blend
        test_pred_df[pred_cols[0]] = test_blend
    else:
        for j, c in enumerate(pred_cols):
            oof_df[c] = oof_blend[:, j]
            test_pred_df[c] = test_blend[:, j]

    oof_active_df = oof_df[active_mask].copy()
    params: dict[str, Any] = (
        (postprocess(oof_active_df.copy(), test_pred_df.copy()) or {})
        if postprocess is not None
        else {}
    )

    if method == "mean" and not params:
        cv_post = cv_raw
        cv_honest = cv_raw
    else:
        thr_full = float(params.get("threshold", 0.5))
        cv_post = (
            float(
                score(
                    y_all[score_mask],
                    oof_blend[score_mask],
                    cfg0.metric,
                    cfg0.task,
                    threshold=thr_full,
                    classes=data.classes,
                    clip_range=clip_range,
                )
            )
            if int(np.sum(score_mask)) > 0
            else float("nan")
        )

        eval_folds_present = [
            k for k in eval_folds if int(np.sum(score_mask & (folds == k))) > 0
        ]
        if not eval_folds_present and int(np.sum(score_mask)) > 0:
            eval_folds_present = sorted(np.unique(folds[score_mask]).tolist())

        honest_preds_full = oof_blend.copy()
        for k in eval_folds_present:
            mask_k = score_mask & (folds == k)
            if cfg0.cv_strategy == "timeseries":
                src_mask = active_mask & (folds >= 0) & (folds < k)
            else:
                src_mask = active_mask & (folds != k)

            if int(np.sum(src_mask)) > 0:
                w_k = _optimize_weights(src_mask) if method == "weights" else w_uniform
                pred_k_all = _weighted_sum(w_k, oof_arrs)
                if postprocess is not None:
                    src_df = oof_df.loc[src_mask].copy()
                    if len(pred_cols) == 1:
                        src_df[pred_cols[0]] = pred_k_all[src_mask]
                    else:
                        for j, c in enumerate(pred_cols):
                            src_df[c] = pred_k_all[src_mask, j]
                    params_k = postprocess(src_df, None) or {}
                else:
                    params_k = {}
            else:
                w_k = w_uniform
                pred_k_all = oof_uniform
                params_k = {}

            fold_pred = pred_k_all[mask_k]
            thr_k = float(params_k.get("threshold", 0.5))
            if (
                postprocess is not None
                and cfg0.task == "binary"
                and m_desc.input_type == "label"
                and fold_pred.ndim == 1
            ):
                fold_pred = (fold_pred >= thr_k).astype(float)
            honest_preds_full[mask_k] = fold_pred

        cv_honest = (
            float(
                score(
                    y_all[score_mask],
                    honest_preds_full[score_mask],
                    cfg0.metric,
                    cfg0.task,
                    threshold=0.5,
                    classes=data.classes,
                    clip_range=clip_range,
                )
            )
            if int(np.sum(score_mask)) > 0
            else float("nan")
        )

    cv = cv_honest

    per_fold: dict[str, dict[str, Any]] = {}
    for k in range(cfg0.n_folds):
        mask_fk = active_mask & (folds == k)
        if int(np.sum(mask_fk)) == 0:
            continue
        raw_k = float(
            score(
                y_all[mask_fk],
                oof_blend[mask_fk],
                cfg0.metric,
                cfg0.task,
                threshold=0.5,
                classes=data.classes,
                clip_range=clip_range,
            )
        )
        per_fold[str(k)] = {
            "raw": raw_k,
            "n_rows": int(np.sum(mask_fk)),
            "duration_sec": 0.0,
        }

    n_folds_done = int(len(per_fold))
    eval_fold_vals = [
        per_fold[str(k)]["raw"]
        for k in eval_folds
        if str(k) in per_fold and not np.isnan(per_fold[str(k)]["raw"])
    ]
    cv_std = float(np.std(eval_fold_vals)) if len(eval_fold_vals) > 0 else 0.0

    status: Literal["ok", "partial"] = (
        "partial" if (data.has_partial or n_folds_done < cfg0.n_folds) else "ok"
    )

    cfg_ens = BaseConfig(
        task=cfg0.task,
        target=cfg0.target,
        metric=cfg0.metric,
        id_col=cfg0.id_col,
        run_name=run_name or f"blend_{method}",
        owner=cfg0.owner,
        modality="ensemble",
        seed=cfg0.seed,
        fold_seed=cfg0.fold_seed,
        n_folds=cfg0.n_folds,
        cv_strategy=cfg0.cv_strategy,
        group_col=cfg0.group_col,
        time_col=cfg0.time_col,
        ts_gap=cfg0.ts_gap,
        test_mode="fold_avg",
        runs_dir=str(runs_dir),
        debug=False,
        notes=f"Blend ({method}) of {[d.name for d in data.run_dirs]}",
        extra={
            "method": method,
            "sources": [d.name for d in data.run_dirs],
            "weights": [float(x) for x in w_full],
        },
    )
    cfg_ens.resolve()
    cfg_ens.validate()

    model_name = f"blend_{method}"
    rm = RunManager(cfg_ens, model_name=model_name)

    scores: dict[str, Any] = {
        "model": model_name,
        "timestamp": rm.timestamp,
        "owner": str(rm.cfg.owner),
        "minutes": 0.0,
        "data_hash": str(data.folds_hash),
        "metric": str(rm.cfg.metric),
        "direction": str(m_desc.direction),
        "per_fold": per_fold,
        "refit_duration_sec": 0.0,
        "cv": float(cv),
        "cv_raw": float(cv_raw),
        "cv_post": float(cv_post),
        "cv_honest": float(cv_honest),
        "cv_std": float(cv_std),
        "cv_raw_all": float(cv_raw_all),
        "eval_folds": list(eval_folds),
        "n_rows_eval": int(n_rows_eval),
        "folds_hash": str(data.folds_hash),
        "test_ids_hash": str(data.test_ids_hash),
        "classes": list(data.classes) if data.classes is not None else None,
        "status": str(status),
        "test_pred_source": "fold_avg",
        "n_folds_done": int(n_folds_done),
    }

    rm.set_status(status)
    rm.save_oof(oof_df)
    rm.save_test_pred(test_pred_df)
    rm.save_postprocess(params)
    rm.save_scores(scores)

    generate_report(
        rm.run_dir,
        rm.cfg,
        scores,
        oof_df,
        params,
        importances=None,
        classes=data.classes,
    )
    rm.append_leaderboard(scores)

    return RunResult(
        run_dir=rm.run_dir,
        oof=oof_df,
        test_pred=test_pred_df,
        scores=scores,
        params=params,
        cfg=copy.deepcopy(rm.cfg),
        classes=list(data.classes) if data.classes is not None else None,
    )


__all__ = [
    "EnsembleData",
    "load_runs",
    "blend",
]
