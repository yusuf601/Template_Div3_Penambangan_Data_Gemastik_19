"""BasePipeline with 3 primary hooks, injectable clock time budget, resume, refit guard, and RunResult."""

from collections import defaultdict
import copy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Optional, Union
import warnings

import numpy as np
import pandas as pd

from utils.core.config import BaseConfig
from utils.core.cv import (
    compute_ids_target_hash,
    debug_subsample,
    get_folds,
    split_fold,
)
from utils.core.errors import DataError, HookContractError, SubmissionError
from utils.core.io import canon_id
from utils.core.metrics import get_metric, score
from utils.core.report import generate_report
from utils.core.run import RunManager
from utils.core.seed import seed_everything

# 1.25x matches 5-fold CV where full train is 1.25x of a 4/5 train fold;
# acts as a conservative lower bound for 3-fold CV where full train is ~1.5x.
REFIT_TIME_MARGIN: float = 1.25


def _extract_raw_preds(df_slice: pd.DataFrame, pred_cols: list[str]) -> np.ndarray:
    """Extract 1D or 2D float numpy array of predictions from df_slice."""
    if len(pred_cols) == 1:
        return df_slice[pred_cols[0]].to_numpy(dtype=float)
    return df_slice[pred_cols].to_numpy(dtype=float)


def compute_scores(
    oof_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    pred_cols: list[str],
    cfg: BaseConfig,
    postprocess_fn: Callable[..., Optional[dict[str, Any]]],
    model_name: str,
    timestamp: str,
    minutes: float,
    refit_duration_sec: float,
    fold_info: dict[int, dict[str, Any]],
    folds_hash: str,
    test_ids_hash: str,
    data_hash: str,
    status: str,
    test_pred_source: str,
    classes: Optional[list[Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute per-fold and pooled OOF scores (cv_raw, cv_post, cv_honest, cv) on eval_folds."""
    eval_folds = (
        [k for k in range(1, cfg.n_folds)]
        if cfg.cv_strategy == "timeseries"
        else [k for k in range(cfg.n_folds)]
    )

    valid_pred_mask = oof_df[pred_cols].notna().all(axis=1)
    oof_active_df = oof_df[(oof_df["fold"] >= 0) & valid_pred_mask].copy()
    oof_eval_df = oof_df[oof_df["fold"].isin(eval_folds) & valid_pred_mask].copy()
    n_rows_eval = int(len(oof_eval_df))

    clip_range: Optional[tuple[int, int]] = None
    if cfg.metric == "qwk" and cfg.task == "regression":
        valid_targets = oof_df[cfg.target].dropna()
        if len(valid_targets) > 0:
            clip_range = (int(np.min(valid_targets)), int(np.max(valid_targets)))

    per_fold: dict[str, dict[str, Any]] = {}
    for k in range(cfg.n_folds):
        fk_df = oof_active_df[oof_active_df["fold"] == k]
        if len(fk_df) == 0:
            continue
        raw_k = score(
            fk_df[cfg.target].to_numpy(),
            _extract_raw_preds(fk_df, pred_cols),
            cfg.metric,
            cfg.task,
            threshold=0.5,
            classes=classes,
            clip_range=clip_range,
        )
        info_k = fold_info.get(k, fold_info.get(str(k), {})) or {}  # type: ignore[arg-type]
        per_fold[str(k)] = {
            "raw": float(raw_k),
            "n_rows": int(len(fk_df)),
            "duration_sec": float(info_k.get("duration_sec", 0.0)),
        }

    n_folds_done = int(len(per_fold))
    eval_fold_vals = [
        per_fold[str(k)]["raw"]
        for k in eval_folds
        if str(k) in per_fold and not np.isnan(per_fold[str(k)]["raw"])
    ]
    cv_std = float(np.std(eval_fold_vals)) if len(eval_fold_vals) > 0 else 0.0

    cv_raw_all = (
        float(
            score(
                oof_active_df[cfg.target].to_numpy(),
                _extract_raw_preds(oof_active_df, pred_cols),
                cfg.metric,
                cfg.task,
                threshold=0.5,
                classes=classes,
                clip_range=clip_range,
            )
        )
        if len(oof_active_df) > 0
        else float("nan")
    )

    eval_target_df = oof_eval_df if len(oof_eval_df) > 0 else oof_active_df
    cv_raw = (
        float(
            score(
                eval_target_df[cfg.target].to_numpy(),
                _extract_raw_preds(eval_target_df, pred_cols),
                cfg.metric,
                cfg.task,
                threshold=0.5,
                classes=classes,
                clip_range=clip_range,
            )
        )
        if len(eval_target_df) > 0
        else float("nan")
    )

    params: dict[str, Any] = postprocess_fn(oof_active_df.copy(), test_pred_df.copy()) or {}

    if not params:
        cv_post = cv_raw
        cv_honest = cv_raw
    else:
        thr_full = float(params.get("threshold", 0.5))
        cv_post = (
            float(
                score(
                    eval_target_df[cfg.target].to_numpy(),
                    _extract_raw_preds(eval_target_df, pred_cols),
                    cfg.metric,
                    cfg.task,
                    threshold=thr_full,
                    classes=classes,
                    clip_range=clip_range,
                )
            )
            if len(eval_target_df) > 0
            else float("nan")
        )

        m = get_metric(cfg.metric)
        eval_folds_present = [
            k for k in eval_folds if (eval_target_df["fold"] == k).any()
        ]
        if not eval_folds_present and len(eval_target_df) > 0:
            eval_folds_present = sorted(eval_target_df["fold"].unique().tolist())

        honest_raw = _extract_raw_preds(eval_target_df, pred_cols).copy()
        fold_arr = eval_target_df["fold"].to_numpy(dtype=int)

        for k in eval_folds_present:
            if cfg.cv_strategy == "timeseries":
                src_oof = oof_active_df[
                    (oof_active_df["fold"] >= 0) & (oof_active_df["fold"] < k)
                ].copy()
            else:
                src_oof = oof_active_df[oof_active_df["fold"] != k].copy()

            params_k = (postprocess_fn(src_oof, None) or {}) if len(src_oof) > 0 else {}
            thr_k = float(params_k.get("threshold", 0.5))
            if cfg.task == "binary" and m.input_type == "label" and honest_raw.ndim == 1:
                mask_k = fold_arr == k
                honest_raw[mask_k] = (honest_raw[mask_k] >= thr_k).astype(float)

        cv_honest = (
            float(
                score(
                    eval_target_df[cfg.target].to_numpy(),
                    honest_raw,
                    cfg.metric,
                    cfg.task,
                    threshold=0.5,
                    classes=classes,
                    clip_range=clip_range,
                )
            )
            if len(eval_target_df) > 0
            else float("nan")
        )

    cv = cv_honest
    m_desc = get_metric(cfg.metric)

    scores: dict[str, Any] = {
        "model": str(model_name),
        "timestamp": str(timestamp),
        "owner": str(cfg.owner),
        "minutes": float(minutes),
        "data_hash": str(data_hash),
        "metric": str(cfg.metric),
        "direction": str(m_desc.direction),
        "per_fold": per_fold,
        "refit_duration_sec": float(refit_duration_sec),
        "cv": float(cv),
        "cv_raw": float(cv_raw),
        "cv_post": float(cv_post),
        "cv_honest": float(cv_honest),
        "cv_std": float(cv_std),
        "cv_raw_all": float(cv_raw_all),
        "eval_folds": list(eval_folds),
        "n_rows_eval": int(n_rows_eval),
        "folds_hash": str(folds_hash),
        "test_ids_hash": str(test_ids_hash),
        "classes": list(classes) if classes is not None else None,
        "status": str(status),
        "test_pred_source": str(test_pred_source),
        "n_folds_done": int(n_folds_done),
    }
    return scores, params


@dataclass
class RunResult:
    """Container for completed pipeline run artifacts and submission generator."""

    run_dir: Path
    oof: pd.DataFrame
    test_pred: pd.DataFrame
    scores: dict[str, Any]
    params: dict[str, Any]
    cfg: BaseConfig
    classes: Optional[list[Any]] = None

    def make_submission(
        self,
        sample_sub: pd.DataFrame,
        path: Union[str, Path],
        fn: Optional[Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Validate sample_sub against test IDs, align row order, format predictions, and write CSV."""
        id_col = self.cfg.id_col
        if id_col not in sample_sub.columns:
            raise SubmissionError(
                f"Kolom id {id_col!r} tidak ditemukan pada sample_sub.",
                f"Pastikan sample_sub memiliki kolom {id_col!r} sesuai cfg.id_col.",
            )

        # Check NaN on id_col FIRST before duplicate or set comparison
        if sample_sub[id_col].isna().any():
            raise SubmissionError(
                f"Kolom id {id_col!r} pada sample_sub mengandung nilai NaN.",
                f"Hapus atau perbaiki nilai NaN pada kolom {id_col!r} di sample_sub.",
            )

        sub_cid = canon_id(sample_sub[id_col])
        if sub_cid.duplicated().any():
            raise SubmissionError(
                f"Kolom id {id_col!r} pada sample_sub mengandung ID duplikat.",
                f"Pastikan seluruh nilai pada kolom {id_col!r} di sample_sub unik.",
            )

        test_cid = canon_id(self.test_pred[id_col])
        if len(sub_cid) != len(test_cid) or set(sub_cid.to_numpy()) != set(test_cid.to_numpy()):
            raise SubmissionError(
                f"Himpunan ID pada sample_sub (n={len(sub_cid)}) tidak cocok dengan ID pada test (n={len(test_cid)}).",
                "Pastikan sample_sub berasal dari test set yang sama dengan yang diberikan ke pipeline.run(train, test).",
            )

        pred_cols = [c for c in self.test_pred.columns if c != id_col]
        tp_indexed = self.test_pred.set_index(test_cid.to_numpy())
        aligned_tp = tp_indexed.loc[sub_cid.to_numpy()].reset_index(drop=True)
        aligned_tp[id_col] = sample_sub[id_col].to_numpy()

        if fn is not None:
            out_df = fn(aligned_tp.copy(), copy.deepcopy(self.params))
            if not isinstance(out_df, pd.DataFrame) or id_col not in out_df.columns:
                raise SubmissionError(
                    "Fungsi fn pada make_submission tidak mengembalikan DataFrame valid dengan kolom id.",
                    f"Pastikan fn(test_pred, params) mengembalikan pd.DataFrame dengan kolom {id_col!r}.",
                )
            out_cid = canon_id(out_df[id_col])
            if len(out_cid) != len(sub_cid) or list(out_cid) != list(sub_cid):
                raise SubmissionError(
                    "Urutan atau nilai ID hasil fn pada make_submission tidak sesuai dengan sample_sub.",
                    "Jangan mengubah urutan atau nilai kolom id di dalam fn.",
                )
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(p, index=False)
            return out_df

        sub_cols = [c for c in sample_sub.columns if c != id_col]
        out_df = sample_sub.copy()
        m = get_metric(self.cfg.metric)

        if len(sub_cols) == 1:
            tcol = sub_cols[0]
            if self.cfg.task == "multiclass":
                probs = aligned_tp[pred_cols].to_numpy(dtype=float)
                idx = np.argmax(probs, axis=1)
                if self.classes is not None and len(self.classes) == len(pred_cols):
                    out_df[tcol] = [self.classes[i] for i in idx]
                else:
                    out_df[tcol] = idx
            elif self.cfg.task == "binary":
                raw = aligned_tp[pred_cols[0]].to_numpy(dtype=float)
                if m.input_type == "label":
                    thr = float((self.params or {}).get("threshold", 0.5))
                    lbl_idx = (raw >= thr).astype(int)
                    if self.classes is not None and len(self.classes) == 2:
                        out_df[tcol] = [self.classes[i] for i in lbl_idx]
                    else:
                        out_df[tcol] = lbl_idx
                else:
                    out_df[tcol] = raw
            else:
                raw = aligned_tp[pred_cols[0]].to_numpy(dtype=float)
                if m.input_type == "label":
                    out_df[tcol] = np.rint(raw).astype(int)
                else:
                    out_df[tcol] = raw

        elif (
            self.cfg.task == "multiclass"
            and len(sub_cols) == len(pred_cols)
            and len(sub_cols) > 1
        ):
            if self.classes is not None and set(map(str, sub_cols)) == set(
                map(str, self.classes)
            ):
                cls_to_pcol = {
                    str(cls_val): f"pred_{i}" for i, cls_val in enumerate(self.classes)
                }
                for col in sub_cols:
                    out_df[col] = aligned_tp[cls_to_pcol[str(col)]].to_numpy(dtype=float)
            else:
                warnings.warn(
                    "Nama kolom sample_sub tidak cocok dengan classes; mengisi kolom secara posisional (pred_0..pred_{K-1}).",
                    UserWarning,
                    stacklevel=2,
                )
                for i, col in enumerate(sub_cols):
                    out_df[col] = aligned_tp[pred_cols[i]].to_numpy(dtype=float)
        else:
            raise SubmissionError(
                f"Format kolom sample_sub {sub_cols!r} tidak dapat dipetakan otomatis dari prediksi {pred_cols!r}.",
                "Berikan fn= ke make_submission(sample_sub, path, fn=...) untuk memformat kolom submission khusus.",
            )

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(p, index=False)
        return out_df


class BasePipeline:
    """Base experiment pipeline with 3 primary hooks, time budget guard, and checkpoint resume."""

    model_name: str = "base"

    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg: BaseConfig = copy.deepcopy(cfg)
        self.fold_info: defaultdict[int, dict[str, Any]] = defaultdict(dict)
        self.importances: Optional[pd.Series] = None
        self.classes_: Optional[list[Any]] = None
        self._clock: Callable[[], float] = time.monotonic

    # --- 3 Primary Hooks ---

    def build_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Global feature engineering without target leakage. Do not add or drop rows."""
        return train, test

    def fit_transform_fold(
        self,
        tr: pd.DataFrame,
        va: pd.DataFrame,
        te: pd.DataFrame,
        fold: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Per-fold stateful transformations fit only on tr. When fold == -1, va is empty."""
        return tr, va, te

    def build_model(self, fold: int) -> Any:
        """Return a fresh unfitted model instance for the given fold."""
        raise NotImplementedError(
            "[Masalah]   build_model(fold) belum diimplementasikan.\n"
            "[Perbaikan] Override method build_model(self, fold) pada subclass BasePipeline."
        )

    # --- 4 Advanced Hooks & Helpers ---

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return feature column names excluding id, target, group, time, and fold."""
        excluded = {
            self.cfg.id_col,
            self.cfg.target,
            self.cfg.group_col,
            self.cfg.time_col,
            "fold",
        }
        return [c for c in df.columns if c not in excluded]

    def fit_predict_fold(
        self,
        model: Any,
        tr: pd.DataFrame,
        va: pd.DataFrame,
        te: pd.DataFrame,
        fold: int,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Fit sklearn-compatible model on tr and predict on va (if non-empty) and te."""
        feat_cols = self.get_feature_cols(tr)
        if not feat_cols:
            raise HookContractError(
                "Tidak ada kolom fitur yang tersisa pada tr untuk melatih model.",
                "Pastikan DataFrame memiliki minimal 1 kolom fitur numerik selain id/target/group/time/fold.",
            )

        if len(va) > 0:
            missing_va = [c for c in feat_cols if c not in va.columns]
            if missing_va:
                raise HookContractError(
                    f"Kolom fitur {missing_va!r} hilang pada va di fold {fold}.",
                    "Pastikan fit_transform_fold menghasilkan kolom fitur yang sama pada tr, va, dan te.",
                )

        missing_te = [c for c in feat_cols if c not in te.columns]
        if missing_te:
            raise HookContractError(
                f"Kolom fitur {missing_te!r} hilang pada te di fold {fold}.",
                "Pastikan build_features dan fit_transform_fold menghasilkan kolom fitur yang sama pada tr dan te.",
            )

        non_num_tr = [
            c for c in feat_cols if not pd.api.types.is_numeric_dtype(tr[c].dtype)
        ]
        non_num_va = (
            [c for c in feat_cols if not pd.api.types.is_numeric_dtype(va[c].dtype)]
            if len(va) > 0
            else []
        )
        non_num_te = [
            c for c in feat_cols if not pd.api.types.is_numeric_dtype(te[c].dtype)
        ]
        bad_cols = list(dict.fromkeys(non_num_tr + non_num_va + non_num_te))
        if bad_cols:
            raise HookContractError(
                f"Kolom fitur non-numerik ditemukan: {bad_cols!r}.",
                f"Encode atau buang kolom {bad_cols!r} di build_features / fit_transform_fold, atau override get_feature_cols.",
            )

        model.fit(tr[feat_cols], tr[self.cfg.target])

        if fold >= 0 and "importances" not in self.fold_info[fold]:
            if hasattr(model, "feature_importances_"):
                imp_arr = np.asarray(model.feature_importances_, dtype=float).ravel()
                if len(imp_arr) == len(feat_cols):
                    self.fold_info[fold]["importances"] = {
                        str(c): float(v) for c, v in zip(feat_cols, imp_arr)
                    }
            elif hasattr(model, "coef_"):
                coef_arr = np.mean(np.abs(np.atleast_2d(model.coef_)), axis=0).ravel()
                if len(coef_arr) == len(feat_cols):
                    self.fold_info[fold]["importances"] = {
                        str(c): float(v) for c, v in zip(feat_cols, coef_arr)
                    }

        def _predict_part(df_part: pd.DataFrame) -> np.ndarray:
            if len(df_part) == 0:
                if self.cfg.task == "multiclass":
                    n_cls = len(self.classes_) if self.classes_ is not None else 0
                    return np.empty((0, n_cls), dtype=float)
                return np.empty(0, dtype=float)

            X_part = df_part[feat_cols]
            if self.cfg.task == "binary":
                if hasattr(model, "predict_proba"):
                    proba = np.asarray(model.predict_proba(X_part), dtype=float)
                    if (
                        hasattr(model, "classes_")
                        and self.classes_ is not None
                        and len(self.classes_) == 2
                    ):
                        pos_cls = self.classes_[1]
                        m_cls = list(model.classes_)
                        if pos_cls in m_cls:
                            return proba[:, m_cls.index(pos_cls)]
                    return proba[:, 1]
                return np.asarray(model.predict(X_part), dtype=float).ravel()

            if self.cfg.task == "multiclass":
                proba = np.asarray(model.predict_proba(X_part), dtype=float)
                if hasattr(model, "classes_") and self.classes_ is not None:
                    m_cls = list(model.classes_)
                    if m_cls != list(self.classes_):
                        aligned = np.zeros((len(df_part), len(self.classes_)), dtype=float)
                        for j, c_val in enumerate(self.classes_):
                            if c_val in m_cls:
                                aligned[:, j] = proba[:, m_cls.index(c_val)]
                        return aligned
                return proba

            return np.asarray(model.predict(X_part), dtype=float).ravel()

        va_pred = _predict_part(va)
        te_pred = _predict_part(te)
        return model, va_pred, te_pred

    def postprocess(
        self, oof_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
    ) -> dict[str, Any]:
        """Pure postprocessing parameter estimation (e.g., threshold optimization)."""
        return {}

    def refit_iterations(self) -> Optional[int]:
        """Return round(mean(best_iteration) * cfg.refit_iter_mult) across completed folds, or None."""
        iters = [
            float(info["best_iteration"])
            for k, info in self.fold_info.items()
            if int(k) >= 0 and info.get("best_iteration") is not None
        ]
        if not iters:
            return None
        return int(round(float(np.mean(iters)) * float(self.cfg.refit_iter_mult)))

    def fit_full_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Fit on 100% of train (fold=-1, empty va) and return test predictions aligned to test."""
        id_col = self.cfg.id_col
        empty_va = train.iloc[:0].copy()
        tr_f, va_f, te_f = self.fit_transform_fold(
            train.copy(), empty_va, test.copy(), -1
        )

        if len(va_f) != 0:
            raise HookContractError(
                f"fit_transform_fold pada fold=-1 (refit) harus mengembalikan va kosong, tetapi len(va)={len(va_f)}.",
                "Saat fold == -1 (atau len(va) == 0), lewati transformasi va dan kembalikan va apa adanya.",
            )

        if id_col not in te_f.columns or te_f[id_col].isna().any():
            raise HookContractError(
                f"fit_transform_fold pada fold=-1 menghilangkan atau merusak kolom {id_col!r} pada te.",
                f"Jangan menghapus atau mengisi NaN pada kolom {id_col!r} di fit_transform_fold.",
            )

        orig_te_cid = canon_id(test[id_col])
        new_te_cid = canon_id(te_f[id_col])
        if len(new_te_cid) != len(orig_te_cid) or set(new_te_cid.to_numpy()) != set(
            orig_te_cid.to_numpy()
        ):
            raise HookContractError(
                "fit_transform_fold pada fold=-1 mengubah himpunan ID atau jumlah baris te.",
                "Jangan membuang atau menambah baris te di dalam fit_transform_fold.",
            )

        model = self.build_model(-1)
        _, _, te_pred = self.fit_predict_fold(model, tr_f, va_f, te_f, -1)

        if list(new_te_cid.to_numpy()) != list(orig_te_cid.to_numpy()):
            order_map = pd.Series(np.arange(len(new_te_cid)), index=new_te_cid.to_numpy())
            perm = order_map.loc[orig_te_cid.to_numpy()].to_numpy(dtype=int)
            te_pred = te_pred[perm]

        return te_pred

    # --- Main Execution Flow ---

    def run(self, train: pd.DataFrame, test: pd.DataFrame) -> RunResult:
        """Execute full CV pipeline with fold checkpointing, time budget guard, report, and leaderboard."""
        self.fold_info = defaultdict(dict)
        self.importances = None
        self.classes_ = None

        self.cfg.resolve()
        self.cfg.validate()
        seed_everything(self.cfg.seed)
        session_start = self._clock()

        rm = RunManager(self.cfg, model_name=self.model_name)
        self.cfg = copy.deepcopy(rm.cfg)

        try:
            id_col = self.cfg.id_col
            target_col = self.cfg.target

            if id_col not in train.columns:
                raise DataError(
                    f"Kolom id {id_col!r} tidak ditemukan pada train.",
                    f"Pastikan kolom {id_col!r} ada di train atau sesuaikan cfg.id_col.",
                )
            if id_col not in test.columns:
                raise DataError(
                    f"Kolom id {id_col!r} tidak ditemukan pada test.",
                    f"Pastikan kolom {id_col!r} ada di test atau sesuaikan cfg.id_col.",
                )
            if target_col not in train.columns:
                raise DataError(
                    f"Kolom target {target_col!r} tidak ditemukan pada train.",
                    f"Pastikan kolom {target_col!r} ada di train atau sesuaikan cfg.target.",
                )

            # Check NaN on id_col FIRST before duplicate checks
            if train[id_col].isna().any():
                raise DataError(
                    f"Kolom id {id_col!r} pada train mengandung nilai NaN.",
                    f"Hapus atau isi nilai NaN pada kolom {id_col!r} di train.",
                )
            if test[id_col].isna().any():
                raise DataError(
                    f"Kolom id {id_col!r} pada test mengandung nilai NaN.",
                    f"Hapus atau isi nilai NaN pada kolom {id_col!r} di test.",
                )

            if canon_id(train[id_col]).duplicated().any():
                raise DataError(
                    f"Kolom id {id_col!r} pada train mengandung ID duplikat.",
                    f"Pastikan setiap baris di train memiliki nilai unik pada kolom {id_col!r}.",
                )
            if canon_id(test[id_col]).duplicated().any():
                raise DataError(
                    f"Kolom id {id_col!r} pada test mengandung ID duplikat.",
                    f"Pastikan setiap baris di test memiliki nilai unik pada kolom {id_col!r}.",
                )

            if train[target_col].isna().any():
                raise DataError(
                    f"Kolom target {target_col!r} pada train mengandung nilai NaN.",
                    f"Hapus baris dengan target NaN pada kolom {target_col!r} sebelum menjalankan pipeline.",
                )

            train_work = train.copy()
            test_work = test.copy()

            if self.cfg.debug:
                train_work = debug_subsample(train_work, self.cfg, max_rows=2000)

            dv = rm.save_data_version(train_work, test_work)

            snap_cols = [id_col, target_col]
            if self.cfg.group_col and self.cfg.group_col in train_work.columns:
                snap_cols.append(self.cfg.group_col)
            if self.cfg.time_col and self.cfg.time_col in train_work.columns:
                snap_cols.append(self.cfg.time_col)
            snap = train_work[list(dict.fromkeys(snap_cols))].copy()

            if rm.is_resumed and (rm.run_dir / "folds.csv").exists():
                cfg_folds = copy.deepcopy(self.cfg)
                cfg_folds.folds_path = str(rm.run_dir / "folds.csv")
                folds_df, meta = get_folds(snap, cfg_folds, source_run=rm.run_dir.name)
            else:
                folds_df, meta = get_folds(snap, self.cfg, source_run=rm.run_dir.name)
                rm.save_folds(folds_df, meta)

            if self.cfg.task == "multiclass":
                self.classes_ = sorted(snap[target_col].unique().tolist())
            elif self.cfg.task == "binary":
                uniq_y = sorted(snap[target_col].unique().tolist())
                self.classes_ = uniq_y if uniq_y != [0, 1] else None

            train_fe, test_fe = self.build_features(train_work.copy(), test_work.copy())

            if (
                not isinstance(train_fe, pd.DataFrame)
                or not isinstance(test_fe, pd.DataFrame)
                or id_col not in train_fe.columns
                or id_col not in test_fe.columns
            ):
                raise HookContractError(
                    f"build_features harus mengembalikan (train, test) DataFrame yang memuat kolom {id_col!r}.",
                    f"Pastikan build_features tidak menghapus kolom {id_col!r}.",
                )

            if train_fe[id_col].isna().any() or test_fe[id_col].isna().any():
                raise HookContractError(
                    f"build_features menghasilkan nilai NaN pada kolom {id_col!r}.",
                    f"Jangan mengubah kolom {id_col!r} menjadi NaN di dalam build_features.",
                )

            snap_cid = canon_id(snap[id_col])
            tr_fe_cid = canon_id(train_fe[id_col])
            if len(tr_fe_cid) != len(snap_cid) or set(tr_fe_cid.to_numpy()) != set(
                snap_cid.to_numpy()
            ):
                raise HookContractError(
                    f"build_features mengubah himpunan ID atau jumlah baris train (sebelum={len(snap_cid)}, sesudah={len(tr_fe_cid)}).",
                    "Jangan membuang/menambah baris atau mengubah ID di build_features; biarkan nilai kosong sebagai NaN.",
                )

            te_orig_cid = canon_id(test_work[id_col])
            te_fe_cid = canon_id(test_fe[id_col])
            if len(te_fe_cid) != len(te_orig_cid) or set(te_fe_cid.to_numpy()) != set(
                te_orig_cid.to_numpy()
            ):
                raise HookContractError(
                    f"build_features mengubah himpunan ID atau jumlah baris test (sebelum={len(te_orig_cid)}, sesudah={len(te_fe_cid)}).",
                    "Jangan membuang/menambah baris atau mengubah ID test di build_features.",
                )

            # Check if target/group/time changed compared to pre-build_features snapshot
            check_cols = [c for c in snap.columns if c != id_col]
            mutated = False
            for c in check_cols:
                if c not in train_fe.columns:
                    mutated = True
                    break
            if not mutated:
                fe_hash = compute_ids_target_hash(train_fe, self.cfg)
                if fe_hash != meta["ids_target_hash"]:
                    mutated = True

            if mutated:
                warnings.warn(
                    "build_features mengubah kolom target/group/time dibanding snapshot awal; "
                    "model dilatih pada target hasil build_features, tetapi skor CV/OOF tetap dihitung "
                    "terhadap target mentah dari snapshot sehingga fit_predict_fold wajib mengembalikan prediksi dalam skala mentah.",
                    UserWarning,
                    stacklevel=2,
                )

            pred_cols = (
                [f"pred_{i}" for i in range(len(self.classes_))]
                if (self.cfg.task == "multiclass" and self.classes_ is not None)
                else ["pred"]
            )

            folds_cid = canon_id(folds_df[id_col])
            fold_map = pd.Series(
                folds_df["fold"].to_numpy(dtype=int), index=folds_cid.to_numpy()
            )
            aligned_folds = fold_map.loc[snap_cid.to_numpy()].to_numpy(dtype=int)

            oof_df = pd.DataFrame(
                {
                    id_col: snap[id_col].to_numpy(),
                    "fold": aligned_folds,
                    target_col: snap[target_col].to_numpy(),
                }
            )
            for c in pred_cols:
                oof_df[c] = np.nan

            snap_pos_map = pd.Series(
                np.arange(len(snap_cid)), index=snap_cid.to_numpy()
            )

            active_folds = list(range(self.cfg.n_folds))
            completed_folds: list[int] = []
            test_fold_preds: dict[int, pd.DataFrame] = {}

            for k in active_folds:
                ckpt = rm.load_fold_checkpoint(k) if rm.is_resumed else None
                if ckpt is not None:
                    oof_k_df, test_k_df, info_k = ckpt
                    self.fold_info[k].update(info_k)
                    k_cid = canon_id(oof_k_df[id_col]).to_numpy()
                    pos = snap_pos_map.loc[k_cid].to_numpy(dtype=int)
                    for c in pred_cols:
                        oof_df.loc[pos, c] = oof_k_df[c].to_numpy(dtype=float)
                    test_fold_preds[k] = test_k_df
                    completed_folds.append(k)
                    rm.log(f"Fold {k} loaded from checkpoint")
                    continue

                if len(completed_folds) >= 1 and self.cfg.max_minutes is not None:
                    elapsed_sec = self._clock() - session_start
                    if elapsed_sec >= self.cfg.max_minutes * 60.0:
                        rm.log(
                            f"Stopping before fold {k}: elapsed {elapsed_sec:.1f}s >= max_minutes ({self.cfg.max_minutes}m)"
                        )
                        break

                t0 = self._clock()
                tr_ids, va_ids = split_fold(snap, folds_df, k, self.cfg)
                tr_raw = train_fe[tr_fe_cid.isin(tr_ids)].copy()
                va_raw = train_fe[tr_fe_cid.isin(va_ids)].copy()
                te_raw = test_fe.copy()

                tr_f, va_f, te_f = self.fit_transform_fold(tr_raw, va_raw, te_raw, k)

                if (
                    id_col not in va_f.columns
                    or id_col not in te_f.columns
                    or va_f[id_col].isna().any()
                    or te_f[id_col].isna().any()
                ):
                    raise HookContractError(
                        f"fit_transform_fold pada fold {k} menghapus atau merusak kolom {id_col!r} pada va/te.",
                        f"Jangan menghapus atau mengubah kolom {id_col!r} di dalam fit_transform_fold.",
                    )

                va_f_cid = canon_id(va_f[id_col])
                if len(va_f_cid) != len(va_ids) or set(va_f_cid.to_numpy()) != va_ids:
                    raise HookContractError(
                        f"fit_transform_fold pada fold {k} mengubah himpunan ID atau jumlah baris va.",
                        "Boleh membuang baris pada tr, tetapi dilarang membuang/menambah baris pada va.",
                    )

                te_f_cid = canon_id(te_f[id_col])
                if len(te_f_cid) != len(te_orig_cid) or set(te_f_cid.to_numpy()) != set(
                    te_orig_cid.to_numpy()
                ):
                    raise HookContractError(
                        f"fit_transform_fold pada fold {k} mengubah himpunan ID atau jumlah baris te.",
                        "Dilarang membuang/menambah baris pada te di dalam fit_transform_fold.",
                    )

                model = self.build_model(k)
                model, va_pred, te_pred = self.fit_predict_fold(
                    model, tr_f, va_f, te_f, k
                )
                duration_sec = float(self._clock() - t0)
                self.fold_info[k]["duration_sec"] = duration_sec

                va_pred_arr = np.asarray(va_pred, dtype=float)
                te_pred_arr = np.asarray(te_pred, dtype=float)

                oof_k_df = pd.DataFrame({id_col: va_f[id_col].to_numpy()})
                test_k_df = pd.DataFrame({id_col: te_f[id_col].to_numpy()})
                if len(pred_cols) == 1:
                    oof_k_df[pred_cols[0]] = va_pred_arr.ravel()
                    test_k_df[pred_cols[0]] = te_pred_arr.ravel()
                else:
                    for j, c in enumerate(pred_cols):
                        oof_k_df[c] = va_pred_arr[:, j]
                        test_k_df[c] = te_pred_arr[:, j]

                rm.save_fold_checkpoint(k, oof_k_df, test_k_df, dict(self.fold_info[k]))

                pos = snap_pos_map.loc[va_f_cid.to_numpy()].to_numpy(dtype=int)
                for c in pred_cols:
                    oof_df.loc[pos, c] = oof_k_df[c].to_numpy(dtype=float)

                test_fold_preds[k] = test_k_df
                completed_folds.append(k)

                clip_range = None
                if self.cfg.metric == "qwk" and self.cfg.task == "regression":
                    valid_t = oof_df[target_col].dropna()
                    if len(valid_t) > 0:
                        clip_range = (int(np.min(valid_t)), int(np.max(valid_t)))
                fold_score = score(
                    oof_df.loc[pos, target_col].to_numpy(),
                    _extract_raw_preds(oof_df.loc[pos], pred_cols),
                    self.cfg.metric,
                    self.cfg.task,
                    threshold=0.5,
                    classes=self.classes_,
                    clip_range=clip_range,
                )
                rm.log(
                    f"Fold {k} done in {duration_sec:.2f}s | {self.cfg.metric}={fold_score:.6f}"
                )

            n_folds_done = len(completed_folds)
            if n_folds_done == 0:
                raise DataError(
                    "Tidak ada fold yang berhasil diselesaikan.",
                    "Naikkan cfg.max_minutes agar minimal 1 fold selesai dijalankan.",
                )

            imp_dicts = [
                self.fold_info[k]["importances"]
                for k in completed_folds
                if isinstance(self.fold_info[k].get("importances"), dict)
                and len(self.fold_info[k]["importances"]) > 0
            ]
            if imp_dicts:
                self.importances = (
                    pd.DataFrame(imp_dicts).mean(axis=0).sort_values(ascending=False)
                )

            status = "ok" if n_folds_done == len(active_folds) else "partial"
            rm.set_status(status)

            def _average_fold_test_preds() -> pd.DataFrame:
                acc = np.zeros((len(test_work), len(pred_cols)), dtype=float)
                for fk in completed_folds:
                    tk_df = test_fold_preds[fk]
                    tk_cid = canon_id(tk_df[id_col]).to_numpy()
                    tk_indexed = tk_df.set_index(tk_cid).loc[te_orig_cid.to_numpy()]
                    acc += tk_indexed[pred_cols].to_numpy(dtype=float)
                acc /= float(len(completed_folds))
                out = pd.DataFrame({id_col: test_work[id_col].to_numpy()})
                for j, c in enumerate(pred_cols):
                    out[c] = acc[:, j]
                return out

            refit_duration_sec = 0.0
            if self.cfg.test_mode == "refit_full":
                max_fold_dur = max(
                    (
                        float(self.fold_info[k].get("duration_sec", 0.0))
                        for k in completed_folds
                    ),
                    default=0.0,
                )
                rem_sec = (
                    self.cfg.max_minutes * 60.0 - (self._clock() - session_start)
                    if self.cfg.max_minutes is not None
                    else float("inf")
                )
                if status == "partial" or (
                    self.cfg.max_minutes is not None
                    and rem_sec < max_fold_dur * REFIT_TIME_MARGIN
                ):
                    warnings.warn(
                        "refit_full dilewati karena status partial atau sisa waktu kurang dari margin; "
                        "test_pred memakai rata-rata dari fold yang selesai.",
                        UserWarning,
                        stacklevel=2,
                    )
                    test_pred_df = _average_fold_test_preds()
                    test_pred_source = "fold_avg_fallback"
                else:
                    t_refit = self._clock()
                    refit_pred = np.asarray(
                        self.fit_full_predict(train_fe, test_fe), dtype=float
                    )
                    refit_duration_sec = float(self._clock() - t_refit)
                    test_pred_df = pd.DataFrame({id_col: test_work[id_col].to_numpy()})
                    if len(pred_cols) == 1:
                        test_pred_df[pred_cols[0]] = refit_pred.ravel()
                    else:
                        for j, c in enumerate(pred_cols):
                            test_pred_df[c] = refit_pred[:, j]
                    test_pred_source = "refit"
            else:
                test_pred_df = _average_fold_test_preds()
                test_pred_source = "fold_avg"

            total_fold_sec = sum(
                float(self.fold_info[k].get("duration_sec", 0.0))
                for k in completed_folds
            )
            minutes = float((total_fold_sec + refit_duration_sec) / 60.0)

            scores, params = compute_scores(
                oof_df=oof_df,
                test_pred_df=test_pred_df,
                pred_cols=pred_cols,
                cfg=self.cfg,
                postprocess_fn=self.postprocess,
                model_name=self.model_name,
                timestamp=rm.timestamp,
                minutes=minutes,
                refit_duration_sec=refit_duration_sec,
                fold_info=dict(self.fold_info),
                folds_hash=str(meta["folds_hash"]),
                test_ids_hash=str(dv["test_ids_hash"]),
                data_hash=str(dv["raw_train_hash"]),
                status=status,
                test_pred_source=test_pred_source,
                classes=self.classes_,
            )

            rm.save_oof(oof_df)
            rm.save_test_pred(test_pred_df)
            rm.save_postprocess(params)
            rm.save_scores(scores)
            generate_report(
                rm.run_dir,
                self.cfg,
                scores,
                oof_df,
                params,
                importances=self.importances,
                classes=self.classes_,
            )
            rm.append_leaderboard(scores)

            return RunResult(
                run_dir=rm.run_dir,
                oof=oof_df,
                test_pred=test_pred_df,
                scores=scores,
                params=params,
                cfg=self.cfg,
                classes=self.classes_,
            )

        except BaseException as exc:
            rm.mark_failed(exc)
            raise


__all__ = [
    "REFIT_TIME_MARGIN",
    "compute_scores",
    "RunResult",
    "BasePipeline",
]
