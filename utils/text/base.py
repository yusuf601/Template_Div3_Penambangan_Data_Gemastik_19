"""Shared base pipeline for text modality experiments."""

from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.core.errors import ConfigError, DataError
from utils.core.metrics import get_metric, is_better, score
from utils.core.pipeline import BasePipeline
from utils.text.config import TextConfig
from utils.text.features import extract_pair_stats, extract_text_stats
from utils.text.preprocessing import clean_text_series, load_slang_map


def tune_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str,
    classes: Optional[list[Any]] = None,
) -> float:
    """Find threshold in [0.05, 0.95] optimizing metric on binary predictions."""
    best_t = 0.5
    best_s = None
    for t in np.linspace(0.05, 0.95, 181):
        s_val = float(score(y_true, y_prob, metric, "binary", threshold=float(t), classes=classes))
        if best_s is None or is_better(s_val, best_s, metric):
            best_s = s_val
            best_t = float(t)
    return best_t


class TextPipelineBase(BasePipeline):
    """Base pipeline providing input validation, text cleaning, tabular stats, and postprocess thresholding."""

    _use_tabular_stats: bool = True

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__(cfg)
        self.cfg: TextConfig = cfg
        self._stat_cols: list[str] = []

    def estimate_fold_seconds(self, completed_folds: list[int]) -> float:
        """Estimate next fold duration as max of completed fold durations times fold_time_margin."""
        durations = [float(self.fold_info[fk].get("duration_sec", 0.0)) for fk in completed_folds]
        return max(durations, default=0.0) * float(self.cfg.fold_time_margin)

    def build_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate input columns, binary targets, normalize text, and extract row-wise stats."""
        # 1. Validate required columns exist in both train and test
        required_cols = list(dict.fromkeys(self.cfg.text_cols + self.cfg.num_cols + self.cfg.cat_cols))
        missing_train = [c for c in required_cols if c not in train.columns]
        if missing_train:
            raise DataError(
                f"Kolom {missing_train!r} tidak ditemukan pada train.",
                f"Pastikan kolom {missing_train!r} tersedia di train DataFrame.",
            )

        missing_test = [c for c in required_cols if c not in test.columns]
        if missing_test:
            raise DataError(
                f"Kolom {missing_test!r} tidak ditemukan pada test.",
                f"Pastikan kolom {missing_test!r} tersedia di test DataFrame.",
            )

        # 2. Binary target contract validation
        if self.cfg.task == "binary":
            target_series = train[self.cfg.target].dropna()
            is_num_or_bool = pd.api.types.is_numeric_dtype(target_series.dtype) or pd.api.types.is_bool_dtype(
                target_series.dtype
            )
            uniq_vals = set(target_series.unique())
            if not is_num_or_bool or not uniq_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                raise ConfigError(
                    f"Kolom target '{self.cfg.target}' pada task binary memuat nilai selain 0/1: {sorted(list(uniq_vals))[:5]!r}.",
                    "Ubah kolom target biner ke integer 0/1 sebelum memanggil run(), dan simpan pemetaan labelnya.",
                )

        train_out = train.copy()
        test_out = test.copy()

        # 3. Text cleaning
        slang_map = (
            load_slang_map(self.cfg.slang_dict_path)
            if self.cfg.clean_mode == "social_id"
            else None
        )
        for col in self.cfg.text_cols:
            clean_col_name = f"{col}__clean"
            train_out[clean_col_name] = clean_text_series(
                train_out[col],
                mode=self.cfg.clean_mode,
                lowercase=self.cfg.lowercase,
                slang_map=slang_map,
            )
            test_out[clean_col_name] = clean_text_series(
                test_out[col],
                mode=self.cfg.clean_mode,
                lowercase=self.cfg.lowercase,
                slang_map=slang_map,
            )

        self._stat_cols = []

        # 4. Tabular stats (text stats & pair stats)
        if self._use_tabular_stats:
            if self.cfg.add_text_stats:
                train_out, s_cols = extract_text_stats(train_out, self.cfg.text_cols)
                test_out, _ = extract_text_stats(test_out, self.cfg.text_cols)
                self._stat_cols.extend(s_cols)

            if self.cfg.add_pair_features and len(self.cfg.text_cols) >= 2:
                col_a_clean = f"{self.cfg.text_cols[0]}__clean"
                col_b_clean = f"{self.cfg.text_cols[1]}__clean"
                train_out, p_cols = extract_pair_stats(train_out, col_a_clean, col_b_clean)
                test_out, _ = extract_pair_stats(test_out, col_a_clean, col_b_clean)
                self._stat_cols.extend(p_cols)

            self._stat_cols = list(dict.fromkeys(self._stat_cols))

        return train_out, test_out

    def fit_transform_fold(
        self, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, fold: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Per-fold stateful scaling of numeric and stat features fit strictly on tr."""
        tr_out = tr.copy()
        va_out = va.copy()
        te_out = te.copy()

        if self._use_tabular_stats:
            num_stat_cols = list(dict.fromkeys(self.cfg.num_cols + self._stat_cols))
            if num_stat_cols:
                # Impute median fit only on tr
                medians = tr_out[num_stat_cols].median().fillna(0.0)
                tr_out[num_stat_cols] = tr_out[num_stat_cols].fillna(medians)
                if len(va_out) > 0:
                    va_out[num_stat_cols] = va_out[num_stat_cols].fillna(medians)
                te_out[num_stat_cols] = te_out[num_stat_cols].fillna(medians)

                scaler = StandardScaler()
                tr_out[num_stat_cols] = scaler.fit_transform(tr_out[num_stat_cols])
                if len(va_out) > 0:
                    va_out[num_stat_cols] = scaler.transform(va_out[num_stat_cols])
                te_out[num_stat_cols] = scaler.transform(te_out[num_stat_cols])

            if self.cfg.cat_cols:
                for col in self.cfg.cat_cols:
                    tr_out[col] = tr_out[col].fillna("__NA__").astype(str)
                    if len(va_out) > 0:
                        va_out[col] = va_out[col].fillna("__NA__").astype(str)
                    te_out[col] = te_out[col].fillna("__NA__").astype(str)

        return tr_out, va_out, te_out

    def postprocess(
        self, oof_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
    ) -> dict[str, Any]:
        """Tune binary decision threshold on OOF predictions after dropping NaN rows."""
        if (
            self.cfg.tune_threshold
            and self.cfg.task == "binary"
            and get_metric(self.cfg.metric).input_type == "label"
        ):
            valid_df = oof_df.dropna(subset=[self.cfg.target, "pred"])
            if len(valid_df) == 0:
                return {}

            thr = tune_binary_threshold(
                valid_df[self.cfg.target].to_numpy(dtype=float),
                valid_df["pred"].to_numpy(dtype=float),
                self.cfg.metric,
                classes=self.classes_,
            )
            return {"threshold": float(thr)}

        return {}


__all__ = [
    "tune_binary_threshold",
    "TextPipelineBase",
]
