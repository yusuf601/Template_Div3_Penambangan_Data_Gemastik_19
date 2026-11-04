"""TextConfig dataclass with auto-resolution, validation, and BaseConfig extra serialization."""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from utils.core.config import BaseConfig
from utils.core.errors import ConfigError
from utils.core.io import load_json, save_json
from utils.core.metrics import get_metric

VALID_CLEAN_MODES = ("none", "minimal", "social_id")
VALID_ESTIMATORS = ("logistic", "linear_svc", "sgd", "ridge", "lgbm")
VALID_PRECISIONS = ("auto", "fp32", "fp16", "bf16")

TEXT_SPECIFIC_FIELDS = (
    "text_cols",
    "num_cols",
    "cat_cols",
    "text_sep",
    "clean_mode",
    "lowercase",
    "slang_dict_path",
    "add_text_stats",
    "add_pair_features",
    "tune_threshold",
    "estimator",
    "word_ngram_range",
    "char_ngram_range",
    "use_char_ngrams",
    "max_features_word",
    "max_features_char",
    "min_df",
    "sublinear_tf",
    "c_reg",
    "svd_components",
    "top_k_importances",
    "model_name_or_path",
    "local_files_only",
    "select_metric",
    "fold_time_margin",
    "max_length",
    "epochs",
    "train_batch_size",
    "eval_batch_size",
    "grad_accum_steps",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "precision",
    "serialize_meta_to_text",
    "save_best_weights",
    "num_workers",
)


@dataclass(kw_only=True)
class TextConfig(BaseConfig):
    modality: str = "text"
    text_cols: list[str] = field(default_factory=lambda: ["text"])
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    text_sep: str = " [SEP] "

    # Stateless text preprocessing & row-wise feature flags
    clean_mode: str = "minimal"
    lowercase: bool = True
    slang_dict_path: Optional[str] = None
    add_text_stats: bool = True
    add_pair_features: bool = True

    # Postprocessing
    tune_threshold: bool = True

    # TF-IDF specific parameters
    estimator: Optional[str] = None
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    use_char_ngrams: bool = True
    max_features_word: int = 50_000
    max_features_char: int = 50_000
    min_df: int = 2
    sublinear_tf: bool = True
    c_reg: float = 1.0
    svd_components: int = 64
    top_k_importances: int = 50

    # Transformer specific parameters
    model_name_or_path: str = "assets/models/indobert-base-p1"
    local_files_only: bool = True
    select_metric: Optional[str] = None
    fold_time_margin: float = 1.1
    max_length: int = 256
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "auto"
    serialize_meta_to_text: bool = True
    save_best_weights: bool = False
    num_workers: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.word_ngram_range, (list, tuple)):
            self.word_ngram_range = tuple(int(x) for x in self.word_ngram_range)
        if isinstance(self.char_ngram_range, (list, tuple)):
            self.char_ngram_range = tuple(int(x) for x in self.char_ngram_range)

    def resolve(self) -> list[str]:
        """Auto-resolve BaseConfig fields, ngram ranges, and select_metric."""
        reasons = super().resolve()

        if isinstance(self.word_ngram_range, (list, tuple)):
            self.word_ngram_range = tuple(int(x) for x in self.word_ngram_range)
        if isinstance(self.char_ngram_range, (list, tuple)):
            self.char_ngram_range = tuple(int(x) for x in self.char_ngram_range)

        if self.select_metric is None:
            self.select_metric = "rmse" if self.task == "regression" else "logloss"
            reasons.append(f"select_metric={self.select_metric!r} (auto from task={self.task!r})")

        if self.estimator is None:
            self.estimator = "ridge" if self.task == "regression" else "logistic"
            reasons.append(f"estimator={self.estimator!r} (auto from task={self.task!r})")

        return reasons

    def validate(self) -> None:
        """Validate configuration and raise ConfigError with actionable remediation."""
        super().validate()

        if not isinstance(self.text_cols, list) or len(self.text_cols) == 0:
            raise ConfigError(
                f"text_cols={self.text_cols!r} harus berupa list non-kosong.",
                "Berikan list nama kolom teks yang akan diproses pada text_cols.",
            )

        if len(self.text_cols) != len(set(self.text_cols)):
            raise ConfigError(
                f"text_cols={self.text_cols!r} memuat nama kolom duplikat.",
                "Pastikan setiap elemen pada text_cols unik.",
            )

        forbidden_overlap = {self.id_col, self.target}
        if self.group_col:
            forbidden_overlap.add(self.group_col)
        if self.time_col:
            forbidden_overlap.add(self.time_col)

        overlap = set(self.text_cols).intersection(forbidden_overlap)
        if overlap:
            raise ConfigError(
                f"text_cols={self.text_cols!r} bertabrakan dengan kolom id/target/group/time: {list(overlap)!r}.",
                "Hapus kolom id, target, group_col, atau time_col dari daftar text_cols.",
            )

        if self.clean_mode not in VALID_CLEAN_MODES:
            raise ConfigError(
                f"clean_mode={self.clean_mode!r} tidak valid.",
                f"Set clean_mode ke salah satu dari {VALID_CLEAN_MODES}.",
            )

        if self.estimator not in VALID_ESTIMATORS:
            raise ConfigError(
                f"estimator={self.estimator!r} tidak valid.",
                f"Set estimator ke salah satu dari {VALID_ESTIMATORS}.",
            )

        if self.task == "regression" and self.estimator in ("logistic", "linear_svc"):
            raise ConfigError(
                f"estimator={self.estimator!r} hanya untuk klasifikasi, tidak cocok untuk task='regression'.",
                "Gunakan estimator='ridge', 'sgd', atau 'lgbm' untuk task regresi.",
            )

        if self.precision not in VALID_PRECISIONS:
            raise ConfigError(
                f"precision={self.precision!r} tidak valid.",
                f"Set precision ke salah satu dari {VALID_PRECISIONS}.",
            )

        if self.select_metric is not None:
            get_metric(self.select_metric)

        for name, rng in [("word_ngram_range", self.word_ngram_range), ("char_ngram_range", self.char_ngram_range)]:
            if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
                raise ConfigError(
                    f"{name}={rng!r} harus berupa tuple 2 elemen (min, max).",
                    f"Contoh format yang benar: {name}=(1, 2).",
                )
            if int(rng[0]) < 1 or int(rng[0]) > int(rng[1]):
                raise ConfigError(
                    f"{name}={rng!r} tidak valid (min harus >= 1 dan min <= max).",
                    f"Perbaiki rentang nilai {name}.",
                )

        if self.max_length < 8:
            raise ConfigError(f"max_length={self.max_length} terlalu kecil (minimal 8).", "Set max_length >= 8.")
        if self.epochs < 1:
            raise ConfigError(f"epochs={self.epochs} terlalu kecil (minimal 1).", "Set epochs >= 1.")
        if self.train_batch_size < 1:
            raise ConfigError(f"train_batch_size={self.train_batch_size} minimal 1.", "Set train_batch_size >= 1.")
        if self.eval_batch_size < 1:
            raise ConfigError(f"eval_batch_size={self.eval_batch_size} minimal 1.", "Set eval_batch_size >= 1.")
        if self.grad_accum_steps < 1:
            raise ConfigError(f"grad_accum_steps={self.grad_accum_steps} minimal 1.", "Set grad_accum_steps >= 1.")
        if self.lr <= 0:
            raise ConfigError(f"lr={self.lr} harus bernilai positif.", "Set lr > 0 (misalnya 2e-5).")
        if self.c_reg <= 0:
            raise ConfigError(f"c_reg={self.c_reg} harus bernilai positif.", "Set c_reg > 0 (misalnya 1.0).")
        if self.svd_components < 2:
            raise ConfigError(f"svd_components={self.svd_components} minimal 2.", "Set svd_components >= 2.")
        if self.top_k_importances < 1:
            raise ConfigError(f"top_k_importances={self.top_k_importances} minimal 1.", "Set top_k_importances >= 1.")
        if self.fold_time_margin < 1.0:
            raise ConfigError(f"fold_time_margin={self.fold_time_margin} minimal 1.0.", "Set fold_time_margin >= 1.0.")

        if self.slang_dict_path is not None and not Path(self.slang_dict_path).exists():
            raise ConfigError(
                f"slang_dict_path={self.slang_dict_path!r} tidak ditemukan di disk.",
                "Periksa kembali path kamus slang atau set slang_dict_path=None.",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to BaseConfig format with extra['_text']."""
        base_dict = super().to_dict()
        extra_copy = copy.deepcopy(base_dict.get("extra", {}))

        text_dict: dict[str, Any] = {}
        for fld in TEXT_SPECIFIC_FIELDS:
            val = getattr(self, fld)
            if isinstance(val, tuple):
                val = list(val)
            text_dict[fld] = val

        extra_copy["_text"] = text_dict
        base_dict["extra"] = extra_copy

        # Remove text-specific attributes from base_dict if present so only 23 BaseConfig keys remain
        for fld in TEXT_SPECIFIC_FIELDS:
            base_dict.pop(fld, None)

        return base_dict

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextConfig":
        """Load configuration from JSON file restoring extra and TextConfig fields."""
        data = load_json(path)
        data_copy = copy.deepcopy(data)
        text_fields = data_copy.get("extra", {}).pop("_text", {})

        if "word_ngram_range" in text_fields and isinstance(text_fields["word_ngram_range"], list):
            text_fields["word_ngram_range"] = tuple(text_fields["word_ngram_range"])
        if "char_ngram_range" in text_fields and isinstance(text_fields["char_ngram_range"], list):
            text_fields["char_ngram_range"] = tuple(text_fields["char_ngram_range"])

        return cls(**data_copy, **text_fields)


__all__ = [
    "VALID_CLEAN_MODES",
    "VALID_ESTIMATORS",
    "VALID_PRECISIONS",
    "TextConfig",
]
