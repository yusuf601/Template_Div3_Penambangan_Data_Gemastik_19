"""BaseConfig dataclass with auto-resolution and actionable validation."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
import re
import socket
from typing import Any, Optional, Union
import warnings

from utils.core.errors import ConfigError
from utils.core.io import load_json, save_json
from utils.core.metrics import get_metric

VALID_TASKS = ("binary", "multiclass", "regression")
VALID_CV_STRATEGIES = ("kfold", "stratified", "group", "stratified_group", "timeseries")
VALID_TEST_MODES = ("fold_avg", "refit_full")


@dataclass(kw_only=True)
class BaseConfig:
    """Competition configuration with sensible defaults and auto-resolution."""

    task: str
    target: str
    metric: str
    id_col: str = "id"
    run_name: str = "baseline"
    owner: str = ""
    modality: str = "generic"
    seed: int = 42
    fold_seed: int = 42
    n_folds: int = 5
    cv_strategy: Optional[str] = None
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    ts_gap: int = 0
    folds_path: Optional[str] = None
    test_mode: Optional[str] = None
    refit_iter_mult: float = 1.1
    max_minutes: Optional[float] = None
    resume_from: Optional[str] = None
    runs_dir: str = "runs"
    debug: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> list[str]:
        """Auto-resolve owner, debug overrides, cv_strategy, and test_mode."""
        reasons: list[str] = []

        if not self.owner:
            raw_host = socket.gethostname().lower()
            cleaned = re.sub(r"[^a-z0-9]+", "", raw_host)[:12]
            self.owner = cleaned or "local"
            reasons.append(f"owner={self.owner!r} (from hostname)")

        if self.debug:
            self.n_folds = 2
            self.folds_path = None
            self.resume_from = None
            reasons.append("debug=True -> n_folds=2, folds_path=None, resume_from=None")

        if self.cv_strategy is None:
            is_clf = self.task in ("binary", "multiclass")
            if self.group_col:
                self.cv_strategy = "stratified_group" if is_clf else "group"
                reasons.append(
                    f"cv_strategy={self.cv_strategy!r} (auto from task={self.task!r}, group_col={self.group_col!r})"
                )
            else:
                self.cv_strategy = "stratified" if is_clf else "kfold"
                reasons.append(f"cv_strategy={self.cv_strategy!r} (auto from task={self.task!r})")

        if self.time_col is not None and self.cv_strategy != "timeseries":
            warnings.warn(
                f"time_col={self.time_col!r} diisi tetapi cv_strategy={self.cv_strategy!r} (bukan 'timeseries'). "
                "Set cv_strategy='timeseries' secara eksplisit jika ingin split berbasis waktu.",
                UserWarning,
                stacklevel=2,
            )

        if self.test_mode is None:
            self.test_mode = "refit_full" if self.cv_strategy == "timeseries" else "fold_avg"
            reasons.append(
                f"test_mode={self.test_mode!r} (auto from cv_strategy={self.cv_strategy!r})"
            )

        return reasons

    def validate(self) -> None:
        """Validate configuration and raise ConfigError with concrete remediation steps."""
        if self.task not in VALID_TASKS:
            raise ConfigError(
                f"task={self.task!r} tidak dikenal.",
                f"Set task ke salah satu dari {VALID_TASKS}.",
            )

        if not self.target:
            raise ConfigError(
                "target tidak boleh kosong.",
                "Isi nama kolom target pada BaseConfig(target=...).",
            )

        # Validate metric exists in registry
        get_metric(self.metric)

        if self.cv_strategy not in VALID_CV_STRATEGIES:
            raise ConfigError(
                f"cv_strategy={self.cv_strategy!r} tidak valid (atau resolve() belum dipanggil).",
                f"Set cv_strategy ke salah satu dari {VALID_CV_STRATEGIES} atau biarkan None lalu panggil cfg.resolve().",
            )

        if self.cv_strategy in ("group", "stratified_group") and not self.group_col:
            raise ConfigError(
                f"cv_strategy={self.cv_strategy!r} membutuhkan group_col, tetapi group_col={self.group_col!r}.",
                "Isi group_col dengan nama kolom grup, atau ganti cv_strategy ke 'stratified' / 'kfold'.",
            )

        if self.cv_strategy == "timeseries" and not self.time_col:
            raise ConfigError(
                "cv_strategy='timeseries' membutuhkan time_col, tetapi time_col=None.",
                "Isi time_col dengan nama kolom waktu/tanggal, atau ganti cv_strategy.",
            )

        if self.cv_strategy in ("stratified", "stratified_group") and self.task == "regression":
            suggested = "group" if self.cv_strategy == "stratified_group" else "kfold"
            raise ConfigError(
                f"cv_strategy={self.cv_strategy!r} tidak dapat dipakai untuk task='regression'.",
                f"Set cv_strategy='{suggested}' untuk regresi.",
            )

        if self.n_folds < 2:
            raise ConfigError(
                f"n_folds={self.n_folds} terlalu kecil (minimal 2).",
                "Set n_folds >= 2 (misalnya n_folds=5).",
            )

        if self.ts_gap < 0:
            raise ConfigError(
                f"ts_gap={self.ts_gap} tidak boleh negatif.",
                "Set ts_gap >= 0.",
            )

        if self.test_mode not in VALID_TEST_MODES:
            raise ConfigError(
                f"test_mode={self.test_mode!r} tidak valid.",
                f"Set test_mode ke salah satu dari {VALID_TEST_MODES} atau biarkan None lalu panggil cfg.resolve().",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseConfig":
        """Load configuration from JSON file."""
        data = load_json(path)
        return cls(**data)


__all__ = ["BaseConfig"]
