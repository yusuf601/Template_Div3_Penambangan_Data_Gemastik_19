"""Core utilities for Gemastik template."""

from utils.core.config import BaseConfig
from utils.core.errors import (
    ConfigError,
    DataError,
    FoldMismatchError,
    HookContractError,
    SubmissionError,
    TemplateError,
)
from utils.core.io import (
    canon_hash,
    canon_id,
    fast_df_hash,
    load_json,
    load_parquet,
    save_json,
    save_parquet,
)
from utils.core.metrics import (
    Metric,
    get_metric,
    is_better,
    list_metrics,
    register_metric,
    score,
)
from utils.core.seed import seed_everything

__all__ = [
    "TemplateError",
    "ConfigError",
    "DataError",
    "FoldMismatchError",
    "HookContractError",
    "SubmissionError",
    "canon_id",
    "canon_hash",
    "fast_df_hash",
    "save_json",
    "load_json",
    "save_parquet",
    "load_parquet",
    "seed_everything",
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "is_better",
    "score",
    "BaseConfig",
]
