"""Core utilities for Gemastik template."""

from utils.core.config import BaseConfig
from utils.core.cv import (
    compute_folds_hash,
    compute_ids_target_hash,
    debug_subsample,
    get_folds,
    load_folds,
    make_folds,
    make_folds_file,
    save_folds,
    split_fold,
)
from utils.core.ensemble import EnsembleData, blend, load_runs
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
from utils.core.pipeline import (
    REFIT_TIME_MARGIN,
    BasePipeline,
    RunResult,
    compute_scores,
)
from utils.core.report import compare_runs, generate_report
from utils.core.run import (
    LEADERBOARD_COLS,
    RunManager,
    safe_name,
    sync_leaderboard,
    update_lb_score,
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
    "compute_ids_target_hash",
    "compute_folds_hash",
    "make_folds",
    "save_folds",
    "load_folds",
    "get_folds",
    "make_folds_file",
    "split_fold",
    "debug_subsample",
    "LEADERBOARD_COLS",
    "safe_name",
    "RunManager",
    "update_lb_score",
    "sync_leaderboard",
    "generate_report",
    "compare_runs",
    "REFIT_TIME_MARGIN",
    "compute_scores",
    "RunResult",
    "BasePipeline",
    "EnsembleData",
    "load_runs",
    "blend",
]
