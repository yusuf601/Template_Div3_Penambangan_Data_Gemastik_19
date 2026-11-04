# tests/core/test_public_api.py
import utils.core as core


EXPECTED_EXPORTS = [
    # Task 1: errors + io
    "TemplateError", "ConfigError", "DataError", "FoldMismatchError",
    "HookContractError", "SubmissionError",
    "canon_id", "canon_hash", "fast_df_hash",
    "save_json", "load_json", "save_parquet", "load_parquet", "seed_everything",
    # Task 2: metrics
    "Metric", "register_metric", "get_metric", "list_metrics", "is_better", "score",
    # Task 3: config + cv
    "BaseConfig", "compute_ids_target_hash", "compute_folds_hash",
    "make_folds", "save_folds", "load_folds", "get_folds",
    "make_folds_file", "split_fold", "debug_subsample",
    # Task 4: run + report
    "LEADERBOARD_COLS", "safe_name", "RunManager", "update_lb_score", "sync_leaderboard",
    "generate_report", "compare_runs",
    # Task 5: pipeline
    "REFIT_TIME_MARGIN", "compute_scores", "RunResult", "BasePipeline",
    # Task 6: ensemble
    "EnsembleData", "load_runs", "blend",
]


def test_utils_core_public_exports():
    assert hasattr(core, "__all__")
    assert set(core.__all__) == set(EXPECTED_EXPORTS)
    for name in EXPECTED_EXPORTS:
        assert hasattr(core, name), f"Missing export on utils.core: {name}"
