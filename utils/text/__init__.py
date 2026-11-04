"""Text modality module for offline-first ML pipelines."""

from utils.text.analysis import extract_top_errors, run_multilabel
from utils.text.base import TextPipelineBase, tune_binary_threshold
from utils.text.config import (
    VALID_CLEAN_MODES,
    VALID_ESTIMATORS,
    VALID_PRECISIONS,
    TextConfig,
)
from utils.text.features import extract_pair_stats, extract_text_stats
from utils.text.preprocessing import (
    DEFAULT_ID_SLANG_MAP,
    clean_text_series,
    load_slang_map,
)
from utils.text.tfidf import TfidfPipeline
from utils.text.transformer import (
    TextTokenDataset,
    TransformerPipeline,
    make_dynamic_collator,
    validate_local_model_dir,
)

__all__ = [
    "VALID_CLEAN_MODES",
    "VALID_ESTIMATORS",
    "VALID_PRECISIONS",
    "TextConfig",
    "DEFAULT_ID_SLANG_MAP",
    "load_slang_map",
    "clean_text_series",
    "extract_text_stats",
    "extract_pair_stats",
    "TextPipelineBase",
    "tune_binary_threshold",
    "TfidfPipeline",
    "validate_local_model_dir",
    "TextTokenDataset",
    "make_dynamic_collator",
    "TransformerPipeline",
    "extract_top_errors",
    "run_multilabel",
]
