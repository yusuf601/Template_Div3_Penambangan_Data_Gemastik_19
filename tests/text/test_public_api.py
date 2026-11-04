# tests/text/test_public_api.py
import utils.text as text_mod


EXPECTED_TEXT_EXPORTS = [
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


def test_utils_text_public_exports():
    assert hasattr(text_mod, "__all__")
    assert set(text_mod.__all__) == set(EXPECTED_TEXT_EXPORTS)
    for name in EXPECTED_TEXT_EXPORTS:
        assert hasattr(text_mod, name), f"Missing export on utils.text: {name}"
