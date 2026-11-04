# Gemastik 2026 — Sub-project #2: Text Modality (`utils/text/`) Design Specification

**Date:** 2026-11-03  
**Status:** Approved & Finalized  
**Target Directory:** `/home/kali/machine_learning/template_gemastik`  
**Depends On:** Sub-project #1 Core-lite (`utils/core/`, 44 tests passing + Task 0 incremental Core hooks)

---

## 1. Context, Intent, and Hardware Reality

### 1.1 Why This Exists
The final modality for the Gemastik Data Mining competition has been confirmed as **Text**. Teams have **5 hours** of live offline coding to deliver both competitive leaderboard submissions and final presentation slides.

Because the exact task schema is only revealed when the 5-hour timer starts, `utils/text/` must handle all standard text problem shapes out of the box on top of `utils/core/`:
- **Tasks:** `binary`, `multiclass`, and `regression` directly via `BasePipeline`, plus a `run_multilabel` wrapper if the dataset has multiple binary target columns (`[y_1, y_2, ..., y_K]`).
- **Text Columns (`text_cols`):**
  - Single text column (e.g., `["text"]` for sentiment, emotion, hoax, toxicity, topic, rating).
  - Text pair (`["text_a", "text_b"]` or `["title", "body"]` for NLI, duplicate/matching, relevance, stance).
  - Multi-field text (`> 2` text columns).
- **Auxiliary Tabular Metadata:** Optional numeric (`num_cols`) or low-cardinality categorical (`cat_cols`) columns alongside text (e.g., repost count, platform source, author verified flag).

### 1.2 Confirmed Competition Hardware
Each team receives 1 competition PC:
- **CPU:** Intel Core i7-12700 (12 cores / 20 threads)
- **RAM:** 32 GB DDR5 4800 MHz
- **Storage:** 1 TB PCIe NVMe SSD
- **GPU:** NVIDIA GeForce RTX 3070 LHR (**8 GB GDDR6 VRAM**, Ampere — native `fp16` and `bf16`)
- **Support Machines:** Team laptops (CPU-only, Windows/Linux) connected via flashdrive/LAN folder sync (`sync_leaderboard` and `blend`).
- **Network:** Strictly **offline** during the competition (`local_files_only=True` by default, pointing to local folder `assets/models/indobert-base-p1`).

---

## 2. Goals, Non-Goals, and Anti-Goals

### 2.1 Goals
1. **Task 0 — Backward-Compatible Core Hooks (`utils/core/pipeline.py`, `utils/core/report.py`):**
   - Expose `self._run_start: float` on `BasePipeline` at the start of `run()` so subclasses share the exact same session clock origin as Core (no second timer in `build_features`).
   - Call `seed_everything(self.cfg.seed + max(k, 0))` right before `build_model(k)` in `BasePipeline.run()` (and `seed_everything(self.cfg.seed)` before `build_model(-1)` in `fit_full_predict`) so fold `k` is 100% deterministic whether run uninterrupted or via `resume_from`.
   - Add hook `estimate_fold_seconds(self, completed_folds: list[int]) -> float` (default `0.0` in `BasePipeline`) and stop before starting fold `k >= 1` when `rem_sec < self.estimate_fold_seconds(completed_folds)`.
   - Propagate `fold_info[k].get("truncated", False)` into `scores["per_fold"][str(k)]["truncated"]` and render `| {k} (truncated) |` in `summary.md`'s Per-Fold Breakdown table.
   - Export `safe_name` in `utils/core/__init__.py` (`__all__`).
   - Criterion: all 44 existing Core tests remain green, plus new unit tests for `estimate_fold_seconds`, `_run_start`, per-fold seeding on resume, and `truncated` in `scores.json` & `summary.md`.
2. **Model Family #1 — Fast Sparse TF-IDF Pipeline (`TfidfPipeline`):**
   - Word + character n-gram TF-IDF vectorization kept strictly as `scipy.sparse.csr_matrix` inside `fit_predict_fold` (never materialized as a dense DataFrame).
   - Supports multiple fast estimators: `"ridge"`, `"linear_svc"` (with safe `CalibratedClassifierCV` adaptive `cv=min(3, min_class_count)` and fallback to `LogisticRegression` if `min_class_count < 2`), `"sgd"`, `"logistic"`, and `"lgbm"`.
   - Aligns multiclass `predict_proba` to the full `K = len(self.classes_)` columns using `estimator.classes_` (filling `0.0` for any rare class absent in `tr` and re-normalizing rows).
   - Computes leak-free per-fold TF-IDF pair similarity features (word & char cosine similarity between `text_cols[0]` and `text_cols[1]`) inside `fit_predict_fold` when `len(text_cols) >= 2`.
   - Exports top-K (`top_k_importances = 50`) absolute coefficients/importances per fold into `self.fold_info[fold]["importances"]` so `plots/importance.png` is generated automatically by Core's `generate_report` without bloating `info.json`.
3. **Model Family #3 — End-to-End HuggingFace Transformer Fine-Tuning (`TransformerPipeline`):**
   - Validates at the start of `build_features` (when `local_files_only=True`) that `cfg.model_name_or_path` exists on disk and contains `config.json`, tokenizer files, and model weights (`model.safetensors` or `pytorch_model.bin`), failing in milliseconds with `ConfigError` and `[Perbaikan]`.
   - Loads a **fresh pretrained model from disk in `build_model(fold)` on every fold** (caching only the tokenizer) and explicitly frees memory (`del model; gc.collect(); torch.cuda.empty_cache()`) at the end of each fold so weights never leak across folds.
   - Native PyTorch training loop with `precision: "auto" | "fp32" | "fp16" | "bf16"` (`"auto"` selects `"bf16"` on Ampere+, `"fp16"` on older CUDA, `"fp32"` on CPU), dynamic batch padding, gradient accumulation, linear/cosine warmup schedule, and gradient clipping.
   - Evaluates validation predictions at the end of every epoch using `cfg.select_metric` (defaulting to smooth loss metrics: `"logloss"` for classification and `"rmse"` for regression), tracks `best_epoch` (`best_iteration` in `fold_info[fold]`), and restores the best model weights for `va_pred` and `te_pred`.
   - **Two-Tier `max_minutes` Guard (using `self._run_start` from Core):**
     1. **Pre-Fold Guard (Primary):** `TextPipelineBase.estimate_fold_seconds(completed_folds)` returns `max(duration_sec of completed folds) * cfg.fold_time_margin` (default `1.1`). If remaining session time is less than the estimated fold duration, Core stops before starting fold `k`, records `status="partial"`, and runs fold `k` cleanly from scratch on `resume_from`. On fold 0, measures the average duration of **batches 3 through 10** (indices `2..9`, skipping batch 0/1 warmup overhead) and emits a `UserWarning` if the estimated 1-epoch duration exceeds remaining `max_minutes`.
     2. **Intra-Fold Emergency Cutoff (Safety Net):** If a fold runs slower than estimated and hits `max_minutes` mid-fold after completing at least 1 epoch, stops early, marks `self.fold_info[fold]["truncated"] = True` (rendered as `(truncated)` in `summary.md`), restores the best epoch weights so far, and emits a `UserWarning`.
   - Handles `test_mode="refit_full"` (`fold == -1`, `va` empty) by training a fresh model for `self.refit_iterations()` epochs.
   - Scales continuous targets per fold (`(y - mean) / std`) when `task == "regression"` and unscales predictions back to raw target scale before returning.
4. **Stateless Row-Wise Text Cleaning & Engineering (`utils/text/preprocessing.py`, `utils/text/features.py`):**
   - Configurable text normalization (`clean_mode`: `"none"`, `"minimal"`, `"social_id"`) including optional Indonesian slang dictionary normalization (`slang_dict_path` or built-in compact colloquial map), URL/mention/email placeholders, whitespace normalization, and optional lowercase.
   - Leak-free row-wise text statistics (`add_text_stats=True`) and row-wise pair features (`add_pair_features=True` when `len(text_cols) >= 2`).
5. **Postprocessing, Error Analysis & Multi-Label Helper:**
   - Automatic binary threshold tuning (`tune_threshold=True`) in `postprocess(oof_df, test_df)` after `dropna` on OOF predictions/targets, seamlessly evaluated via Core's honest scoring (`cv_honest`).
   - `extract_top_errors(result, train_df, n=20)` saving `error_analysis.csv` inside `run_dir`.
   - `run_multilabel(..., folds_dir="folds")` helper that creates/reuses per-label `folds/{label}/folds.csv` so each label has its own valid `ids_target_hash` while allowing `TfidfPipeline` and `TransformerPipeline` runs for the same label to share identical folds and be blended via `load_runs` + `blend`.

### 2.2 Non-Goals (Deferred)
- **Model Family #2 (Frozen Transformer Embeddings + GBDT):** Deferred until after Families #1 and #3 are complete and tested.
- **Experimental Transformer Tweaks (`multisample_dropout`, `head_lr`, `grad_checkpointing`):** Omitted from v1 config to keep the surface area minimal and verified.
- **Seq2Seq / Generative LLM Fine-Tuning (T5 / GPT / LoRA):** Out of scope.

### 2.3 Anti-Goals (Failure Modes to Prevent)
- **Weight Leakage Across Folds:** Reusing a fine-tuned model object from fold 0 in fold 1 invalidates OOF and ensemble blending.
- **Data Leakage in Pair TF-IDF Features:** Any vectorizer-based cosine similarity feature **must** be fitted on `tr` inside `fit_predict_fold`, never in `build_features`.
- **Dense TF-IDF DataFrames & Bloated `info.json`:** Keep TF-IDF strictly `scipy.sparse` and cap stored importances to `top_k_importances` (default 50).
- **Hidden Internet Calls or Late Model-Path Failures:** Default `model_name_or_path` to local directory `"assets/models/indobert-base-p1"` with `local_files_only=True` and pre-check directory contents before fold training starts.

---

## 3. Architectural Approaches Considered

| Approach | Description | Trade-offs | Decision |
|---|---|---|---|
| **A. Native PyTorch AMP Loop + Sparse-in-Fold `TfidfPipeline` sharing `TextConfig`** | Single `TextConfig(BaseConfig)` dataclass; `TfidfPipeline` keeps matrices as `scipy.sparse` inside `fit_predict_fold`; `TransformerPipeline` implements a ~200-line native PyTorch AMP training/eval loop directly against `AutoModelForSequenceClassification`. | (+) Zero dependency on `accelerate` or `datasets`; exact integration with Core's `score()`, `fold_info["best_iteration"]`, `refit_full` (`va` empty), and `self._run_start` / `self._clock()` time budget; CPU unit tests run in seconds using tiny offline models. (-) Requires writing the ~200-line PyTorch training loop ourselves. | **Chosen** |
| **B. Wrapping `transformers.Trainer` + `TrainingArguments`** | Delegates Transformer training to HuggingFace `Trainer` with custom callbacks for metrics and time limits. | (+) Shorter training loop code. (-) Requires `accelerate` offline; brittle when `va` is empty during `refit_full`; writes heavy intermediate checkpoint folders; harder to test deterministically with `FakeClock`. | Rejected |

---

## 4. Initial Runtime Assumptions & Estimator Selection Guide (5-Hour Budget)

> **Catatan Penting (Asumsi Awal — Wajib Diukur Saat Gladi Resik):** Angka di bawah ini adalah **estimasi awal** pada perangkat keras lomba (**i7-12700 CPU + RTX 3070 8GB GPU**) untuk dataset **25.000 baris** (rata-rata 40–80 kata/baris). Tim wajib mencatat waktu aktual 1 fold saat gladi resik di PC lomba untuk menyesuaikan `max_minutes` dan `n_folds`.

| Pipeline & Estimator | Typical Features / Seq Len | Est. Time / Fold | Est. 5-Fold Total | Notes & Recommendations |
|---|---|---|---|---|
| `TfidfPipeline(estimator="ridge")` | 100k word+char n-grams | ~3 – 8 sec | **~20 – 40 sec** | Fastest baseline for regression/F1/accuracy. **Warning:** for classification, `ridge` applies sigmoid/softmax to uncalibrated margins; for `logloss` or probability blending, use `"logistic"`, `"sgd"`, or `"linear_svc"`. |
| `TfidfPipeline(estimator="sgd")` | 100k word+char n-grams | ~4 – 12 sec | **~25 – 60 sec** | Fast linear classifier (`loss="log_loss"`) / regressor with calibrated log-loss probabilities. |
| `TfidfPipeline(estimator="linear_svc")` | 100k word+char n-grams | ~15 – 45 sec | **~1.5 – 4 min** | Fits `CalibratedClassifierCV(LinearSVC(), cv=min(3, min_cls))` (~3 internal fits per fold) for calibrated probabilities. |
| `TfidfPipeline(estimator="logistic")` (`lbfgs`) | 100k word+char n-grams | ~15 – 45 sec | **~1.5 – 4 min** | Accurate probability baseline (`solver="lbfgs"`, **avoid** `solver="saga"` on 100k features). |
| `TfidfPipeline(estimator="lgbm")` | SVD (`svd_components=64`) + stats | ~15 – 40 sec | **~1.5 – 3.5 min** | Captures non-linear interactions between text stats, metadata, and 64 SVD components. |
| `TransformerPipeline` (`indobert-base`, `bf16`/`fp16`) | `max_length=128`, `epochs=3` | ~1.5 – 3 min | **~8 – 15 min** | Primary GPU workhorse on RTX 3070 for short/medium texts (tweets, reviews, titles). |
| `TransformerPipeline` (`indobert-base`, `bf16`/`fp16`) | `max_length=256`, `epochs=3` | ~3 – 6 min | **~15 – 30 min** | Primary GPU model for longer paragraphs/news articles. |
| `TransformerPipeline` (`indobert-large` / `xlm-r-base`) | `max_length=256`, `epochs=3` | ~10 – 20 min | **~50 – 100 min** | High-capacity model on 8 GB VRAM; use `n_folds=3` or `max_minutes` budget guard. |

---

## 5. Module Structure & Public Exports (`utils/text/`)

```text
utils/text/
├── __init__.py          # Public API exports (__all__)
├── config.py            # TextConfig(BaseConfig) with validation and extra["_text"] serialization
├── preprocessing.py     # DEFAULT_ID_SLANG_MAP, load_slang_map, clean_text_series
├── features.py          # extract_text_stats, extract_pair_stats (stateless row-wise)
├── base.py              # TextPipelineBase(BasePipeline)
├── tfidf.py             # TfidfPipeline(TextPipelineBase)
├── transformer.py       # TransformerPipeline(TextPipelineBase), TextDataset, make_dynamic_collator
└── analysis.py          # tune_binary_threshold, extract_top_errors, run_multilabel
```

---

## 6. Detailed Component Design

### 6.0 Task 0: Incremental Core Hooks (`utils/core/pipeline.py`, `utils/core/report.py`, `utils/core/__init__.py`)

1. **`self._run_start` and Per-Fold Seeding in `BasePipeline` (`utils/core/pipeline.py`):**
   - In `BasePipeline.__init__`: initializes `self._run_start: float = 0.0`.
   - In `BasePipeline.run()`: sets `self._run_start = session_start` immediately after `session_start = self._clock()`.
   - Inside the fold loop `for k in active_folds:` right before `model = self.build_model(k)`: calls `seed_everything(self.cfg.seed + max(int(k), 0))`.
   - In `BasePipeline.fit_full_predict()` right before `model = self.build_model(-1)`: calls `seed_everything(self.cfg.seed)`.
2. **Pre-Fold Time Budget Hook (`estimate_fold_seconds`) in `BasePipeline` (`utils/core/pipeline.py`):**
   - Adds method `def estimate_fold_seconds(self, completed_folds: list[int]) -> float: return 0.0`.
   - In `BasePipeline.run()` before starting fold `k` (when `len(completed_folds) >= 1` and `self.cfg.max_minutes is not None`):
     ```python
     elapsed_sec = self._clock() - session_start
     rem_sec = self.cfg.max_minutes * 60.0 - elapsed_sec
     est_sec = float(self.estimate_fold_seconds(completed_folds))
     if rem_sec <= 0.0 or rem_sec < est_sec:
         rm.log(
             f"Stopping before fold {k}: remaining {rem_sec:.1f}s < required {est_sec:.1f}s (max_minutes={self.cfg.max_minutes}m)"
         )
         break
     ```
3. **`truncated` Flag Propagation (`utils/core/pipeline.py` & `utils/core/report.py`):**
   - In `compute_scores`: if `info_k.get("truncated")` is truthy, adds `"truncated": True` to `per_fold[str(k)]`.
   - In `generate_report`: formats fold label in the Per-Fold Breakdown table as `f"{k} (truncated)" if f_info.get("truncated") else f"{k}"`.
4. **Export `safe_name` in `utils/core/__init__.py`:**
   - Adds `"safe_name"` to `utils.core.__all__` (and updates `tests/core/test_public_api.py` accordingly).

---

### 6.1 `TextConfig(BaseConfig)` (`utils/text/config.py`)

Inherits all fields and behavior from `BaseConfig` (`modality` defaults to `"text"`).

```python
VALID_CLEAN_MODES = ("none", "minimal", "social_id")
VALID_ESTIMATORS = ("logistic", "linear_svc", "sgd", "ridge", "lgbm")
VALID_PRECISIONS = ("auto", "fp32", "fp16", "bf16")


@dataclass(kw_only=True)
class TextConfig(BaseConfig):
    modality: str = "text"
    text_cols: list[str] = field(default_factory=lambda: ["text"])
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    text_sep: str = " [SEP] "

    # Stateless text preprocessing & row-wise feature flags (run in build_features)
    clean_mode: str = "minimal"          # "none" | "minimal" | "social_id"
    lowercase: bool = True
    slang_dict_path: Optional[str] = None
    add_text_stats: bool = True
    add_pair_features: bool = True       # active when len(text_cols) >= 2

    # Postprocessing
    tune_threshold: bool = True          # active when task == "binary" and metric.input_type == "label"

    # TF-IDF specific parameters
    estimator: str = "logistic"          # "logistic" | "linear_svc" | "sgd" | "ridge" | "lgbm"
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    use_char_ngrams: bool = True
    max_features_word: int = 50_000
    max_features_char: int = 50_000
    min_df: int = 2
    sublinear_tf: bool = True
    c_reg: float = 1.0                   # Inverse regularization C (or alpha = 1 / c_reg for ridge/sgd)
    svd_components: int = 64             # Used when estimator == "lgbm" (TruncatedSVD on TF-IDF)
    top_k_importances: int = 50

    # Transformer specific parameters
    model_name_or_path: str = "assets/models/indobert-base-p1"
    local_files_only: bool = True
    select_metric: Optional[str] = None  # Auto-resolves to "logloss" (clf) or "rmse" (reg)
    fold_time_margin: float = 1.1        # Pre-fold time guard multiplier on previous fold duration_sec
    max_length: int = 256
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "auto"              # "auto" | "fp32" | "fp16" | "bf16"
    serialize_meta_to_text: bool = True  # Prepends num_cols/cat_cols as "col: val" prefix for Transformer
    save_best_weights: bool = False      # If True, saves fold_k/best_model.pt on disk
    num_workers: int = 0                 # 0 avoids multiprocessing fork deadlocks in notebooks/Windows
```

**Serialization & `load_runs` / `resume_from` Compatibility (`TextConfig.to_dict()`, `save()`, & `load()`):**
1. `TextConfig.resolve()` calls `super().resolve()`, normalizes `word_ngram_range` and `char_ngram_range` to `tuple[int, int]`, and auto-resolves `select_metric` to `"logloss"` (for `"binary"` / `"multiclass"`) or `"rmse"` (for `"regression"`) when `select_metric is None`.
2. `TextConfig.to_dict()` returns a dictionary containing the exact 22 top-level keys of `BaseConfig`, preserving user keys in `d["extra"]` while storing all `TextConfig`-specific fields under the reserved namespace `d["extra"]["_text"]` with tuples converted to JSON-native `list`s (e.g., `"word_ngram_range": [1, 2]`).
3. `TextConfig.save(path)` writes `save_json(self.to_dict(), path)`.
4. `TextConfig.load(path)` reads `config.json`, deep-copies `data`, extracts `text_fields = data.get("extra", {}).pop("_text", {})` so `cfg.extra` is restored to the exact user dictionary without `_text` pollution, and constructs `TextConfig(**data, **text_fields)`.

**Validation Rules (`TextConfig.validate()`):**
Calls `super().validate()`, then enforces:
1. `text_cols` must be a non-empty `list[str]` of unique column names and must not overlap with `id_col`, `target`, `group_col`, or `time_col`.
2. `clean_mode` must be in `VALID_CLEAN_MODES`.
3. `estimator` must be in `VALID_ESTIMATORS`, and classification-only estimators (`"logistic"`, `"linear_svc"`) raise `ConfigError` when `task == "regression"`.
4. `precision` must be in `VALID_PRECISIONS`.
5. `select_metric` must exist in the Core metric registry (`get_metric(self.select_metric)`).
6. `word_ngram_range` and `char_ngram_range` must be 2-element sequences `(low, high)` with `1 <= int(low) <= int(high)`.
7. `max_length >= 8`, `epochs >= 1`, `train_batch_size >= 1`, `eval_batch_size >= 1`, `grad_accum_steps >= 1`, `lr > 0`, `c_reg > 0`, `svd_components >= 2`, `top_k_importances >= 1`, `fold_time_margin >= 1.0`.
8. If `slang_dict_path` is provided and does not exist on disk, raise `ConfigError` with `[Perbaikan]`.

---

### 6.2 Stateless Text Cleaning & Row-Wise Features (`preprocessing.py`, `features.py`)

Executed inside `TextPipelineBase.build_features(train, test)` so raw text is preserved for inspection while cleaned text and numeric row-wise features are added without any target or cross-row leakage:

1. **Missing Value Handling on `text_cols`:**
   - Any `NaN` or `None` in `df[col]` is filled with `""` (empty string) and an indicator column `{col}__is_na` (`0.0`/`1.0` float) is added when `add_text_stats=True`.
2. **Text Cleaning (`clean_text_series(s, mode, lowercase, slang_map)`):**
   - Stored in `{col}__clean` so raw text in `{col}` is never overwritten.
   - `"none"`: `s.fillna("").astype(str)` (plus `.str.lower()` if `lowercase=True`).
   - `"minimal"`: normalizes Unicode NFC, strips HTML tags/entities, collapses repeated whitespace/newlines to a single space, optional lowercase.
   - `"social_id"`: `"minimal"` + replaces URLs with `<url>`, `@mentions` with `<user>`, emails with `<email>`, collapses `>= 3` repeated characters (e.g., `"baguuusss"` → `"baguus"`), and normalizes common Indonesian colloquial/slang tokens via built-in `DEFAULT_ID_SLANG_MAP` (merged with custom JSON/CSV `slang_dict_path` if given).
3. **Stateless Row-Wise Statistics (`extract_text_stats(df, text_cols)`):**
   - Computed on raw text (before lowercasing/cleaning strips casing or URLs):
     - `{col}__is_na`, `{col}__char_len`, `{col}__word_count`, `{col}__mean_word_len`
     - `{col}__upper_ratio`, `{col}__digit_ratio`, `{col}__punct_count`
     - `{col}__excl_count` (`!`), `{col}__quest_count` (`?`), `{col}__url_count`, `{col}__mention_count`, `{col}__hashtag_count`
4. **Stateless Row-Wise Pair Features (`extract_pair_stats(df, col_a, col_b)` when `len(text_cols) >= 2` and `add_pair_features=True`):**
   - Computed on `{col_a}__clean` and `{col_b}__clean` row by row:
     - `pair__char_len_diff`, `pair__char_len_ratio`
     - `pair__word_len_diff`, `pair__word_len_ratio`
     - `pair__exact_match` (`1.0` if identical, else `0.0`)
     - `pair__word_jaccard` (token set intersection / union)
     - `pair__char3_jaccard` (character 3-gram set intersection / union)

---

### 6.3 Shared Base Class: `TextPipelineBase(BasePipeline)` (`utils/text/base.py`)

Provides shared input/target validation, pre-fold time estimation, `build_features`, numeric feature scaling (`fit_transform_fold`), and `postprocess` threshold tuning:

1. **Input Column & Binary Target Validation (at the top of `build_features(train, test)`):**
   - Verifies every column in `cfg.text_cols`, `cfg.num_cols`, and `cfg.cat_cols` exists in both `train` and `test`, raising `DataError` with `[Perbaikan]` if any is missing.
   - **Binary Target Contract:** When `cfg.task == "binary"`, verifies that `train[cfg.target]` has numeric/boolean dtype and its unique non-NaN values are a subset of `{0, 1}`. If `train[cfg.target]` contains string labels or values outside `{0, 1}`, raises `ConfigError` with `[Perbaikan]` (*"Ubah kolom target biner ke integer 0/1 sebelum memanggil run(), dan simpan pemetaan labelnya"*).
2. **Pre-Fold Time Budget Hook (`estimate_fold_seconds(completed_folds)`):**
   - Overrides `BasePipeline.estimate_fold_seconds(completed_folds)` to return:
     `max((float(self.fold_info[fk].get("duration_sec", 0.0)) for fk in completed_folds), default=0.0) * float(self.cfg.fold_time_margin)`
   - Because `self.fold_info` is populated from `fold{fk}_info.json` on `resume_from`, this pre-fold guard works identically on fresh and resumed runs.
3. **`build_features(train, test)`:**
   - Cleans each `col` in `cfg.text_cols` into `{col}__clean`.
   - When `self._use_tabular_stats` is `True` (`True` for `TfidfPipeline`, `False` for `TransformerPipeline`): runs `extract_text_stats` (if `cfg.add_text_stats`) and `extract_pair_stats` on the first two `text_cols` (if `len(cfg.text_cols) >= 2` and `cfg.add_pair_features`), storing generated numeric column names in `self._stat_cols`.
4. **`fit_transform_fold(tr, va, te, fold)`:**
   - For `TfidfPipeline` (`self._use_tabular_stats == True`), scales `num_stat_cols = list(dict.fromkeys(cfg.num_cols + self._stat_cols))`: fills `NaN` with the median computed strictly on `tr` (or `0.0` if all `NaN`), then applies `StandardScaler` fit on `tr` (skipping `va` transform when `len(va) == 0` on `fold == -1`).
   - Categorical metadata columns (`cfg.cat_cols`) are sanitized as `fillna("__NA__").astype(str)` in `fit_transform_fold` and encoded inside `TfidfPipeline.fit_predict_fold` via `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`.
5. **`postprocess(oof_df, test_df=None)`:**
   - When `cfg.tune_threshold=True`, `cfg.task == "binary"`, and `get_metric(cfg.metric).input_type == "label"`:
     - Drops rows with `NaN` in `cfg.target` or `"pred"` (`valid_df = oof_df.dropna(subset=[cfg.target, "pred"])`) so partial runs or `fold == -1` rows never break threshold search.
     - If `len(valid_df) == 0`, returns `{}`.
     - Searches thresholds in `np.linspace(0.05, 0.95, 181)` on `valid_df` to optimize `score(y_true, y_pred, cfg.metric, "binary", threshold=t, classes=self.classes_)`.
     - Returns `{"threshold": float(best_t)}`.
   - Otherwise returns `{}`.

---

### 6.4 Model Family #1: `TfidfPipeline(TextPipelineBase)` (`utils/text/tfidf.py`)

`model_name` defaults to `f"tfidf_{cfg.estimator}"`.

1. **Per-Fold Sparse Vectorization inside `fit_predict_fold(model, tr, va, te, fold)`:**
   - For each `col` in `cfg.text_cols`:
     - Fits `word_vec = TfidfVectorizer(ngram_range=cfg.word_ngram_range, max_features=cfg.max_features_word, min_df=cfg.min_df, sublinear_tf=cfg.sublinear_tf)` strictly on `tr[f"{col}__clean"]` (falling back to `min_df=1` if `len(tr) < cfg.min_df` or empty vocabulary).
     - If `cfg.use_char_ngrams`: fits `char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=cfg.char_ngram_range, max_features=cfg.max_features_char, min_df=cfg.min_df, sublinear_tf=cfg.sublinear_tf)` strictly on `tr[f"{col}__clean"]`.
   - **Leak-Free Pair TF-IDF Cosine Similarity (when `len(cfg.text_cols) >= 2` and `cfg.add_pair_features=True`):**
     - Fits a shared word `TfidfVectorizer` (and char `TfidfVectorizer` if `use_char_ngrams=True`) on the concatenation of `tr[col_a__clean]` and `tr[col_b__clean]` (**only from `tr`**).
     - Transforms `(col_a, col_b)` for `tr`, `va` (if non-empty), and `te`, and computes row-wise L2-normalized dot product `(V_a.multiply(V_b)).sum(axis=1)` as `pair__tfidf_word_cos` and `pair__tfidf_char_cos`.
   - Combines all sparse TF-IDF blocks, `OneHotEncoder` sparse `cat_cols` (if any), and scaled numeric/stat/pair features via `scipy.sparse.hstack(..., format="csr")`.
2. **Estimator Execution & Per-Fold Seed (`fold_seed = cfg.seed + max(fold, 0)`):**
   - `"logistic"`: `LogisticRegression(C=cfg.c_reg, solver="lbfgs", max_iter=1000, random_state=fold_seed)`.
   - `"linear_svc"`:
     - Computes `min_class_count = int(pd.Series(y_tr).value_counts().min())`.
     - If `min_class_count < 2`: emits `UserWarning` and falls back to `LogisticRegression(C=cfg.c_reg, solver="lbfgs", max_iter=1000, random_state=fold_seed)`.
     - Otherwise fits `CalibratedClassifierCV(LinearSVC(C=cfg.c_reg, random_state=fold_seed), cv=min(3, min_class_count))` on `tr`.
   - `"sgd"`: `SGDClassifier(loss="log_loss", alpha=1.0 / (cfg.c_reg * max(len(tr), 1)), random_state=fold_seed)` for classification; `SGDRegressor(alpha=1.0 / (cfg.c_reg * max(len(tr), 1)), random_state=fold_seed)` for regression.
   - `"ridge"`: `Ridge(alpha=1.0 / cfg.c_reg, random_state=fold_seed)` for regression; for binary/multiclass uses `RidgeClassifier(alpha=1.0 / cfg.c_reg, random_state=fold_seed)` converting `decision_function` via sigmoid (binary) or softmax (multiclass).
   - `"lgbm"`:
     - Fits `TruncatedSVD(n_components=min(cfg.svd_components, max(1, X_sparse.shape[1] - 1)), random_state=fold_seed)` on `tr`'s sparse TF-IDF matrix (or uses `X_sparse.toarray()` directly if `X_sparse.shape[1] <= cfg.svd_components`), concatenates SVD components with numeric/stat/pair features.
     - **When `fold >= 0` (`va` non-empty):** fits `LGBMClassifier` / `LGBMRegressor` (`n_estimators=500`, `random_state=fold_seed`, `n_jobs=-1`, `verbosity=-1`) with `eval_set=[(X_va, y_va)]` and `callbacks=[early_stopping(30, verbose=False)]`, recording `self.fold_info[fold]["best_iteration"] = int(model.best_iteration_ or 500)`.
     - **When `fold == -1` (`refit_full`, `va` empty):** sets `n_estimators = max(1, self.refit_iterations() or 100)` (which already includes `cfg.refit_iter_mult`) and fits on 100% of `tr` **without** early stopping (`eval_set=None`).
3. **Multiclass K-Column Probability Alignment:**
   - For `task == "multiclass"`, aligns estimator probabilities to the full `K = len(self.classes_)` columns via `estimator.classes_`: any class absent in `tr` receives column `0.0`, and rows are re-normalized to sum to `1.0` so `pred_0..pred_{K-1}` never shift positionally.
4. **Top-K Importances Extraction:**
   - Extracts feature names across all vectorizers + numeric/SVD features, computes `|coef|` (averaged across classes for multiclass) or `feature_importances_`, selects the top `cfg.top_k_importances` (default 50), and saves `{name: float(val)}` to `self.fold_info[fold]["importances"]`.

---

### 6.5 Model Family #3: `TransformerPipeline(TextPipelineBase)` (`utils/text/transformer.py`)

`model_name` defaults to `f"hf_{Path(cfg.model_name_or_path).name}"`.

1. **Fast Local Directory Pre-Check (at the start of `build_features`):**
   - Before cleaning text or starting folds, validates `cfg.model_name_or_path`:
   - When `cfg.local_files_only=True`: verifies that `Path(cfg.model_name_or_path)` is an existing directory containing `config.json`, at least one tokenizer file (`vocab.txt`, `tokenizer.json`, `sentencepiece.bpe.model`, `spiece.model`, or `tokenizer_config.json`), and at least one weight file (`model.safetensors` or `pytorch_model.bin`). If missing, raises `ConfigError` immediately with `[Perbaikan]` showing how to download or point `model_name_or_path` to the offline model folder.
2. **Fresh Pretrained Model per Fold (`build_model(fold)`) & Memory Cleanup:**
   - `AutoTokenizer` is loaded once and cached on `self._tokenizer`.
   - `build_model(fold)` (called after Core's `seed_everything(cfg.seed + max(fold, 0))`) loads a **brand-new** `AutoModelForSequenceClassification.from_pretrained(cfg.model_name_or_path, config=hf_cfg, local_files_only=cfg.local_files_only)` from disk on **every** fold `k` (and `fold == -1`), guaranteeing zero weight carryover between folds.
   - At the end of `fit_predict_fold`, deletes the model (`del model`), runs `gc.collect()`, and calls `torch.cuda.empty_cache()` (returning `None` as the model handle).
3. **Tokenization, Metadata Serialization, Deterministic `DataLoader` & Dynamic Padding:**
   - If `cfg.serialize_meta_to_text=True` and `(cfg.num_cols or cfg.cat_cols)` is non-empty, formats `"col1: val1 | col2: val2 | "` per row and prepends it to `{cfg.text_cols[0]}__clean`.
   - If `len(cfg.text_cols) == 1`: tokenizes `first_col_texts` with `truncation=True, max_length=cfg.max_length, padding=False`.
   - If `len(cfg.text_cols) == 2`: tokenizes `(first_col_texts, df[f"{col_b}__clean"].tolist())` as a native pair (`truncation=True, max_length=cfg.max_length, padding=False`).
   - If `len(cfg.text_cols) > 2`: joins cleaned columns with `cfg.text_sep` and tokenizes with `truncation=True, max_length=cfg.max_length, padding=False`.
   - Uses a seeded PyTorch generator `g = torch.Generator(); g.manual_seed(cfg.seed + max(fold, 0))` in `DataLoader(..., shuffle=True, generator=g)` so batch order on a resumed fold `k` is identical to an uninterrupted run.
   - Batch collator pads each batch dynamically to the longest sequence in that batch (up to `cfg.max_length`).
4. **Loss Function, Target Scaling & Prediction Activation:**
   - `task == "binary"`: targets validated as `0` / `1`; `BCEWithLogitsLoss()`; predictions via `torch.sigmoid(logits)`.
   - `task == "multiclass"`: targets mapped to `0..K-1` via `self.classes_`; `CrossEntropyLoss()`; predictions via `torch.softmax(logits, dim=-1)`.
   - `task == "regression"`: targets standardized per fold (`y_scaled = (y - y_mean) / y_std`, where `y_std = max(std, 1e-8)`); `SmoothL1Loss()`; predictions unscaled via `logits * y_std + y_mean`.
5. **Training Loop, `precision`, `select_metric` Best-Epoch Selection, `refit_full`, and Two-Tier Time Guard:**
   - Optimizer: `AdamW` (`lr=cfg.lr`, `weight_decay=cfg.weight_decay` excluded on `bias` and `LayerNorm.weight`).
   - Scheduler: linear warmup for `int(total_steps * cfg.warmup_ratio)` steps followed by cosine decay to 0.
   - Precision resolution:
     - `"fp32"` (or when running on CPU): AMP disabled.
     - `"fp16"`: `torch.amp.autocast("cuda", dtype=torch.float16)` + `GradScaler("cuda", enabled=True)`.
     - `"bf16"`: `torch.amp.autocast("cuda", dtype=torch.bfloat16)` + `GradScaler("cuda", enabled=False)`.
     - `"auto"`: selects `"bf16"` if CUDA is available and `torch.cuda.is_bf16_supported()`, `"fp16"` if CUDA is available without bf16, else `"fp32"` on CPU.
   - **When `fold >= 0` (CV fold with non-empty `va`):**
     - **Fold-0 Batches 3..10 Timing Warning:** During epoch 0 of fold 0 (when `cfg.max_minutes is not None`), records batch durations for batch indices `2..min(9, len(train_loader) - 1)` (the 3rd through 10th batches, avoiding batch 0/1 warmup). Once batch index `min(9, len(train_loader) - 1)` finishes (if `len(train_loader) >= 3`), computes `est_epoch_sec = mean_batch_sec * len(train_loader)`. If `(self._clock() - self._run_start) + est_epoch_sec > cfg.max_minutes * 60.0`, emits a `UserWarning` (*"Estimasi 1 epoch (~{est_min:.2f}m) melebihi sisa max_minutes; cutoff darurat akan aktif setelah epoch pertama selesai"*).
     - Runs up to `cfg.epochs` epochs.
     - At the end of each epoch, computes `va_pred` and evaluates `val_select = score(y_va_raw, va_pred, cfg.select_metric, cfg.task, classes=self.classes_)`.
     - If `best_select is None` or `is_better(val_select, best_select, self.cfg.select_metric)`, snapshots the best state dict in CPU RAM (`copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})`), updates `best_select = val_select` and `best_epoch = epoch + 1` (`self.fold_info[fold]["best_iteration"] = best_epoch`), and optionally writes `best_model.pt` if `cfg.save_best_weights=True`.
     - **Emergency Intra-Fold Cutoff (Safety Net):** After each epoch (or batch once `epoch >= 1` so `best_state_dict is not None`), checks `self._clock() - self._run_start >= cfg.max_minutes * 60.0`. If triggered, sets `self.fold_info[fold]["truncated"] = True`, emits a `UserWarning`, breaks out of the loop, restores `best_state_dict`, and computes `va_pred` and `te_pred`.
   - **When `fold == -1` (`refit_full` with empty `va`):**
     - Determines `n_epochs = max(1, self.refit_iterations() or cfg.epochs)`.
     - Trains a fresh model for `n_epochs` on 100% of `tr` and returns `(None, empty_va_pred, te_pred)`.

---

### 6.6 Error Analysis & Multi-Label Helper (`utils/text/analysis.py`)

1. **`extract_top_errors(result: RunResult, train_df: pd.DataFrame, n: int = 20, text_cols: Optional[list[str]] = None) -> pd.DataFrame`:**
   - Aligns `train_df` with `result.oof` by `canon_id(id_col)`, dropping rows with `NaN` predictions.
   - Computes per-row error magnitude:
     - `binary`: `|y_true_01 - pred|`
     - `multiclass`: `1.0 - pred_{true_class_idx}`
     - `regression`: `|y_true - pred|`
   - Sorts descending by `error`, selects top `n` rows keeping `[id_col, "fold", *text_cols, target, *pred_cols, "error"]`, saves to `result.run_dir / "error_analysis.csv"`, and returns the DataFrame.
2. **`run_multilabel(pipeline_cls, base_cfg: TextConfig, train_df: pd.DataFrame, test_df: pd.DataFrame, target_cols: list[str], folds_dir: Union[str, Path] = "folds", sample_sub: Optional[pd.DataFrame] = None, sub_path: Optional[Union[str, Path]] = None) -> dict[str, RunResult]`:**
   - Validates all `target_cols` exist in `train_df`.
   - Imports `safe_name` from `utils.core.run` (`safe_name(s: Any, default: str) -> str`).
   - For each `tcol` in `target_cols`:
     - Resolves `label_folds_dir = Path(folds_dir) / safe_name(tcol, "label")` and `label_folds_csv = label_folds_dir / "folds.csv"`.
     - Clones `base_cfg` into `cfg_t` with `task="binary"`, `target=tcol`, `run_name=f"{base_cfg.run_name}_{tcol}"`.
     - If `label_folds_csv` does not exist yet (and `not cfg_t.debug`), calls `make_folds_file(train_df, cfg_t, out_dir=label_folds_dir)` to create it once.
     - Sets `cfg_t.folds_path = str(label_folds_csv)` (when `not cfg_t.debug`), runs `pipeline_cls(cfg_t).run(train_df, test_df)`, and stores the `RunResult` under `results[tcol]`.
   - If `sample_sub` and `sub_path` are provided, aligns test predictions across all `target_cols` by `canon_id` and writes the combined multi-label submission CSV.
   - Adds a concise note in `README.md` explaining per-label `folds/{label}/folds.csv` sharing for multi-label blending.

---

## 7. Testing & Verification Strategy (`tests/text/`)

Each test file is designed to run **100% offline on CPU in < 10 seconds per test file**, using `tests/text/__init__.py` and shared fixtures in `tests/text/helpers.py` (which builds a tiny local `BertForSequenceClassification` with `hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32` and a local `vocab.txt` containing `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, plus synthetic Indonesian domain words):

1. **Task 0 Core Tests (`tests/core/test_pipeline.py`, `tests/core/test_public_api.py`):**
   - Verifies all 44 existing Core tests pass, plus new tests for `estimate_fold_seconds` pre-fold stop with `FakeClock`, `self._run_start`, per-fold seeding on `resume_from`, `truncated` propagation into `scores.json` & `summary.md`, and `safe_name` export in `utils.core.__all__`.
2. **`tests/text/test_config_and_preprocessing.py`:**
   - Tests `TextConfig(..., extra={"foo": 1}) -> config.json -> BaseConfig.load -> TextConfig.load` round-trip identity (preserving user `extra={"foo": 1}` without `_text` leakage) and `resume_from` compatibility.
   - Tests `clean_text_series` across `"none"`, `"minimal"`, and `"social_id"`.
   - Tests `extract_text_stats` and `extract_pair_stats`.
   - Tests `ConfigError` when `task="binary"` is given string target labels (`["neg", "pos"]`).
3. **`tests/text/test_tfidf_pipeline.py`:**
   - Tests `TfidfPipeline` across `binary`, `multiclass` (with string labels, including rare-class fallback on `linear_svc` and missing-class K-column alignment), and `regression`.
   - Tests all estimators (`"logistic"`, `"linear_svc"`, `"sgd"`, `"ridge"`, `"lgbm"` including `fold == -1` refit using `refit_iterations()`).
   - Tests 2-column text pair mode (`text_cols=["q1", "q2"]`) with leak-free per-fold `pair__tfidf_word_cos` and `pair__tfidf_char_cos`.
   - Verifies `self.fold_info[0]["importances"]` is capped at `top_k_importances` and `plots/importance.png` is generated.
   - Verifies binary threshold tuning in `postprocess` handles `NaN` rows (partial OOF) and updates `params["threshold"]` and `scores["cv_honest"]`.
4. **`tests/text/test_transformer_pipeline.py`:**
   - Uses `n_folds=2, epochs=1` with the tiny local model from `tests/text/helpers.py`.
   - **Fast Pre-Check Test:** Verifies `ConfigError` with `[Perbaikan]` in `build_features` when `model_name_or_path` is missing or lacks `config.json`/weights.
   - **Fresh Model per Fold Test:** Verifies that initial weights at the start of fold 1 match the pretrained weights on disk (`torch.equal`), not the fine-tuned weights from fold 0.
   - **Deterministic Resume Test:** Verifies that fold 1 predictions from a resumed run (`resume_from` after fold 0) are identical (`np.allclose`) to fold 1 predictions from an uninterrupted full run.
   - **Two-Tier `max_minutes` Tests with `FakeClock`:**
     - Pre-fold guard stops before fold 1 when remaining session time `< fold_0_duration * fold_time_margin`, yielding `status == "partial"`.
     - Batches 3..10 slow timing on fold 0 emits `UserWarning` (not a false `DataError`).
     - Emergency mid-fold cutoff sets `fold_info[k]["truncated"] = True` and writes `(truncated)` in `summary.md`.
   - Tests `binary`, `multiclass` (string labels), `regression` (target scaling/unscaling), pair tokenization (`["text_a", "text_b"]`), and `test_mode="refit_full"`.
5. **`tests/text/test_analysis_and_multilabel.py`:**
   - Tests `extract_top_errors` output columns, sorting, and `error_analysis.csv` creation.
   - Tests `run_multilabel` with **2 target labels and 2 pipelines** (`TfidfPipeline` and `TransformerPipeline`), verifying per-label `folds/{label}/folds.csv` reuse, blending both pipelines per label via `load_runs` + `blend`, and combined multi-label submission CSV alignment.
6. **`tests/text/test_public_api.py`:**
   - Verifies `utils.text.__all__` exports all public classes and helpers.
