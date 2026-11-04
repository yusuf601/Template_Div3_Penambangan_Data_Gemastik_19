# Template Gemastik - Divisi Penambangan Data (Data Mining)

Template standar kompetisi penambangan data berbasis **`utils.core` (Core-lite Universal ML Pipeline Framework)**. Dilengkapi dengan invariant guardrails anti data-leakage, time-budget guard, resume/checkpoint otomatis, metric registry, leaderboard upsert, serta OOF ensembling.

---

## Daftar Isi (Table of Contents)

1. [Persiapan Lingkungan (Setup & Installation)](#1-persiapan-lingkungan-setup--installation)
   - [Instalasi Dependencies](#instalasi-dependencies)
   - [Setup Local LLM (Ollama)](#setup-local-llm-ollama)
2. [Arsitektur & Konsep utils.core](#2-arsitektur--konsep-utilscore)
3. [Quickstart (Contoh Cepat End-to-End)](#3-quickstart-contoh-cepat-end-to-end)
4. [Panduan Komponen Utama](#4-panduan-komponen-utama)
   - [BaseConfig (Manajemen Konfigurasi)](#a-baseconfig-manajemen-konfigurasi)
   - [BasePipeline (Siklus Hidup & Hooks)](#b-basepipeline-siklus-hidup--hooks)
   - [Checkpointing & Resume](#c-checkpointing--resume)
   - [RunManager & Leaderboard](#d-runmanager--leaderboard)
   - [OOF Blending & Ensembling](#e-oof-blending--ensembling)
   - [Generasi Submission](#f-generasi-submission)
5. [Struktur Folder & Hasil Run](#5-struktur-folder--hasil-run)
6. [Modalitas Teks (utils.text)](#6-modalitas-teks-utilstext)
   - [TextConfig & Preprocessing](#a-textconfig--preprocessing)
   - [TfidfPipeline](#b-tfidfpipeline-sparse--classical-ml)
   - [TransformerPipeline](#c-transformerpipeline-deep-learning-offline)
   - [Multi-label Recipe & Shared Folds](#d-multi-label-recipe--shared-folds-run_multilabel)
   - [Error Analysis](#e-error-analysis)
7. [Testing & Verifikasi](#7-testing--verifikasi)

---

## 1. Persiapan Lingkungan (Setup & Installation)

### Instalasi Dependencies

Gunakan [uv](https://github.com/astral-sh/uv) atau `pip` standar untuk menginstal dependencies:

```bash
uv pip install -r requirements.txt
# atau
pip install -r requirements.txt
```

### Setup Local LLM (Ollama)

Jika alur kerja kompetisi memerlukan bantuan LLM lokal (misal: ekstraksi fitur teks / prompting):

```bash
# Instal Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Jalankan model DeepSeek
ollama run deepseek-r1:1.5b
```

---

## 2. Arsitektur & Konsep utils.core

`utils.core` dirancang universal untuk semua modalitas (Tabular, NLP, Vision, Audio, TimeSeries) dengan jaminan kepatuhan kompetisi:
- **Anti Data-Leakage:** Preprocessing per-fold dipisah ketat antara train, val, dan test (`fit_transform_fold`). Target dan ID di-snapshot sebelum manipulasi fitur.
- **Actionable Error:** Semua exception turunan `TemplateError` memberikan penjelasan terstruktur `[Masalah]` dan solusi `[Perbaikan]`.
- **Portabilitas & Determinisme:** Canonical ID hashing (`canon_id`, `fast_df_hash`) bekerja konsisten lintas OS (Linux & Windows).
- **Time-Budget & Checkpoint:** Menyimpan hasil OOF tiap fold seketika; aman dari timeout atau interupsi tanpa kehilangan progres.

---

## 3. Quickstart (Contoh Cepat End-to-End)

File runnable tersedia di `example_usage.py`. Anda dapat langsung mencobanya:

```bash
python example_usage.py
```

### Cuplikan Kode Minimal

```python
import pandas as pd
from sklearn.linear_model import Ridge
from utils.core import BaseConfig, BasePipeline, seed_everything

# 1. Buat Pipeline turunan BasePipeline
class MyPipeline(BasePipeline):
    model_name = "ridge_regressor"

    def build_model(self, fold: int):
        return Ridge(alpha=1.0, random_state=self.cfg.seed + fold)

# 2. Inisialisasi Konfigurasi
cfg = BaseConfig(
    run_name="baseline_ridge",
    owner="tim_gemastik",
    modality="tabular",
    task="regression",
    target="target",
    metric="rmse",
    n_folds=5,
    seed=42,
)

# 3. Jalankan Training & Evaluasi CV
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

pipeline = MyPipeline(cfg)
result = pipeline.run(train_df, test_df)

print(f"CV Score: {result.scores['cv']:.4f}")

# 4. Generate Submission
sample_sub = pd.read_csv("sample_submission.csv")
sub_df = result.make_submission(sample_sub, "runs/submission.csv")
```

---

## 4. Panduan Komponen Utama

### A. BaseConfig (Manajemen Konfigurasi)

`BaseConfig` otomatis memvalidasi parameter saat inisialisasi:
- **Parameter Wajib:** `run_name`, `target`.
- **Parameter Umum:**
  - `task`: `"binary"`, `"multiclass"`, `"regression"`, `"multilabel"`.
  - `metric`: `"rmse"`, `"mae"`, `"logloss"`, `"auc"`, `"f1_macro"`, `"qwk"`, dll.
  - `cv_strategy`: `"stratified"`, `"kfold"`, `"group"`, `"timeseries"`.
  - `n_folds`: Jumlah split cross-validation (default `5`).
  - `max_minutes`: Batas waktu running sebelum time-budget guard aktif.

### B. BasePipeline (Siklus Hidup & Hooks)

Turunkan `BasePipeline` dan implementasikan hook sesuai kebutuhan:

| Hook | Sifat | Penjelasan |
|---|---|---|
| `build_model(fold)` | **Wajib** | Mengembalikan instance model baru untuk fold tertentu. |
| `build_features(tr_df, te_df)` | Opsional | Feature engineering global sebelum splitting fold. |
| `fit_transform_fold(tr, va, te, fold)` | Opsional | Preprocessing per-fold (fit hanya pada train). |
| `fit_predict_fold(tr, va, te, fold)` | Default Scikit-Learn | Fit model lalu predict val dan test. Bisa di-override untuk PyTorch/LightGBM. |
| `postprocess_fold(preds, val_df, fold)` | Opsional | Postprocessing prediksi (misal: thresholding, clipping). |

### C. Checkpointing & Resume

Jika training terputus di tengah jalan (misal pada fold 3 dari 5), cukup tambahkan parameter `resume_from` pada config:

```python
cfg = BaseConfig(
    run_name="lanjutkan_training",
    resume_from="runs/20261031_1200_tim_gemastik_tabular_baseline_ridge",
    # ... parameter lain tetap sama ...
)
```
Pipeline akan otomatis mendeteksi checkpoint fold yang telah selesai dan langsung melanjutkan ke fold berikutnya.

### D. RunManager & Leaderboard

Setiap run tersimpan rapi di dalam direktori `runs/`:
- Menyimpan konfigurasi (`config.json`), metrik skor (`scores.json`), laporan slide (`report.html`), dan artefak prediksi (`oof.parquet`, `test.parquet`).
- Otomatis melakukan update pada `runs/leaderboard.csv`.
- Gunakan `sync_leaderboard(runs_dir="runs")` untuk merekonstruksi tabel leaderboard secara konsisten.

### E. OOF Blending & Ensembling

Menggabungkan beberapa model secara objektif menggunakan OOF (Out-of-Fold) blending tanpa risiko leakage:

```python
from utils.core import load_runs, blend

# Load artefak dari direktori run
runs = load_runs([
    "runs/20261031_run_ridge",
    "runs/20261031_run_lgbm"
])

# Blending dengan bobot optimal (SLSQP solver)
ensemble_result = blend(runs, method="weights", run_name="ensemble_ridge_lgbm")
print("Ensemble CV Score:", ensemble_result.scores["cv"])
```

### F. Generasi Submission

Method `make_submission` memverifikasi keselarasan ID, jumlah baris, dan mencegah nilai NaN:

```python
sub_df = result.make_submission(sample_sub_df, path="runs/final_sub.csv")
```

---

## 5. Struktur Folder & Hasil Run

Setelah menjalankan pipeline, direktori output akan terbentuk seperti berikut:

```
runs/
├── leaderboard.csv                  # Tabel performa seluruh run
└── <timestamp>_<owner>_<modality>_<run_name>/
    ├── config.json                  # Konfigurasi lengkap run
    ├── scores.json                  # Metrik evaluasi (cv, raw, per-fold)
    ├── report.html                  # Slide deck laporan visual & diagnostik
    ├── fold0_oof.parquet            # Prediksi validasi per-fold
    ├── fold0_test.parquet           # Prediksi test per-fold
    ├── oof.parquet                  # Gabungan Out-of-Fold utuh
    └── test.parquet                 # Rata-rata prediksi test seluruh fold
```

---

## 6. Modalitas Teks (utils.text)

Modul `utils.text` dibangun di atas `utils.core` dengan filosofi **offline-first**, deterministik, dan anti-leakage untuk kompetisi NLP / Text Mining.

### A. TextConfig & Preprocessing

`TextConfig` mewarisi `BaseConfig` dengan validasi parameter spesifik NLP:
- `clean_mode`: `"none"`, `"minimal"` (menjaga casing/punctuation), atau `"aggressive"` (lowercasing, slang normalization via map kamus slang bahasa Indonesia, regex sanitization).
- `stateless_features`: Ekstraksi statistik teks bebas data leakage (`char_length`, `word_count`, `digit_ratio`, dll) serta pair stats (`len_diff`, `jaccard_similarity`, `levenshtein_distance`).

### B. TfidfPipeline (Sparse & Classical ML)

Pipeline klasikal berbasis TF-IDF (`TfidfPipeline`) bekerja langsung dengan sparse CSR matrix tanpa densifikasi untuk efisiensi memori (RAM-friendly):
- Pilihan `estimator`: `"ridge"`, `"logistic"`, `"lgbm"`, `"sgd"`, `"linear_svm"`.
- Mendukung fitur pair teks (`text_pair_col`) dengan TF-IDF Cosine Similarity otomatis.
- Menjamin alignment K-kolom probabilitas konsisten pada validasi/test saat terdapat kelas langka pada fold tertentu.

```python
from utils.text import TextConfig, TfidfPipeline

cfg = TextConfig(
    run_name="tfidf_baseline",
    task="binary",
    target="label",
    metric="f1_macro",
    estimator="logistic",
    clean_mode="aggressive",
    tfidf_max_features=10000,
    n_folds=5,
)
pipe = TfidfPipeline(cfg)
result = pipe.run(train_df, test_df)
```

### C. TransformerPipeline (Deep Learning Offline)

`TransformerPipeline` mendukung fine-tuning model HuggingFace lokal (PyTorch) secara offline:
- **Offline Guard:** `validate_local_model_dir` memeriksa kelengkapan file weights dan tokenizer secara instan sebelum training dimulai.
- **Backbone Freezing:** Opsi `freeze_backbone=True` dan `unfreeze_last_n_layers` untuk pelatihan cepat atau transfer learning.
- **Dynamic Batch Collator:** Padding dinamis per-batch (`make_dynamic_collator`) menghemat VRAM hingga ~40% dibandingkan static max_length.
- **Precision:** Mendukung `"fp32"`, `"fp16"`, dan `"bf16"` (menggunakan `torch.amp.autocast`).
- **Emergency Truncation & Profiling:** Batch profiling otomatis memprediksi runtime per fold.

```python
from utils.text import TextConfig, TransformerPipeline

cfg = TextConfig(
    run_name="indobert_run",
    task="multiclass",
    target="category",
    metric="f1_macro",
    model_name_or_path="models/indobert-base-p1",  # Direktori lokal
    local_files_only=True,
    max_length=128,
    train_batch_size=16,
    eval_batch_size=32,
    gradient_accumulation_steps=2,
    precision="fp16",
    epochs=3,
    n_folds=5,
)
pipe = TransformerPipeline(cfg)
result = pipe.run(train_df, test_df)
```

### D. Multi-label Recipe & Shared Folds (`run_multilabel`)

Untuk kompetisi multi-label (Binary Relevance), `run_multilabel` melatih model independen per target label dan memastikan pembagian fold konsisten melalui `folds_dir`:

```python
from utils.text import TextConfig, TfidfPipeline, run_multilabel

cfg = TextConfig(
    run_name="multilabel_tfidf",
    task="binary",
    target="dummy",
    metric="auc",
    estimator="logistic",
    n_folds=5,
)

results = run_multilabel(
    TfidfPipeline,
    cfg,
    train_df,
    test_df,
    target_cols=["toxic", "severe_toxic", "obscene", "threat"],
    folds_dir="folds",  # Folds disimpan per-label: folds/{label}/folds.csv
    sample_sub=sample_sub_df,
    sub_path="runs/multilabel_submission.csv",
)
```

> [!TIP]
> Dengan menyimpan split fold di `folds/{label}/folds.csv`, arsitektur model berbeda (misal `TfidfPipeline` dan `TransformerPipeline`) akan menggunakan partisi data yang persis sama untuk setiap label, sehingga hasil OOF dapat langsung digabungkan menggunakan `utils.core.blend` tanpa terjadi `FoldMismatchError`.

### E. Error Analysis

Gunakan `extract_top_errors` untuk mengidentifikasi sampel dengan residual/loss prediksi terbesar untuk iterasi error analysis:

```python
from utils.text import extract_top_errors

err_df = extract_top_errors(result, train_df, n=20)
# Menyimpan hasil otomatis ke <run_dir>/error_analysis.csv
```

---

## 7. Testing & Verifikasi

Seluruh komponen `utils.core` dan `utils.text` telah dilengkapi dengan test suite lengkap (unit dan contract tests):

```bash
# Test Core Pipeline
pytest tests/core/ -v

# Test Text Modality Suite
pytest tests/text/ -v
```
