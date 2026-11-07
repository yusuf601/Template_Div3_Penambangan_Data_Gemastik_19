# Template Gemastik - Divisi Penambangan Data (Data Mining)

Template standar kompetisi penambangan data berbasis **`utils.core` (Core-lite Universal ML Pipeline Framework)**. Dilengkapi dengan invariant guardrails anti data-leakage, time-budget guard, resume/checkpoint otomatis, metric registry, leaderboard upsert, serta OOF ensembling.

---

## Daftar Isi (Table of Contents)

1. [Persiapan Lingkungan (Setup & Installation)](#1-persiapan-lingkungan-setup--installation)
   - [Instalasi Dependencies](#instalasi-dependencies)
   - [Download Model Transformer Offline (RTX 3070 8GB)](#download-model-transformer-offline-rtx-3070-8gb)
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

### Download Model Transformer Offline (RTX 3070 8GB)

Sesuai spesifikasi PC lomba (**Intel Core i7-12700, RAM 32 GB DDR5, 1 TB NVMe SSD, NVIDIA GeForce RTX 3070 LHR 8 GB VRAM**), unduh **4 model Tier Utama** (~2.4 GB total) ke `assets/models/` sebelum lomba agar siap dijalankan secara 100% offline (`local_files_only=True`):

| # | Model HuggingFace | Direktori Lokal (`model_name_or_path`) | Ukuran | Peran Utama | VRAM & Waktu (5-Fold `bf16`) |
| :-: | :--- | :--- | :---: | :--- | :--- |
| **1** | `indobenchmark/indobert-base-p1` | `assets/models/indobert-base-p1` | ~440 MB | Baseline utama bahasa Indonesia formal/berita/umum (WordPiece) | ~2.8 GB (`bs=16, len=256`), ~10–15 mnt |
| **2** | `indolem/indobertweet-base-uncased` | `assets/models/indobertweet-base-uncased` | ~440 MB | Spesialis sosmed, ulasan e-commerce, komentar netizen, slang | ~2.8 GB (`bs=16, len=128–256`), ~8–14 mnt |
| **3** | `LazarusNLP/NusaBERT-base` *(atau `indobert-base-p2`)* | `assets/models/nusabert-base` | ~440 MB | Kosakata lebih baru & lintas bahasa daerah/nusantara | ~2.8 GB (`bs=16, len=256`), ~10–15 mnt |
| **4** | `microsoft/mdeberta-v3-base` | `assets/models/mdeberta-v3-base` | ~1.1 GB | Pasangan ensemble terbaik untuk IndoBERT (DeBERTa-v3 + SentencePiece; kuat di *text-pair* & *code-mixing*) | ~5.5–6.5 GB (`bs=8, accum=2, len=256`), ~20–30 mnt |

```bash
huggingface-cli download indobenchmark/indobert-base-p1 --local-dir assets/models/indobert-base-p1
huggingface-cli download indolem/indobertweet-base-uncased --local-dir assets/models/indobertweet-base-uncased
huggingface-cli download LazarusNLP/NusaBERT-base --local-dir assets/models/nusabert-base
huggingface-cli download microsoft/mdeberta-v3-base --local-dir assets/models/mdeberta-v3-base
```

> Panduan lengkap Model Cards, skrip verifikasi offline otomatis, dan konfigurasi per model tersedia di **[`docs/offline_models.md`](docs/offline_models.md)**.

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

`BaseConfig` (dan turunannya `TextConfig`) otomatis memvalidasi dan me-*resolve* parameter saat `pipeline.run()` dipanggil:
- **Parameter Wajib:** `task` (`"binary"`, `"multiclass"`, `"regression"`), `target`, dan `metric`.
- **Parameter CV & Eksekusi:**
  - `cv_strategy`: `"stratified"`, `"kfold"`, `"group"`, `"stratified_group"`, `"timeseries"` (otomatis dipilih jika `None`).
  - `n_folds`: Jumlah split cross-validation (default `5`).
  - `folds_path`: Path ke `folds/folds.csv` agar seluruh model menggunakan split CV yang identik.
  - `test_mode`: `"fold_avg"` (rata-rata semua fold) atau `"refit_full"` (latih ulang pada 100% train menggunakan `refit_iter_mult`).
  - `max_minutes`: Batas waktu running sebelum *Two-Tier Time-Budget Guard* aktif.
  - `debug`: Jika `True`, otomatis memotong data maks 2.000 baris (`n_folds=2`) dan menyimpan hasil ke `runs/_debug/`.

> [!IMPORTANT]
> **Dokumentasi Lengkap Pengaturan Training & File `.json`:**
> Penjelasan menyeluruh tentang seluruh parameter `BaseConfig` & `TextConfig`, struktur semua file `.json` (`config.json`, `scores.json`, `folds_meta.json`, `data_version.json`, `postprocess.json`, `lb.json`, `fold{k}_info.json`, serta kamus `slang_id.json` & `emoji_map.json`), dan resep alur kerja lomba tersedia di **[`docs/training_and_config_guide.md`](docs/training_and_config_guide.md)**.

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

Setelah menjalankan pipeline, direktori output akan terbentuk seperti berikut (lihat penjelasan detail isi setiap file `.json` di **[`docs/training_and_config_guide.md`](docs/training_and_config_guide.md#3-referensi-lengkap-seluruh-file-json-di-proyek)**):

```text
runs/
├── leaderboard.csv                  # Tabel performa seluruh run
└── <timestamp>_<owner>_<modality>_<run_name>/
    ├── config.json                  # Snapshot konfigurasi lengkap (extra["_text"] untuk TextConfig)
    ├── data_version.json            # Hash & dimensi dataset train/test
    ├── folds.csv                    # Pembagian fold per ID
    ├── folds_meta.json              # Hash & metadata partisi CV (diverifikasi saat blend)
    ├── postprocess.json             # Parameter post-processing (mis. threshold optimal)
    ├── scores.json                  # Metrik evaluasi (cv, cv_raw, cv_post, cv_honest, per_fold)
    ├── lb.json                      # Skor Public/Private LB (via update_lb_score)
    ├── oof.parquet                  # Gabungan Out-of-Fold utuh
    ├── test_pred.parquet            # Prediksi test gabungan seluruh fold
    ├── summary.md                   # Laporan ringkas Markdown
    ├── plots/                       # Plot diagnostik (oof_dist.png, importance.png, dll.)
    └── fold_preds/                  # Checkpoint instan per fold (fold{k}_oof/test.parquet & fold{k}_info.json)
```

---

## 6. Modalitas Teks (utils.text)

Modul `utils.text` dibangun di atas `utils.core` dengan filosofi **offline-first**, deterministik, dan anti-leakage untuk kompetisi NLP / Text Mining.

### A. TextConfig & Preprocessing

`TextConfig` mewarisi `BaseConfig` dengan validasi parameter spesifik NLP:
- `clean_mode`: `"none"`, `"minimal"` (normalisasi NFC, strip HTML, rapikan spasi), atau `"social_id"` (masking `<url>`/`<email>`/`<user>`, normalisasi slang bahasa Indonesia, serta peringkasan karakter berulang).
- `emoji_mode`: `"keep"` (default, tanpa dependensi tambahan), `"strip"` (hapus seluruh sequence emoji atomik), atau `"map"` (petakan emoji univalen menggunakan `assets/dictionaries/emoji_map.json`).
- `add_text_stats` & `add_pair_features`: Ekstraksi statistik teks bebas data leakage (`char_len`, `word_count`, `digit_ratio`, dll.) serta pair stats (`word_jaccard`, `char3_jaccard`, dll.).

### B. TfidfPipeline (Sparse & Classical ML)

Pipeline klasikal berbasis TF-IDF (`TfidfPipeline`) bekerja langsung dengan sparse CSR matrix tanpa densifikasi untuk efisiensi memori (RAM-friendly):
- Pilihan `estimator`: `"ridge"`, `"logistic"`, `"lgbm"`, `"sgd"`, `"linear_svc"`.
- Mendukung fitur pair teks (`len(text_cols) >= 2`) dengan TF-IDF Cosine Similarity per-fold bebas leakage.
- Menjamin alignment K-kolom probabilitas konsisten pada validasi/test saat terdapat kelas langka pada fold tertentu.

```python
from utils.text import TextConfig, TfidfPipeline

cfg = TextConfig(
    run_name="tfidf_baseline",
    task="binary",
    target="label",
    metric="f1_macro",
    estimator="logistic",
    clean_mode="social_id",
    emoji_mode="map",
    max_features_word=50_000,
    n_folds=5,
)
pipe = TfidfPipeline(cfg)
result = pipe.run(train_df, test_df)
```

### C. TransformerPipeline (Deep Learning Offline)

`TransformerPipeline` mendukung fine-tuning model HuggingFace lokal (PyTorch) secara offline (lihat detail Model Cards & batas VRAM RTX 3070 8GB di **[`docs/offline_models.md`](docs/offline_models.md)**):
- **Offline Guard:** `validate_local_model_dir` memeriksa kelengkapan `config.json`, file tokenizer, dan file weights secara instan sebelum training dimulai.
- **Dynamic Batch Collator:** Padding dinamis per-batch (`make_dynamic_collator`) menghemat VRAM secara signifikan dibandingkan static `max_length`.
- **Precision:** Mendukung `"auto"`, `"fp32"`, `"fp16"`, dan `"bf16"` (menggunakan `torch.amp.autocast`).
- **Two-Tier Time Guard:** Estimasi durasi fold otomatis (`estimate_fold_seconds`) dan cutoff darurat antar-epoch jika melewati `max_minutes`.

```python
from utils.text import TextConfig, TransformerPipeline

cfg = TextConfig(
    run_name="indobert_run",
    task="multiclass",
    target="category",
    metric="f1_macro",
    model_name_or_path="assets/models/indobert-base-p1",  # Direktori lokal
    local_files_only=True,
    max_length=256,
    train_batch_size=16,      # Gunakan 8 + grad_accum_steps=2 untuk mdeberta-v3-base
    eval_batch_size=32,
    grad_accum_steps=1,
    precision="auto",
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

### F. Fast EDA (`quick_text_eda` & CLI)

Lakukan audit kilat dataset teks di menit-menit awal lomba untuk mendapatkan ringkasan statistik teks, deteksi noise sosmed/emoji, imbalance target, dan rekomendasi otomatis parameter `clean_mode`, `emoji_mode`, serta `max_length`:

```python
from utils.text import quick_text_eda

eda = quick_text_eda(
    train_df=train_df,
    test_df=test_df,
    text_cols=["text"],
    target="label",
    verbose=True,
)
# eda["recommendations"] -> {'clean_mode': 'social_id', 'emoji_mode': 'compare_keep_strip_map', 'max_length': 192}
```

Tersedia juga standalone CLI script untuk inspeksi via terminal atau ekspor laporan JSON:
```bash
python scripts/quick_eda.py --train train.csv --test test.csv --text-cols text --target label --out eda.json
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
