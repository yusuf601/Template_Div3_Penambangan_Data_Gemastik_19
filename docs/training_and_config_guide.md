# Panduan Pengaturan Training, Parameter Konfigurasi, & Referensi File `.json`

Dokumen ini menjelaskan seluruh pengaturan training pada **`BaseConfig`** ([`utils/core/config.py`](../utils/core/config.py)) dan **`TextConfig`** ([`utils/text/config.py`](../utils/text/config.py)), cara kerja auto-resolution, anatomi seluruh file `.json` (baik kamus input di `assets/` maupun artefak output di `runs/`), serta resep praktis pengaturan eksperimen saat lomba.

---

## Daftar Isi

1. [Referensi Lengkap Parameter Konfigurasi (`BaseConfig` & `TextConfig`)](#1-referensi-lengkap-parameter-konfigurasi-baseconfig--textconfig)
   - [A. Parameter Inti (`BaseConfig`)](#a-parameter-inti-baseconfig)
   - [B. Parameter Preprocessing & Fitur Teks (`TextConfig`)](#b-parameter-preprocessing--fitur-teks-textconfig)
   - [C. Parameter Training `TfidfPipeline`](#c-parameter-training-tfidfpipeline)
   - [D. Parameter Training `TransformerPipeline` (Deep Learning)](#d-parameter-training-transformerpipeline-deep-learning)
2. [Aturan Auto-Resolution (`cfg.resolve()`)](#2-aturan-auto-resolution-cfgresolve)
3. [Referensi Lengkap Seluruh File `.json` di Proyek](#3-referensi-lengkap-seluruh-file-json-di-proyek)
   - [A. File `.json` Input (`assets/`)](#a-file-json-input-assets)
   - [B. File `.json` Output & Checkpoint (`runs/<run_dir>/`)](#b-file-json-output--checkpoint-runsrun_dir)
4. [Alur Kerja & Resep Pengaturan Training Saat Lomba](#4-alur-kerja--resep-pengaturan-training-saat-lomba)

---

## 1. Referensi Lengkap Parameter Konfigurasi (`BaseConfig` & `TextConfig`)

`TextConfig` mewarisi seluruh field dari `BaseConfig`, sehingga dalam kompetisi berbasis teks Anda cukup menginisialisasi **`TextConfig(...)`** untuk mengatur segalanya.

### A. Parameter Inti (`BaseConfig`)

| Parameter | Tipe & Default | Fungsi & Aturan Validasi |
| :--- | :--- | :--- |
| `task` | `str` *(Wajib)* | Jenis masalah: `"binary"`, `"multiclass"`, atau `"regression"`. (Untuk multi-label beberapa kolom biner, gunakan helper [`run_multilabel`](../utils/text/analysis.py) dengan `task="binary"`). |
| `target` | `str` *(Wajib)* | Nama kolom target pada `train_df`. Pada `task="binary"`, nilai target di `train_df` **wajib** berupa angka `0` / `1` (atau `bool`). |
| `metric` | `str` *(Wajib)* | Metrik kompetisi terdaftar: `"f1"`, `"f1_macro"`, `"f1_micro"`, `"f1_weighted"`, `"auc"`, `"logloss"`, `"accuracy"`, `"rmse"`, `"mae"`, `"r2"`, `"mape"`, `"qwk"`. |
| `id_col` | `str = "id"` | Nama kolom ID unik pada `train_df`, `test_df`, dan `sample_sub`. Tidak boleh mengandung `NaN` atau duplikat. |
| `run_name` | `str = "baseline"` | Nama deskriptif eksperimen (menjadi akhiran nama folder di `runs/`). |
| `owner` | `str = ""` | Nama anggota tim / mesin. Jika dikosongkan (`""`), otomatis diisi nama hostname komputer saat `resolve()`. |
| `modality` | `str = "text"` | Penanda modalitas pada `TextConfig` (default `"text"`). |
| `seed` | `int = 42` | Seed global untuk reproduksibilitas model, numpy, dan PyTorch (`seed + fold` dipakai di setiap fold). |
| `fold_seed` | `int = 42` | Seed khusus untuk pengacakan pembagian Cross-Validation (`make_folds`). |
| `n_folds` | `int = 5` | Jumlah fold CV (minimal `2`). |
| `cv_strategy` | `Optional[str] = None` | `"stratified"`, `"kfold"`, `"group"`, `"stratified_group"`, atau `"timeseries"`. Jika `None`, dipilih otomatis oleh `resolve()`. |
| `group_col` | `Optional[str] = None` | Nama kolom grup (mis. `user_id` / `author_id`) agar baris dari grup yang sama tidak bocor antar train dan val. |
| `time_col` | `Optional[str] = None` | Nama kolom waktu/urutan untuk `cv_strategy="timeseries"`. |
| `ts_gap` | `int = 0` | Jarak baris (*gap*) antara train dan validation pada `cv_strategy="timeseries"` (`>= 0`). |
| `folds_path` | `Optional[str] = None` | Path ke file `folds.csv` eksternal (mis. `"folds/folds.csv"`) agar seluruh model memakai partisi fold yang 100% identik. |
| `test_mode` | `Optional[str] = None` | `"fold_avg"` (rata-rata prediksi seluruh fold) atau `"refit_full"` (latih ulang 1 model pada 100% data train di `fold=-1`). |
| `refit_iter_mult` | `float = 1.1` | Pengali jumlah iterasi/epoch saat `test_mode="refit_full"`: `round(mean(best_iteration) * refit_iter_mult)`. |
| `max_minutes` | `Optional[float] = None` | Batas waktu maksimal eksekusi `run()` dalam menit. Mengaktifkan *Pre-Fold Time Guard* dan *Intra-Fold Emergency Cutoff*. |
| `resume_from` | `Optional[str] = None` | Path atau nama folder run yang ingin dilanjutkan (melompati fold yang sudah memiliki checkpoint lengkap di `fold_preds/`). |
| `runs_dir` | `str = "runs"` | Direktori induk tempat seluruh folder hasil run dan `leaderboard.csv` disimpan. |
| `debug` | `bool = False` | Mode uji cepat: memotong `train_df` maks 2.000 baris, memaksa `n_folds=2`, menyimpan ke `runs/_debug/`, dan melewati penulisan `leaderboard.csv`. |
| `notes` | `str = ""` | Catatan bebas untuk eksperimen ini. |
| `extra` | `dict[str, Any] = {}` | Dictionary bebas untuk menyimpan parameter kustom tambahan pengguna. |

---

### B. Parameter Preprocessing & Fitur Teks (`TextConfig`)

| Parameter | Tipe & Default | Fungsi & Aturan Validasi |
| :--- | :--- | :--- |
| `text_cols` | `list[str] = ["text"]` | Daftar nama kolom teks. `1` kolom = single text; `2` kolom = native text-pair (NLI / similarity); `>2` kolom = digabung dengan `text_sep`. Tidak boleh bertabrakan dengan `id_col`, `target`, `group_col`, atau `time_col`. |
| `num_cols` | `list[str] = []` | Daftar kolom numerik pendamping teks (di-impute median & `StandardScaler` per-fold pada `TfidfPipeline`, atau diserialisasi ke teks pada `TransformerPipeline`). |
| `cat_cols` | `list[str] = []` | Daftar kolom kategorikal pendamping teks (`OneHotEncoder` per-fold pada `TfidfPipeline`, atau diserialisasi ke teks pada `TransformerPipeline`). |
| `text_sep` | `str = " [SEP] "` | Pemisah saat menggabungkan `>2` kolom teks. |
| `clean_mode` | `str = "minimal"` | Mode pembersihan teks ([`clean_text_series`](../utils/text/preprocessing.py)):<br>• `"none"`: tanpa pembersihan (hanya lowercase jika `lowercase=True`).<br>• `"minimal"`: normalisasi Unicode NFC + hapus tag/entitas HTML + rapikan spasi.<br>• `"social_id"`: `"minimal"` + masking `<url>`, `<email>`, `<user>` + penanganan emoji + ringkas karakter berulang (`>=3` jadi `2`, mis. `baguuusss` → `baguus`) + normalisasi slang Indonesia. |
| `lowercase` | `bool = True` | Mengubah teks menjadi huruf kecil. (Tip: set `False` untuk model SentencePiece *cased* seperti `mdeberta-v3-base`). |
| `slang_dict_path` | `Optional[str] = None` | Path ke kamus slang tambahan JSON/CSV/TSV (mis. `"assets/dictionaries/slang_id.json"`). Digabung di atas `DEFAULT_ID_SLANG_MAP`. |
| `emoji_mode` | `str = "keep"` | Mode penanganan emoji pada `clean_mode="minimal"` / `"social_id"`:<br>• `"keep"`: emoji dibiarkan apa adanya (no-op, tanpa impor library `emoji`).<br>• `"strip"`: seluruh sequence emoji atomik dihapus menjadi spasi.<br>• `"map"`: emoji yang ada di kamus diubah menjadi ` kata ` (spasi di kanan-kiri agar terpisah dari kata yang menempel seperti `gk👍`), sisanya dihapus. Karakter `©`, `®`, `™` selalu dilindungi. |
| `emoji_map_path` | `Optional[str] = "assets/dictionaries/emoji_map.json"` | Path ke kamus pemetaan emoji `{emoji: kata}` saat `emoji_mode="map"`. Otomatis dicari relatif terhadap root proyek via `resolve_asset_path()`. |
| `add_text_stats` | `bool = True` | Mengekstrak 12 fitur statistik per kolom teks (`char_len`, `word_count`, `upper_ratio`, `digit_ratio`, `excl_count`, `url_count`, `hashtag_count`, dll.) pada `TfidfPipeline`. |
| `add_pair_features` | `bool = True` | Jika `len(text_cols) >= 2`, mengekstrak fitur perbandingan pasangan teks (`word_jaccard`, `char3_jaccard`, `char_len_diff`, serta `pair__tfidf_word_cos` & `pair__tfidf_char_cos` per-fold di `TfidfPipeline`). |
| `tune_threshold` | `bool = True` | Jika `task="binary"` dan metrik berbasis label (mis. `f1`, `accuracy`), mencari threshold optimal di rentang `[0.05, 0.95]` pada OOF dan mengevaluasinya secara *honest* (`cv_honest`). |

---

### C. Parameter Training `TfidfPipeline`

| Parameter | Tipe & Default | Fungsi & Aturan Validasi |
| :--- | :--- | :--- |
| `estimator` | `Optional[str] = None` | Pilihan model di atas fitur TF-IDF:<br>• `"logistic"`: `LogisticRegression(solver="lbfgs")` (default klasifikasi).<br>• `"linear_svc"`: `LinearSVC` terkalibrasi via `CalibratedClassifierCV` (otomatis fallback ke `LogisticRegression` jika ada kelas dengan `<2` sampel di fold train).<br>• `"sgd"`: `SGDClassifier(loss="log_loss")` / `SGDRegressor`.<br>• `"ridge"`: `Ridge` (default regresi) / `RidgeClassifier`.<br>• `"lgbm"`: `TruncatedSVD(svd_components)` + `LGBMClassifier` / `LGBMRegressor` dengan early stopping 30 iterasi. |
| `word_ngram_range` | `tuple[int, int] = (1, 2)` | Rentang n-gram kata untuk `TfidfVectorizer`. |
| `char_ngram_range` | `tuple[int, int] = (3, 5)` | Rentang n-gram karakter (`analyzer="char_wb"`). |
| `use_char_ngrams` | `bool = True` | Mengaktifkan TF-IDF n-gram karakter (sangat efektif untuk typo dan variasi morfologi bahasa Indonesia). |
| `max_features_word` | `int = 50_000` | Maksimum ukuran kosakata word TF-IDF per kolom teks. |
| `max_features_char` | `int = 50_000` | Maksimum ukuran kosakata char TF-IDF per kolom teks. |
| `min_df` | `int = 2` | Frekuensi dokumen minimum token TF-IDF. |
| `sublinear_tf` | `bool = True` | Menggunakan skala logaritmik `1 + log(tf)` pada frekuensi kata. |
| `c_reg` | `float = 1.0` | Kekuatan regularisasi terbalik `C` (atau `alpha = 1 / c_reg` pada Ridge/SGD; harus `> 0`). |
| `svd_components` | `int = 64` | Jumlah komponen `TruncatedSVD` saat `estimator="lgbm"` (`>= 2`). |
| `top_k_importances` | `int = 50` | Jumlah fitur terpenting per fold yang disimpan ke `fold{k}_info.json` dan diplot ke `plots/importance.png`. |

---

### D. Parameter Training `TransformerPipeline` (Deep Learning)

| Parameter | Tipe & Default | Fungsi & Aturan Validasi |
| :--- | :--- | :--- |
| `model_name_or_path` | `str = "assets/models/indobert-base-p1"` | Path direktori model HuggingFace lokal. |
| `local_files_only` | `bool = True` | Memaksa pemuatan offline dan memicu pre-check cepat [`validate_local_model_dir`](../utils/text/transformer.py) sebelum training. |
| `select_metric` | `Optional[str] = None` | Metrik untuk memilih epoch terbaik (`best_iteration`) di tiap fold. Jika `None`, otomatis `"logloss"` (klasifikasi) atau `"rmse"` (regresi). |
| `fold_time_margin` | `float = 1.1` | Pengali estimasi durasi fold berikutnya (`max(duration_sec) * fold_time_margin >= 1.0`) untuk *Pre-Fold Time Guard*. |
| `max_length` | `int = 256` | Panjang maksimum token per baris (`>= 8`). Batch tetap di-pad secara dinamis mengikuti kalimat terpanjang di dalam batch tersebut. |
| `epochs` | `int = 3` | Jumlah epoch maksimum per fold (`>= 1`). |
| `train_batch_size` | `int = 16` | Ukuran batch training per step (`>= 1`). Gunakan `16` untuk IndoBERT dan `8` untuk `mdeberta-v3-base` pada RTX 3070 8GB. |
| `eval_batch_size` | `int = 32` | Ukuran batch saat inferensi validasi dan test (`>= 1`). |
| `grad_accum_steps` | `int = 1` | Akumulasi gradien sebelum `optimizer.step()`. Ukuran batch efektif = `train_batch_size * grad_accum_steps`. |
| `lr` | `float = 2e-5` | Learning rate `AdamW` (`> 0`). |
| `weight_decay` | `float = 0.01` | Weight decay `AdamW` (dikecualikan otomatis pada `bias` dan `LayerNorm.weight`). |
| `warmup_ratio` | `float = 0.1` | Proporsi total step training untuk linear warmup sebelum peluruhan cosine. |
| `max_grad_norm` | `float = 1.0` | Batas *gradient clipping* (`clip_grad_norm_`). |
| `precision` | `str = "auto"` | `"auto"` (otomatis `"bf16"` di GPU Ampere RTX 3070, `"fp16"` di GPU CUDA lama, `"fp32"` di CPU), `"fp32"`, `"fp16"`, atau `"bf16"`. |
| `serialize_meta_to_text` | `bool = True` | Menyisipkan nilai `num_cols` dan `cat_cols` sebagai teks `"kolom: nilai | "` di awal teks utama sebelum tokenisasi. |
| `save_best_weights` | `bool = False` | Opsi penyimpanan bobot terbaik. |
| `num_workers` | `int = 0` | Jumlah worker `DataLoader` (`0` mencegah *deadlock* multiprocessing di Jupyter Notebook / Windows). |

---

## 2. Aturan Auto-Resolution (`cfg.resolve()`)

Saat `pipeline.run(train_df, test_df)` dipanggil, `cfg.resolve()` dieksekusi otomatis sebelum `cfg.validate()` dengan aturan berikut:
1. **`owner`**: Jika kosong, diisi dari `socket.gethostname()` (hanya huruf kecil & angka, maks 12 karakter).
2. **`debug=True`**: Memaksa `n_folds = 2`, `folds_path = None`, dan `resume_from = None`.
3. **`cv_strategy`**: Jika `None`:
   - Ada `group_col` → `"stratified_group"` (untuk `binary`/`multiclass`) atau `"group"` (untuk `regression`).
   - Tanpa `group_col` → `"stratified"` (untuk `binary`/`multiclass`) atau `"kfold"` (untuk `regression`).
4. **`test_mode`**: Jika `None`, dipilih `"refit_full"` bila `cv_strategy == "timeseries"`, atau `"fold_avg"` untuk strategi lainnya.
5. **`select_metric`** (`TextConfig`): Jika `None`, dipilih `"rmse"` untuk `regression` atau `"logloss"` untuk `binary`/`multiclass`.
6. **`estimator`** (`TextConfig`): Jika `None`, dipilih `"ridge"` untuk `regression` atau `"logistic"` untuk `binary`/`multiclass`.

---

## 3. Referensi Lengkap Seluruh File `.json` di Proyek

### A. File `.json` Input (`assets/`)

#### 1. `assets/dictionaries/slang_id.json`
- **Fungsi:** Kamus normalisasi kata tidak baku / singkatan bahasa Indonesia yang dipakai saat `clean_mode="social_id"`.
- **Format JSON:** Objek flat `{ "slang": "baku" }` (kunci huruf kecil).
  ```json
  {
    "gk": "tidak",
    "bgt": "banget",
    "mantul": "mantap betul",
    "brg": "barang"
  }
  ```
- **Perilaku:** Saat `load_slang_map("assets/dictionaries/slang_id.json")` dipanggil, isinya digabungkan di atas `DEFAULT_ID_SLANG_MAP`. Anda bebas menambahkan singkatan baru yang ditemukan saat EDA ke file ini.

#### 2. `assets/dictionaries/emoji_map.json`
- **Fungsi:** Kamus pemetaan emoji ke kata bahasa Indonesia saat `emoji_mode="map"`.
- **Format JSON:** Objek flat `{ "emoji": "kata" }` tanpa kode `U+FE0F`.
  ```json
  {
    "👍": "bagus",
    "👎": "buruk",
    "❤": "suka",
    "😡": "marah",
    "🤮": "jijik",
    "💔": "kecewa"
  }
  ```
- **Perilaku & Aturan Penting:**
  - `load_emoji_map()` otomatis membuang variation selector (`U+FE0E`, `U+FE0F`) dan skin-tone (`U+1F3FB–U+1F3FF`) baik pada kunci JSON maupun pada teks yang sedang dibersihkan, sehingga `"❤️"` dan `"👍🏽"` tetap cocok ke `"❤"` dan `"👍"`.
  - Karena langkah emoji berjalan **sebelum** normalisasi slang di `clean_text_series`, pastikan **kata hasil pemetaan emoji bukan merupakan kunci di `slang_id.json`**.

#### 3. `assets/models/<nama_model>/config.json` & `tokenizer_config.json`
- **Fungsi:** Konfigurasi arsitektur dan tokenizer bawaan HuggingFace untuk pemuatan offline.
- **Perilaku:** Diperiksa oleh `validate_local_model_dir()` sebelum training dimulai. Anda tidak perlu mengubah `num_labels` di dalam file ini karena `TransformerPipeline.build_model()` mengatur `num_labels` dan `problem_type` secara dinamis di RAM.

---

### B. File `.json` Output & Checkpoint (`runs/<run_dir>/`)

Setiap eksekusi `pipeline.run(train_df, test_df)` menghasilkan struktur folder run berikut:

```text
runs/
├── leaderboard.csv                       # Rekap seluruh eksperimen tim
└── 20261108_0930_tim01_text_indobert/    # <run_dir>
    ├── config.json                       # 1. Snapshot konfigurasi lengkap
    ├── data_version.json                 # 2. Hash & dimensi dataset train/test
    ├── folds.csv                         # Partisi fold per ID
    ├── folds_meta.json                   # 3. Hash & metadata pembagian fold
    ├── postprocess.json                  # 4. Parameter post-processing (threshold)
    ├── scores.json                       # 5. Skor CV lengkap (raw, post, honest, per_fold)
    ├── lb.json                           # 6. Skor Public/Private LB (opsional via update_lb_score)
    ├── oof.parquet                       # Prediksi Out-of-Fold gabungan
    ├── test_pred.parquet                 # Prediksi Test gabungan
    ├── summary.md                        # Laporan ringkas Markdown
    ├── log.txt                           # Log eksekusi berstempel waktu
    ├── plots/                            # Visualisasi otomatis (oof_dist.png, importance.png, dll.)
    └── fold_preds/                       # Checkpoint instan per fold
        ├── fold0_oof.parquet
        ├── fold0_test.parquet
        └── fold0_info.json               # 7. Durasi, best_iteration, & feature importances fold 0
```

Berikut penjelasan detail ketujuh file `.json` di dalam `<run_dir>`:

#### 1. `config.json`
Menyimpan seluruh konfigurasi run. Agar kompatibel dengan `BaseConfig.load()` (yang dipakai oleh fungsi universal seperti `load_runs` dan `blend`), `TextConfig.to_dict()` menyimpan 23 field `BaseConfig` di level teratas, dan membungkus seluruh field spesifik teks di dalam `extra["_text"]`:
```json
{
  "task": "binary",
  "target": "label",
  "metric": "f1",
  "id_col": "id",
  "run_name": "indobert_social",
  "owner": "pc1",
  "modality": "text",
  "seed": 42,
  "fold_seed": 42,
  "n_folds": 5,
  "cv_strategy": "stratified",
  "test_mode": "fold_avg",
  "extra": {
    "_text": {
      "text_cols": ["text"],
      "clean_mode": "social_id",
      "emoji_mode": "map",
      "emoji_map_path": "assets/dictionaries/emoji_map.json",
      "model_name_or_path": "assets/models/indobert-base-p1",
      "max_length": 256,
      "epochs": 3,
      "train_batch_size": 16,
      "grad_accum_steps": 1,
      "lr": 2e-05,
      "precision": "auto"
    }
  }
}
```
- **Membaca kembali ke `TextConfig`:** Gunakan `cfg = TextConfig.load("runs/<run_dir>/config.json")` (otomatis mengekstrak `extra["_text"]` kembali menjadi atribut `TextConfig` dan mengembalikan tuple seperti `word_ngram_range`).
- **Kontrak `resume_from`:** Saat Anda memanggil `resume_from="runs/<run_dir>"`, `RunManager` memverifikasi bahwa konfigurasi baru **sama persis** dengan `config.json` pada folder tersebut, kecuali 5 field yang diizinkan berubah: `max_minutes`, `resume_from`, `notes`, `runs_dir`, dan `owner`.

#### 2. `data_version.json`
Mencatat hash kanonik dan bentuk tabel `train_df` serta `test_df` saat run dimulai:
```json
{
  "raw_train_hash": "a1b2c3d4e5f60718",
  "test_ids_hash": "9f8e7d6c5b4a3210",
  "train_shape": [25000, 4],
  "test_shape": [5000, 3],
  "train_cols": ["id", "text", "platform", "label"],
  "test_cols": ["id", "text", "platform"]
}
```

#### 3. `folds_meta.json`
Berpasangan dengan `folds.csv`. Mencatat identitas kriptografis partisi Cross-Validation:
```json
{
  "folds_hash": "4f3c2b1a9e8d7c6b",
  "ids_target_hash": "1122334455667788",
  "cv_strategy": "stratified",
  "n_folds": 5,
  "fold_seed": 42,
  "n_rows": 25000
}
```
> [!IMPORTANT]
> Fungsi [`blend(runs)`](../utils/core/ensemble.py) memeriksa `folds_hash` dan `test_ids_hash` dari setiap run yang digabungkan. Jika dua model dilatih dengan pembagian fold yang berbeda, `blend()` akan menolak penggabungan dengan `FoldMismatchError` demi mencegah *data leakage*.

#### 4. `postprocess.json`
Menyimpan parameter hasil post-processing OOF (seperti threshold klasifikasi biner optimal):
```json
{
  "threshold": 0.44
}
```
Jika `postprocess()` tidak aktif (mis. metrik probabilitas `auc`/`logloss` atau `tune_threshold=False`), file ini berisi `{}`.

#### 5. `scores.json`
Rekap lengkap metrik evaluasi run:
```json
{
  "model": "hf_indobert-base-p1",
  "timestamp": "2026-11-08T09:30:12+07:00",
  "owner": "pc1",
  "minutes": 11.42,
  "data_hash": "a1b2c3d4e5f60718",
  "metric": "f1",
  "direction": "max",
  "per_fold": {
    "0": {"raw": 0.831, "n_rows": 5000, "duration_sec": 135.2},
    "1": {"raw": 0.842, "n_rows": 5000, "duration_sec": 136.8}
  },
  "refit_duration_sec": 0.0,
  "cv": 0.8405,
  "cv_raw": 0.8365,
  "cv_post": 0.8431,
  "cv_honest": 0.8405,
  "cv_std": 0.0055,
  "cv_raw_all": 0.8365,
  "eval_folds": [0, 1, 2, 3, 4],
  "n_rows_eval": 25000,
  "folds_hash": "4f3c2b1a9e8d7c6b",
  "test_ids_hash": "9f8e7d6c5b4a3210",
  "classes": null,
  "status": "ok",
  "test_pred_source": "fold_avg",
  "n_folds_done": 5
}
```
- **Perbedaan `cv_raw`, `cv_post`, dan `cv_honest` (`cv`):**
  - `cv_raw`: Skor OOF sebelum postprocessing (threshold default `0.5`).
  - `cv_post`: Skor OOF dengan satu threshold yang dicari di seluruh OOF sekaligus (sedikit optimistik).
  - `cv_honest` (dipakai sebagai `cv` utama): Skor OOF dengan *out-of-fold threshold* (threshold untuk fold `k` dicari hanya dari fold selain `k`).

#### 6. `lb.json`
Dibuat saat Anda mencatat skor hasil submit ke portal kompetisi menggunakan `update_lb_score`:
```json
{
  "lb_score": 0.8451,
  "note": "Submission ke-2 (IndoBERT + social_id)",
  "updated_at": "2026-11-08T10:15:00+07:00"
}
```
Karena tersimpan sebagai file tersendiri di dalam folder run, skor LB tetap aman saat folder `runs/` disalin antar komputer dan disinkronkan ulang dengan `sync_leaderboard()`.

#### 7. `fold_preds/fold{k}_info.json`
Menyimpan metadata spesifik fold `k`:
```json
{
  "duration_sec": 135.2,
  "best_iteration": 2,
  "truncated": false,
  "importances": {
    "word__text__bagus": 2.412,
    "char__text__mant": 1.891
  }
}
```

---

## 4. Alur Kerja & Resep Pengaturan Training Saat Lomba

### Langkah 0: Fast EDA (`quick_text_eda` & `scripts/quick_eda.py`)
Sebelum memulai training atau preprocessing apa pun, jalankan Fast EDA untuk memeriksa distribusi teks, mendeteksi noise sosmed/emoji, dan mendapatkan rekomendasi parameter `TextConfig` (`clean_mode`, `emoji_mode`, `max_length`, `task`) secara instan.

**Opsi A: Menggunakan CLI Script (`scripts/quick_eda.py`)**
```bash
python scripts/quick_eda.py \
    --train train.csv \
    --test test.csv \
    --text-cols text \
    --target label \
    --task binary \
    --n-folds 5 \
    --out eda_summary.json
```

**Opsi B: Menggunakan Python Helper Function (`quick_text_eda`)**
```python
import pandas as pd
from utils.text import quick_text_eda

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

eda = quick_text_eda(
    train_df=train_df,
    test_df=test_df,
    text_cols=["text"],
    target="label",
    task="binary",
    verbose=True,  # Cetak ringkasan visual ke console
)

# Gunakan rekomendasi otomatis untuk inisialisasi TextConfig:
print("Rekomendasi TextConfig:", eda["recommendations"])
# Contoh output:
# {'clean_mode': 'social_id', 'emoji_mode': 'compare_keep_strip_map', 'max_length': 192, 'task': 'binary'}
```

### Langkah 1: Buat & Kunci File `folds.csv` Bersama di Awal Lomba
Lakukan ini **satu kali** di awal agar seluruh eksperimen TF-IDF dan Transformer menggunakan pembagian fold yang sama:

```python
import pandas as pd
from utils.core import BaseConfig, make_folds_file

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

base_cfg = BaseConfig(
    task="binary",          # atau "multiclass" / "regression"
    target="label",
    metric="f1",
    id_col="id",
    n_folds=5,
    fold_seed=42,
)
folds_csv_path = make_folds_file(train_df, base_cfg, out_dir="folds")
# Menghasilkan folds/folds.csv dan folds/folds_meta.json
```

### Langkah 2: Smoke Test Cepat dengan `debug=True`
Sebelum menjalankan training penuh, pastikan tidak ada error kolom atau tipe data dalam beberapa detik:

```python
from utils.text import TextConfig, TfidfPipeline

cfg_debug = TextConfig(
    run_name="smoke_test",
    task="binary",
    target="label",
    metric="f1",
    text_cols=["text"],
    clean_mode="social_id",
    emoji_mode="map",
    debug=True,             # Otomatis <=2000 baris, 2 fold, masuk ke runs/_debug/
)
TfidfPipeline(cfg_debug).run(train_df, test_df)
```

### Langkah 3: Jalankan Baseline Cepat TF-IDF (`logistic` & `lgbm`)

```python
for est in ["logistic", "lgbm"]:
    cfg_tfidf = TextConfig(
        run_name=f"tfidf_{est}",
        task="binary",
        target="label",
        metric="f1",
        folds_path="folds/folds.csv",
        text_cols=["text"],
        clean_mode="social_id",
        emoji_mode="map",
        slang_dict_path="assets/dictionaries/slang_id.json",
        estimator=est,
        n_folds=5,
    )
    res_tfidf = TfidfPipeline(cfg_tfidf).run(train_df, test_df)
    print(est, "CV:", res_tfidf.scores["cv"])
```

### Langkah 4: Jalankan `TransformerPipeline` dengan Guard Waktu & Resume Otomatis

```python
from utils.text import TransformerPipeline

cfg_hf = TextConfig(
    run_name="indobertweet_s1",
    task="binary",
    target="label",
    metric="f1",
    folds_path="folds/folds.csv",
    text_cols=["text"],
    clean_mode="social_id",
    emoji_mode="map",
    model_name_or_path="assets/models/indobertweet-base-uncased",
    max_length=192,
    train_batch_size=16,
    eval_batch_size=32,
    grad_accum_steps=1,
    lr=2e-5,
    epochs=3,
    precision="auto",
    max_minutes=20.0,       # Berhenti aman sebelum fold berikutnya jika waktu hampir habis
)
res_hf = TransformerPipeline(cfg_hf).run(train_df, test_df)

# Jika status "partial" (mis. baru selesai 3 dari 5 fold karena batas max_minutes):
if res_hf.scores["status"] == "partial":
    cfg_resume = TextConfig.load(res_hf.run_dir / "config.json")
    cfg_resume.resume_from = str(res_hf.run_dir)
    cfg_resume.max_minutes = 20.0
    res_hf = TransformerPipeline(cfg_resume).run(train_df, test_df)
```

### Langkah 5: Catat Skor LB, Bandingkan Run, & Blending Akhir

```python
from utils.core import update_lb_score, sync_leaderboard, compare_runs, load_runs, blend

# 1. Buat file submission dari model tunggal
res_hf.make_submission(sample_sub, "runs/sub_indobertweet.csv")

# 2. Catat skor Public LB setelah submit ke portal
update_lb_score(res_hf.run_dir, lb_score=0.8520, note="IndoBERTweet 5-fold")

# 3. Sinkronkan leaderboard (terutama setelah copy folder run dari laptop teman)
lb_df = sync_leaderboard("runs")
print(compare_runs("runs", top_k=10))

# 4. Blending OOF lintas keluarga model (TF-IDF + IndoBERTweet + mDeBERTa)
runs = load_runs([
    "runs/<folder_run_tfidf_logistic>",
    "runs/<folder_run_tfidf_lgbm>",
    "runs/<folder_run_indobertweet>",
    "runs/<folder_run_mdeberta>",
])
ens_res = blend(runs, method="weights", run_name="final_ensemble")
print("Final Ensemble CV:", ens_res.scores["cv"])
ens_res.make_submission(sample_sub, "runs/sub_final_ensemble.csv")
```
