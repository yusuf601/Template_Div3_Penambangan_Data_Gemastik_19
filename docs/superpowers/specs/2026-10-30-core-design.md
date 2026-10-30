# Template Final Gemastik Data — Spec #1: Core

- **Tanggal:** 2026-10-30
- **Status:** Draft — menunggu review user
- **Scope spec ini:** `utils/core/` (fondasi bersama). Modul tabular, image, text punya spec terpisah (lihat Roadmap).

---

## 1. Latar Belakang & Tujuan

Final Gemastik Data kemungkinan memakai salah satu dari tiga tipe data: **text**, **citra**, atau **tabular (termasuk time series)**. Template ini harus memungkinkan tim:

1. Langsung bikin baseline yang valid dalam hitungan menit di notebook.
2. Override bagian mana pun (parameter maupun logika) tanpa mengedit library.
3. Eksperimen *reproducible*: seed, fold, versi data tersimpan otomatis.
4. Menyimpan OOF + prediksi test per run dengan format seragam, sehingga **blending/stacking lintas run (dan lintas modalitas)** valid dan tidak bocor.

### Keputusan yang sudah disepakati

| Topik | Keputusan |
|---|---|
| Bentuk | Notebook tipis yang `import` helper dari package `utils/` |
| Environment | Semua dependency di-install di awal; compute tidak jadi constraint |
| Mekanisme override | **A + C**: dataclass config (nilai) + subclass hook (logika) |
| Tracking | Folder run lokal + `runs/leaderboard.csv` (tanpa MLflow/W&B) |
| Reproducibility | Wajib: seed, `folds.csv`, versi data, env disimpan per run |
| Artefak | OOF + test pred **mentah** (proba / nilai regresi) per run |

---

## 2. Struktur Project

```
template_gemastik/
├── utils/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # BaseConfig dataclass
│   │   ├── seed.py        # seed_everything
│   │   ├── cv.py          # make_folds, load_folds, get_folds, split_fold
│   │   ├── metrics.py     # Metric, registry, register_metric, score
│   │   ├── run.py         # RunManager (folder run, artefak, leaderboard)
│   │   ├── pipeline.py    # BasePipeline (hook + orkestrasi CV)
│   │   ├── ensemble.py    # load_runs, blend, stack
│   │   └── io.py          # hash data, save/load parquet/json, env info
│   ├── tabular/           # spec #2
│   ├── image/             # spec #3
│   └── text/              # spec #4
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 10_tabular.ipynb
│   ├── 20_image.ipynb
│   ├── 30_text.ipynb
│   └── 90_ensemble.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── runs/
│   └── leaderboard.csv
├── tests/
│   └── core/
└── docs/superpowers/{specs,plans}/
```

Prinsip: setiap file di `core/` punya satu tanggung jawab dan bisa dites sendiri. `pipeline.py` adalah satu-satunya modul yang mengorkestrasi modul lain.

---

## 3. Komponen

### 3.1 `config.py` — `BaseConfig`

Dataclass. Modalitas lain men-subclass (`TabularConfig`, `ImageConfig`, `TextConfig`).

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `run_name` | `str` | `"baseline"` | Nama eksperimen |
| `modality` | `str` | `"generic"` | `tabular` / `image` / `text` / `ensemble` |
| `task` | `str` | wajib | `binary` / `multiclass` / `regression` |
| `target` | `str` | wajib | Nama kolom target |
| `id_col` | `str` | `"id"` | Kolom ID unik; dipakai untuk semua join |
| `metric` | `str` | wajib | Nama metric di registry |
| `seed` | `int` | `42` | Seed global |
| `n_folds` | `int` | `5` | Jumlah fold |
| `cv_strategy` | `str` | `"stratified"` | `kfold` / `stratified` / `group` / `stratified_group` / `timeseries` |
| `group_col` | `str \| None` | `None` | Wajib untuk `group` / `stratified_group` |
| `time_col` | `str \| None` | `None` | Wajib untuk `timeseries` |
| `ts_gap` | `int` | `0` | Jumlah baris/periode jeda antara train dan valid (timeseries) |
| `ts_max_train_size` | `int \| None` | `None` | `None` = expanding window; angka = sliding window |
| `folds_path` | `str \| None` | `None` | **Path `folds.csv` dari run sebelumnya. Jika diisi, fold di-load, bukan dibuat ulang.** |
| `test_mode` | `str` | `"fold_avg"` | `fold_avg` (rata-rata prediksi model per fold) / `refit_full` (latih ulang di full train) |
| `runs_dir` | `str` | `"runs"` | Root folder run |
| `save_models` | `bool` | `False` | Simpan checkpoint per fold |
| `debug` | `bool` | `False` | Subsample data + 2 fold untuk smoke test cepat |
| `notes` | `str` | `""` | Ditulis ke `notes.md` |
| `extra` | `dict` | `{}` | Bebas, untuk parameter custom user |

Method:
- `to_dict()` / `save(path)` / `BaseConfig.load(path)` — serialisasi JSON.
- `validate()` — dipanggil di awal pipeline; raise `ValueError` jika kombinasi invalid (mis. `cv_strategy="group"` tanpa `group_col`, `task="regression"` dengan `cv_strategy="stratified"`).

### 3.2 `seed.py`

`seed_everything(seed: int)` — set `random`, `numpy`, `PYTHONHASHSEED`, dan (jika terinstal) `torch` + `torch.cuda` + `cudnn.deterministic=True`, `cudnn.benchmark=False`. Tidak gagal jika torch tidak terinstal.

### 3.3 `cv.py` — Fold yang bisa dipakai ulang

**Representasi fold:** DataFrame `[id_col, "fold"]`. `fold ∈ {0..n_folds-1}`; untuk timeseries, baris yang tidak pernah masuk valid diberi `fold = -1`.

**Fungsi:**
- `make_folds(df, cfg) -> pd.DataFrame` — membuat split baru sesuai `cv_strategy` dan `seed`.
  - `timeseries`: data diurutkan menurut `time_col`, dibagi menjadi `n_folds + 1` blok berurutan; blok pertama hanya train (`fold=-1`), blok berikutnya menjadi valid fold 0..n-1.
- `load_folds(path, df, cfg) -> pd.DataFrame` — membaca `folds.csv` + `folds_meta.json` di direktori yang sama, lalu **validasi ketat** (lihat bawah).
- `get_folds(df, cfg)` — `load_folds` jika `cfg.folds_path` diisi, selain itu `make_folds`.
- `split_fold(df, folds, fold, cfg) -> (train_idx, valid_idx)`:
  - non-timeseries: train = semua `fold != k`, valid = `fold == k`.
  - timeseries: valid = `fold == k`; train = baris dengan `time_col` < waktu minimum valid dikurangi `ts_gap`, dipotong ke `ts_max_train_size` terakhir jika diisi. (Baris fold > k **tidak** masuk train.)

**`folds_meta.json`** (ditulis bersama `folds.csv`):
```json
{
  "n_folds": 5,
  "cv_strategy": "stratified",
  "seed": 42,
  "group_col": null,
  "time_col": null,
  "id_target_hash": "sha256...",
  "folds_hash": "sha256..."
}
```
- `id_target_hash` = hash dari pasangan `(id, target)` terurut menurut id. Sengaja **bukan** hash seluruh DataFrame, supaya menambah kolom fitur tidak membatalkan reuse fold.
- `folds_hash` = hash dari `(id, fold)` terurut; dipakai `ensemble.py` untuk memastikan semua run memakai split yang sama.

**Validasi saat load (semua → raise `FoldMismatchError`):**
1. Set `id` di data == set `id` di `folds.csv` (tidak lebih, tidak kurang).
2. `n_folds` di meta == `cfg.n_folds`.
3. `id_target_hash` di meta == hash data sekarang.
4. Join selalu via `id_col`, **tidak pernah** via index/urutan baris.

### 3.4 `metrics.py` — Registry

```python
@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    direction: Literal["maximize", "minimize"]
    input_type: Literal["proba", "label", "value"]
```

- `input_type`:
  - `proba` — fungsi menerima probabilitas (AUC, logloss).
  - `label` — fungsi menerima label kelas (F1, accuracy, QWK). Konversi dari proba dilakukan di `score()`.
  - `value` — regresi (RMSE, MAE, R2).
- `register_metric(name, direction, input_type)` — decorator untuk metric custom dari notebook.
- `get_metric(name) -> Metric`.
- `score(y_true, y_raw, metric, task, params=None) -> float` — `y_raw` selalu output mentah. Jika `input_type="label"`:
  - binary: `label = y_raw >= params.get("threshold", 0.5)`
  - multiclass: `label = argmax(y_raw, axis=1)`
- `is_better(a, b, metric) -> bool`.

**Built-in:** `auc`, `logloss`, `accuracy`, `f1`, `f1_macro`, `f1_weighted`, `qwk`, `rmse`, `mae`, `mape`, `r2`.

**Aturan utama:** `input_type` hanya menentukan **cara scoring**, bukan apa yang disimpan. OOF dan test pred **selalu** disimpan mentah.

### 3.5 `run.py` — `RunManager`

Membuat folder `runs/{YYYYMMDD_HHMM}_{modality}_{run_name}/` (tambah suffix `_2`, `_3` jika bentrok) dan menulis:

| File | Isi | Kapan ditulis |
|---|---|---|
| `config.json` | Seluruh config setelah override | Awal run |
| `env.json` | Versi python, numpy, pandas, sklearn, torch, timm, transformers, lightgbm, xgboost, catboost (yang terinstal), GPU name, git commit (jika ada) | Awal run |
| `data_version.json` | Hash raw train & test (sebelum `build_features`), shape, daftar kolom, dan md5 file sumber jika path diberikan | Awal run |
| `folds.csv` + `folds_meta.json` | Fold yang dipakai (selalu ditulis, juga saat di-load dari run lain) | Setelah fold dibuat/di-load |
| `oof.parquet` | `id`, `fold`, `target`, kolom pred | Akhir run |
| `test_pred.parquet` | `id`, kolom pred | Akhir run |
| `scores.json` | Skor per fold, `cv_mean`, `cv_std`, `cv_overall` (skor di seluruh OOF, baris NaN/`fold=-1` dikecualikan), metric, direction, `folds_hash`, `classes` (multiclass) | Akhir run |
| `postprocess.json` | Parameter dari hook `postprocess` (mis. threshold) | Akhir run |
| `models/fold{k}.*` | Checkpoint (opsional, `save_models=True`) | Per fold |
| `notes.md` | `cfg.notes` | Awal run |
| `log.txt` | Log run | Sepanjang run |

**Format kolom pred (seragam untuk semua modalitas):**
- binary / regression: satu kolom `pred`.
- multiclass: `pred_0 … pred_{K-1}` (probabilitas per kelas, urutan kelas disimpan di `scores.json` → `classes`).
- Baris timeseries dengan `fold=-1`: pred = NaN.

**Leaderboard:** setiap run yang selesai menambah satu baris ke `runs/leaderboard.csv`:
`run_dir, timestamp, modality, run_name, model, metric, cv_mean, cv_std, cv_overall, folds_hash, data_hash, seed`.
Run yang gagal tidak masuk leaderboard; foldernya diberi file `FAILED` berisi traceback.

### 3.6 `pipeline.py` — `BasePipeline` (hook)

```python
class BasePipeline:
    def __init__(self, cfg: BaseConfig): ...

    # ---- Hooks (override sesuai kebutuhan) ----
    def load_data(self, train, test):              # default: return apa adanya
    def build_features(self, train, test):         # GLOBAL, tidak boleh pakai target. default: identity
    def fit_transform_fold(self, tr, va, te, fold):# DI DALAM FOLD: target encoding, scaler,
                                                   # imputer, feature selection. default: identity
    def build_model(self, fold):                   # WAJIB di-override (subclass modalitas menyediakan default)
    def fit_predict_fold(self, model, tr, va, te, fold) -> (model, va_pred, te_pred)  # WAJIB
    def fit_full_predict(self, train, test) -> te_pred  # untuk test_mode="refit_full"
    def postprocess(self, oof_df, test_df) -> dict # default: {}; mis. optimasi threshold di OOF
    def get_feature_cols(self, df) -> list[str]    # default: semua kolom kecuali id, target, fold, group, time

    # ---- Orkestrasi (biasanya tidak di-override) ----
    def run(self, train, test) -> RunResult
```

**Alur `run()`:**
```
cfg.validate()
seed_everything(cfg.seed)
rm = RunManager(cfg)                     # config.json, env.json, notes.md
train, test = load_data(train, test)
rm.save_data_version(train, test)        # hash SEBELUM feature engineering
train, test = build_features(train, test)
folds = get_folds(train, cfg); rm.save_folds(folds)
for k in folds aktif:
    tr_idx, va_idx = split_fold(train, folds, k, cfg)
    tr, va, te = fit_transform_fold(train[tr_idx], train[va_idx], test, k)
    model = build_model(k)
    model, va_pred, te_pred = fit_predict_fold(model, tr, va, te, k)
    simpan va_pred ke oof; akumulasi te_pred; skor fold k; log
test_pred = mean(te_pred per fold)  atau  fit_full_predict(...) jika refit_full
params = postprocess(oof, test_pred);  rm.save_postprocess(params)
scores = hitung skor per fold + overall (dengan params)
rm.save_oof / save_test_pred / save_scores / append_leaderboard
return RunResult(run_dir, oof, test_pred, scores, params)
```

**Aturan anti-leakage (didokumentasikan di docstring hook):**
- `build_features` hanya boleh berisi transformasi yang tidak melihat target dan tidak "fit" statistik lintas fold (lag, rolling, date parts, text cleaning, resize).
- Semua yang di-*fit* (target encoding, scaler, imputer statistik, feature selection, PCA) **wajib** di `fit_transform_fold`, di-fit hanya pada `tr`.

**`RunResult`:** dataclass `run_dir, oof, test_pred, scores, params`, plus helper `make_submission(sample_sub, path, fn=None)` yang menerapkan `params` (mis. threshold/argmax) ke `test_pred` dan menulis CSV submission.

### 3.7 `ensemble.py`

- `load_runs(run_dirs) -> EnsembleData` — load OOF & test pred semua run, join via `id`.
  - **Raise** `FoldMismatchError` jika `folds_hash` antar run berbeda.
  - Raise jika `task`/`metric`/`classes` berbeda.
  - Baris dengan pred NaN (timeseries `fold=-1`) dikecualikan dari optimasi & scoring.
- `blend(data, method)`:
  - `mean`, `rank_mean`, `geo_mean`
  - `weights` — optimasi bobot non-negatif, jumlah 1, memaksimalkan/meminimalkan metric di OOF (`scipy.optimize.minimize`).
  - `hill_climb` — greedy forward selection dengan replacement.
- `stack(data, meta_model, cfg)` — meta-model dilatih dengan **`folds.csv` yang sama**, input = OOF level-1; menghasilkan OOF level-2 yang valid.
- Hasil blend/stack disimpan sebagai run baru (`modality="ensemble"`) lewat `RunManager`, termasuk daftar run sumber + bobot di `config.json`.

### 3.8 `io.py`

`hash_df(df, cols=None)`, `md5_file(path)`, `save_json/load_json`, `save_parquet/load_parquet`, `get_env_info()`.

---

## 4. Contoh Pemakaian (Notebook)

```python
from utils.core import BaseConfig, BasePipeline, register_metric
from utils.tabular import TabularConfig, TabularPipeline   # spec #2

cfg = TabularConfig(task="binary", target="y", metric="f1",
                    cv_strategy="stratified", run_name="lgbm_te")
cfg.seed = 7
cfg.folds_path = "runs/20261030_1400_tabular_baseline/folds.csv"  # reuse split

class MyPipe(TabularPipeline):
    def build_features(self, train, test):
        ...                                   # fitur global tanpa target
        return train, test
    def fit_transform_fold(self, tr, va, te, fold):
        ...                                   # target encoding di dalam fold
        return tr, va, te
    def postprocess(self, oof, test):
        return {"threshold": best_threshold(oof)}

res = MyPipe(cfg).run(train_df, test_df)
res.make_submission(sample_sub, "submission.csv")
```

---

## 5. Error Handling

| Kondisi | Perilaku |
|---|---|
| Config invalid | `ValueError` di `cfg.validate()` sebelum folder run dibuat |
| `folds_path` tidak cocok (id/n_folds/hash) | `FoldMismatchError` — **stop**, bukan warning |
| ID duplikat di train/test | `ValueError` |
| Metric tidak dikenal | `KeyError` dengan daftar metric yang tersedia |
| Exception di tengah fold | Traceback ditulis ke `FAILED` di folder run, tidak masuk leaderboard, exception dilempar ulang |
| Ensemble dengan `folds_hash` berbeda | `FoldMismatchError` |
| Library opsional (torch, lgbm, dll.) tidak ada | `env.json` mencatat `null`; core tetap jalan |

---

## 6. Testing

`pytest` di `tests/core/` dengan data sintetis kecil (tanpa GPU, tanpa download):

- `test_cv.py` — setiap strategi menghasilkan fold yang benar; stratified menjaga proporsi; group tidak bocor antar fold; timeseries: train selalu lebih awal dari valid, `ts_gap` dihormati, `fold=-1` untuk blok awal; reuse `folds_path` menghasilkan fold identik; mismatch id / n_folds / hash → `FoldMismatchError`; data dengan urutan baris diacak tetap mendapat fold yang sama (join via id).
- `test_metrics.py` — nilai built-in cocok dengan sklearn; konversi proba→label (threshold & argmax); `register_metric` custom.
- `test_run.py` — semua file artefak terbentuk dengan skema benar; leaderboard bertambah satu baris; run gagal menulis `FAILED` dan tidak masuk leaderboard.
- `test_pipeline.py` — smoke test end-to-end dengan pipeline dummy (sklearn `LogisticRegression`/`Ridge`) untuk binary, multiclass, regression, timeseries; `test_mode` fold_avg vs refit_full; hook `fit_transform_fold` benar-benar hanya melihat data train fold.
- `test_ensemble.py` — blend mean/weights/hill_climb; stacking memakai fold yang sama; `folds_hash` berbeda → error; NaN timeseries ditangani.
- `test_seed.py` — dua run dengan seed sama → OOF identik.

---

## 7. Di Luar Scope Spec Ini (YAGNI)

- UI dashboard, MLflow, W&B (bisa ditambah sebagai hook nanti).
- Locking `leaderboard.csv` untuk penulisan paralel dari banyak mesin.
- Hyperparameter tuning (Optuna) — dibahas di spec modalitas bila perlu.
- Detail feature engineering, model, augmentasi, tokenizer — spec #2–#4.

---

## 8. Roadmap Sub-Project

Masing-masing mengikuti siklus spec → plan → implementasi sendiri.

| # | Sub-project | Bergantung pada |
|---|---|---|
| 1 | **Core** (spec ini) | — |
| 2 | Tabular + Time Series + **Feature Selection** (forward, backward, RFE/RFECV, permutation, SHAP, Boruta, null importance) | Core |
| 3 | Image (timm full: backbone, augmentasi, scheduler, AMP, TTA) | Core |
| 4 | Text (TF-IDF baseline + fine-tune transformer) | Core |
| 5 | Notebook template `00`–`90` + README cara pakai | 1–4 |
