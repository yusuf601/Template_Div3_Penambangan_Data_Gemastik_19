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
| `seed` | `int` | `42` | Seed untuk model/training (`seed_everything`) |
| `fold_seed` | `int` | `42` | Seed khusus `make_folds`; terpisah dari `seed` supaya ganti seed model tidak mengubah split |
| `n_folds` | `int` | `5` | Jumlah fold |
| `cv_strategy` | `str \| None` | `None` | `kfold` / `stratified` / `group` / `stratified_group` / `timeseries`; `None` = di-resolve otomatis (lihat bawah) |
| `group_col` | `str \| None` | `None` | Wajib untuk `group` / `stratified_group` |
| `time_col` | `str \| None` | `None` | Wajib untuk `timeseries` |
| `ts_gap` | `int` | `0` | Jeda antara train dan valid, dalam **jumlah periode unik** `time_col` (rank waktu), bukan satuan waktu mentah |
| `ts_max_train_size` | `int \| None` | `None` | Dalam periode unik. `None` = expanding window; angka = sliding window |
| `folds_path` | `str \| None` | `None` | **Path `folds.csv` dari run sebelumnya. Jika diisi, fold di-load, bukan dibuat ulang** (`fold_seed` diabaikan, hanya di-log) |
| `test_mode` | `str \| None` | `None` | `fold_avg` (rata-rata prediksi model per fold) / `refit_full` (latih ulang di full train); `None` = di-resolve otomatis |
| `refit_iter_mult` | `float` | `1.1` | Pengali rata-rata `best_iteration` antar fold untuk `refit_full` (GBDT) |
| `runs_dir` | `str` | `"runs"` | Root folder run |
| `save_models` | `bool` | `False` | Simpan checkpoint per fold |
| `debug` | `bool` | `False` | Subsample data + 2 fold untuk smoke test cepat; mengabaikan `folds_path`, menulis ke `runs/_debug/`, dan tidak menambah `leaderboard.csv` |
| `notes` | `str` | `""` | Ditulis ke `notes.md` |
| `extra` | `dict` | `{}` | Bebas, untuk parameter custom user |

Method:
- `to_dict()` / `save(path)` / `BaseConfig.load(path)` — serialisasi JSON.
- `resolve() -> list[str]` — mengisi field `None` dan mengembalikan daftar alasan (ditulis ke `log.txt`). Dipanggil di awal `run()`; `config.json` selalu berisi **nilai hasil resolve**, bukan `None`.
  - `cv_strategy is None`:
    1. `group_col` diisi → `stratified_group` (binary/multiclass) / `group` (regression). Log: `"cv_strategy=None + group_col='<col>' → <hasil>"`. Kalau `group_col` dipakai untuk keperluan lain, set `cv_strategy` eksplisit.
    2. selain itu → `stratified` (binary/multiclass) / `kfold` (regression).
    3. `timeseries` **tidak pernah** dipilih otomatis. Jika `time_col` diisi tapi hasil resolve bukan `timeseries` → warning: set `cv_strategy="timeseries"` eksplisit bila memang time series.
  - `test_mode is None` → `refit_full` jika `cv_strategy == "timeseries"`, selain itu `fold_avg`. `fold_avg` tetap bisa dipilih eksplisit untuk timeseries.
  - `debug is True` → paksa `n_folds = 2`; jika `folds_path` diisi, reset `folds_path = None` dan log alasan (karena subsample mengubah set id dan `ids_target_hash`).
- `validate()` — dipanggil setelah `resolve()`; raise `ValueError` jika kombinasi invalid (mis. `cv_strategy="group"` tanpa `group_col`, `task="regression"` dengan `stratified`/`stratified_group`, `timeseries` tanpa `time_col`, `n_folds < 2`, metric tidak dikenal).

**Multi-seed** bukan fitur khusus: jika perlu, jalankan beberapa run dengan `seed` berbeda dan `fold_seed` sama, lalu blend.

### 3.2 `seed.py`

`seed_everything(seed: int)` — set `random`, `numpy`, `PYTHONHASHSEED`, dan (jika terinstal) `torch` + `torch.cuda` + `cudnn.deterministic=True`, `cudnn.benchmark=False`. Tidak gagal jika torch tidak terinstal.

### 3.3 `cv.py` — Fold yang bisa dipakai ulang

**Representasi fold:** DataFrame `[id_col, "fold"]`. `fold ∈ {0..n_folds-1}`; untuk timeseries, baris yang tidak pernah masuk valid diberi `fold = -1`.

**Fungsi:**
- `make_folds(df, cfg) -> pd.DataFrame` — membuat split baru sesuai `cv_strategy` dan **`fold_seed`** (bukan `seed`).
  - `kfold` / `stratified` / `stratified_group`: `shuffle=True, random_state=fold_seed`.
  - `group`: `GroupKFold` (deterministik, tidak memakai seed — didokumentasikan).
  - `timeseries`: dibagi berdasarkan **nilai unik `time_col`**, bukan baris. Waktu unik diurutkan, dipecah menjadi `n_folds + 1` blok berurutan (`np.array_split` atas waktu unik); semua baris dengan timestamp yang sama selalu berada di blok yang sama (aman untuk data panel, mis. provinsi × tanggal). Blok pertama hanya train (`fold=-1`), blok berikutnya menjadi valid fold 0..n-1. Tidak memakai seed.
  - Validasi data (raise `ValueError` dengan pesan jelas): `stratified*` → jumlah sampel kelas terkecil ≥ `n_folds`; `group*` → jumlah grup unik ≥ `n_folds`; `timeseries` → jumlah waktu unik ≥ `n_folds + 1`.
- `load_folds(path, df, cfg) -> pd.DataFrame` — membaca `folds.csv` + `folds_meta.json` di direktori yang sama, lalu **validasi ketat** (lihat bawah).
- `get_folds(df, cfg)` — `load_folds` jika `cfg.folds_path` diisi, selain itu `make_folds`. Selalu dipanggil dengan snapshot `[id, target, group_col?, time_col?]` dari **sebelum** `build_features`.
- `split_fold(df, folds, fold, cfg) -> (train_idx, valid_idx)` — fold diselaraskan ke `df` via id, bukan posisi:
  - non-timeseries: train = semua `fold != k`, valid = `fold == k`.
  - timeseries: setiap baris diberi `t_rank` = rank padat (dense rank) waktu uniknya. valid = `fold == k`; dengan `r0` = `t_rank` minimum di valid, train = baris dengan `t_rank < r0 - ts_gap`, dan jika `ts_max_train_size` diisi, hanya `t_rank ≥ r0 - ts_gap - ts_max_train_size`. (Baris fold > k **tidak** masuk train.) Pada deret tidak teratur (ada tanggal hilang), gap 7 periode ≠ 7 hari kalender — trade-off yang diterima.

**`folds_meta.json`** (ditulis bersama `folds.csv`):
```json
{
  "source_run": "runs/20261030_1400_tabular_baseline",
  "n_folds": 5,
  "cv_strategy": "stratified",
  "fold_seed": 42,
  "group_col": null,
  "time_col": null,
  "ids_target_hash": "sha256...",
  "folds_hash": "sha256..."
}
```

Dua hash dengan peran berbeda:

| Hash | Isi | Dipakai untuk |
|---|---|---|
| `ids_target_hash` | `id`, `target` **mentah**, plus `group_col` jika `cv_strategy ∈ {group, stratified_group}` dan `time_col` jika `timeseries` | Validasi bahwa data sekarang sama dengan data saat fold dibuat |
| `folds_hash` | Pasangan `id → fold` + `n_folds` + `cv_strategy` | Identitas split; dibandingkan oleh `ensemble.py` |

- Sengaja **bukan** hash seluruh DataFrame: yang menentukan sah tidaknya fold adalah baris, label, dan kolom penentu split, bukan kolom fitur. Menambah/mengubah fitur tidak membatalkan reuse fold.
- `ids_target_hash` dihitung **setelah `load_data`, sebelum `build_features`**, dari snapshot yang sama dengan yang dipakai `get_folds`. Transformasi target (mis. `log1p`) tidak boleh dilakukan di DataFrame sebelum `run()`; lakukan di hook `fit_predict_fold` / subclass modalitas, supaya hash selalu dari target mentah.

**Hash kanonik** (`io.canonical_hash(df, cols, sort_by)`), supaya urutan baris dan perbedaan dtype tidak menimbulkan mismatch palsu:
1. Ambil hanya `cols`; setiap nilai dikonversi ke string kanonik:
   - NaN/None → `"<NA>"`;
   - `bool` / `np.bool_` (dicek **sebelum** `int` karena `bool` adalah subclass `int` di Python) → `"1"` jika `True`, `"0"` jika `False` (sehingga target biner `True`/`False`, `1`/`0`, dan `1.0`/`0.0` punya bentuk kanonik yang sama);
   - tipe integer (`int`, `np.integer`, `uint*`) → selalu `str(int(x))` tanpa batas `2**53` (sehingga ID integer 64-bit / 18 digit tidak jatuh ke `.6g` dan tidak pernah bertabrakan);
   - tipe float (`float`, `np.floating`) yang bernilai bulat (`x.is_integer()`) dan `|x| < 2**53` → `str(int(x))` (jadi `7.0` → `"7"`);
   - float lain → `f"{x:.6g}"` (6 digit signifikan; aman untuk perbedaan float32 vs float64 — pembulatan desimal tetap seperti 10 desimal **tidak** cukup karena `float32(0.1)` = `0.10000000149`);
   - datetime → ISO 8601 (`Timestamp.isoformat()`);
   - lainnya → `str(x)`.
2. Sort berdasarkan kolom id kanonik (string).
3. `sha256` dari baris-baris yang digabung (`"\x1f"` antar kolom, `"\n"` antar baris).

Konsekuensi: perubahan target di digit signifikan ke-7 ke atas tidak terdeteksi — diterima sebagai trade-off.

**Validasi saat load:**

| Cek | Jika gagal |
|---|---|
| Set `id` data == set `id` di `folds.csv` (dibandingkan sebagai id kanonik string) | raise `FoldMismatchError` |
| `n_folds`, `cv_strategy`, `group_col`, `time_col` di meta == config sekarang | raise `FoldMismatchError` |
| `ids_target_hash` di meta == hash data sekarang | raise `FoldMismatchError` |
| Hash data mentah di `data_version.json` run asal == hash data mentah sekarang | **warning** (`FoldDataChangedWarning`), menyebut `source_run`; fold tetap dipakai |

- Join selalu via id kanonik, **tidak pernah** via index/urutan baris.
- Strictness **tidak** bisa dimatikan di v1. Membuang baris (mis. outlier) sebelum training mengubah set id → wajib membuat fold baru. (Opsi `allow_subset` ada di §7.)

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

Membuat folder `runs/{YYYYMMDD_HHMM}_{modality}_{run_name}/` (atau `runs/_debug/{YYYYMMDD_HHMM}_{modality}_{run_name}/` jika `cfg.debug=True`; tambah suffix `_2`, `_3` jika bentrok) dan menulis:

| File | Isi | Kapan ditulis |
|---|---|---|
| `config.json` | Seluruh config setelah override | Awal run |
| `env.json` | Versi python, numpy, pandas, sklearn, torch, timm, transformers, lightgbm, xgboost, catboost (yang terinstal), GPU name, git commit (jika ada) | Awal run |
| `data_version.json` | `raw_train_hash`, `raw_test_hash` (hash kanonik semua kolom, setelah `load_data`, sebelum `build_features`), `ids_target_hash`, `test_ids_hash`, shape, daftar kolom, dan md5 file sumber jika path diberikan | Awal run |
| `folds.csv` + `folds_meta.json` | Fold yang dipakai (selalu ditulis, juga saat di-load dari run lain) | Setelah fold dibuat/di-load |
| `oof.parquet` | `id`, `fold`, `target`, kolom pred | Akhir run |
| `test_pred.parquet` | `id`, kolom pred | Akhir run |
| `scores.json` | Struktur skor raw / postprocessed / honest (lihat §3.6 "Skema skor"), metric, direction, `folds_hash`, `test_ids_hash`, `classes` (multiclass) | Akhir run |
| `postprocess.json` | Parameter dari hook `postprocess` (mis. threshold) | Akhir run |
| `models/fold{k}.*` | Checkpoint (opsional, `save_models=True`) | Per fold |
| `notes.md` | `cfg.notes` | Awal run |
| `log.txt` | Log run | Sepanjang run |

**Format kolom pred (seragam untuk semua modalitas):**
- binary / regression: satu kolom `pred`.
- multiclass: `pred_0 … pred_{K-1}` (probabilitas per kelas, urutan kelas disimpan di `scores.json` → `classes`).
- Baris timeseries dengan `fold=-1`: pred = NaN.

**Leaderboard:** setiap run yang selesai menambah satu baris ke `runs/leaderboard.csv`:
`run_dir, timestamp, modality, run_name, model, metric, cv_raw, cv_postprocessed, cv_honest, n_rows_eval, cv_raw_all, lb_score, folds_hash, data_hash, seed, fold_seed`.
- Tiga skor `cv_raw`/`cv_postprocessed`/`cv_honest` dihitung pada **subset fold yang sama** (`eval_folds`, lihat §3.6) supaya sebanding; `cv_raw_all` = skor raw di semua fold aktif. **Ranking memakai `cv_honest`.**
- `lb_score` awalnya `NaN`/kosong saat run selesai, dan diisi setelah submit ke leaderboard lomba via helper `update_lb_score(run_dir_or_name, lb_score, runs_dir="runs")` untuk mengecek korelasi CV vs Public LB.
- `data_hash` = `data_version["raw_train_hash"]` untuk run model tunggal; untuk run ensemble diisi `folds_hash` jika `raw_train_hash` antar run sumber berbeda, atau `raw_train_hash` yang sama jika semua run sumber berasal dari data mentah yang identik.
- **Pemetaan tiga skor untuk run ensemble:**
  - `blend`: `cv_raw` = rata-rata seragam (`mean`) tanpa optimasi bobot dan tanpa `postprocess`; `cv_postprocessed` = skor `in_sample` (bobot + `postprocess` dioptimasi dan dinilai di seluruh `eval_folds`); `cv_honest` = skor `honest` (untuk tiap fold $k \in \text{eval\_folds}$, bobot + `postprocess` dioptimasi di `honest_source(k)` lalu dinilai di fold $k$).
  - `stack`: karena prediksi OOF level-2 dihasilkan oleh `meta_model` yang dilatih di `honest_source(k)`, prediksi OOF level-2 sudah out-of-fold. Maka `cv_raw` = OOF level-2 tanpa `postprocess`, `cv_postprocessed` = OOF level-2 + `postprocess` in-sample, `cv_honest` = OOF level-2 + `postprocess` honest per fold.
- Run yang gagal tidak masuk leaderboard; foldernya diberi file `FAILED` berisi traceback. Run dengan `cfg.debug=True` disimpan di `runs/_debug/` dan **tidak** ditambahkan ke `leaderboard.csv`.

### 3.6 `pipeline.py` — `BasePipeline` (hook)

```python
class BasePipeline:
    model_name: str = "base"                       # dicatat di leaderboard
    def __init__(self, cfg: BaseConfig): ...
    fold_info: dict[int, dict]                     # diisi fit_predict_fold, mis. {"best_iteration": 812}

    # ---- Hooks (override sesuai kebutuhan) ----
    def load_data(self, train, test):              # default: return apa adanya
    def build_features(self, train, test):         # GLOBAL, tidak boleh pakai target,
                                                   # TIDAK boleh menambah/menghapus/mengubah id. default: identity
    def fit_transform_fold(self, tr, va, te, fold):# DI DALAM FOLD: target encoding, scaler,
                                                   # imputer, feature selection. default: identity
    def build_model(self, fold):                   # WAJIB di-override (subclass modalitas menyediakan default)
    def fit_predict_fold(self, model, tr, va, te, fold) -> (model, va_pred, te_pred)  # WAJIB
    def fit_full_predict(self, train, test) -> te_pred  # untuk test_mode="refit_full"
    def postprocess(self, oof_df, test_df=None) -> dict  # PURE; default: {}; mis. optimasi threshold
    def apply_params(self, y_raw, params) -> np.ndarray  # menerapkan params ke pred; default: threshold (binary)
    def get_feature_cols(self, df) -> list[str]    # default: semua kolom kecuali id, target, fold, group, time
    def refit_iterations(self) -> int | None       # helper: round(mean(best_iteration) * cfg.refit_iter_mult)

    # ---- Orkestrasi (biasanya tidak di-override) ----
    def run(self, train, test) -> RunResult
```

**Alur `run()`:**
```
self.fold_info = {}; self.classes_ = None # reset state dari run sebelumnya pada instance yang sama
reasons = cfg.resolve(); cfg.validate()
seed_everything(cfg.seed)
rm = RunManager(cfg)                     # config.json (hasil resolve), env.json, notes.md; log reasons
train, test = load_data(train, test)
cek: id unik di train & test, target tidak NaN      → ValueError
rm.save_data_version(train, test)        # hash SEBELUM feature engineering
snap = train[[id, target, group_col?, time_col?]]
folds, meta = get_folds(snap, cfg); rm.save_folds(folds, meta)   # fold dibuat SEBELUM build_features
te_check_cols = [id] + ([time_col] if cv_strategy == "timeseries" else [])
te_snap_hash = canonical_hash(test, te_check_cols, sort_by=id)
train, test = build_features(train, test)
assert compute_ids_target_hash(train, cfg) == meta["ids_target_hash"] \
   dan canonical_hash(test, te_check_cols, sort_by=id) == te_snap_hash   → ValueError (sebut hook build_features)
for k in folds aktif:
    tr_idx, va_idx = split_fold(train, folds, k, cfg)            # selaras via id
    tr, va, te = fit_transform_fold(train[tr_idx], train[va_idx], test, k)
    model = build_model(k)
    model, va_pred, te_pred = fit_predict_fold(model, tr, va, te, k)
    simpan va_pred ke oof (dipetakan via id dari va yang dikembalikan); akumulasi te_pred; log
test_pred = mean(te_pred per fold)  atau  fit_full_predict(train, test) jika refit_full
params = postprocess(oof, test_pred);  rm.save_postprocess(params)
scores = hitung skema skor (raw / postprocessed / honest, lihat bawah)
rm.save_oof / save_test_pred / save_scores / append_leaderboard
return RunResult(run_dir, oof, test_pred, scores, params)
```

**Aturan baris & anti-leakage (didokumentasikan di docstring hook):**
- `build_features` hanya boleh berisi transformasi yang tidak melihat target dan tidak "fit" statistik lintas fold (lag, rolling, date parts, text cleaning, resize).
- `build_features` **tidak boleh** menambah, menghapus, atau mengubah `id`, `target`, `group_col` (jika group CV), atau `time_col` (jika timeseries) di train, maupun `id` (dan `time_col` jika timeseries) di test. Dicek dengan membandingkan ulang `compute_ids_target_hash(train, cfg)` terhadap `meta["ids_target_hash"]` dan `canonical_hash(test, [id, time_col?])` sebelum vs sesudah `build_features`. Baris yang tidak punya nilai fitur (mis. lag di awal deret) dibiarkan **NaN**, tidak di-drop. **Urutan** baris boleh berubah (mis. sort per grup × waktu untuk lag), karena semua penyelarasan via id.
- Semua yang di-*fit* (target encoding, scaler, imputer statistik, feature selection, PCA) **wajib** di `fit_transform_fold`, di-fit hanya pada `tr`.
- `fit_transform_fold` **boleh** membuang baris dari `tr` (mis. baris warm-up dengan lag NaN), tetapi **tidak boleh** membuang baris dari `va` atau `te` (set id harus tetap; dicek, `ValueError` jika dilanggar).

**Hook `postprocess` (kontrak):**
- Signature `postprocess(oof_df, test_df=None) -> dict`. Harus **pure**: hanya bergantung pada argumen, tanpa side effect, dan hanya **mengembalikan params** (mis. `{"threshold": 0.37}`).
- Pada pass honest, hook dipanggil dengan subset OOF dan `test_df=None` — hook yang diam-diam memakai data test akan gagal/ketahuan.
- Penerapan params ke prediksi **bukan** tugas hook; dilakukan oleh `apply_params` yang dipakai `score()` dan `make_submission`.

**Skema skor:**
- `eval_folds` — fold yang dipakai untuk membandingkan ketiga skor:
  - non-timeseries: semua fold aktif.
  - timeseries: fold ≥ 1 (fold 0 tidak punya skor honest).
- Untuk setiap `k ∈ eval_folds`, `honest_source(k)`:
  - non-timeseries: fold aktif `≠ k`.
  - timeseries: fold `< k` (tidak pernah memakai fold masa depan).
- Skor:
  - `raw` — prediksi mentah, tanpa params (default `apply_params({})`, mis. threshold 0.5).
  - `postprocessed` — params dari `postprocess(oof penuh)`. **Optimistis**: params dioptimasi dan dinilai di OOF yang sama.
  - `honest` — untuk tiap fold k: `params_k = postprocess(oof[honest_source(k)], None)`, lalu nilai fold k dengan `params_k`. *(Catatan: pada non-timeseries, model fold $j \neq k$ dilatih menggunakan baris fold $k$, sehingga OOF fold $j$ sudah melihat fold $k$ secara tidak langsung; efeknya kecil dan standar dalam praktik lomba tanpa biaya nested CV, namun `cv_honest` tidak sepenuhnya bebas bias seperti data test baru).*
  - Jika `postprocess` mengembalikan `{}`, `postprocessed = honest = raw` (pass honest dilewati).
- `scores.json`:
```json
{
  "metric": "f1", "direction": "maximize",
  "per_fold": {"1": {"raw": 0.71, "postprocessed": 0.74, "honest": 0.73, "n_rows": 1200}},
  "all_folds": {"raw": {"mean": 0.70, "std": 0.02, "overall": 0.70, "n_rows": 6000}},
  "eval": {"folds": [1, 2, 3, 4], "n_rows": 4800,
           "raw": {"mean": 0.0, "std": 0.0, "overall": 0.0},
           "postprocessed": {"mean": 0.0, "std": 0.0, "overall": 0.0},
           "honest": {"mean": 0.0, "std": 0.0, "overall": 0.0}},
  "folds_hash": "...", "test_ids_hash": "...", "classes": null
}
```
  `overall` = metric atas gabungan OOF fold terkait (untuk honest, tiap fold memakai `params_k`-nya). Leaderboard memakai `eval.*.overall` dan `all_folds.raw.overall`.

**`refit_full` (aturan hook & jumlah iterasi, tertulis di docstring `fit_transform_fold`, `fit_predict_fold`, dan `fit_full_predict`):**
- Default: `tr, va_empty, te = fit_transform_fold(train, train.iloc[:0], test, fold=-1)`, `model = build_model(-1)`, lalu `fit_predict_fold(model, tr, va_empty, te, -1)`. `fold=-1` dan `len(va) == 0` menandakan refit; tidak ada early stopping.
- **Kontrak wajib hook saat `fold == -1` / `len(va) == 0`:**
  - `fit_transform_fold(tr, va, te, fold)` wajib menangani `len(va) == 0` tanpa memanggil `.transform(va)` pada transformer sklearn/encoder yang menolak 0 baris (kembalikan `tr.iloc[:0].copy()` supaya skema kolom `va` tetap identik dengan `tr`).
  - `fit_predict_fold(model, tr, va, te, fold)` wajib menangani `fold == -1` / `len(va) == 0` dengan melewati `eval_set`/early stopping dan mengembalikan `va_pred` kosong (`np.empty(0)` atau `np.empty((0, K))` untuk multiclass).
- **Reset `fold_info`:** `self.fold_info` selalu di-reset ke `{}` di awal `run()` supaya `refit_iterations()` tidak memakai sisa `best_iteration` dari run sebelumnya pada instance pipeline yang sama.
- **GBDT:** `fit_predict_fold` mencatat `self.fold_info[k]["best_iteration"]` saat CV (`fold >= 0`); saat `fold == -1`, jumlah iterasi = `refit_iterations()` = `round(mean(best_iteration antar fold) × cfg.refit_iter_mult)` (default `1.1`).
- **Deep learning:** jumlah epoch tetap dari config modalitas (mis. `cfg.epochs`), bukan early stopping.
- Jika tidak ada `best_iteration` yang tercatat, `refit_iterations()` mengembalikan `None` dan model dilatih dengan setting di config apa adanya.

**`RunResult`:** dataclass `run_dir, oof, test_pred, scores, params`, plus helper `make_submission(sample_sub, path, fn=None)` yang menerapkan `params` lewat `apply_params` (mis. threshold/argmax) ke `test_pred` dan menulis CSV submission.

### 3.7 `ensemble.py`

- `load_runs(run_dirs) -> EnsembleData` — load OOF & test pred semua run, join via id kanonik.
  - **Raise** `FoldMismatchError` jika `folds_hash` antar run berbeda.
  - **Raise** `FoldMismatchError` jika `test_ids_hash` antar run berbeda (set id test harus sama supaya `test_pred` sejajar saat blending).
  - Raise `ValueError` jika `task`/`metric`/`classes` berbeda.
  - Baris dengan pred NaN (timeseries `fold=-1`) dikecualikan dari optimasi & scoring.
- `blend(data, method, postprocess=None)`:
  - `mean`, `rank_mean`, `geo_mean` — tanpa optimasi.
  - `weights` — optimasi bobot non-negatif, jumlah 1, memaksimalkan/meminimalkan metric di OOF (`scipy.optimize.minimize`).
  - `hill_climb` — greedy forward selection dengan replacement.
  - `postprocess` opsional (fungsi pure `oof_df -> params`, mis. threshold untuk blend binary F1).
  - Skor memakai **skema yang sama dengan §3.6** sehingga terpetakan langsung ke kolom leaderboard (`eval.raw`, `eval.postprocessed`, `eval.honest`):
    - `raw` = rata-rata seragam (`mean`) tanpa optimasi bobot dan tanpa `postprocess`.
    - `postprocessed` (`in_sample`) = bobot + `postprocess` dioptimasi dan dinilai di `eval_folds` yang sama (optimistis).
    - `honest` = untuk tiap `k ∈ eval_folds`, bobot + `postprocess` dioptimasi di `honest_source(k)`, dinilai di fold k. `eval_folds` / `honest_source` mengikuti `cv_strategy` run sumber (timeseries → hanya fold < k, fold ≥ 1). Untuk metode tanpa optimasi (`mean`, `rank_mean`, `geo_mean`) dan tanpa `postprocess`, `honest = postprocessed`. Bobot final (untuk test) dioptimasi di seluruh OOF.
- `stack(data, meta_model, cfg, postprocess=None)` — meta-model dilatih dengan **`folds.csv` yang sama** di `honest_source(k)`, input = OOF level-1; menghasilkan OOF level-2 yang valid (`eval.raw` = OOF level-2 mentah, `eval.postprocessed` = OOF level-2 + `postprocess` in-sample, `eval.honest` = OOF level-2 + `postprocess` honest per fold).
- Hasil blend/stack disimpan sebagai run baru (`modality="ensemble"`) lewat `RunManager`, termasuk daftar run sumber + bobot di `config.json`.

### 3.8 `io.py`

`canonical_hash(df, cols, sort_by)` (aturan di §3.3), `md5_file(path)`, `save_json/load_json`, `save_parquet/load_parquet`, `get_env_info()`.

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
    def postprocess(self, oof, test=None):
        return {"threshold": best_threshold(oof)}

res = MyPipe(cfg).run(train_df, test_df)
res.make_submission(sample_sub, "submission.csv")
```

---

## 5. Error Handling

| Kondisi | Perilaku |
|---|---|
| Config invalid / kelas terkecil < `n_folds` / grup unik < `n_folds` / waktu unik < `n_folds + 1` | `ValueError` dengan pesan jelas |
| `folds_path` tidak cocok (id / n_folds / cv_strategy / group_col / time_col / `ids_target_hash`) | `FoldMismatchError` — **stop**, bukan warning |
| `folds_path` cocok, tapi hash data mentah berbeda dari run asal | `FoldDataChangedWarning` yang menyebut `source_run`; run lanjut |
| ID duplikat di train/test, atau target NaN | `ValueError` |
| `build_features` mengubah `id`, `target`, `group_col`, atau `time_col` di train/test | `ValueError` yang menyebut hook `build_features` |
| `fit_transform_fold` mengubah set id di `va` atau `te` | `ValueError` yang menyebut hook `fit_transform_fold` |
| Metric tidak dikenal | `KeyError` dengan daftar metric yang tersedia |
| Exception di tengah fold | Traceback ditulis ke `FAILED` di folder run, tidak masuk leaderboard, exception dilempar ulang |
| Ensemble dengan `folds_hash` atau `test_ids_hash` berbeda | `FoldMismatchError` |
| Library opsional (torch, lgbm, dll.) tidak ada | `env.json` mencatat `null`; core tetap jalan |

---

## 6. Testing

`pytest` di `tests/core/` dengan data sintetis kecil (tanpa GPU, tanpa download):

- `test_config.py` — `resolve()` otomatis memilih `stratified` / `kfold` / `stratified_group` / `group` dan `test_mode` (`refit_full` untuk timeseries, `fold_avg` untuk lainnya); warning saat `time_col` diisi tapi `cv_strategy` bukan `timeseries`; `validate()` menolak kombinasi invalid.
- `test_io.py` — `canonical_hash` sama untuk: urutan baris diacak, id `int` vs `str` vs `float` bulat (`7`/`"7"`/`7.0`), target biner `bool` (`True`/`False`) vs `int` (`1`/`0`) vs `float` (`1.0`/`0.0`), target `float32` vs `float64`, datetime vs ISO string yang sama; ID integer 64-bit ≥ `2**53` (18 digit) yang berbeda 1 tetap menghasilkan string/hash berbeda; berbeda jika satu label berubah atau satu id hilang.
- `test_cv.py` — setiap strategi menghasilkan fold yang benar; stratified menjaga proporsi; group tidak bocor antar fold; timeseries: bagi per waktu unik (data panel tidak terbelah antar blok), train selalu lebih awal dari valid, `ts_gap` dan `ts_max_train_size` dalam rank periode unik dihormati, `fold=-1` untuk blok awal; reuse `folds_path` menghasilkan fold identik; `fold_seed` terpisah dari `seed`; mismatch id / n_folds / cv_strategy / hash → `FoldMismatchError`; isi `group_col` (group CV) atau `time_col` (timeseries) diubah → `FoldMismatchError`; hanya kolom fitur berubah → fold lolos + `FoldDataChangedWarning`; data dengan urutan baris diacak tetap mendapat fold yang sama (join via id).
- `test_metrics.py` — nilai built-in cocok dengan sklearn; konversi proba→label (threshold & argmax); `register_metric` custom.
- `test_run.py` — semua file artefak terbentuk dengan skema benar (`config.json` menyimpan nilai hasil resolve); leaderboard bertambah satu baris dengan kolom `cv_raw`, `cv_postprocessed`, `cv_honest`, `n_rows_eval`, `cv_raw_all`, `lb_score`, `data_hash` (= `raw_train_hash`), `seed`, `fold_seed`; `update_lb_score` memperbarui `lb_score` untuk run terkait; run gagal menulis `FAILED` dan tidak masuk leaderboard; `debug=True` menulis ke `runs/_debug/` dan tidak menambah leaderboard.
- `test_pipeline.py` — smoke test end-to-end dengan pipeline dummy (sklearn `LogisticRegression`/`Ridge`) untuk binary, multiclass, regression, timeseries; `test_mode` fold_avg vs refit_full (`refit_iterations`, reset `fold_info` saat instance dipanggil ulang, dan `fit_transform_fold` dengan `StandardScaler`/target encoding saat `va` kosong & `fold=-1`); `build_features` mengubah `id`, `target`, atau `time_col` → `ValueError`; `build_features` mengacak urutan baris → tetap selaras via id; `fit_transform_fold` boleh buang baris `tr` tapi tidak `va`/`te`; `debug=True` mengabaikan `folds_path`, memakai 2 fold di `runs/_debug/`, dan tidak menambah leaderboard; skema skor raw vs postprocessed vs honest (fold ≠ k untuk non-TS, fold < k untuk TS, `test_df=None` pada pass honest).
- `test_ensemble.py` — blend mean/rank_mean/geo_mean/weights/hill_climb + pemetaan `eval.raw` (mean tanpa optimasi), `eval.postprocessed` (`in_sample`), dan `eval.honest` (`honest`) di `scores.json` & `leaderboard.csv`; stacking memakai fold yang sama; `folds_hash` berbeda → error; set id test berbeda → error; NaN timeseries ditangani.
- `test_seed.py` — dua run dengan `seed` sama → OOF identik; dua run dengan `seed` berbeda tapi `fold_seed` sama → `folds_hash` identik dan bisa di-blend.

---

## 7. Di Luar Scope Spec Ini (YAGNI)

- UI dashboard, MLflow, W&B (bisa ditambah sebagai hook nanti).
- Locking `leaderboard.csv` untuk penulisan paralel dari banyak mesin.
- `allow_subset` (reuse fold pada sebagian baris, OOF NaN untuk baris yang dibuang) — v1 tetap strict; buang baris = buat fold baru.
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
