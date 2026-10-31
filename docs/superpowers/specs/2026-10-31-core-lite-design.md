# Template Final Gemastik Data — Spec #1 (revisi): Core-lite

- **Tanggal:** 2026-10-31
- **Status:** Approved (2026-10-31) — plan: `docs/superpowers/plans/2026-10-31-core-lite.md`
- **Menggantikan:** `2026-10-30-core-design.md` (terlalu rumit untuk lomba 5 jam; disimpan sebagai arsip keputusan)
- **Scope:** `utils/core/`. Modul tabular, image, text, dan kit offline punya spec terpisah (§9).

---

## 1. Konteks Lomba (mengikat semua keputusan)

| Kondisi | Dampak desain |
|---|---|
| Durasi **5 jam** | Prioritas: baseline valid dalam menit, sedikit konsep, error yang langsung memberi solusi |
| Data **baru dibuka saat lomba**, modalitas belum diketahui | Core generik; tiga modalitas harus siap; gladi resik dengan dataset tiruan |
| GPU **hanya 1 di 1 PC** (minimal RTX 3050, VRAM 4 GB); **laptop lain CPU-only (support)** | `max_minutes` + resume per fold di PC GPU; laptop support menjalankan EDA, pembuatan `folds.csv`, tabular GBDT CPU, TF-IDF baseline, blending, dan pembuatan slide; folder run harus **portabel lintas mesin** (path relatif + `sync_leaderboard`) |
| **Tanpa internet** (local LLM boleh) | Core tidak pernah mengakses jaringan; dependensi hanya numpy, pandas, scikit-learn, scipy, pyarrow, matplotlib |
| **Ada presentasi** | Tiap run otomatis menulis `summary.md` + plot dasar |

**Prinsip Core-lite:**
1. Biaya terbesar adalah bug dan kebingungan di bawah tekanan, bukan waktu komputasi.
2. Cek yang gagalnya **pasti merusak hasil** → raise. Cek yang sifatnya **menjaga kebersihan** → warning keras.
3. **Setiap error berisi langkah perbaikan** (§7).
4. Pemakai baru cukup paham **3 hook** (§5.2).

---

## 2. Yang Dipertahankan vs Dibuang (dibanding spec 2026-10-30)

| Dipertahankan | Disederhanakan | Dibuang dari v1 |
|---|---|---|
| Fold tetap + `folds_path` | Canonical id/hash (aturan pendek, §4.1) | `stack`, `hill_climb`, `rank_mean`, `geo_mean` |
| OOF + test_pred mentah per run | Skor honest hanya dihitung jika `postprocess` di-override | `FoldDataChangedWarning`, md5 file sumber, `env.json` |
| Leaderboard lokal + `lb_score` | Hook: 3 utama + 4 lanjutan (dari 11) | `load_data`, `apply_params` |
| `make_submission` dengan cek id & urutan | Cek target/group/time setelah `build_features` → warning | `ts_max_train_size` |
| Anti-leakage di hook | `resolve()` hanya 2 aturan | Validasi `ids_target_hash` di ensemble |
| `fold_seed` terpisah dari `seed` | | |
| Split timeseries per waktu unik + `ts_gap` | | |
| `refit_full` + `refit_iterations` | | |
| `debug` terisolasi | | |
| **Baru:** `max_minutes`, resume per fold, `summary.md` + plot | | |

---

## 3. Struktur Project

```
template_gemastik/
├── utils/core/
│   ├── __init__.py     # ekspor publik
│   ├── errors.py       # TemplateError + subclass (pesan masalah + perbaikan)
│   ├── io.py           # canon_id, canon_hash, json/parquet helper
│   ├── config.py       # BaseConfig
│   ├── metrics.py      # registry metric + score
│   ├── cv.py           # make_folds, load_folds, get_folds, split_fold
│   ├── run.py          # RunManager, leaderboard, update_lb_score
│   ├── report.py       # summary.md + plot
│   ├── pipeline.py     # BasePipeline, RunResult
│   └── ensemble.py     # load_runs, blend
├── tests/core/
└── runs/  (leaderboard.csv, _debug/)
```

---

## 4. Komponen

### 4.1 `io.py` — id dan hash kanonik

- `canon_id(series) -> pd.Series[str]` — dipakai untuk **semua** join; **divektorisasi per dtype** agar cepat pada jutaan baris:
  - dtype `bool` → vektorisasi map `{True: "1", False: "0"}` (dicek sebelum integer)
  - dtype integer (`int*`, `uint*`) → `series.astype(str)`, tanpa batas `2**53`
  - dtype float (`float*`) → vektorisasi: mask nilai bulat dengan `|x| < 2**53` dikonversi ke `int64` lalu `astype(str)`, float lain diformat `f"{x:.6g}"`, NaN → `"<NA>"`
  - dtype `datetime64*` → ISO 8601 (`Timestamp.isoformat()`); NaN/NaT → `"<NA>"`
  - hanya kolom `object` / bertipe campuran yang memakai konversi per elemen.
- `canon_hash(df, cols, sort_by) -> str` — terapkan `canon_id` hanya pada `cols`, sort berdasarkan `sort_by`, `sha256` atas baris yang digabung (`"\x1f"` antar kolom, `"\n"` antar baris). Dipakai **hanya untuk kolom sedikit** (`ids_target_hash`, `folds_hash`, `test_ids_hash`).
- `fast_df_hash(df) -> str` — `sha256` atas `pd.util.hash_pandas_object(df, index=False).values.tobytes()`, dibungkus `try / except Exception: return "n/a"` agar kolom berisi objek tak-hashable (`list`/`dict`) tidak menggagalkan run. Sangat cepat untuk tabel lebar (mis. 1 juta baris × 50 kolom); dipakai untuk `raw_train_hash` yang sifatnya informatif.
- `save_json`, `load_json` (mendukung tipe numpy), `save_parquet`, `load_parquet`.

### 4.2 `config.py` — `BaseConfig`

Dataclass `kw_only=True`. Modalitas men-subclass.

| Field | Default | Keterangan |
|---|---|---|
| `task` | wajib | `binary` / `multiclass` / `regression` |
| `target` | wajib | Kolom target |
| `metric` | wajib | Nama di registry |
| `id_col` | `"id"` | |
| `run_name` | `"baseline"` | |
| `owner` | `""` | Nama anggota/mesin; jika kosong otomatis diisi dari `socket.gethostname()` saat `resolve()` agar nama folder unik antar-laptop tanpa perlu diingat |
| `modality` | `"generic"` | |
| `seed` | `42` | Seed model |
| `fold_seed` | `42` | Seed `make_folds` |
| `n_folds` | `5` | |
| `cv_strategy` | `None` | `kfold` / `stratified` / `group` / `stratified_group` / `timeseries`; `None` → auto |
| `group_col` | `None` | |
| `time_col` | `None` | |
| `ts_gap` | `0` | Jeda train→valid dalam jumlah **periode unik** |
| `folds_path` | `None` | Reuse `folds.csv` dari run lain |
| `test_mode` | `None` | `fold_avg` / `refit_full`; `None` → auto |
| `refit_iter_mult` | `1.1` | |
| `max_minutes` | `None` | Batas waktu run (§5.4) |
| `resume_from` | `None` | Folder run yang dilanjutkan (§5.4) |
| `runs_dir` | `"runs"` | |
| `debug` | `False` | §5.5 |
| `notes` | `""` | Masuk `summary.md` |
| `extra` | `{}` | Bebas |

**Auto-resolve** (di awal `run()`; hasil ditulis ke `config.json` dan dicatat di log):
- `owner==""`: diisi otomatis dari `socket.gethostname()` (dibersihkan ke alfanumerik lowercase).
- `cv_strategy=None`: `group_col` diisi → `stratified_group` (klasifikasi) / `group` (regresi); selain itu → `stratified` / `kfold`. `timeseries` tidak pernah dipilih otomatis; jika `time_col` diisi tanpa `timeseries` → warning.
- `test_mode=None`: `refit_full` untuk `timeseries`, selain itu `fold_avg`.

**`validate()`** → `ConfigError` (dengan perbaikan) untuk: task/metric/strategi tidak dikenal, `group*` tanpa `group_col`, `timeseries` tanpa `time_col`, `stratified*` pada regresi, `n_folds < 2`.

### 4.3 `metrics.py`

- `Metric(name, fn, direction, input_type)` dengan `input_type ∈ {proba, label, value}`.
- `register_metric(name, direction, input_type)` (decorator), `get_metric(name)`, `list_metrics()`.
- `score(y_true, y_raw, metric, task, threshold=0.5, classes=None, clip_range=None) -> float`: buang baris pred NaN; `label` + binary → `y_raw >= threshold`; `label` + multiclass → argmax dipetakan ke `classes`.
  - Khusus `metric == "qwk"`:
    - Untuk `task == "regression"`: bulatkan dan clip prediksi kontinu `np.clip(np.rint(y_raw), y_min, y_max).astype(int)` memakai `clip_range=(y_min, y_max)` yang dihitung dari **seluruh target train** (atau fallback `(int(np.min(y_true)), int(np.max(y_true)))` jika `clip_range=None`), agar pass honest per fold dan pass penuh memakai batas clip yang sama.
    - Untuk klasifikasi (`binary`/`multiclass`): jika `classes` berisi string, keluarkan `UserWarning` bahwa `qwk` mengasumsikan kelas ordinal terurut sedangkan urutan alfabet string belum tentu bermakna ordinal.
- Built-in: `auc`, `logloss`, `accuracy`, `f1`, `f1_macro`, `f1_weighted`, `qwk`, `rmse`, `mae`, `mape`, `r2`.
- OOF dan test_pred **selalu** disimpan mentah; `input_type` hanya memengaruhi scoring dan submission.

### 4.4 `cv.py`

- `make_folds(snap, cfg) -> (folds_df, meta)` — `folds_df = [id_col, fold]`.
  - Cek `snap[cfg.id_col].isna().any()` terlebih dahulu (sebelum cek duplikat) → `DataError`.
  - `kfold` / `stratified` / `stratified_group`: shuffle dengan `fold_seed`; `group`: `GroupKFold`.
  - `timeseries`: waktu **unik** diurutkan, `np.array_split` menjadi `n_folds + 1` blok; blok 0 → `fold=-1` (hanya train), blok k+1 → fold k.
  - Hasil tidak bergantung pada urutan baris input (sort berdasarkan `canon_id` dulu).
  - `ValueError` dengan perbaikan jika kelas terkecil / grup unik < `n_folds`, atau waktu unik < `n_folds + 1`.
- `meta` (`folds_meta.json`): `source_run`, `n_folds`, `cv_strategy`, `fold_seed`, `group_col`, `time_col`, `ids_target_hash` (`canon_hash` atas id, target, + group/time bila dipakai split), `folds_hash` (`canon_hash` atas id→fold + `n_folds` + `cv_strategy`).
- `make_folds`, `load_folds`, `get_folds`, dan `make_folds_file` selalu menjalankan `resolve()` + `validate()` pada salinan `cfg` (`copy.deepcopy(cfg)`) jika belum di-resolve, sehingga aman dipanggil langsung dengan `BaseConfig(...)` baru.
- `load_folds(path, snap, cfg)` → `FoldMismatchError` (dengan perbaikan "set `folds_path=None`") jika set id, `n_folds`, `cv_strategy`, atau `ids_target_hash` berbeda.
- `get_folds(snap, cfg)` — load jika `folds_path` diisi, selain itu make.
- `make_folds_file(train_df, cfg, out_dir="folds") -> Path` — API mandiri untuk membuat `folds/folds.csv` + `folds/folds_meta.json` (`source_run="standalone"`) di awal lomba sebelum menjalankan pipeline apa pun. Seluruh mesin kemudian cukup mengarahkan `cfg.folds_path = "folds/folds.csv"` (path relatif terhadap root project).
- `split_fold(snap, folds_df, k, cfg) -> (train_ids, valid_ids)` — mengembalikan **set id kanonik**, dihitung dari **snapshot** sebelum `build_features` (sehingga `time_col` yang diubah hook tidak memengaruhi split).
  - non-timeseries: train = fold ≠ k (dan ≠ -1), valid = fold == k.
  - timeseries: `t_rank` = dense rank waktu; `r0` = rank minimum valid; train = `t_rank < r0 - ts_gap`. Train kosong → `ValueError` dengan perbaikan "kurangi `ts_gap` atau `n_folds`".

### 4.5 `run.py` — `RunManager`

Folder: `runs/{YYYYMMDD_HHMM}_{safe_owner_}{safe_modality}_{safe_run_name}/` (satu fungsi pembersih `safe_name(s, default)` = `re.sub(r"[^A-Za-z0-9_-]+", "-", str(s)).strip("-") or default` dipakai untuk `owner`, `modality`, dan `run_name` dengan `{safe_owner_}` = `f"{safe_name(cfg.owner, '')}_"` jika tidak kosong, sehingga nama folder selalu valid di Windows maupun Linux tanpa spasi/titik di ujung, sedangkan nilai asli `cfg` tetap disimpan di `config.json`; timestamp selalu menggunakan `zoneinfo.ZoneInfo("Asia/Jakarta")` agar urutan waktu konsisten lintas mesin; suffix `_2`, `_3` jika bentrok; `runs/_debug/...` jika debug; folder yang sama jika `resume_from`).
Atribut publik `RunManager`: `self.cfg: BaseConfig`, `self.model_name: str`, `self.run_dir: Path`, `self.timestamp: str`, `self.is_resumed: bool` (saat `resume_from`, `self.cfg.owner` dan `self.timestamp` mempertahankan nilai asli dari folder run yang di-resume).

| File | Isi |
|---|---|
| `config.json` | Config hasil resolve |
| `folds.csv`, `folds_meta.json` | Fold yang dipakai |
| `data_version.json` | `raw_train_hash` (`fast_df_hash` semua kolom train, sebelum `build_features`), `test_ids_hash` (`canon_hash` kolom id test), shape, kolom |
| `fold_preds/fold{k}_oof.parquet`, `fold{k}_test.parquet`, `fold{k}_info.json` | Ditulis **segera setelah fold k selesai** (dasar resume). `fold{k}_info.json` menyimpan `best_iteration`, `duration_sec`, dan `importances` (`dict[str, float]`) fold k |
| `oof.parquet` | `id`, `fold`, target, `pred` / `pred_0..pred_{K-1}` |
| `test_pred.parquet` | `id`, kolom pred |
| `scores.json` | §5.3 (mencakup seluruh metadata yang dibutuhkan baris leaderboard: `model`, `timestamp`, `owner`, `minutes`, `data_hash`, `status`, `test_pred_source`, sehingga `sync_leaderboard` dapat membangun ulang baris leaderboard secara utuh dan identik) |
| `postprocess.json` | Params dari `postprocess` (mis. threshold) |
| `lb.json` | Opsional; ditulis oleh `update_lb_score` (`{"lb_score": float, "note": str, "updated_at": str}`) agar skor LB ikut terbawa saat folder run disalin antar-laptop |
| `summary.md`, `plots/` | §4.6 |
| `log.txt` | Log run |
| `FAILED` / `PARTIAL` | Penanda status (`PARTIAL` **hanya** ditulis jika ada fold yang belum selesai; dihapus otomatis jika `resume_from` menyelesaikan seluruh fold) |

**Leaderboard** `runs/leaderboard.csv` — satu baris per run selesai (termasuk `PARTIAL`, tidak termasuk debug/gagal):
`run_dir, timestamp, owner, modality, run_name, model, metric, cv, cv_raw, cv_post, cv_honest, cv_std, n_folds_done, n_rows_eval, status, test_pred_source, lb_score, minutes, folds_hash, data_hash, seed, fold_seed`
- `run_dir` disimpan sebagai **nama folder relatif** (mis. `20261031_1300_yusuf_tabular_lgbm`), **bukan** absolute path, supaya folder run dari laptop support yang di-copy via flashdisk/LAN ke PC utama tetap langsung terbaca.
- `append_leaderboard` melakukan **upsert berdasarkan `run_dir`**: jika `run_dir` sudah ada di `leaderboard.csv` (mis. sebelumnya `partial` lalu dilanjutkan via `resume_from`), baris lama diperbarui di tempat (tidak menambah baris ganda) dan `lb_score` yang sudah ada dipertahankan.
- `cv` = kolom ranking = selalu sama dengan `cv_honest` pada `eval_folds` (jika `postprocess` tidak di-override, `cv_post = cv_honest = cv_raw` pada `eval_folds`, sehingga semua run dinilai pada populasi baris `eval_folds` yang persis sama; lihat §5.3).
- `data_hash` = `raw_train_hash`.
- `lb_score` kosong di awal; diisi via `update_lb_score(run, score, note="", runs_dir="runs")` yang menulis `lb.json` di dalam folder run **dan** meng-update `runs/leaderboard.csv` (lempar `ConfigError` dengan `[Perbaikan]` jika folder run tidak ditemukan).
- `status ∈ {ok, partial}` (murni menunjukkan apakah semua fold aktif selesai: `n_folds_done == n_active_folds`).
- `test_pred_source ∈ {fold_avg, refit, fold_avg_fallback}` (terpisah dari `status`; lihat §5.4).
- `sync_leaderboard(runs_dir="runs") -> pd.DataFrame`: memindai semua `runs/*/scores.json` + `config.json` + `lb.json` (mengabaikan `_debug` dan folder berisi `FAILED`), mengurutkan baris secara deterministik berdasarkan `["timestamp", "run_dir"]` naik (urutan yang sama juga dipakai `append_leaderboard`), dan menulis ulang `runs/leaderboard.csv` yang **100% identik** kolom, urutan, dan nilainya dengan yang ditulis `append_leaderboard`. Dipakai saat menggabungkan folder run dari laptop support dan PC GPU.

### 4.6 `report.py` — bahan presentasi

Ditulis di akhir setiap run (gagal render → warning, run tetap sukses):
- `summary.md`: run_name, model, notes, config inti (task, metric, cv_strategy, n_folds, seed), tabel skor per fold + cv (`eval_folds`), status, durasi, top-20 feature importance (rata-rata lintas fold selesai dari `fold{k}_info.json`).
- `plots/`: `fold_scores.png`; klasifikasi → `confusion_matrix.png` (dari OOF dengan threshold/argmax); regresi → `residuals.png`; `importance.png` jika ada.
- `compare_runs(runs_dir="runs", top=10, include_partial=False) -> pd.DataFrame` — tabel leaderboard terurut `cv` untuk slide; secara default **hanya menampilkan `status == "ok"`** supaya run `partial` yang baru menyelesaikan sebagian fold tidak merusak perbandingan.

### 4.7 `ensemble.py`

- `load_runs(run_dirs, allow_partial: bool = False) -> EnsembleData` — join via `canon_id`; `FoldMismatchError` (dengan perbaikan "jalankan ulang run dengan `folds_path` yang sama") jika `folds_hash` atau `test_ids_hash` berbeda; `ConfigError` jika task/metric/classes berbeda.
  - Secara default (`allow_partial=False`), jika ada run dengan `scores["status"] != "ok"` (sumber kebenaran status adalah `scores.json`, konsisten dengan `sync_leaderboard`), lempar `ConfigError` dengan perbaikan ("selesaikan run dengan `resume_from`, keluarkan dari daftar blend, atau set `allow_partial=True`").
  - Jika `allow_partial=True` dan ada run sumber yang `status == "partial"`, maka run ensemble yang dihasilkan juga diberi `status = "partial"` (dan menulis `PARTIAL`). Evaluasi dan optimasi bobot selalu memakai **irisan baris** pada `eval_folds` yang memiliki prediksi non-NaN di seluruh run sumber, serta mencatat `n_rows_eval` yang sebenarnya.
- `blend(data, method="mean"|"weights", postprocess=None, run_name=None) -> RunResult` — `weights`: bobot non-negatif berjumlah 1 via `scipy.optimize.minimize` (SLSQP). Disimpan sebagai run `modality="ensemble"`, `model=f"blend_{method}"`, `data_hash=data.folds_hash`, sumber + bobot di `config.extra`.
  - Skor: `cv_raw` = `mean` tanpa optimasi; `cv_post` = bobot (+ postprocess) dioptimasi dan dinilai di OOF yang sama; `cv_honest` = untuk tiap fold k, bobot (+ postprocess) dioptimasi di fold sumber honest (§5.3) lalu dinilai di fold k. Untuk `mean` tanpa postprocess, `cv_post = cv_honest = cv_raw`.

---

## 5. `pipeline.py` — `BasePipeline`

Konstanta modul: `REFIT_TIME_MARGIN: float = 1.25` (angka `1.25` pas untuk 5-fold CV di mana full train berukuran $1.25\times$ dari $4/5$ fold train; untuk 3-fold CV yang full train-nya $\approx 1.5\times$, nilai ini bertindak sebagai batas bawah konservatif).

### 5.1 Alur `run(train, test) -> RunResult`

```
reset state: fold_info = defaultdict(dict), importances = None, classes_ = None
resolve + validate config; seed_everything(cfg.seed); mulai timer via self._clock() (default time.monotonic)
rm = RunManager(cfg)            # atau buka folder resume_from
cek id train/test tidak NaN (lebih dulu!), lalu cek id unik, lalu target tidak NaN → raise DataError
debug: subsample train sesuai cv_strategy (§5.5, ≤ 2000 baris, fold_seed)
rm.save_data_version(train, test)
snap = train[[id, target, group?, time?]]
folds, meta = get_folds(snap, cfg); rm.save_folds(...)
train, test = build_features(train, test)
set id train/test berubah                                       → raise HookContractError
id/target/group/time berubah (canon_hash vs meta)               → warning keras (sebutkan bahwa model dilatih pada target hasil build_features, tetapi skor CV/OOF tetap dihitung terhadap target mentah dari snapshot sehingga fit_predict_fold wajib mengembalikan prediksi dalam skala mentah)
oof[target] selalu diisi dari snap[target] (via canon_id)
multiclass: classes_ = sorted(unique snap[target])
for k in fold aktif:
    jika fold k sudah ada di fold_preds (resume) → load oof, test, & fold{k}_info.json (fold_info[k] + importances), lanjut
    jika n_folds_done >= 1 dan max_minutes terlampaui → hentikan, status = partial
    t0 = self._clock()
    tr, va, te = fit_transform_fold(train[train_ids], train[valid_ids], test, k)
    set id va/te berubah                                        → raise HookContractError
    model = build_model(k)
    model, va_pred, te_pred = fit_predict_fold(model, tr, va, te, k)
    fold_info[k]["duration_sec"] = self._clock() - t0
    tulis fold_preds/fold{k}_{oof,test}.parquet + fold{k}_info.json; log skor fold (terhadap snap[target])
jika n_folds_done == 0 → raise DataError (perbaikan: "naikkan max_minutes")
aggregasi self.importances = rata-rata importances lintas fold selesai (jika ada)
status = "ok" jika semua fold aktif selesai (hapus file PARTIAL jika ada dari sebelum resume), selain itu "partial" (tulis file PARTIAL)
jika test_mode == "refit_full":
    jika status == "partial" ATAU (max_minutes diisi dan sisa_waktu < max(duration_sec) * REFIT_TIME_MARGIN):
        warning keras ("refit_full dilewati, test_pred memakai rata-rata fold selesai")
        test_pred = rata-rata te_pred dari fold yang selesai
        test_pred_source = "fold_avg_fallback"
    selain itu:
        t_refit = self._clock()
        test_pred = fit_full_predict(train, test)
        refit_duration_sec = self._clock() - t_refit
        test_pred_source = "refit"
selain itu:
    test_pred = rata-rata te_pred dari fold yang selesai
    test_pred_source = "fold_avg"
minutes = (sum(duration_sec semua fold selesai) + refit_duration_sec) / 60.0
scores, params = compute_scores(oof, test_pred, ...)   # panggil postprocess(oof_aktif, test_pred) sekali untuk pass penuh, dan postprocess(src_oof, None) untuk pass honest
simpan oof / test_pred / scores / postprocess; report; leaderboard (kecuali debug)
exception di mana pun → tulis FAILED + traceback, lempar ulang
```

### 5.2 Hook

**Utama (90% pemakaian — satu-satunya yang ada di README & notebook):**

| Hook | Default | Aturan |
|---|---|---|
| `build_features(train, test)` | identity | Fitur global tanpa target (lag, rolling, date parts, cleaning). Jangan buang/tambah baris; nilai kosong biarkan NaN. Urutan baris boleh berubah. |
| `fit_transform_fold(tr, va, te, fold)` | identity | Semua yang di-*fit* (target encoding, scaler, imputer, oversampling, feature selection) di sini, fit hanya di `tr`. Boleh buang baris `tr`, **tidak boleh** `va`/`te`. Saat `fold == -1` (refit), `va` kosong: lewati transform `va`, kembalikan `tr.iloc[:0]`. |
| `build_model(fold)` | raise `NotImplementedError` dengan perbaikan | Kembalikan model baru per fold. |

**Lanjutan (bagian "Advanced"):**

| Hook | Default |
|---|---|
| `fit_predict_fold(model, tr, va, te, fold)` | API sklearn: pertama-tama pastikan seluruh kolom fitur (`get_feature_cols(tr)`) juga ada di `va` (jika tidak kosong) dan `te`, serta semuanya bertipe numerik (`pd.api.types.is_numeric_dtype`, yang sudah `True` untuk `bool`); jika ada kolom yang hilang atau non-numerik, lempar `HookContractError` yang menyebut nama kolom tersebut beserta langkah perbaikannya. Lalu `fit(X, y)`; binary → `predict_proba[:, 1]`; multiclass → `predict_proba` diselaraskan ke `classes_`; regresi → `predict`. `va` kosong → pred kosong. Modalitas meng-override (GBDT early stopping, training torch). `self.fold_info` adalah `defaultdict(dict)` sehingga subclass bebas menulis `self.fold_info[fold]["best_iteration"]` atau `self.fold_info[fold]["importances"]` tanpa `KeyError`. |
| `postprocess(oof_df, test_df=None) -> dict` | `{}`. Harus pure; hanya mengembalikan params (`threshold` untuk binary). Nilai kembali `None` dianggap `{}`. Pada **pass penuh** dipanggil satu kali dengan `postprocess(oof_df, test_pred_df)` (`test_df` tidak `None`); pada **pass honest** per fold dipanggil dengan `postprocess(src_oof_df, test_df=None)`. |
| `get_feature_cols(df)` | Semua kolom kecuali id, target, group, time, fold |
| `fit_full_predict(train, test)` | `fit_transform_fold(train, train.iloc[:0], test, -1)` → `build_model(-1)` → `fit_predict_fold(..., -1)` |

Helper: `refit_iterations()` = `round(mean(best_iteration) × refit_iter_mult)` atau `None`.

### 5.3 Skor (konsisten lintas semua run di `eval_folds`)

- `eval_folds` ditentukan **hanya oleh `cv_strategy`**, bukan oleh apakah `postprocess` di-override:
  - non-timeseries: semua fold (`[0..n_folds-1]`).
  - timeseries: selalu fold ≥ 1 (`[1..n_folds-1]`), karena fold 0 tidak punya fold masa lalu untuk evaluasi honest.
- Semua skor utama (`cv_raw`, `cv_post`, `cv_honest`, dan `cv`) **selalu dihitung pada baris OOF di `eval_folds`** (yang sudah selesai / tidak NaN), terhadap **target mentah dari snapshot** (`snap[cfg.target]`), sehingga setiap run pada dataset yang sama dinilai pada populasi baris dan target yang persis sama (`n_rows_eval`):
  - `cv_raw`: metric pada `eval_folds` tanpa params (threshold 0.5 / argmax).
  - Panggilan penuh `params = postprocess(oof_aktif, test_pred_df) or {}` (`oof_aktif` = seluruh baris OOF yang selesai / `fold >= 0`, `test_pred_df` = DataFrame `test_pred` sehingga `test_df is not None`).
  - Jika `params` tidak kosong:
    - `cv_post`: `params` dari panggilan penuh, dinilai pada `eval_folds` (**optimistis**).
    - `cv_honest`: untuk tiap fold $k \in \text{eval\_folds}$, `params_k = postprocess(src_oof, test_df=None) or {}` dioptimasi pada fold sumber honest (non-timeseries: fold aktif $\neq k$; timeseries: fold aktif $< k$, termasuk fold 0), lalu dinilai di fold $k$.
    - Catatan: pada non-timeseries, `cv_honest` masih sedikit optimistis karena model fold lain dilatih memakai fold $k$.
  - Jika `params` kosong (`{}` atau `None`): `cv_post = cv_honest = cv_raw` (ketiganya bernilai sama, bukan NaN, dan pass honest per fold tidak perlu dijalankan ulang).
  - `cv` (kolom ranking di `leaderboard.csv`) selalu = `cv_honest`.
  - `cv_raw_all`: metric raw di seluruh fold aktif (termasuk fold 0 pada timeseries), dicatat di `scores.json` sebagai informasi tambahan.

`scores.json`: `model`, `timestamp`, `owner`, `minutes`, `data_hash`, `metric`, `direction`, `per_fold` (skor + n_rows + duration_sec), `refit_duration_sec`, `cv`, `cv_raw`, `cv_post`, `cv_honest`, `cv_std`, `cv_raw_all`, `eval_folds`, `n_rows_eval`, `folds_hash`, `test_ids_hash`, `classes`, `status`, `test_pred_source`, `n_folds_done`.

### 5.4 Batas waktu, `refit_full`, & resume

- **Pemisahan `status` vs `test_pred_source`:**
  - `status ∈ {"ok", "partial"}` **hanya** ditentukan oleh apakah seluruh fold aktif selesai (`n_folds_done == n_active_folds`). File `PARTIAL` hanya ada jika `status == "partial"`, dan dihapus otomatis saat `resume_from` menuntaskan fold sisanya.
  - `test_pred_source ∈ {"fold_avg", "refit", "fold_avg_fallback"}` mencatat asal `test_pred`. Melewati `refit_full` karena sisa waktu tidak cukup **tidak** mengubah `status` menjadi `partial` (sehingga run timeseries yang seluruh fold-nya selesai tetap berstatus `ok` dan tetap muncul di `compare_runs`), melainkan mencatat `test_pred_source = "fold_avg_fallback"`.
- `max_minutes` (**berlaku per sesi `run()`**, bukan akumulatif lintas sesi resume; sedangkan `minutes` yang tercatat di `scores.json` & leaderboard adalah **jumlah `duration_sec` semua fold selesai ditambah `refit_duration_sec`**, bukan hanya durasi sesi terakhir):
  1. Fold pertama (`n_folds_done == 0`) **selalu dijalankan** agar setiap run menghasilkan minimal 1 fold selesai. Sebelum memulai fold berikutnya (`n_folds_done >= 1`), dicek waktu yang berjalan di sesi `run()` saat ini (`self._clock() - session_start`). Jika waktu sesi ≥ `max_minutes`, fold sisanya dilewati, `status = "partial"`, file `PARTIAL` ditulis, OOF fold yang belum jalan = NaN. Jika karena alasan apa pun `n_folds_done == 0` setelah loop, lempar `DataError` dengan perbaikan `"naikkan max_minutes"`.
  2. Sebelum `refit_full`: jika `status == "partial"` **atau** sisa waktu sesi `< max(duration_sec fold selesai) × REFIT_TIME_MARGIN` (memakai durasi fold **terbesar**, karena pada timeseries expanding window fold terakhir paling besar dan mendekati ukuran full train), `refit_full` dilewati, muncul warning keras bahwa prediksi test bukan hasil refit, `test_pred_source = "fold_avg_fallback"`, dan `test_pred` memakai **rata-rata `te_pred` dari fold yang selesai**.
  3. Pemotongan di tengah fold (per epoch) adalah tanggung jawab modul image/text.
- `resume_from=<run_dir>`:
  - Memakai folder dan `folds.csv` run tersebut; untuk setiap fold yang memiliki `fold_preds/fold{k}_{oof,test}.parquet` dan `fold{k}_info.json`, ketiganya di-load ke memori (`fold_info[k]` terisi kembali termasuk `best_iteration` dan `duration_sec`, begitu pula `importances` per fold).
  - **Validasi penuh `config.json`:** membandingkan **seluruh** `config.json` yang tersimpan di folder run terhadap config sekarang setelah `resolve()`, **kecuali** `max_minutes`, `resume_from`, `notes`, `runs_dir`, dan `owner` (diabaikan agar run dari laptop A yang di-resume di PC GPU tidak gagal karena perbedaan `hostname`; nilai `owner` dan `timestamp` awal dipertahankan dari run asal). Jika ada field lain yang berbeda (termasuk hyperparameter model atau `extra`), lempar `ConfigError` dengan perbaikan ("kembalikan parameter config seperti semula, atau jalankan sebagai run baru dengan `resume_from=None`") agar fold lama dan fold baru tidak pernah mencampur model/fitur yang berbeda.
  - Catatan di README: kode di dalam hook (`build_features`, `fit_transform_fold`, `build_model`) tidak boleh diubah di antara resume karena perubahan kode hook tidak dapat dideteksi dari `config.json`.

### 5.5 Debug (Subsample Sadar-Struktur CV)

`debug=True` → `n_folds=2`, `folds_path` dan `resume_from` diabaikan, folder `runs/_debug/`, tidak masuk leaderboard. Subsample hingga maksimal `N = 2000` baris dilakukan sesuai `cv_strategy` (dengan `fold_seed`) agar tidak pernah memicu `DataError`:
- `timeseries`: urutkan berdasarkan `time_col`, ambil periode-periode unik paling akhir (ekor deret waktu) secara **utuh per periode** hingga ≤ 2000 baris, dengan jaminan minimal `n_folds + 1` (3) periode unik.
- `group` / `stratified_group`: sample grup secara **utuh** (semua baris dalam grup terpilih ikut diambil) hingga ≤ 2000 baris, dengan jaminan minimal `n_folds` grup per kelas (untuk `stratified_group`) atau `n_folds` grup total (untuk `group`).
- `stratified`: stratified sample proporsional hingga ≤ 2000 baris dengan jaminan setiap kelas memiliki minimal `n_folds` baris.
- `kfold`: random sample ≤ 2000 baris.

### 5.6 `RunResult`

Dataclass `run_dir, oof, test_pred, scores, params, cfg, classes`, dengan
`make_submission(sample_sub, path, fn=None) -> pd.DataFrame`:
- Cek `sample_sub[id_col].isna().any()` terlebih dahulu (sebelum cek duplikat / pencocokan set id) → `SubmissionError` dengan perbaikan.
- Set id `sample_sub` harus sama dengan set id test → `SubmissionError` dengan perbaikan.
- Urutan baris mengikuti `sample_sub`.
- Kolom target tunggal: metric `label` → threshold/argmax lalu dipetakan ke label asli; metric `proba`/`value` → nilai mentah.
- Multiclass proba dengan $K$ kolom (selain `id_col`):
  - Jika nama kolom template (`sub_cols`) sama dengan himpunan `classes` (`set(map(str, sub_cols)) == set(map(str, classes))`), isi setiap kolom **berdasarkan kecocokan nama kelas** (tidak bergantung pada urutan kolom di `sample_sub`).
  - Jika nama kolom tidak cocok tetapi jumlah kolom selain `id_col` tepat $K = \text{len(classes)}$, isi berdasarkan urutan posisi `pred_0..pred_{K-1}` dan keluarkan `UserWarning` bahwa kolom diisi berdasarkan urutan karena nama kolom tidak cocok dengan `classes`.
- Format lain → wajib `fn(test_pred, params) -> DataFrame`; jika tidak diberikan → `SubmissionError` dengan perbaikan.

---

## 6. Contoh Notebook

```python
from utils.core import BaseConfig, BasePipeline
from sklearn.ensemble import HistGradientBoostingClassifier

cfg = BaseConfig(task="binary", target="y", metric="f1", run_name="hgb_v1", max_minutes=20)

class Pipe(BasePipeline):
    def build_features(self, train, test):
        return train, test                       # fitur global
    def fit_transform_fold(self, tr, va, te, fold):
        return tr, va, te                        # target encoding dsb. di sini
    def build_model(self, fold):
        return HistGradientBoostingClassifier(random_state=self.cfg.seed)

res = Pipe(cfg).run(train_df, test_df)
res.make_submission(sample_sub, "submission.csv")
```

---

## 7. Error Handling

Semua error turunan `TemplateError(ValueError)` dengan format:
```
[Masalah]   <apa yang salah, dengan angka konkret>
[Perbaikan] <langkah yang bisa langsung dilakukan>
```

| Error | Kondisi | Contoh perbaikan |
|---|---|---|
| `ConfigError` | Config invalid, config resume berbeda | "set `cv_strategy='kfold'` untuk regresi" |
| `DataError` | Id duplikat, target NaN, kelas/grup/waktu kurang, train timeseries kosong | "kurangi `n_folds` menjadi 3" |
| `FoldMismatchError` | `folds_path` tidak cocok; ensemble beda fold/test id | "set `folds_path=None` untuk membuat fold baru" |
| `HookContractError` | Set id berubah di `build_features`, atau set id `va`/`te` berubah di `fit_transform_fold` | "jangan drop baris; isi NaN" |
| `SubmissionError` | Id `sample_sub` ≠ id test, format tidak dikenali | "berikan `fn=` ke `make_submission`" |

Warning (tidak menghentikan run): `time_col` tanpa `timeseries`; target/group/time diubah `build_features`; refit dilewati karena partial; gagal render report.

---

## 8. Testing (`pytest tests/core/`, data sintetis, CPU, offline)

- `test_io.py` — `canon_id` ter-vektorisasi per dtype menyamakan `7`/`"7"`/`7.0`, `True`/`1`/`1.0`; int64 18 digit tidak bertabrakan; `canon_hash` tidak terpengaruh urutan baris dan float32/float64; `fast_df_hash` cepat dan deterministik.
- `test_config.py` — auto-resolve (`owner` dari `socket.gethostname()`, stratified, kfold, group, stratified_group, refit_full); warning `time_col`; `validate` gagal dengan pesan berisi "[Perbaikan]".
- `test_metrics.py` — built-in cocok dengan sklearn; threshold/argmax + classes string; metric custom.
- `test_cv.py` — tiap strategi; timeseries panel tidak terbelah, `ts_gap` dihormati, train kosong → `DataError`; `make_folds_file` menulis `folds/folds.csv` + `folds/folds_meta.json` (`source_run="standalone"`) yang bisa langsung dipakai via `folds_path="folds/folds.csv"`; reuse fold identik meski baris diacak; `fold_seed` vs `seed`; mismatch → `FoldMismatchError` berisi "[Perbaikan]".
- `test_run.py` — artefak lengkap; `owner` masuk ke nama folder + timezone `Asia/Jakarta`; `run_dir` relatif; leaderboard kolom (`status`, `test_pred_source`, `owner`, `minutes`, `model`) & `cv` ranking; `append_leaderboard` melakukan upsert berdasarkan `run_dir`; `update_lb_score` menulis `lb.json` di folder run + meng-update CSV; **hapus `leaderboard.csv` lalu jalankan `sync_leaderboard()` dan pastikan hasilnya identik dengan sebelum dihapus**; `FAILED`; debug di `_debug` tanpa leaderboard.
- `test_pipeline.py` — binary/multiclass/regression/timeseries end-to-end dengan model sklearn; refit dengan scaler di `fit_transform_fold` (va kosong); `build_features` drop baris → `HookContractError`; ubah target → warning; drop `va` → `HookContractError`; `max_minutes` kecil di tengah fold → `status="partial"` + `test_pred_source="fold_avg_fallback"` saat `refit_full`; semua fold selesai tapi sisa waktu `< 1.25 × max(durasi_fold)` sebelum refit → refit dilewati + warning + `status` tetap `"ok"` + `test_pred_source="fold_avg_fallback"` (tetap muncul di `compare_runs`); `resume_from` melewati fold yang sudah ada, memuat ulang `fold{k}_info.json`, menghapus `PARTIAL` saat tuntas, dan melakukan upsert leaderboard; `resume_from` menolak perubahan config selain `max_minutes`/`resume_from`/`notes` dengan `ConfigError`; `debug=True` aman untuk kelas langka, group CV, dan timeseries; skor `cv_raw`, `cv_post`, `cv_honest` selalu di `eval_folds` yang sama (`cv_post = cv_honest = cv_raw` bila tanpa `postprocess`); `make_submission` mengikuti urutan `sample_sub` dan menolak id berbeda.
- `test_report.py` — `summary.md` dan plot terbentuk untuk klasifikasi & regresi; `compare_runs` secara default menyaring hanya `status == "ok"`; error render tidak menggagalkan run.
- `test_ensemble.py` — `mean` & `weights` + tiga skor; beda `folds_hash`/test id → `FoldMismatchError`; NaN partial/timeseries ditangani.

---

## 9. Roadmap (revisi)

| # | Sub-project | Isi inti |
|---|---|---|
| 1 | **Core-lite** (spec ini) | — |
| 2 | Tabular + time series | GBDT preset (LGBM/XGB/CatBoost) dengan early stopping, fitur waktu/lag/rolling, **feature selection cepat**: importance threshold, permutation importance, satu putaran drop-column |
| 3 | Image | `timm` preset ringan untuk 3050 (efficientnet_b0, convnext_nano, resnet34), AMP, batch kecil + grad accumulation, `max_minutes` per epoch, checkpoint & resume, TTA flip |
| 4 | Text | TF-IDF + linear baseline cepat; fine-tune IndoBERT fp16, `max_len` 128–256, `max_minutes`, checkpoint |
| 5 | Kit offline & gladi resik | Skrip unduh bobot timm/HF + wheel, verifikasi dengan internet mati, dataset tiruan per modalitas, cheat sheet, kerangka slide |
