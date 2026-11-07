# Panduan Model Transformer Offline & Download Cards (RTX 3070 8GB)

Dokumen ini memuat daftar model Transformer (HuggingFace) yang wajib diunduh sebelum lomba ke dalam direktori lokal `assets/models/`, disesuaikan dengan spesifikasi perangkat keras resmi kompetisi dan kontrak offline [`TransformerPipeline`](../utils/text/transformer.py).

---

## 1. Spesifikasi PC Kompetisi & Batas Operasional

Setiap tim mendapatkan **1 unit PC** dengan spesifikasi berikut:

| Komponen | Spesifikasi Resmi | Implikasi untuk `TransformerPipeline` |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i7-12700 (12 Cores / 20 Threads) | Tokenisasi cepat, TF-IDF + SVD + LightGBM selesai dalam hitungan detik. |
| **Memory (RAM)** | 32 GB DDR5 4800 UDIMM (16 GB + 16 GB) | Sangat lega untuk caching dataset, OOF matriks, dan state dict CPU per fold. |
| **Storage (SSD)** | 1 TB 2280 PCIe NVMe SSD | **Bukan bottleneck.** Menyimpan 4–6 model hanya membutuhkan ~2.5–4.0 GB (<0.5% kapasitas). |
| **Graphics Card (GPU)** | NVIDIA GeForce RTX 3070 LHR (**8 GB GDDR6 VRAM**) | Arsitektur **Ampere** (mendukung native `bf16` & `fp16`). Batas utama adalah **8 GB VRAM** dan waktu lomba **5 jam**. |

> [!IMPORTANT]
> Karena `local_files_only=True` aktif secara default pada `TextConfig`, seluruh model **wajib** sudah tersimpan di dalam `assets/models/` sebelum lomba dimulai. Fungsi [`validate_local_model_dir`](../utils/text/transformer.py) akan memeriksa keberadaan `config.json`, file tokenizer (`vocab.txt` / `tokenizer.json` / `sentencepiece.bpe.model` / `spm.model` / `tokenizer_config.json`), serta file bobot (`model.safetensors` atau `pytorch_model.bin`).

---

## 2. Daftar Wajib Download (Tier Utama — Cepat & Kuat di 8 GB VRAM)

Keempat model berikut dipilih agar saling melengkapi dari sisi **domain bahasa** (formal vs sosmed vs daerah vs multibahasa) dan **keberagaman arsitektur/tokenizer** (WordPiece BERT vs SentencePiece DeBERTa-v3) untuk memaksimalkan kenaikan skor saat di-*blend* dengan [`utils.core.blend`](../utils/core/ensemble.py).

| # | HuggingFace Repo ID | Direktori Lokal (`model_name_or_path`) | Ukuran | Arsitektur & Tokenizer | Estimasi VRAM (`bf16`) | Estimasi 5-Fold (`max_length=256`) |
| :-: | :--- | :--- | :---: | :--- | :---: | :---: |
| **1** | [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1) | `assets/models/indobert-base-p1` | ~440 MB | BERT-base (124M, WordPiece) | ~2.8 GB (`bs=16`) | ~10–15 menit |
| **2** | [`indolem/indobertweet-base-uncased`](https://huggingface.co/indolem/indobertweet-base-uncased) | `assets/models/indobertweet-base-uncased` | ~440 MB | BERT-base (110M, WordPiece) | ~2.8 GB (`bs=16`) | ~8–14 menit |
| **3** | [`LazarusNLP/NusaBERT-base`](https://huggingface.co/LazarusNLP/NusaBERT-base) *(atau `indobenchmark/indobert-base-p2`)* | `assets/models/nusabert-base` | ~440 MB | BERT-base (124M, WordPiece) | ~2.8 GB (`bs=16`) | ~10–15 menit |
| **4** | [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base) | `assets/models/mdeberta-v3-base` | ~1.1 GB | DeBERTa-v3 (278M, SentencePiece) | ~5.5–6.5 GB (`bs=8, accum=2`) | ~20–30 menit |

---

## 3. Model Cards & Konfigurasi per Model

### Card 1: `indobenchmark/indobert-base-p1`
- **Spesifikasi:** ~440 MB | BERT-base, 124M parameter, WordPiece (30.522 token)
- **Peran:** Baseline utama bahasa Indonesia formal, berita, artikel, atau teks umum. Merupakan default `model_name_or_path` pada `TextConfig`.
- **VRAM & Waktu (5-fold, `bf16`):** ~2.8 GB VRAM (`train_batch_size=16, max_length=256`), **~10–15 menit**.
- **Download Command:**
  ```bash
  huggingface-cli download indobenchmark/indobert-base-p1 --local-dir assets/models/indobert-base-p1
  ```
- **Contoh `TextConfig`:**
  ```python
  cfg = TextConfig(
      run_name="indobert_base_p1",
      task="multiclass",
      target="label",
      metric="f1_macro",
      model_name_or_path="assets/models/indobert-base-p1",
      clean_mode="minimal",
      max_length=256,
      train_batch_size=16,
      eval_batch_size=32,
      grad_accum_steps=1,
      lr=2e-5,
      epochs=3,
      precision="auto",
  )
  ```

---

### Card 2: `indolem/indobertweet-base-uncased`
- **Spesifikasi:** ~440 MB | BERT-base, 110M parameter, WordPiece uncased
- **Peran:** Spesialis **media sosial, ulasan e-commerce, komentar netizen, slang/singkatan** (dilatih pada jutaan tweet bahasa Indonesia). Sering mengalahkan IndoBERT standar pada data noisy/percakapan.
- **VRAM & Waktu (5-fold, `bf16`):** ~2.8 GB VRAM (`train_batch_size=16, max_length=128–256`), **~8–14 menit**.
- **Download Command:**
  ```bash
  huggingface-cli download indolem/indobertweet-base-uncased --local-dir assets/models/indobertweet-base-uncased
  ```
- **Contoh `TextConfig`:**
  ```python
  cfg = TextConfig(
      run_name="indobertweet_social",
      task="binary",
      target="label",
      metric="f1",
      model_name_or_path="assets/models/indobertweet-base-uncased",
      clean_mode="social_id",
      emoji_mode="map",
      lowercase=True,
      max_length=192,
      train_batch_size=16,
      eval_batch_size=32,
      grad_accum_steps=1,
      lr=2e-5,
      epochs=3,
      precision="auto",
  )
  ```

---

### Card 3: `LazarusNLP/NusaBERT-base` *(atau `indobenchmark/indobert-base-p2`)*
- **Spesifikasi:** ~440 MB | BERT-base, 124M parameter, WordPiece
- **Peran:** Korpus lebih baru dan mencakup kosakata lintas daerah/nusantara (Jawa, Sunda, Bugis, Melayu, dll.) selain bahasa Indonesia baku. Sangat berguna ketika teks mengandung istilah lokal/daerah.
- **VRAM & Waktu (5-fold, `bf16`):** ~2.8 GB VRAM (`train_batch_size=16, max_length=256`), **~10–15 menit**.
- **Download Command:**
  ```bash
  huggingface-cli download LazarusNLP/NusaBERT-base --local-dir assets/models/nusabert-base
  # Alternatif:
  # huggingface-cli download indobenchmark/indobert-base-p2 --local-dir assets/models/indobert-base-p2
  ```
- **Contoh `TextConfig`:**
  ```python
  cfg = TextConfig(
      run_name="nusabert_base",
      task="multiclass",
      target="label",
      metric="f1_macro",
      model_name_or_path="assets/models/nusabert-base",
      clean_mode="social_id",
      max_length=256,
      train_batch_size=16,
      eval_batch_size=32,
      grad_accum_steps=1,
      lr=2e-5,
      epochs=3,
      precision="auto",
  )
  ```

---

### Card 4: `microsoft/mdeberta-v3-base`
- **Spesifikasi:** ~1.1 GB | DeBERTa-v3 Multilingual, 278M parameter (86M backbone + 192M embedding), SentencePiece (250.000 token)
- **Peran:** **Pasangan ensemble terbaik untuk IndoBERT.** Arsitektur *disentangled relative attention* DeBERTa-v3 + tokenizer SentencePiece menghasilkan representasi dan distribusi prediksi yang berbeda dari keluarga BERT, sehingga *blending* (`IndoBERT` + `mDeBERTa-v3` + `TF-IDF`) memberikan lonjakan skor CV/LB paling konsisten—terutama pada tugas **text-pair** (NLI, deteksi kemiripan, hoax vs klaim) maupun teks campuran Indonesia–Inggris (*code-mixing*).
- **VRAM & Waktu (5-fold, `bf16`):** ~5.5–6.5 GB VRAM (gunakan `train_batch_size=8, grad_accum_steps=2` agar ukuran batch efektif tetap 16 tanpa melebihi 8 GB VRAM), **~20–30 menit**.
- **Download Command:**
  ```bash
  huggingface-cli download microsoft/mdeberta-v3-base --local-dir assets/models/mdeberta-v3-base
  ```
- **Contoh `TextConfig`:**
  ```python
  cfg = TextConfig(
      run_name="mdeberta_v3_base",
      task="multiclass",
      target="label",
      metric="f1_macro",
      model_name_or_path="assets/models/mdeberta-v3-base",
      clean_mode="minimal",
      lowercase=False,          # SentencePiece mDeBERTa peka huruf besar/kecil
      max_length=256,
      train_batch_size=8,       # Hemat VRAM 8 GB RTX 3070
      eval_batch_size=16,
      grad_accum_steps=2,       # Effective batch size = 8 * 2 = 16
      lr=1.5e-5,
      epochs=3,
      precision="auto",
  )
  ```

---

## 4. Script One-Liner Download & Verifikasi Offline Sekaligus

Jalankan perintah berikut sebelum hari lomba (saat masih terhubung ke internet) untuk mengunduh keempat model ke `assets/models/` dan langsung memvalidasi bahwa setiap folder lolos pemeriksaan [`validate_local_model_dir`](../utils/text/transformer.py):

```bash
python -c '
from pathlib import Path
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from utils.text import validate_local_model_dir

MODELS = {
    "indobenchmark/indobert-base-p1": "assets/models/indobert-base-p1",
    "indolem/indobertweet-base-uncased": "assets/models/indobertweet-base-uncased",
    "LazarusNLP/NusaBERT-base": "assets/models/nusabert-base",
    "microsoft/mdeberta-v3-base": "assets/models/mdeberta-v3-base",
}

for repo_id, local_dir in MODELS.items():
    print(f"Downloading {repo_id} -> {local_dir} ...")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(repo_id)
    tok.save_pretrained(local_dir)
    cfg = AutoConfig.from_pretrained(repo_id)
    cfg.save_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, config=cfg)
    model.save_pretrained(local_dir)
    validate_local_model_dir(local_dir, local_files_only=True)
    print(f"OK: {local_dir} verified for offline use.")
'
```

---

## 5. Strategi Urutan Eksekusi & OOF Blending (Budget Waktu 5 Jam)

1. **Menit 0–15:** Jalankan `TfidfPipeline` (`estimator="logistic"` dan `estimator="lgbm"`) → amankan baseline cepat + submission pertama.
2. **Menit 15–35:** Jalankan `TransformerPipeline` dengan **Model #1 (`indobert-base-p1`)** atau **Model #2 (`indobertweet-base-uncased`)** sesuai karakter teks (formal vs sosmed).
3. **Menit 35–65:** Jalankan **Model #4 (`mdeberta-v3-base`)** dengan `train_batch_size=8, grad_accum_steps=2` pada fold yang sama (`folds_path` identik).
4. **Menit 65–70:** Lakukan **OOF Blending** lintas keluarga model:
   ```python
   from utils.core import load_runs, blend

   runs = load_runs([
       "runs/<run_tfidf_logistic>",
       "runs/<run_indobert_or_indobertweet>",
       "runs/<run_mdeberta_v3_base>",
   ])
   ens = blend(runs, method="weights", run_name="blend_tfidf_indobert_mdeberta")
   print("Ensemble CV:", ens.scores["cv"])
   ens.make_submission(sample_sub_df, "runs/submission_ensemble.csv")
   ```
