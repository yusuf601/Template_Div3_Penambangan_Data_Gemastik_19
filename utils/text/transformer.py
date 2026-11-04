"""Native PyTorch fine-tuning pipeline for HuggingFace sequence classification models."""

import copy
import gc
from pathlib import Path
import time
from typing import Any, Callable, Optional, Union
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from utils.core.errors import ConfigError
from utils.core.metrics import is_better, score
from utils.text.base import TextPipelineBase
from utils.text.config import TextConfig


def validate_local_model_dir(model_name_or_path: Union[str, Path], local_files_only: bool = True) -> None:
    """Validate that local pretrained model directory exists and has config, tokenizer, and weight files."""
    if not local_files_only:
        return

    p = Path(model_name_or_path)
    if not p.exists() or not p.is_dir():
        raise ConfigError(
            f"Direktori model lokal {str(model_name_or_path)!r} tidak ditemukan.",
            "Pastikan model_name_or_path menunjuk ke folder lokal yang berisi model HuggingFace atau download model terlebih dahulu.",
        )

    if not (p / "config.json").exists():
        raise ConfigError(
            f"File 'config.json' tidak ditemukan di {str(p)!r}.",
            "Pastikan folder model memuat file konfigurasi HuggingFace 'config.json'.",
        )

    tok_files = ["vocab.txt", "tokenizer.json", "sentencepiece.bpe.model", "spiece.model", "tokenizer_config.json"]
    if not any((p / f).exists() for f in tok_files):
        raise ConfigError(
            f"File tokenizer tidak ditemukan di {str(p)!r}.",
            f"Pastikan folder model memuat minimal salah satu dari file tokenizer: {tok_files}.",
        )

    weight_files = ["model.safetensors", "pytorch_model.bin"]
    if not any((p / f).exists() for f in weight_files):
        raise ConfigError(
            f"File bobot model tidak ditemukan di {str(p)!r}.",
            f"Pastikan folder model memuat file bobot 'model.safetensors' atau 'pytorch_model.bin'.",
        )


class TextTokenDataset(Dataset):
    """PyTorch Dataset wrapping HuggingFace tokenized encodings and target labels."""

    def __init__(self, encodings: dict[str, Any], labels: Optional[np.ndarray] = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


def make_dynamic_collator(pad_token_id: int) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Create dynamic padding collator for variable-length batches."""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        max_len = max(len(x["input_ids"]) for x in batch)
        keys = [k for k in batch[0].keys() if k != "labels"]

        batch_out: dict[str, torch.Tensor] = {}
        for k in keys:
            pad_val = pad_token_id if k == "input_ids" else 0
            padded_list = []
            for item in batch:
                seq = item[k]
                diff = max_len - len(seq)
                if diff > 0:
                    pad = torch.full((diff,), pad_val, dtype=seq.dtype)
                    seq = torch.cat([seq, pad], dim=0)
                padded_list.append(seq)
            batch_out[k] = torch.stack(padded_list, dim=0)

        if "labels" in batch[0]:
            batch_out["labels"] = torch.stack([item["labels"] for item in batch], dim=0)

        return batch_out

    return collate_fn


class TransformerPipeline(TextPipelineBase):
    """End-to-end HuggingFace Transformer fine-tuning pipeline with native PyTorch loop."""

    _use_tabular_stats: bool = False

    def __init__(self, cfg: TextConfig) -> None:
        self.model_name = f"hf_{Path(cfg.model_name_or_path).name}"
        self._tokenizer: Optional[Any] = None
        self.initial_fold_weights_: dict[int, torch.Tensor] = {}
        super().__init__(cfg)

    def build_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pre-check local model directory exists before running features."""
        validate_local_model_dir(self.cfg.model_name_or_path, self.cfg.local_files_only)
        return super().build_features(train, test)

    def build_model(self, fold: int) -> Any:
        """Load tokenizer once and fresh pretrained model weights per fold from disk."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name_or_path,
                local_files_only=self.cfg.local_files_only,
            )

        num_labels = len(self.classes_) if self.cfg.task == "multiclass" and self.classes_ else 1
        hf_cfg = AutoConfig.from_pretrained(
            self.cfg.model_name_or_path,
            num_labels=num_labels,
            local_files_only=self.cfg.local_files_only,
        )
        if self.cfg.task == "regression":
            hf_cfg.problem_type = "regression"
        elif self.cfg.task == "binary":
            hf_cfg.problem_type = "single_label_classification"
        elif self.cfg.task == "multiclass":
            hf_cfg.problem_type = "single_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name_or_path,
            config=hf_cfg,
            local_files_only=self.cfg.local_files_only,
            ignore_mismatched_sizes=True,
        )

        self.initial_fold_weights_[fold] = next(iter(model.parameters())).detach().cpu().clone()
        return model

    def _on_train_batch_end(self, fold: int, epoch: int, batch_idx: int) -> None:
        """Hook called at end of each training batch for timing and test injection."""
        pass

    def fit_predict_fold(
        self,
        model: Any,
        tr: pd.DataFrame,
        va: pd.DataFrame,
        te: pd.DataFrame,
        fold: int,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Train transformer on tr, evaluate on va per epoch, and return va/te predictions."""
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        collator = make_dynamic_collator(pad_token_id)

        def _format_texts(df_part: pd.DataFrame):
            if len(df_part) == 0:
                if len(self.cfg.text_cols) == 2:
                    return ([], [])
                return []

            meta_cols = self.cfg.num_cols + self.cfg.cat_cols
            if self.cfg.serialize_meta_to_text and meta_cols:
                prefixes = []
                for _, row in df_part.iterrows():
                    p_str = " | ".join(f"{c}: {row[c]}" for c in meta_cols) + " | "
                    prefixes.append(p_str)
            else:
                prefixes = None

            first_col = f"{self.cfg.text_cols[0]}__clean"
            first_texts = df_part[first_col].tolist()
            if prefixes is not None:
                first_texts = [p + t for p, t in zip(prefixes, first_texts)]

            if len(self.cfg.text_cols) == 1:
                return first_texts
            elif len(self.cfg.text_cols) == 2:
                second_col = f"{self.cfg.text_cols[1]}__clean"
                second_texts = df_part[second_col].tolist()
                return (first_texts, second_texts)
            else:
                other_cols = [f"{c}__clean" for c in self.cfg.text_cols[1:]]
                joined = []
                for i in range(len(df_part)):
                    row_texts = [first_texts[i]] + [str(df_part[c].iloc[i]) for c in other_cols]
                    joined.append(self.cfg.text_sep.join(row_texts))
                return joined

        def _tokenize(texts):
            if isinstance(texts, tuple):
                t_a, t_b = texts
                if not t_a:
                    return {"input_ids": [], "attention_mask": []}
                return self._tokenizer(
                    t_a,
                    text_pair=t_b,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=False,
                )
            else:
                if not texts:
                    return {"input_ids": [], "attention_mask": []}
                return self._tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=False,
                )

        # Scale continuous targets for regression
        y_mean = 0.0
        y_std = 1.0
        if self.cfg.task == "regression":
            y_tr_raw = tr[self.cfg.target].to_numpy(dtype=float)
            y_mean = float(np.mean(y_tr_raw))
            y_std = float(max(np.std(y_tr_raw), 1e-8))
            labels_tr = ((y_tr_raw - y_mean) / y_std).astype(np.float32)
            labels_va = (
                ((va[self.cfg.target].to_numpy(dtype=float) - y_mean) / y_std).astype(np.float32)
                if len(va) > 0
                else None
            )
        elif self.cfg.task == "binary":
            labels_tr = tr[self.cfg.target].to_numpy(dtype=float).astype(np.float32)
            labels_va = (
                va[self.cfg.target].to_numpy(dtype=float).astype(np.float32)
                if len(va) > 0
                else None
            )
        else:
            class_to_idx = {c: i for i, c in enumerate(self.classes_ or [])}
            labels_tr = np.array([class_to_idx[y] for y in tr[self.cfg.target]], dtype=np.int64)
            labels_va = (
                np.array([class_to_idx[y] for y in va[self.cfg.target]], dtype=np.int64)
                if len(va) > 0
                else None
            )

        enc_tr = _tokenize(_format_texts(tr))
        enc_va = _tokenize(_format_texts(va)) if len(va) > 0 else {"input_ids": [], "attention_mask": []}
        enc_te = _tokenize(_format_texts(te))

        ds_tr = TextTokenDataset(enc_tr, labels_tr)
        ds_va = TextTokenDataset(enc_va, labels_va) if len(va) > 0 else None
        ds_te = TextTokenDataset(enc_te, None)

        g = torch.Generator()
        g.manual_seed(self.cfg.seed + max(int(fold), 0))

        train_loader = DataLoader(
            ds_tr,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            generator=g,
            collate_fn=collator,
            num_workers=self.cfg.num_workers,
        )
        val_loader = (
            DataLoader(
                ds_va,
                batch_size=self.cfg.eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=self.cfg.num_workers,
            )
            if ds_va is not None
            else None
        )
        test_loader = DataLoader(
            ds_te,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.cfg.num_workers,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        precision = self.cfg.precision
        if precision == "auto":
            if device.type == "cuda":
                precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            else:
                precision = "fp32"

        use_amp = device.type == "cuda" and precision in ("fp16", "bf16")
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))

        if self.cfg.task == "regression":
            criterion = torch.nn.SmoothL1Loss()
        elif self.cfg.task == "binary":
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.lr)

        total_epochs = self.cfg.epochs if fold >= 0 else max(1, self.refit_iterations() or self.cfg.epochs)
        total_steps = len(train_loader) * total_epochs // self.cfg.grad_accum_steps
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps)
        )

        def _predict_loader(loader: Optional[DataLoader]) -> np.ndarray:
            if loader is None or len(loader.dataset) == 0:
                if self.cfg.task == "multiclass":
                    return np.empty((0, len(self.classes_) if self.classes_ else 0))
                return np.empty(0)

            model.eval()
            all_preds = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        if token_type_ids is not None:
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                            )
                        else:
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    logits = outputs.logits
                    if self.cfg.task == "regression":
                        preds = (logits.squeeze(-1) * y_std + y_mean).float().cpu().numpy()
                    elif self.cfg.task == "binary":
                        preds = torch.sigmoid(logits.squeeze(-1)).float().cpu().numpy()
                    else:
                        preds = torch.softmax(logits.float(), dim=-1).float().cpu().numpy()

                    all_preds.append(preds)

            if not all_preds:
                return np.empty(0)
            if self.cfg.task == "multiclass":
                return np.vstack(all_preds)
            return np.concatenate(all_preds, axis=0)

        best_state_dict: Optional[dict[str, Any]] = None
        best_select: Optional[float] = None
        best_epoch = 1

        # Timing and batch durations on fold 0
        batch_times: list[float] = []

        for epoch in range(total_epochs):
            model.train()
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                t_b_start = self._clock()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                targets = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if token_type_ids is not None:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    logits = outputs.logits
                    if self.cfg.task == "regression":
                        loss = criterion(logits.squeeze(-1), targets)
                    elif self.cfg.task == "binary":
                        loss = criterion(logits.squeeze(-1), targets)
                    else:
                        loss = criterion(logits, targets)

                    if self.cfg.grad_accum_steps > 1:
                        loss = loss / self.cfg.grad_accum_steps

                if use_amp and precision == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.cfg.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if use_amp and precision == "fp16":
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                self._on_train_batch_end(fold, epoch, batch_idx)
                b_dur = float(self._clock() - t_b_start)

                # Batches 3..10 timing guard check on fold 0
                if fold == 0 and epoch == 0 and self.cfg.max_minutes is not None:
                    if 2 <= batch_idx <= 9:
                        batch_times.append(b_dur)
                    if batch_idx == min(9, len(train_loader) - 1) and len(train_loader) >= 3 and batch_times:
                        mean_b = float(np.mean(batch_times))
                        est_epoch_sec = mean_b * len(train_loader)
                        rem_sec = self.cfg.max_minutes * 60.0 - (self._clock() - self._run_start)
                        if est_epoch_sec > rem_sec:
                            warnings.warn(
                                f"Estimasi 1 epoch (~{est_epoch_sec / 60:.2f}m) melebihi sisa max_minutes ({rem_sec / 60:.2f}m); "
                                "cutoff darurat akan aktif setelah epoch pertama selesai.",
                                UserWarning,
                                stacklevel=2,
                            )

            # End of epoch evaluation (for fold >= 0)
            if fold >= 0 and val_loader is not None:
                va_pred_curr = _predict_loader(val_loader)
                y_va_raw = va[self.cfg.target].to_numpy()
                val_select = float(
                    score(
                        y_va_raw,
                        va_pred_curr,
                        self.cfg.select_metric,
                        self.cfg.task,
                        classes=self.classes_,
                    )
                )

                if best_select is None or is_better(val_select, best_select, self.cfg.select_metric):
                    best_select = val_select
                    best_epoch = epoch + 1
                    self.fold_info[fold]["best_iteration"] = best_epoch
                    best_state_dict = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

                # Intra-fold emergency cutoff
                if self.cfg.max_minutes is not None:
                    elapsed = float(self._clock() - self._run_start)
                    if elapsed >= self.cfg.max_minutes * 60.0:
                        self.fold_info[fold]["truncated"] = True
                        warnings.warn(
                            f"Emergency cutoff triggered on fold {fold} at epoch {epoch + 1}: elapsed {elapsed:.1f}s >= max_minutes",
                            UserWarning,
                            stacklevel=2,
                        )
                        break

        # Restore best weights if available
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        va_pred = _predict_loader(val_loader) if val_loader is not None else np.empty(0)
        te_pred = _predict_loader(test_loader)

        # Explicit GPU memory cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None, va_pred, te_pred


__all__ = [
    "validate_local_model_dir",
    "TextTokenDataset",
    "make_dynamic_collator",
    "TransformerPipeline",
]
