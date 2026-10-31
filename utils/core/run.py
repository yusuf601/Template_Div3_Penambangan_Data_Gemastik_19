"""Portable RunManager, upsert leaderboard, LB score tracking, and sync_leaderboard."""

import copy
from datetime import datetime
from pathlib import Path
import re
import traceback
from typing import Any, Literal, Optional, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.core.config import BaseConfig
from utils.core.cv import save_folds as cv_save_folds
from utils.core.errors import ConfigError
from utils.core.io import (
    canon_hash,
    fast_df_hash,
    load_json,
    load_parquet,
    save_json,
    save_parquet,
)

JAKARTA_TZ = ZoneInfo("Asia/Jakarta")

LEADERBOARD_COLS: list[str] = [
    "run_dir",
    "timestamp",
    "owner",
    "modality",
    "run_name",
    "model",
    "metric",
    "cv",
    "cv_raw",
    "cv_post",
    "cv_honest",
    "cv_std",
    "n_folds_done",
    "n_rows_eval",
    "status",
    "test_pred_source",
    "lb_score",
    "minutes",
    "folds_hash",
    "data_hash",
    "seed",
    "fold_seed",
]

RESUME_IGNORED_KEYS = {"max_minutes", "resume_from", "notes", "runs_dir", "owner"}


def safe_name(s: Any, default: str) -> str:
    """Sanitize a string into a portable folder name segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(s)).strip("-")
    return cleaned or default


def _opt_float(val: Any) -> float:
    if val is None or pd.isna(val) or val == "":
        return float(np.nan)
    return float(val)


def _build_leaderboard_row(
    run_dir: Path,
    cfg_dict: dict[str, Any],
    scores: dict[str, Any],
    fallback_lb_score: Any = np.nan,
) -> dict[str, Any]:
    """Construct a canonical leaderboard row dictionary with LEADERBOARD_COLS keys."""
    lb_file = run_dir / "lb.json"
    lb_val: Any = fallback_lb_score
    if lb_file.exists():
        try:
            lb_data = load_json(lb_file)
            if lb_data.get("lb_score") is not None:
                lb_val = float(lb_data["lb_score"])
        except Exception:
            pass

    return {
        "run_dir": str(run_dir.name),
        "timestamp": str(scores.get("timestamp", "")),
        "owner": str(scores.get("owner", cfg_dict.get("owner", ""))),
        "modality": str(cfg_dict.get("modality", "generic")),
        "run_name": str(cfg_dict.get("run_name", "baseline")),
        "model": str(scores.get("model", "base")),
        "metric": str(scores.get("metric", cfg_dict.get("metric", ""))),
        "cv": _opt_float(scores.get("cv")),
        "cv_raw": _opt_float(scores.get("cv_raw")),
        "cv_post": _opt_float(scores.get("cv_post")),
        "cv_honest": _opt_float(scores.get("cv_honest")),
        "cv_std": _opt_float(scores.get("cv_std")),
        "n_folds_done": int(scores.get("n_folds_done", 0)),
        "n_rows_eval": int(scores.get("n_rows_eval", 0)),
        "status": str(scores.get("status", "ok")),
        "test_pred_source": str(scores.get("test_pred_source", "fold_avg")),
        "lb_score": _opt_float(lb_val),
        "minutes": float(scores.get("minutes", 0.0)),
        "folds_hash": str(scores.get("folds_hash", "")),
        "data_hash": str(scores.get("data_hash", "")),
        "seed": int(cfg_dict.get("seed", 42)),
        "fold_seed": int(cfg_dict.get("fold_seed", 42)),
    }


def _recover_timestamp(run_dir: Path) -> str:
    """Recover original run timestamp from scores.json or log.txt header."""
    scores_path = run_dir / "scores.json"
    if scores_path.exists():
        try:
            ts = load_json(scores_path).get("timestamp")
            if ts:
                return str(ts)
        except Exception:
            pass
    log_path = run_dir / "log.txt"
    if log_path.exists():
        try:
            first_line = log_path.read_text(encoding="utf-8").splitlines()[0]
            m = re.search(r"timestamp=([^\s]+)", first_line)
            if m:
                return m.group(1)
        except Exception:
            pass
    return datetime.now(JAKARTA_TZ).isoformat(timespec="seconds")


class RunManager:
    """Manages experiment run directory, fold checkpoints, artifacts, and leaderboard."""

    def __init__(self, cfg: BaseConfig, model_name: str = "base") -> None:
        self.cfg: BaseConfig = copy.deepcopy(cfg)
        self.cfg.resolve()
        self.cfg.validate()
        self.model_name: str = str(model_name)

        if self.cfg.resume_from:
            self.is_resumed: bool = True
            cand = Path(self.cfg.resume_from)
            if not cand.exists():
                alt = Path(self.cfg.runs_dir) / self.cfg.resume_from
                if alt.exists():
                    cand = alt
                else:
                    alt2 = Path(self.cfg.runs_dir) / Path(self.cfg.resume_from).name
                    if alt2.exists():
                        cand = alt2

            cfg_file = cand / "config.json"
            if not cand.exists() or not cand.is_dir() or not cfg_file.exists():
                raise ConfigError(
                    f"Folder resume_from={self.cfg.resume_from!r} atau config.json tidak ditemukan.",
                    "Periksa path resume_from atau jalankan sebagai run baru dengan resume_from=None.",
                )

            saved_cfg = load_json(cfg_file)
            curr_cfg = self.cfg.to_dict()
            all_keys = sorted(set(saved_cfg.keys()) | set(curr_cfg.keys()))
            for k in all_keys:
                if k in RESUME_IGNORED_KEYS:
                    continue
                if saved_cfg.get(k) != curr_cfg.get(k):
                    raise ConfigError(
                        f"Config {k!r} saat resume ({curr_cfg.get(k)!r}) berbeda dengan config run asal ({saved_cfg.get(k)!r}).",
                        "Kembalikan parameter config seperti semula, atau jalankan sebagai run baru dengan resume_from=None.",
                    )

            if saved_cfg.get("owner"):
                self.cfg.owner = str(saved_cfg["owner"])
            self.run_dir: Path = cand
            self.timestamp: str = _recover_timestamp(self.run_dir)
            self.log(f"Resumed run in {self.run_dir.name}")
        else:
            self.is_resumed = False
            now = datetime.now(JAKARTA_TZ)
            self.timestamp = now.isoformat(timespec="seconds")
            ts_prefix = now.strftime("%Y%m%d_%H%M")

            s_owner = safe_name(self.cfg.owner, "")
            owner_part = f"{s_owner}_" if s_owner else ""
            s_mod = safe_name(self.cfg.modality, "generic")
            s_name = safe_name(self.cfg.run_name, "baseline")
            base_name = f"{ts_prefix}_{owner_part}{s_mod}_{s_name}"

            parent = (
                Path(self.cfg.runs_dir) / "_debug"
                if self.cfg.debug
                else Path(self.cfg.runs_dir)
            )
            parent.mkdir(parents=True, exist_ok=True)

            run_dir = parent / base_name
            suffix = 2
            while run_dir.exists():
                run_dir = parent / f"{base_name}_{suffix}"
                suffix += 1

            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.cfg.save(self.run_dir / "config.json")
            with open(self.run_dir / "log.txt", "a", encoding="utf-8") as f:
                f.write(f"[init] timestamp={self.timestamp} run_dir={self.run_dir.name}\n")

    def log(self, msg: str) -> None:
        """Append a timestamped log line to log.txt."""
        ts = datetime.now(JAKARTA_TZ).isoformat(timespec="seconds")
        with open(self.run_dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    def save_data_version(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Compute and save data_version.json with raw_train_hash and test_ids_hash."""
        raw_train_hash = fast_df_hash(train_df)
        test_ids_hash = canon_hash(test_df, [self.cfg.id_col], sort_by=self.cfg.id_col)
        dv: dict[str, Any] = {
            "raw_train_hash": raw_train_hash,
            "test_ids_hash": test_ids_hash,
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "train_cols": [str(c) for c in train_df.columns],
            "test_cols": [str(c) for c in test_df.columns],
        }
        save_json(dv, self.run_dir / "data_version.json")
        return dv

    def save_folds(self, folds_df: pd.DataFrame, meta: dict[str, Any]) -> Path:
        """Save folds.csv and folds_meta.json into run_dir."""
        return cv_save_folds(folds_df, meta, self.run_dir)

    def save_fold_checkpoint(
        self,
        fold: int,
        oof_fold_df: pd.DataFrame,
        test_fold_df: pd.DataFrame,
        info: dict[str, Any],
    ) -> None:
        """Save per-fold OOF, test predictions, and fold info immediately after fold finishes."""
        fp_dir = self.run_dir / "fold_preds"
        fp_dir.mkdir(parents=True, exist_ok=True)
        save_parquet(oof_fold_df, fp_dir / f"fold{fold}_oof.parquet")
        save_parquet(test_fold_df, fp_dir / f"fold{fold}_test.parquet")
        save_json(info, fp_dir / f"fold{fold}_info.json")

    def load_fold_checkpoint(
        self, fold: int
    ) -> Optional[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
        """Load per-fold OOF, test predictions, and fold info if all 3 files exist."""
        fp_dir = self.run_dir / "fold_preds"
        oof_p = fp_dir / f"fold{fold}_oof.parquet"
        test_p = fp_dir / f"fold{fold}_test.parquet"
        info_p = fp_dir / f"fold{fold}_info.json"
        if oof_p.exists() and test_p.exists() and info_p.exists():
            return load_parquet(oof_p), load_parquet(test_p), load_json(info_p)
        return None

    def save_oof(self, oof_df: pd.DataFrame) -> Path:
        """Save aggregated OOF predictions to oof.parquet."""
        path = self.run_dir / "oof.parquet"
        save_parquet(oof_df, path)
        return path

    def save_test_pred(self, test_pred_df: pd.DataFrame) -> Path:
        """Save test predictions to test_pred.parquet."""
        path = self.run_dir / "test_pred.parquet"
        save_parquet(test_pred_df, path)
        return path

    def save_postprocess(self, params: dict[str, Any]) -> Path:
        """Save postprocessing parameters to postprocess.json."""
        path = self.run_dir / "postprocess.json"
        save_json(params, path)
        return path

    def save_scores(self, scores: dict[str, Any]) -> Path:
        """Save scores dictionary to scores.json."""
        path = self.run_dir / "scores.json"
        save_json(scores, path)
        return path

    def set_status(self, status: Literal["ok", "partial"]) -> None:
        """Create or remove the PARTIAL marker file based on run completion status."""
        partial_path = self.run_dir / "PARTIAL"
        if status == "partial":
            partial_path.write_text("partial\n", encoding="utf-8")
        elif status == "ok":
            if partial_path.exists():
                partial_path.unlink()

    def mark_failed(self, exc: BaseException) -> Path:
        """Write FAILED marker file with traceback and log the error."""
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        failed_path = self.run_dir / "FAILED"
        failed_path.write_text(tb, encoding="utf-8")
        self.log(f"FAILED: {type(exc).__name__}: {exc}")
        return failed_path

    def append_leaderboard(self, scores: dict[str, Any]) -> Optional[Path]:
        """Upsert run row into runs/leaderboard.csv (skipped when debug=True)."""
        if self.cfg.debug:
            return None

        runs_path = Path(self.cfg.runs_dir)
        runs_path.mkdir(parents=True, exist_ok=True)
        lb_path = runs_path / "leaderboard.csv"

        existing_lb_score: Any = np.nan
        rows: list[dict[str, Any]] = []
        if lb_path.exists():
            old_df = pd.read_csv(lb_path)
            for _, r in old_df.iterrows():
                r_dict = r.to_dict()
                if str(r_dict.get("run_dir")) == self.run_dir.name:
                    existing_lb_score = r_dict.get("lb_score", np.nan)
                else:
                    rows.append(r_dict)

        new_row = _build_leaderboard_row(
            self.run_dir,
            self.cfg.to_dict(),
            scores,
            fallback_lb_score=existing_lb_score,
        )
        rows.append(new_row)

        df = pd.DataFrame(rows, columns=LEADERBOARD_COLS)
        df = df.sort_values(by=["timestamp", "run_dir"], ascending=[True, True], kind="mergesort")
        df = df.reset_index(drop=True)
        df.to_csv(lb_path, index=False)
        return lb_path


def sync_leaderboard(runs_dir: Union[str, Path] = "runs") -> pd.DataFrame:
    """Scan runs_dir and rebuild runs/leaderboard.csv deterministically."""
    base = Path(runs_dir)
    rows: list[dict[str, Any]] = []

    if base.exists() and base.is_dir():
        for sub in sorted(base.iterdir(), key=lambda p: p.name):
            if not sub.is_dir() or sub.name == "_debug" or sub.name.startswith("."):
                continue
            if (sub / "FAILED").exists():
                continue
            scores_p = sub / "scores.json"
            cfg_p = sub / "config.json"
            if not scores_p.exists() or not cfg_p.exists():
                continue
            try:
                scores = load_json(scores_p)
                cfg_dict = load_json(cfg_p)
                row = _build_leaderboard_row(sub, cfg_dict, scores)
                rows.append(row)
            except Exception:
                continue

    df = pd.DataFrame(rows, columns=LEADERBOARD_COLS)
    if not df.empty:
        df = df.sort_values(by=["timestamp", "run_dir"], ascending=[True, True], kind="mergesort")
        df = df.reset_index(drop=True)

    if base.exists():
        df.to_csv(base / "leaderboard.csv", index=False)
    return df


def update_lb_score(
    run_dir_or_name: Union[str, Path],
    lb_score: float,
    note: str = "",
    runs_dir: Union[str, Path] = "runs",
) -> Path:
    """Record public/private LB score in lb.json inside the run folder and update leaderboard.csv."""
    base = Path(runs_dir)
    cand = Path(run_dir_or_name)
    if not (cand.exists() and cand.is_dir()):
        cand = base / Path(run_dir_or_name).name

    if not cand.exists() or not cand.is_dir() or not (cand / "config.json").exists():
        raise ConfigError(
            f"Folder run {str(run_dir_or_name)!r} tidak ditemukan di {str(base)!r}.",
            f"Periksa nama folder run di {str(base)!r} atau jalankan sync_leaderboard(runs_dir={str(base)!r}).",
        )

    lb_payload = {
        "lb_score": float(lb_score),
        "note": str(note),
        "updated_at": datetime.now(JAKARTA_TZ).isoformat(timespec="seconds"),
    }
    lb_file = cand / "lb.json"
    save_json(lb_payload, lb_file)

    lb_csv = base / "leaderboard.csv"
    if lb_csv.exists():
        df = pd.read_csv(lb_csv)
        if "run_dir" in df.columns and (df["run_dir"] == cand.name).any():
            df.loc[df["run_dir"] == cand.name, "lb_score"] = float(lb_score)
            df.to_csv(lb_csv, index=False)
        else:
            sync_leaderboard(runs_dir=base)
    else:
        sync_leaderboard(runs_dir=base)

    return lb_file


__all__ = [
    "LEADERBOARD_COLS",
    "safe_name",
    "RunManager",
    "update_lb_score",
    "sync_leaderboard",
]
