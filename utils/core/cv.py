"""Cross-validation splitting, standalone fold files, and CV-aware debug subsampling."""

import copy
import hashlib
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from utils.core.config import BaseConfig
from utils.core.errors import DataError, FoldMismatchError
from utils.core.io import canon_hash, canon_id, load_json, save_json


def _resolve_validate_cfg(cfg: BaseConfig) -> BaseConfig:
    """Return a resolved and validated copy of cfg."""
    c = copy.deepcopy(cfg)
    c.resolve()
    c.validate()
    return c


def _validate_snap(
    snap: pd.DataFrame,
    cfg: BaseConfig,
    require_target: bool = True,
) -> pd.Series:
    """Validate id_col, target, group_col, and time_col on snap and return canonical IDs."""
    if cfg.id_col not in snap.columns:
        raise DataError(
            f"Kolom id {cfg.id_col!r} tidak ditemukan pada DataFrame.",
            f"Pastikan kolom {cfg.id_col!r} ada di data atau sesuaikan cfg.id_col.",
        )

    if snap[cfg.id_col].isna().any():
        raise DataError(
            f"Kolom id {cfg.id_col!r} mengandung nilai NaN.",
            f"Hapus atau isi nilai NaN pada kolom {cfg.id_col!r} sebelum menjalankan CV.",
        )

    cid = canon_id(snap[cfg.id_col])
    if cid.duplicated().any():
        raise DataError(
            f"Kolom id {cfg.id_col!r} mengandung ID duplikat.",
            f"Pastikan setiap baris memiliki nilai unik pada kolom {cfg.id_col!r}.",
        )

    if require_target:
        if cfg.target not in snap.columns:
            raise DataError(
                f"Kolom target {cfg.target!r} tidak ditemukan pada DataFrame.",
                f"Pastikan kolom {cfg.target!r} ada di data train atau sesuaikan cfg.target.",
            )
        if snap[cfg.target].isna().any():
            raise DataError(
                f"Kolom target {cfg.target!r} mengandung nilai NaN.",
                f"Hapus baris dengan nilai NaN pada kolom target {cfg.target!r}.",
            )

    if cfg.cv_strategy in ("group", "stratified_group"):
        if not cfg.group_col or cfg.group_col not in snap.columns:
            raise DataError(
                f"Kolom grup {cfg.group_col!r} tidak ditemukan pada DataFrame.",
                f"Pastikan kolom {cfg.group_col!r} ada di data atau sesuaikan cfg.group_col.",
            )
        if snap[cfg.group_col].isna().any():
            raise DataError(
                f"Kolom grup {cfg.group_col!r} mengandung nilai NaN.",
                f"Hapus atau isi nilai NaN pada kolom grup {cfg.group_col!r}.",
            )

    if cfg.cv_strategy == "timeseries":
        if not cfg.time_col or cfg.time_col not in snap.columns:
            raise DataError(
                f"Kolom waktu {cfg.time_col!r} tidak ditemukan pada DataFrame.",
                f"Pastikan kolom {cfg.time_col!r} ada di data atau sesuaikan cfg.time_col.",
            )
        if snap[cfg.time_col].isna().any():
            raise DataError(
                f"Kolom waktu {cfg.time_col!r} mengandung nilai NaN.",
                f"Hapus atau isi nilai NaN pada kolom waktu {cfg.time_col!r}.",
            )

    return cid


def compute_ids_target_hash(snap: pd.DataFrame, cfg: BaseConfig) -> str:
    """Compute canonical hash over id, target, and active group/time columns."""
    cols = [cfg.id_col, cfg.target]
    if cfg.group_col and cfg.group_col in snap.columns and cfg.group_col not in cols:
        cols.append(cfg.group_col)
    if cfg.time_col and cfg.time_col in snap.columns and cfg.time_col not in cols:
        cols.append(cfg.time_col)
    return canon_hash(snap, cols, sort_by=cfg.id_col)


def compute_folds_hash(folds_df: pd.DataFrame, cfg: BaseConfig) -> str:
    """Compute deterministic hash over id->fold mapping plus n_folds and cv_strategy."""
    mapping_hash = canon_hash(folds_df, [cfg.id_col, "fold"], sort_by=cfg.id_col)
    payload = f"{cfg.n_folds}\x1f{cfg.cv_strategy}\n{mapping_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_folds(
    snap: pd.DataFrame,
    cfg: BaseConfig,
    source_run: str = "standalone",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create row-order-invariant CV folds and metadata dictionary."""
    cfg = _resolve_validate_cfg(cfg)
    cid = _validate_snap(snap, cfg, require_target=True)

    if len(snap) < cfg.n_folds:
        raise DataError(
            f"Jumlah baris data ({len(snap)}) lebih kecil dari n_folds ({cfg.n_folds}).",
            f"Kurangi cfg.n_folds menjadi <= {len(snap)} atau tambah data train.",
        )

    # Sort by canonical ID so fold assignments do not depend on input row order
    order = np.argsort(cid.to_numpy(dtype=str), kind="mergesort")
    work = snap.iloc[order].reset_index(drop=True)
    folds = np.full(len(work), -1, dtype=int)

    strat = cfg.cv_strategy
    if strat == "kfold":
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.fold_seed)
        for k, (_, va_idx) in enumerate(kf.split(work)):
            folds[va_idx] = k

    elif strat == "stratified":
        vc = work[cfg.target].value_counts()
        if len(vc) < 2:
            raise DataError(
                f"Kolom target {cfg.target!r} hanya memiliki {len(vc)} kelas unik.",
                "Pastikan data memiliki minimal 2 kelas atau gunakan cv_strategy='kfold'.",
            )
        min_cls = int(vc.min())
        if min_cls < cfg.n_folds:
            raise DataError(
                f"Kelas terkecil pada {cfg.target!r} hanya memiliki {min_cls} sampel (< n_folds={cfg.n_folds}).",
                f"Kurangi cfg.n_folds menjadi <= {min_cls} atau gunakan cv_strategy='kfold'.",
            )
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.fold_seed)
        for k, (_, va_idx) in enumerate(skf.split(work, work[cfg.target])):
            folds[va_idx] = k

    elif strat == "group":
        assert cfg.group_col is not None
        groups = canon_id(work[cfg.group_col])
        n_groups = int(groups.nunique())
        if n_groups < cfg.n_folds:
            raise DataError(
                f"Jumlah grup unik pada {cfg.group_col!r} ({n_groups}) < n_folds ({cfg.n_folds}).",
                f"Kurangi cfg.n_folds menjadi <= {n_groups} atau ganti cv_strategy='kfold'.",
            )
        gkf = GroupKFold(n_splits=cfg.n_folds)
        for k, (_, va_idx) in enumerate(gkf.split(work, work[cfg.target], groups=groups)):
            folds[va_idx] = k

    elif strat == "stratified_group":
        assert cfg.group_col is not None
        groups = canon_id(work[cfg.group_col])
        n_groups = int(groups.nunique())
        if n_groups < cfg.n_folds:
            raise DataError(
                f"Jumlah grup unik pada {cfg.group_col!r} ({n_groups}) < n_folds ({cfg.n_folds}).",
                f"Kurangi cfg.n_folds menjadi <= {n_groups} atau ganti cv_strategy.",
            )
        vc = work[cfg.target].value_counts()
        if len(vc) < 2 or int(vc.min()) < cfg.n_folds:
            raise DataError(
                f"Kelas terkecil pada {cfg.target!r} kurang dari n_folds ({cfg.n_folds}).",
                f"Kurangi cfg.n_folds atau ganti cv_strategy='group'.",
            )
        grp_per_cls = work.assign(_g=groups).groupby(cfg.target)["_g"].nunique()
        min_grp_cls = int(grp_per_cls.min())
        if min_grp_cls < cfg.n_folds:
            raise DataError(
                f"Kelas terkecil pada {cfg.target!r} hanya muncul di {min_grp_cls} grup unik (< n_folds={cfg.n_folds}).",
                f"Kurangi cfg.n_folds menjadi <= {min_grp_cls} atau gunakan cv_strategy='group'.",
            )
        sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.fold_seed)
        for k, (_, va_idx) in enumerate(sgkf.split(work, work[cfg.target], groups=groups)):
            folds[va_idx] = k

    elif strat == "timeseries":
        assert cfg.time_col is not None
        unique_times = pd.Series(work[cfg.time_col].unique()).sort_values(kind="mergesort").to_numpy()
        if len(unique_times) < cfg.n_folds + 1:
            raise DataError(
                f"Jumlah waktu unik pada {cfg.time_col!r} ({len(unique_times)}) < n_folds + 1 ({cfg.n_folds + 1}).",
                f"Kurangi cfg.n_folds menjadi <= {max(2, len(unique_times) - 1)} atau periksa kolom {cfg.time_col!r}.",
            )
        blocks = np.array_split(unique_times, cfg.n_folds + 1)
        time_to_fold: dict[Any, int] = {}
        for t in blocks[0]:
            time_to_fold[t] = -1
        for k in range(cfg.n_folds):
            for t in blocks[k + 1]:
                time_to_fold[t] = k
        folds = work[cfg.time_col].map(time_to_fold).to_numpy(dtype=int)

    orig_folds = np.empty(len(snap), dtype=int)
    orig_folds[order] = folds
    folds_df = pd.DataFrame({cfg.id_col: snap[cfg.id_col].to_numpy(), "fold": orig_folds})

    meta: dict[str, Any] = {
        "source_run": source_run,
        "n_folds": cfg.n_folds,
        "cv_strategy": cfg.cv_strategy,
        "fold_seed": cfg.fold_seed,
        "group_col": cfg.group_col,
        "time_col": cfg.time_col,
        "ids_target_hash": compute_ids_target_hash(snap, cfg),
        "folds_hash": compute_folds_hash(folds_df, cfg),
    }
    return folds_df, meta


def save_folds(folds_df: pd.DataFrame, meta: dict[str, Any], out_dir: Union[str, Path]) -> Path:
    """Save folds.csv and folds_meta.json to out_dir and return path to folds.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds_csv = out / "folds.csv"
    folds_df.to_csv(folds_csv, index=False)
    save_json(meta, out / "folds_meta.json")
    return folds_csv


def load_folds(
    path: Union[str, Path],
    snap: pd.DataFrame,
    cfg: BaseConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate folds.csv and folds_meta.json against snap and cfg."""
    cfg = _resolve_validate_cfg(cfg)
    snap_cid = _validate_snap(snap, cfg, require_target=True)

    p = Path(path)
    csv_path = p / "folds.csv" if p.is_dir() else p
    meta_path = csv_path.parent / "folds_meta.json"

    if not csv_path.exists() or not meta_path.exists():
        raise FoldMismatchError(
            f"File fold tidak ditemukan di {csv_path} atau {meta_path}.",
            "Set cfg.folds_path=None untuk membuat fold baru atau jalankan make_folds_file terlebih dahulu.",
        )

    raw_folds = pd.read_csv(csv_path)
    meta = load_json(meta_path)

    if cfg.id_col not in raw_folds.columns or "fold" not in raw_folds.columns:
        raise FoldMismatchError(
            f"Kolom {cfg.id_col!r} atau 'fold' tidak ditemukan pada {csv_path}.",
            "Set cfg.folds_path=None untuk membuat fold baru.",
        )

    if raw_folds[cfg.id_col].isna().any():
        raise FoldMismatchError(
            f"Kolom {cfg.id_col!r} pada {csv_path} mengandung nilai NaN.",
            "Set cfg.folds_path=None untuk membuat fold baru.",
        )

    folds_cid = canon_id(raw_folds[cfg.id_col])
    if (
        len(folds_cid) != len(snap_cid)
        or folds_cid.duplicated().any()
        or set(folds_cid.to_numpy()) != set(snap_cid.to_numpy())
    ):
        raise FoldMismatchError(
            "Himpunan ID pada folds.csv tidak cocok dengan ID pada data train.",
            "Set cfg.folds_path=None untuk membuat fold baru yang sesuai dengan data train.",
        )

    if meta.get("n_folds") != cfg.n_folds:
        raise FoldMismatchError(
            f"n_folds pada folds_meta.json ({meta.get('n_folds')}) berbeda dengan cfg.n_folds ({cfg.n_folds}).",
            "Samakan cfg.n_folds atau set cfg.folds_path=None untuk membuat fold baru.",
        )

    if meta.get("cv_strategy") != cfg.cv_strategy:
        raise FoldMismatchError(
            f"cv_strategy pada folds_meta.json ({meta.get('cv_strategy')!r}) berbeda dengan cfg.cv_strategy ({cfg.cv_strategy!r}).",
            "Samakan cfg.cv_strategy atau set cfg.folds_path=None untuk membuat fold baru.",
        )

    expected_hash = compute_ids_target_hash(snap, cfg)
    if meta.get("ids_target_hash") != expected_hash:
        raise FoldMismatchError(
            "ids_target_hash pada folds_meta.json tidak cocok dengan data train saat ini.",
            "Data train (id/target/group/time) telah berubah; set cfg.folds_path=None untuk membuat fold baru.",
        )

    fold_map = pd.Series(raw_folds["fold"].to_numpy(dtype=int), index=folds_cid.to_numpy())
    aligned_folds = fold_map.loc[snap_cid.to_numpy()].to_numpy(dtype=int)
    folds_df = pd.DataFrame({cfg.id_col: snap[cfg.id_col].to_numpy(), "fold": aligned_folds})

    if "folds_hash" in meta and meta["folds_hash"] != compute_folds_hash(folds_df, cfg):
        raise FoldMismatchError(
            "folds_hash pada folds_meta.json tidak cocok dengan isi folds.csv.",
            "Set cfg.folds_path=None atau buat ulang file fold dengan make_folds_file.",
        )

    return folds_df, meta


def get_folds(
    snap: pd.DataFrame,
    cfg: BaseConfig,
    source_run: str = "standalone",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load folds if cfg.folds_path is set, otherwise create new folds."""
    cfg = _resolve_validate_cfg(cfg)
    if cfg.folds_path:
        return load_folds(cfg.folds_path, snap, cfg)
    return make_folds(snap, cfg, source_run=source_run)


def make_folds_file(
    train_df: pd.DataFrame,
    cfg: BaseConfig,
    out_dir: Union[str, Path] = "folds",
) -> Path:
    """Standalone helper to create and save folds.csv + folds_meta.json before running pipelines."""
    cfg = _resolve_validate_cfg(cfg)
    folds_df, meta = make_folds(train_df, cfg, source_run="standalone")
    return save_folds(folds_df, meta, out_dir)


def split_fold(
    snap: pd.DataFrame,
    folds_df: pd.DataFrame,
    k: int,
    cfg: BaseConfig,
) -> tuple[set[str], set[str]]:
    """Return (train_ids, valid_ids) as sets of canonical IDs for fold k."""
    cfg = _resolve_validate_cfg(cfg)
    snap_cid = _validate_snap(snap, cfg, require_target=False)

    folds_cid = canon_id(folds_df[cfg.id_col])
    fold_map = pd.Series(folds_df["fold"].to_numpy(dtype=int), index=folds_cid.to_numpy())
    folds_arr = fold_map.loc[snap_cid.to_numpy()].to_numpy(dtype=int)

    cid_arr = snap_cid.to_numpy()
    valid_mask = folds_arr == k
    valid_ids = set(cid_arr[valid_mask].tolist())

    if cfg.cv_strategy != "timeseries":
        train_mask = (folds_arr != k) & (folds_arr != -1)
        train_ids = set(cid_arr[train_mask].tolist())
    else:
        assert cfg.time_col is not None
        if not valid_mask.any():
            raise DataError(
                f"Valid set kosong pada fold {k}.",
                "Periksa pembagian fold atau kurangi cfg.n_folds.",
            )
        unique_times = pd.Series(snap[cfg.time_col].unique()).sort_values(kind="mergesort").to_numpy()
        time_to_rank = {t: r for r, t in enumerate(unique_times)}
        t_rank = snap[cfg.time_col].map(time_to_rank).to_numpy(dtype=int)
        r0 = int(t_rank[valid_mask].min())
        train_mask = t_rank < (r0 - cfg.ts_gap)
        train_ids = set(cid_arr[train_mask].tolist())
        if len(train_ids) == 0:
            raise DataError(
                f"Train set kosong pada fold {k} karena ts_gap={cfg.ts_gap} terlalu besar.",
                "Kurangi cfg.ts_gap atau cfg.n_folds.",
            )

    return train_ids, valid_ids


def debug_subsample(
    train_df: pd.DataFrame,
    cfg: BaseConfig,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """CV-aware subsampling up to max_rows that preserves CV invariants."""
    cfg = _resolve_validate_cfg(cfg)
    _validate_snap(train_df, cfg, require_target=True)

    if len(train_df) <= max_rows:
        return train_df.copy().reset_index(drop=True)

    rng = np.random.default_rng(cfg.fold_seed)
    strat = cfg.cv_strategy

    if strat == "timeseries":
        assert cfg.time_col is not None
        sorted_df = train_df.sort_values(by=cfg.time_col, kind="mergesort")
        period_counts = sorted_df.groupby(cfg.time_col, sort=False).size()
        periods_rev = period_counts.index.to_numpy()[::-1]

        selected_periods = []
        total_rows = 0
        min_periods = cfg.n_folds + 1
        for p in periods_rev:
            cnt = int(period_counts.loc[p])
            if len(selected_periods) < min_periods or (total_rows + cnt <= max_rows):
                selected_periods.append(p)
                total_rows += cnt
            else:
                break

        mask = sorted_df[cfg.time_col].isin(selected_periods)
        return sorted_df.loc[mask].copy().reset_index(drop=True)

    if strat == "group":
        assert cfg.group_col is not None
        grp_counts = train_df.groupby(cfg.group_col, sort=False).size()
        groups = grp_counts.index.to_numpy().copy()
        rng.shuffle(groups)

        # Sort by size first if needed to fit at least n_folds groups within max_rows
        selected_groups = []
        total_rows = 0
        for g in groups:
            cnt = int(grp_counts.loc[g])
            if len(selected_groups) < cfg.n_folds or (total_rows + cnt <= max_rows):
                selected_groups.append(g)
                total_rows += cnt

        mask = train_df[cfg.group_col].isin(selected_groups)
        return train_df.loc[mask].copy().reset_index(drop=True)

    if strat == "stratified_group":
        assert cfg.group_col is not None
        grp_counts = train_df.groupby(cfg.group_col, sort=False).size()
        all_groups = grp_counts.index.to_numpy().copy()
        rng.shuffle(all_groups)

        selected_groups_set: set[Any] = set()
        total_rows = 0

        # Ensure at least n_folds groups per class
        for cls_val, cls_df in train_df.groupby(cfg.target, sort=False):
            cls_groups = cls_df[cfg.group_col].unique().copy()
            rng.shuffle(cls_groups)
            have = sum(1 for g in cls_groups if g in selected_groups_set)
            for g in cls_groups:
                if have >= cfg.n_folds:
                    break
                if g not in selected_groups_set:
                    selected_groups_set.add(g)
                    total_rows += int(grp_counts.loc[g])
                    have += 1

        for g in all_groups:
            if g in selected_groups_set:
                continue
            cnt = int(grp_counts.loc[g])
            if len(selected_groups_set) < cfg.n_folds or (total_rows + cnt <= max_rows):
                selected_groups_set.add(g)
                total_rows += cnt

        mask = train_df[cfg.group_col].isin(selected_groups_set)
        return train_df.loc[mask].copy().reset_index(drop=True)

    if strat == "stratified":
        guaranteed_idx: list[int] = []
        remaining_idx: list[int] = []
        y_vals = train_df[cfg.target].to_numpy()
        for cls_val in pd.unique(y_vals):
            cls_pos = np.flatnonzero(y_vals == cls_val).copy()
            rng.shuffle(cls_pos)
            n_take = min(len(cls_pos), cfg.n_folds)
            guaranteed_idx.extend(cls_pos[:n_take].tolist())
            remaining_idx.extend(cls_pos[n_take:].tolist())

        rem_budget = max(0, max_rows - len(guaranteed_idx))
        if rem_budget > 0 and len(remaining_idx) > 0:
            rem_arr = np.array(remaining_idx, dtype=int)
            if len(rem_arr) <= rem_budget:
                extra = rem_arr.tolist()
            else:
                extra = rng.choice(rem_arr, size=rem_budget, replace=False).tolist()
            guaranteed_idx.extend(extra)

        selected_idx = np.array(sorted(guaranteed_idx), dtype=int)
        return train_df.iloc[selected_idx].copy().reset_index(drop=True)

    # Default: kfold random sample
    idx = rng.choice(len(train_df), size=max_rows, replace=False)
    idx.sort()
    return train_df.iloc[idx].copy().reset_index(drop=True)


__all__ = [
    "compute_ids_target_hash",
    "compute_folds_hash",
    "make_folds",
    "save_folds",
    "load_folds",
    "get_folds",
    "make_folds_file",
    "split_fold",
    "debug_subsample",
]
