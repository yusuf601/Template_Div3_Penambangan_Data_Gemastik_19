"""Metric registry and unified scoring for binary, multiclass, and regression tasks."""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from utils.core.errors import ConfigError

DirectionType = Literal["maximize", "minimize"]
InputType = Literal["proba", "label", "value"]


@dataclass
class Metric:
    """Descriptor for a registered evaluation metric."""

    name: str
    fn: Callable[..., float]
    direction: DirectionType
    input_type: InputType


_REGISTRY: dict[str, Metric] = {}


def register_metric(
    name: str,
    direction: DirectionType,
    input_type: InputType,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorator to register a metric function in the global registry."""
    if direction not in ("maximize", "minimize"):
        raise ConfigError(
            f"direction metric {name!r} tidak valid: {direction!r}",
            "Gunakan direction='maximize' atau direction='minimize'.",
        )
    if input_type not in ("proba", "label", "value"):
        raise ConfigError(
            f"input_type metric {name!r} tidak valid: {input_type!r}",
            "Gunakan input_type='proba', 'label', atau 'value'.",
        )

    def _decorator(fn: Callable[..., float]) -> Callable[..., float]:
        _REGISTRY[name] = Metric(
            name=name,
            fn=fn,
            direction=direction,
            input_type=input_type,
        )
        return fn

    return _decorator


def list_metrics() -> list[str]:
    """Return list of all registered metric names."""
    return list(_REGISTRY.keys())


def get_metric(name: str) -> Metric:
    """Retrieve a registered Metric by name or raise ConfigError with remediation."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ConfigError(
            f"Metric {name!r} tidak ditemukan di registry.",
            f"Pilih salah satu metric tersedia ({available}) atau daftarkan via @register_metric.",
        )
    return _REGISTRY[name]


def is_better(a: float, b: float, metric: str) -> bool:
    """Return True if score a is strictly better than score b for the given metric."""
    m = get_metric(metric)
    if m.direction == "maximize":
        return bool(a > b)
    return bool(a < b)


# --- Built-in Metrics ---


@register_metric("auc", direction="maximize", input_type="proba")
def _auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim == 2:
        return float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro"))
    return float(roc_auc_score(y_true, y_pred))


@register_metric("logloss", direction="minimize", input_type="proba")
def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true, y_pred))


@register_metric("accuracy", direction="maximize", input_type="label")
def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


@register_metric("f1", direction="maximize", input_type="label")
def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


@register_metric("f1_macro", direction="maximize", input_type="label")
def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("f1_weighted", direction="maximize", input_type="label")
def _f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


@register_metric("qwk", direction="maximize", input_type="label")
def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


@register_metric("rmse", direction="minimize", input_type="value")
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@register_metric("mae", direction="minimize", input_type="value")
def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


@register_metric("mape", direction="minimize", input_type="value")
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred))


@register_metric("r2", direction="maximize", input_type="value")
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _map_to_class_indices(y_true: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    """Map y_true values to integer indices 0..len(classes)-1 matching classes order."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    str_to_idx = {str(c): i for i, c in enumerate(classes)}
    mapped = []
    for v in y_true:
        if v in class_to_idx:
            mapped.append(class_to_idx[v])
        else:
            mapped.append(str_to_idx[str(v)])
    return np.asarray(mapped, dtype=int)


def score(
    y_true: Any,
    y_raw: Any,
    metric: str,
    task: str,
    threshold: float = 0.5,
    classes: Optional[Sequence[Any]] = None,
    clip_range: Optional[tuple[int, int]] = None,
) -> float:
    """Compute metric score after filtering NaN predictions and converting raw outputs."""
    m = get_metric(metric)
    yt = np.asarray(y_true)
    yr = np.asarray(y_raw, dtype=float)

    if yr.ndim == 1:
        valid = ~np.isnan(yr)
    else:
        valid = ~np.isnan(yr).any(axis=1)

    yt = yt[valid]
    yr = yr[valid]

    if len(yt) == 0:
        return float("nan")

    # Special handling for QWK
    if metric == "qwk":
        if task == "regression":
            if clip_range is not None:
                y_min, y_max = int(clip_range[0]), int(clip_range[1])
            else:
                y_min, y_max = int(np.min(yt)), int(np.max(yt))
            y_pred_int = np.clip(np.rint(yr), y_min, y_max).astype(int)
            y_true_int = np.rint(yt).astype(int)
            return float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic"))

        has_str_classes = (
            classes is not None and any(isinstance(c, str) for c in classes)
        ) or (yt.dtype.kind in ("U", "S", "O") and any(isinstance(x, str) for x in yt))
        if has_str_classes:
            warnings.warn(
                "qwk mengasumsikan kelas ordinal terurut sedangkan urutan alfabet/string belum tentu bermakna ordinal.",
                UserWarning,
                stacklevel=2,
            )

        if task == "multiclass":
            y_pred_idx = np.argmax(yr, axis=1) if yr.ndim == 2 else np.rint(yr).astype(int)
            if classes is not None:
                yt_idx = _map_to_class_indices(yt, classes)
                return float(cohen_kappa_score(yt_idx, y_pred_idx, weights="quadratic"))
            return float(cohen_kappa_score(yt, y_pred_idx, weights="quadratic"))

        # binary QWK
        y_pred_bin = (yr >= threshold).astype(int)
        if classes is not None and len(classes) == 2:
            yt_idx = _map_to_class_indices(yt, classes)
            return float(cohen_kappa_score(yt_idx, y_pred_bin, weights="quadratic"))
        return float(cohen_kappa_score(yt.astype(int), y_pred_bin, weights="quadratic"))

    # Standard metric handling by task and input_type
    if task == "multiclass":
        if m.input_type == "label":
            idx = np.argmax(yr, axis=1) if yr.ndim == 2 else yr.astype(int)
            if classes is not None:
                classes_arr = np.asarray(classes, dtype=object)
                y_pred_lbl = classes_arr[idx]
                return float(m.fn(yt.astype(object) if yt.dtype.kind in ("U", "S", "O") else yt, y_pred_lbl))
            return float(m.fn(yt, idx))

        if m.input_type == "proba":
            if classes is not None:
                yt_idx = _map_to_class_indices(yt, classes)
                labels = list(range(len(classes)))
                if metric == "logloss":
                    return float(log_loss(yt_idx, yr, labels=labels))
                if metric == "auc":
                    return float(
                        roc_auc_score(
                            yt_idx, yr, multi_class="ovr", average="macro", labels=labels
                        )
                    )
                return float(m.fn(yt_idx, yr))
            return float(m.fn(yt, yr))

        return float(m.fn(yt, yr))

    if task == "binary":
        if classes is not None and len(classes) == 2 and not np.issubdtype(yt.dtype, np.number):
            yt_eval = _map_to_class_indices(yt, classes)
        else:
            yt_eval = yt

        if m.input_type == "label":
            y_pred_bin = (yr >= threshold).astype(int)
            return float(m.fn(yt_eval, y_pred_bin))

        if m.input_type == "proba":
            if metric == "logloss":
                return float(log_loss(yt_eval, yr, labels=[0, 1]))
            return float(m.fn(yt_eval, yr))

        return float(m.fn(yt_eval, yr))

    # task == "regression"
    if m.input_type == "label":
        return float(m.fn(np.rint(yt).astype(int), np.rint(yr).astype(int)))
    return float(m.fn(yt, yr))


__all__ = [
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "is_better",
    "score",
]
