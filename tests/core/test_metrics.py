import numpy as np
import pytest
from utils.core.errors import ConfigError
from utils.core.metrics import (
    Metric,
    get_metric,
    is_better,
    list_metrics,
    register_metric,
    score,
)


def test_binary_multiclass_and_regression_scoring():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.7, 0.9])
    assert score(y_true, y_prob, "auc", task="binary") == pytest.approx(1.0)
    assert score(y_true, y_prob, "f1", task="binary", threshold=0.5) == pytest.approx(0.8)
    assert score(y_true, y_prob, "f1", task="binary", threshold=0.65) == pytest.approx(1.0)

    classes = ["cat", "dog", "bird"]
    ym_true = np.array(["cat", "dog", "bird", "dog"])
    ym_prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    assert score(ym_true, ym_prob, "f1_macro", task="multiclass", classes=classes) == pytest.approx(1.0)
    assert score(ym_true, ym_prob, "logloss", task="multiclass", classes=classes) < 0.4

    # QWK on regression rounds and clips to global clip_range; QWK on string classes emits warning
    y_ord = np.array([0, 1, 2, 3])
    y_reg_pred = np.array([-0.4, 1.1, 1.9, 4.2])
    assert score(y_ord, y_reg_pred, "qwk", task="regression", clip_range=(0, 3)) == pytest.approx(1.0)
    with pytest.warns(UserWarning, match="ordinal"):
        score(ym_true, ym_prob, "qwk", task="multiclass", classes=classes)

    assert is_better(0.4, 0.6, "rmse") is True
    assert is_better(0.9, 0.8, "f1") is True


def test_builtin_metrics_and_custom_registration():
    expected = {
        "auc", "logloss", "accuracy", "f1", "f1_macro", "f1_weighted",
        "qwk", "rmse", "mae", "mape", "r2"
    }
    assert expected.issubset(set(list_metrics()))

    # Regression metrics
    y_reg = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert score(y_reg, y_pred, "rmse", task="regression") == pytest.approx(0.0)
    assert score(y_reg, y_pred, "mae", task="regression") == pytest.approx(0.0)
    assert score(y_reg, y_pred, "mape", task="regression") == pytest.approx(0.0)
    assert score(y_reg, y_pred, "r2", task="regression") == pytest.approx(1.0)

    # NaN handling in predictions
    y_nan_pred = np.array([1.0, np.nan, 3.0, 4.0])
    assert score(y_reg, y_nan_pred, "rmse", task="regression") == pytest.approx(0.0)
    assert np.isnan(score(y_reg, np.array([np.nan, np.nan, np.nan, np.nan]), "rmse", task="regression"))

    # Custom metric registration
    @register_metric("custom_max_err", direction="minimize", input_type="value")
    def _custom_max_err(yt, yp):
        return float(np.max(np.abs(yt - yp)))

    m = get_metric("custom_max_err")
    assert isinstance(m, Metric)
    assert m.direction == "minimize"
    assert score(y_reg, np.array([1.0, 2.5, 3.0, 4.0]), "custom_max_err", task="regression") == pytest.approx(0.5)

    with pytest.raises(ConfigError, match=r"\[Perbaikan\]"):
        get_metric("nonexistent_metric_xyz")


def test_non_zero_indexed_integer_classes():
    # Multiclass with integer classes [1, 2, 3]
    classes_mc = [1, 2, 3]
    ym_true = np.array([1, 2, 3, 2])
    ym_prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    assert score(ym_true, ym_prob, "f1_macro", task="multiclass", classes=classes_mc) == pytest.approx(1.0)
    assert score(ym_true, ym_prob, "accuracy", task="multiclass", classes=classes_mc) == pytest.approx(1.0)

    # Binary with non-zero-indexed integer classes [1, 2]
    classes_bin = [1, 2]
    yb_true = np.array([1, 1, 2, 2])
    yb_prob = np.array([0.1, 0.4, 0.7, 0.9])
    assert score(yb_true, yb_prob, "f1", task="binary", classes=classes_bin) == pytest.approx(1.0)
    assert score(yb_true, yb_prob, "auc", task="binary", classes=classes_bin) == pytest.approx(1.0)
