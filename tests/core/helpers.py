"""Shared test helpers for core pipeline and ensemble tests."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from utils.core.pipeline import BasePipeline


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SklearnBinaryPipe(BasePipeline):
    model_name = "logreg"

    def build_model(self, fold):
        return LogisticRegression()


class SklearnRegPipe(BasePipeline):
    model_name = "ridge"

    def build_model(self, fold):
        return Ridge()


def make_bin_df(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.4, size=n) > 0).astype(int)
    tr = pd.DataFrame({"id": range(n), "f1": x, "y": y})
    te = pd.DataFrame({"id": range(100, 112), "f1": rng.normal(size=12)})
    return tr, te
