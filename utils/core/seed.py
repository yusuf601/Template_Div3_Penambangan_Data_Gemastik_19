"""Deterministic random seed utility."""

import os
import random
import sys
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds across Python standard library, NumPy, and PyTorch (if present)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


__all__ = ["seed_everything"]
