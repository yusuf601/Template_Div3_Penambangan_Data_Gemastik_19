"""Actionable error hierarchy for competition template."""

from typing import Optional


class TemplateError(ValueError):
    """Base error for all template-related issues, providing actionable fix guidance."""

    def __init__(self, problem: str, fix: str) -> None:
        self.problem = problem
        self.fix = fix
        super().__init__(f"[Masalah]   {problem}\n[Perbaikan] {fix}")


class ConfigError(TemplateError):
    """Raised when configuration validation fails or resume configuration mismatches."""


class DataError(TemplateError):
    """Raised when data structure is invalid (duplicate IDs, NaN targets, insufficient samples)."""


class FoldMismatchError(TemplateError):
    """Raised when fold IDs, splits, or hashes mismatch with expected configurations."""


class HookContractError(TemplateError):
    """Raised when pipeline hooks violate contracts (e.g., dropping rows, changing validation IDs)."""


class SubmissionError(TemplateError):
    """Raised when submission IDs, formats, or columns do not conform to requirements."""


__all__ = [
    "TemplateError",
    "ConfigError",
    "DataError",
    "FoldMismatchError",
    "HookContractError",
    "SubmissionError",
]
