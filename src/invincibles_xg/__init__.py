"""Calibrated expected-goals research from open StatsBomb events."""

from .features import build_shot_table
from .model import evaluate_grouped, fit_model

__all__ = ["build_shot_table", "evaluate_grouped", "fit_model"]

