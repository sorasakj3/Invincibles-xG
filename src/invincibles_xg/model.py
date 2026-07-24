from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["distance", "angle", "under_pressure", "first_time", "header", "open_play"]
CATEGORICAL = ["technique", "body_part"]


@dataclass(frozen=True)
class Evaluation:
    shots: int
    goals: int
    folds: int
    brier: float
    log_loss: float
    roc_auc: float
    calibration_error: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def model_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=0.7, max_iter=2_000),
            ),
        ]
    )


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray) -> float:
    edges = np.linspace(0, 1, 6)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability <= high if high == 1 else probability < high
        )
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(error)


def evaluate_grouped(shots: pd.DataFrame) -> tuple[Evaluation, np.ndarray]:
    groups = shots["match_id"].astype(str)
    folds = min(5, groups.nunique())
    if folds < 2:
        raise ValueError("Grouped evaluation requires shots from at least two matches")
    probabilities = cross_val_predict(
        model_pipeline(),
        shots[NUMERIC + CATEGORICAL],
        shots["goal"],
        groups=groups,
        cv=GroupKFold(n_splits=folds),
        method="predict_proba",
    )[:, 1]
    y = shots["goal"].to_numpy()
    evaluation = Evaluation(
        shots=len(shots),
        goals=int(y.sum()),
        folds=folds,
        brier=float(brier_score_loss(y, probabilities)),
        log_loss=float(log_loss(y, probabilities, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y, probabilities)),
        calibration_error=expected_calibration_error(y, probabilities),
    )
    return evaluation, probabilities


def fit_model(shots: pd.DataFrame) -> Pipeline:
    return model_pipeline().fit(shots[NUMERIC + CATEGORICAL], shots["goal"])
