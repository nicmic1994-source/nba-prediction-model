from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelBundle:
    logistic: object
    boosting: object
    features: list[str]
    metrics: dict


def _models():
    logistic = CalibratedClassifierCV(
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=0.3)),
            ]
        ),
        cv=TimeSeriesSplit(n_splits=5),
        method="sigmoid",
    )
    boosting = CalibratedClassifierCV(
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=350,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        cv=TimeSeriesSplit(n_splits=5),
        method="sigmoid",
    )
    return logistic, boosting


def _metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def train_walk_forward(games: pd.DataFrame, feature_cols: list[str], test_fraction: float = 0.20):
    data = games.dropna(subset=["target_home_win"]).copy()
    data = data.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    n_test = max(1, int(len(data) * test_fraction))
    train = data.iloc[:-n_test]
    test = data.iloc[-n_test:]

    X_train, y_train = train[feature_cols], train["target_home_win"]
    X_test, y_test = test[feature_cols], test["target_home_win"]

    logistic, boosting = _models()
    logistic.fit(X_train, y_train)
    boosting.fit(X_train, y_train)

    p1 = logistic.predict_proba(X_test)[:, 1]
    p2 = boosting.predict_proba(X_test)[:, 1]
    p = 0.5 * p1 + 0.5 * p2

    metrics = {
        "logistic": _metrics(y_test, p1),
        "boosting": _metrics(y_test, p2),
        "ensemble": _metrics(y_test, p),
        "test_start": str(test["GAME_DATE"].min().date()),
        "test_end": str(test["GAME_DATE"].max().date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    bundle = ModelBundle(logistic=logistic, boosting=boosting, features=feature_cols, metrics=metrics)
    return bundle, test.assign(p_logistic=p1, p_boosting=p2, p_ensemble=p)


def save_bundle(bundle: ModelBundle, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(path)


def predict(bundle: ModelBundle, X: pd.DataFrame) -> pd.DataFrame:
    p1 = bundle.logistic.predict_proba(X[bundle.features])[:, 1]
    p2 = bundle.boosting.predict_proba(X[bundle.features])[:, 1]
    probs = np.vstack([p1, p2])

    p = probs.mean(axis=0)
    spread = probs.max(axis=0) - probs.min(axis=0)
    sigma = probs.std(axis=0, ddof=0)

    out = pd.DataFrame(
        {
            "p_home_win": p,
            "model_agreement_sd": sigma,
            "model_probability_range": spread,
            "confidence_index": np.abs(p - 0.5) * 2,
        },
        index=X.index,
    )
    return out
