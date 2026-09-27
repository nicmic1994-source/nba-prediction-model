from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class ModelBundle:
    features: list[str]
    boosting: object
    ensemble: list[object]
    metrics: dict


def _make_boosting(random_state: int = 42):
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=random_state,
    )


def _make_logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            max_iter=2000,
        ),
    )


def _prepare_xy(df: pd.DataFrame, features: list[str]):
    data = df.dropna(subset=features + ["target_home_win"]).copy()

    X = data[features].astype(float)
    y = data["target_home_win"].astype(int)

    return X, y, data


def train_walk_forward(
    games: pd.DataFrame,
    feature_cols: list[str],
):
    """
    Train a time-ordered ensemble.

    The final chronological portion of the data is held out as an untouched
    test set. No random train/test split is used.
    """

    games = games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values(
        ["GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    features = [
        c for c in feature_cols
        if c in games.columns
    ]

    if not features:
        raise ValueError("No usable model features were found.")

    X, y, data = _prepare_xy(games, features)

    if len(data) < 100:
        raise ValueError(
            f"Only {len(data)} usable games are available. "
            "At least 100 games are recommended."
        )

    # Hold out the final 20% chronologically.
    split = int(len(data) * 0.80)

    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    X_train = train[features].astype(float)
    y_train = train["target_home_win"].astype(int)

    X_test = test[features].astype(float)
    y_test = test["target_home_win"].astype(int)

    # Main model used by permutation importance.
    boosting = _make_boosting(42)
    boosting.fit(X_train, y_train)

    # Ensemble models with different random seeds / model specifications.
    ensemble = []

    for seed in [11, 22, 33, 44, 55]:
        model = _make_boosting(seed)
        model.fit(X_train, y_train)
        ensemble.append(model)

    # Include a logistic model to provide a structurally different model.
    logistic = _make_logistic()
    logistic.fit(X_train, y_train)
    ensemble.append(logistic)

    probabilities = []

    for model in ensemble:
        probabilities.append(
            model.predict_proba(X_test)[:, 1]
        )

    probabilities = np.asarray(probabilities)

    # Mean ensemble probability.
    p_mean = probabilities.mean(axis=0)

    # Dispersion between component models.
    agreement_sd = probabilities.std(axis=0)

    # 90% empirical interval from ensemble predictions.
    lower = np.percentile(probabilities, 5, axis=0)
    upper = np.percentile(probabilities, 95, axis=0)

    probability_range = upper - lower

    # Convert spread into an intuitive confidence index.
    # 0 = highly dispersed, 1 = highly consistent.
    confidence_index = np.clip(
        1.0 - probability_range,
        0.0,
        1.0,
    )

    metrics = {
        "test_games": int(len(test)),
        "test_start": str(test["GAME_DATE"].min().date()),
        "test_end": str(test["GAME_DATE"].max().date()),
        "accuracy": float(
            accuracy_score(y_test, p_mean >= 0.5)
        ),
        "log_loss": float(
            log_loss(
                y_test,
                np.clip(p_mean, 1e-6, 1 - 1e-6),
            )
        ),
        "brier_score": float(
            brier_score_loss(y_test, p_mean)
        ),
        "ensemble_models": len(ensemble),
    }

    bundle = ModelBundle(
        features=features,
        boosting=boosting,
        ensemble=ensemble,
        metrics=metrics,
    )

    test = test.copy()

    return bundle, test


def predict(
    bundle: ModelBundle,
    upcoming: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate ensemble predictions for upcoming games.
    """

    missing = [
        c for c in bundle.features
        if c not in upcoming.columns
    ]

    if missing:
        raise ValueError(
            "Upcoming data is missing model features: "
            + ", ".join(missing)
        )

    X = upcoming[bundle.features].astype(float)

    probabilities = []

    for model in bundle.ensemble:
        probabilities.append(
            model.predict_proba(X)[:, 1]
        )

    probabilities = np.asarray(probabilities)

    p_mean = probabilities.mean(axis=0)

    lower = np.percentile(
        probabilities,
        5,
        axis=0,
    )

    upper = np.percentile(
        probabilities,
        95,
        axis=0,
    )

    probability_range = upper - lower

    agreement_sd = probabilities.std(axis=0)

    confidence_index = np.clip(
        1.0 - probability_range,
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "p_home_win": p_mean,
            "model_probability_lower": lower,
            "model_probability_upper": upper,
            "model_probability_range": probability_range,
            "confidence_index": confidence_index,
            "model_agreement_sd": agreement_sd,
        }
    )


def save_bundle(
    bundle: ModelBundle,
    path: Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, path)


def load_bundle(
    path: Path,
) -> ModelBundle:
    return joblib.load(path)
  
