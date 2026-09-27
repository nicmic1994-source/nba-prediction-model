from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import train_walk_forward, save_bundle
from sklearn.inspection import permutation_importance

GAMES = Path("data/processed/matchups.parquet")
OUT = Path("models/nba_model.joblib")

games = pd.read_parquet(GAMES)
feature_cols = [c for c in games.columns if c.startswith("diff_")] + ["home_court"]
feature_cols = [c for c in feature_cols if c in games.columns]

bundle, test = train_walk_forward(games, feature_cols)
save_bundle(bundle, OUT)
print(bundle.metrics)

# Feature influence is measured on the untouched chronological test set rather than the training set.
X_test = test[bundle.features]
y_test = test["target_home_win"]
perm = permutation_importance(
    bundle.boosting,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="neg_log_loss",
)
importance = pd.DataFrame({
    "feature": bundle.features,
    "importance_mean": perm.importances_mean,
    "importance_sd": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
importance.to_csv("data/processed/feature_importance.csv", index=False)

print(importance.head(20).to_string(index=False))
print(f"saved: {OUT}")
