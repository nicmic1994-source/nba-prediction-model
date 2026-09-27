from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model import load_bundle, predict

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")
st.title("NBA Game Predictor")
st.caption("Calibrated win probability + model uncertainty. Historical features are strictly pre-game.")

MODEL_PATH = Path("models/nba_model.joblib")
UPCOMING = Path("data/processed/upcoming_features.parquet")

if not MODEL_PATH.exists():
    st.warning("Train the model first: `python scripts/train_model.py`")
    st.stop()

if not UPCOMING.exists():
    st.info("Create `data/processed/upcoming_features.parquet` from the current schedule + latest team states.")
    st.stop()

bundle = load_bundle(MODEL_PATH)
upcoming = pd.read_parquet(UPCOMING)
pred = predict(bundle, upcoming)
view = upcoming[[c for c in ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"] if c in upcoming.columns]].copy()
view = pd.concat([view.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

view["home_win_pct"] = (100 * view["p_home_win"]).round(1)
view["away_win_pct"] = (100 * (1 - view["p_home_win"])).round(1)
view["confidence_spread_pct"] = (100 * view["model_probability_range"]).round(1)
view["confidence_index_pct"] = (100 * view["confidence_index"]).round(1)
view["model_agreement_pct_sd"] = (100 * view["model_agreement_sd"]).round(1)

st.subheader("Upcoming games")
st.dataframe(
    view[[
        c for c in [
            "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
            "home_win_pct", "away_win_pct",
            "confidence_index_pct", "confidence_spread_pct",
            "model_agreement_pct_sd",
        ] if c in view.columns
    ]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Model validation")
st.json(bundle.metrics)
