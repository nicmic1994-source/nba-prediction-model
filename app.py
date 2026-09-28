from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_client import fetch_team_game_logs
from src.features import build_team_history, make_matchup_dataset
from src.model import train_walk_forward


st.set_page_config(
    page_title="NBA Predictor",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 NBA Game Predictor")

st.caption(
    "Proof of concept using NBA regular-season team data "
    "from 2010–11 onward."
)


# ---------------------------------------------------------
# Download historical data
# ---------------------------------------------------------

@st.cache_data
def load_historical_data():

    seasons = [
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(2010, 2026)
    ]

    frames = []

    progress = st.progress(0)
    status = st.empty()

    for i, season in enumerate(seasons):

        status.write(
            f"Downloading {season}..."
        )

        df = fetch_team_game_logs(
            season,
            "Regular Season",
        )

        frames.append(df)

        progress.progress(
            (i + 1) / len(seasons)
        )

    progress.empty()
    status.empty()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------
# Build features
# ---------------------------------------------------------

@st.cache_data
def build_training_data(raw_data):

    history = build_team_history(
        raw_data
    )

    games = make_matchup_dataset(
        history
    )

    return games


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

@st.cache_resource
def train_model(games):

    feature_cols = [
        c
        for c in games.columns
        if c.startswith("diff_")
    ]

    feature_cols.append("home_court")

    feature_cols = [
        c
        for c in feature_cols
        if c in games.columns
    ]

    bundle, test = train_walk_forward(
        games,
        feature_cols,
    )

    return bundle, test


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

if st.button(
    "Build / refresh model",
    type="primary",
):

    st.session_state["build_model"] = True


if "build_model" not in st.session_state:

    st.info(
        "Click **Build / refresh model** to download the "
        "historical data and train the prediction model."
    )

    st.stop()


with st.spinner(
    "Downloading NBA historical data..."
):

    raw_data = load_historical_data()


st.success(
    f"Historical data loaded: "
    f"{len(raw_data):,} team-game records."
)


with st.spinner(
    "Building pre-game features..."
):

    games = build_training_data(
        raw_data
    )


st.success(
    f"Training dataset created: "
    f"{len(games):,} games."
)


with st.spinner(
    "Training prediction model..."
):

    bundle, test = train_model(
        games
    )


st.success(
    "Model trained successfully."
)


# ---------------------------------------------------------
# Model validation
# ---------------------------------------------------------

st.subheader("Model validation")

metrics = bundle.metrics

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Test games",
    f"{metrics['test_games']:,}",
)

col2.metric(
    "Accuracy",
    f"{metrics['accuracy']:.1%}",
)

col3.metric(
    "Log loss",
    f"{metrics['log_loss']:.3f}",
)

col4.metric(
    "Brier score",
    f"{metrics['brier_score']:.3f}",
)


st.write(
    f"Test period: "
    f"{metrics['test_start']} → "
    f"{metrics['test_end']}"
)


# ---------------------------------------------------------
# Historical predictions
# ---------------------------------------------------------

st.subheader("Model status")

st.write(
    f"The model uses {len(bundle.features)} "
    f"pre-game features and "
    f"{metrics['ensemble_models']} models."
)


with st.expander("Features used"):

    st.write(bundle.features)


with st.expander("Full model metrics"):

    st.json(metrics)
