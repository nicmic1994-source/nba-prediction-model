# NBA Predictor

A research-grade starting point for an NBA game prediction system:

1. ingest historical team and player game logs from NBA.com through `nba_api`;
2. build strictly pre-game rolling team features;
3. train a time-ordered probability model;
4. calibrate probabilities;
5. quantify prediction uncertainty with model ensembles;
6. fetch upcoming games and show predictions in Streamlit.

## Design principles

- **No leakage:** every feature for game N must be calculated only from information available before game N.
- **Walk-forward evaluation:** random train/test splits are deliberately avoided.
- **Probability calibration:** accuracy alone is not enough; evaluate log loss, Brier score and calibration.
- **Era-aware data:** the core model uses statistics that can be built across the full historical period. Tracking/play-by-play features are optional modern-era enrichments.
- **Player availability:** production predictions should include expected minutes and player availability at prediction time.

## Data coverage

NBA.com exposes season-level player and team game logs through public endpoints used by `nba_api`. Some advanced/tracking fields do not exist for every historical era, so the pipeline treats them as optional rather than silently imputing invented values.

For a complete historical backfill, run one season at a time and cache raw responses locally. The NBA endpoints can be slow/rate-limited and may change, so the raw layer is designed to be restartable.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Example: ingest one regular season
python scripts/ingest_season.py --season 2025-26 --season-type "Regular Season"

# Build model features
python scripts/build_features.py

# Train/evaluate using a walk-forward split
python scripts/train_model.py

# Start the dashboard
streamlit run app.py
```

## Recommended production data layers

### Core historical
- games
- team_game_stats
- player_game_stats
- teams
- players
- team_aliases / franchise history

### Modern enrichment
- play-by-play
- shot locations
- lineup/rotation data
- player tracking
- injury/availability snapshots
- probable starters

### Prediction-time context
- rest days
- back-to-back / 3-in-4 / 4-in-6
- home/away
- travel distance/time-zone change
- current roster and projected minutes
- injuries / questionable players
- playoff/regular season flag

## Dashboard metrics

For each upcoming game the UI should show:

- **Win probability:** calibrated probability for the selected team.
- **Model uncertainty:** 90% interval from the ensemble.
- **Confidence spread:** width of that interval; smaller means the model is more internally consistent.
- **Model agreement:** dispersion across component models.
- **Key drivers:** largest positive/negative feature contributions.

Do not call `max(p, 1-p)` “confidence”. That is only the model's implied probability of its chosen side. Confidence should be displayed separately from probability.
