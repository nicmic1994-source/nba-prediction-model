from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2


def current_nba_season() -> str:
    now = datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_schedule(season: str | None = None) -> pd.DataFrame:
    season = season or current_nba_season()
    endpoint = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00")
    return endpoint.season_games.get_data_frame()


def build_upcoming_features(
    schedule: pd.DataFrame,
    team_history: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    schedule = schedule.copy()
    schedule["gameDateTimeUTC"] = pd.to_datetime(schedule["gameDateTimeUTC"], utc=True, errors="coerce")
    upcoming = schedule[
        (schedule["gameDateTimeUTC"] >= now)
        & (schedule["gameStatus"] != 3)
    ].copy()

    # Latest available state for each team. Once current-season games exist, those rows\
    # automatically supersede the previous-season prior.
    history = team_history.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])
    latest = history.groupby("TEAM_ID", as_index=False).tail(1).set_index("TEAM_ID")

    rows: list[dict] = []
    for _, game in upcoming.iterrows():
        home_id = int(game["homeTeam_teamId"])
        away_id = int(game["awayTeam_teamId"])
        if home_id not in latest.index or away_id not in latest.index:
            continue

        h = latest.loc[home_id]
        a = latest.loc[away_id]
        row = {
            "GAME_ID": str(game["gameId"]),
            "GAME_DATE": game["gameDateTimeUTC"],
            "HOME_TEAM": game["homeTeam_teamTricode"],
            "AWAY_TEAM": game["awayTeam_teamTricode"],
        }
        for c in feature_cols:
            if not c.startswith("diff_"):
                continue
            metric = c[len("diff_"):]
            row[c] = h.get(metric, float("nan")) - a.get(metric, float("nan"))
        row["home_court"] = 1.0
        rows.append(row)

    return pd.DataFrame(rows)


def update_upcoming(
    history_path: str | Path = "data/processed/team_history.parquet",
    model_path: str | Path = "models/nba_model.joblib",
    output_path: str | Path = "data/processed/upcoming_features.parquet",
    season: str | None = None,
) -> pd.DataFrame:
    from .model import load_bundle

    history = pd.read_parquet(history_path)
    bundle = load_bundle(model_path)
    schedule = fetch_schedule(season)
    out = build_upcoming_features(schedule, history, bundle.features)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out
