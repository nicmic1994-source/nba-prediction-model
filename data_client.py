from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


SeasonType = Literal[
    "Regular Season",
    "Playoffs",
    "Pre Season",
    "All Star",
]


def _request_with_retry(
    endpoint_factory,
    *,
    tries: int = 5,
    base_sleep: float = 2.0,
):
    last_error: Exception | None = None

    for attempt in range(tries):
        try:
            return endpoint_factory()
        except Exception as exc:
            last_error = exc

            if attempt == tries - 1:
                raise

            time.sleep(base_sleep * (2 ** attempt))

    raise RuntimeError("Unreachable") from last_error


def fetch_team_game_logs(
    season: str,
    season_type: SeasonType = "Regular Season",
) -> pd.DataFrame:

    endpoint = _request_with_retry(
        lambda: leaguegamelog.LeagueGameLog(
            counter=0,
            date_from_nullable="",
            date_to_nullable="",
            direction="ASC",
            league_id="00",
            player_or_team_abbreviation="T",
            season=season,
            season_type_all_star=season_type,
            sorter="DATE",
        )
    )

    df = endpoint.get_data_frames()[0]
    df["SEASON_TYPE"] = season_type

    return df


def ingest_season(
    season: str,
    season_type: SeasonType = "Regular Season",
    raw_dir: str | Path = "data/raw",
) -> Path:

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    path = (
        raw_dir
        / f"team_game_logs_{season}_{season_type.lower().replace(' ', '_')}.parquet"
    )

    if not path.exists():
        df = fetch_team_game_logs(season, season_type)
        df.to_parquet(path, index=False)

    return path
