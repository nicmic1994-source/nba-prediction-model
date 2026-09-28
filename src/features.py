from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def add_box_score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # NBA team-game logs use FGM/FGA, FG3M/FG3A, FTM/FTA, OREB/DREB/REB, AST/TOV etc.
    out["efg_pct"] = safe_div(out["FGM"] + 0.5 * out["FG3M"], out["FGA"])
    out["ts_pct"] = safe_div(out["PTS"], 2 * (out["FGA"] + 0.44 * out["FTA"]))
    out["tov_pct"] = safe_div(out["TOV"], out["FGA"] + 0.44 * out["FTA"] + out["TOV"])
    out["ft_rate"] = safe_div(out["FTA"], out["FGA"])

    # For a team's rebound percentage, use the opposing team's relevant rebound totals later.
    out["points_per_fga"] = safe_div(out["PTS"], out["FGA"])
    return out


def _add_opponent_columns(games: pd.DataFrame) -> pd.DataFrame:
    key = ["GAME_ID"]
    opp = games[["GAME_ID", "TEAM_ID", "REB", "OREB", "DREB", "PTS"]].copy()
    opp = opp.rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "REB": "OPP_REB",
            "OREB": "OPP_OREB",
            "DREB": "OPP_DREB",
            "PTS": "OPP_PTS",
        }
    )

    merged = games.merge(opp, on="GAME_ID", suffixes=("", "_dup"))
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

    merged["orb_pct"] = safe_div(merged["OREB"], merged["OREB"] + merged["OPP_DREB"])
    merged["drb_pct"] = safe_div(merged["DREB"], merged["DREB"] + merged["OPP_OREB"])
    merged["reb_pct"] = safe_div(merged["REB"], merged["REB"] + merged["OPP_REB"])
    merged["net_rating_proxy"] = merged["PTS"] - merged["OPP_PTS"]
    return merged


TEAM_METRICS = [
    "efg_pct",
    "ts_pct",
    "tov_pct",
    "ft_rate",
    "orb_pct",
    "drb_pct",
    "reb_pct",
    "net_rating_proxy",
    "PTS",
    "OPP_PTS",
    "AST",
    "STL",
    "BLK",
    "TOV",
]


def build_team_history(team_logs: pd.DataFrame) -> pd.DataFrame:
    df = add_box_score_metrics(team_logs)
    df = _add_opponent_columns(df)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

if "SEASON_YEAR" not in df.columns and "SEASON_ID" in df.columns:
    df["SEASON_YEAR"] = (
        df["SEASON_ID"]
        .astype(str)
        .str[-4:]
        .astype(int)
    )

df = df.sort_values(
    ["TEAM_ID", "GAME_DATE", "GAME_ID"]
).reset_index(drop=True)

    # Core feature rule: shift first, then roll. This guarantees game N never sees game N's stats.
    for window in (5, 10, 20):
        for metric in TEAM_METRICS:
            df[f"{metric}_l{window}"] = (
                df.groupby("TEAM_ID")[metric]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=max(2, window // 2)).mean())
            )

    # Exponentially weighted form reacts faster than a fixed rolling window.
    for metric in TEAM_METRICS:
        df[f"{metric}_ewm"] = df.groupby("TEAM_ID")[metric].transform(
            lambda s: s.shift(1).ewm(span=15, adjust=False, min_periods=5).mean()
        )

    # Season-to-date baseline, also lagged.
    season_key = ["TEAM_ID", "SEASON_YEAR"] if "SEASON_YEAR" in df else ["TEAM_ID"]
    for metric in TEAM_METRICS:
        df[f"{metric}_season"] = (
            df.groupby(season_key)[metric]
            .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
        )

    return df


def _parse_home(matchup: pd.Series) -> pd.Series:
    # NBA matchup strings are e.g. "LAL vs. BOS" or "LAL @ BOS".
    return ~matchup.str.contains("@", regex=False, na=False)


def make_matchup_dataset(team_history: pd.DataFrame) -> pd.DataFrame:
    df = team_history.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = _parse_home(df["MATCHUP"])

    team_cols = [
        c for c in df.columns
        if any(c.startswith(f"{m}_") for m in TEAM_METRICS)
    ]

    # There are exactly two team rows per completed GAME_ID. Build the matchup from the
    # home row so WL/target are always aligned with the home team.
    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    left_cols = ["GAME_ID", "GAME_DATE", "SEASON_TYPE", "TEAM_ID",
                 "TEAM_ABBREVIATION", "TEAM_NAME", "WL", "PTS", "OPP_PTS"] + team_cols
    left_cols = [c for c in left_cols if c in home.columns]
    right_cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "PTS", "OPP_PTS"] + team_cols
    right_cols = [c for c in right_cols if c in away.columns]

    h = home[left_cols].copy()
    a = away[right_cols].copy()
    a = a.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
        "TEAM_NAME": "AWAY_TEAM_NAME",
        "PTS": "AWAY_PTS",
        "OPP_PTS": "AWAY_OPP_PTS",
        **{c: f"away_{c}" for c in team_cols},
    })

    games = h.merge(a, on="GAME_ID", how="inner", validate="one_to_one")
    games = games.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
        "TEAM_NAME": "HOME_TEAM_NAME",
        "PTS": "HOME_PTS",
        "OPP_PTS": "HOME_OPP_PTS",
    })

    games["target_home_win"] = (games["WL"] == "W").astype(int)
    games["point_diff"] = games["HOME_PTS"] - games["AWAY_PTS"]
    games["home_court"] = 1.0

    feature_cols = []
    for c in team_cols:
        games[f"diff_{c}"] = games[c] - games[f"away_{c}"]
        feature_cols.append(f"diff_{c}")

    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return games
