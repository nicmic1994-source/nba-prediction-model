from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents))
from src.features import build_team_history, make_matchup_dataset

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

team_files = sorted(RAW.glob("team_game_logs_*.parquet"))
if not team_files:
    raise SystemExit("No raw team game logs found. Run scripts/ingest_season.py first.")

team_df = pd.concat([pd.read_parquet(p) for p in team_files], ignore_index=True)
history = build_team_history(team_df)
games = make_matchup_dataset(history)
history.to_parquet(OUT / "team_history.parquet", index=False)
games.to_parquet(OUT / "matchups.parquet", index=False)
print(f"team history rows: {len(history):,}")
print(f"matchup rows:      {len(games):,}")
