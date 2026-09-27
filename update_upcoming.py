from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.schedule import update_upcoming

out = update_upcoming()
print(f"upcoming games: {len(out):,}")
print(out[[c for c in ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"] if c in out.columns]].to_string(index=False))
