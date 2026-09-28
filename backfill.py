from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents))
from data_client import ingest_season


def season_range(start_year: int, end_year: int):
    for y in range(start_year, end_year + 1):
        yield f"{y}-{str(y + 1)[-2:]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1946)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.5)
    args = parser.parse_args()

    season_types = ["Regular Season", "Playoffs"] if args.include_playoffs else ["Regular Season"]
    for season in season_range(args.start_year, args.end_year):
        for season_type in season_types:
            print(f"\n=== {season} / {season_type} ===")
            try:
                paths = ingest_season(season, season_type)
                print(paths[0])
                print(paths[1])
            except Exception as exc:
                print(f"FAILED {season} / {season_type}: {exc}")
            time.sleep(args.sleep)
