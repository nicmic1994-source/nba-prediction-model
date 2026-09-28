from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_client import ingest_season


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        required=True,
        help="Example: 2025-26",
    )

    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=[
            "Regular Season",
            "Playoffs",
            "Pre Season",
            "All Star",
        ],
    )

    args = parser.parse_args()

    path = ingest_season(
        args.season,
        args.season_type,
    )

    print(f"saved: {path}")
