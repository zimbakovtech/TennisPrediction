"""Score upcoming ATP fixtures with the repository's surface-aware Elo tables.

The pipeline maintains pre-match Elo ratings (global and per-surface) keyed by
Sackmann ``player_id``, but nothing in the repo tells you which matches are
NEXT. This script closes that gap: it fetches upcoming ATP fixtures from the
Live Tennis API, joins both players against the committed Elo tables (via a
name -> player_id map built from the ``player_name``/``opponent_name`` columns
of ``data/raw/atp_matches_*.csv``), and emits
surface-aware Elo win probabilities as a CSV plus a printed table.

Vendor disclosure: this script was contributed by the Live Tennis API team
(https://livetennisapi.com). It uses only the free keyed tier (30 requests/
minute, 100 requests/day -- one fixtures call per run, so the daily quota is
a non-issue here). Free keys: https://livetennisapi.com/subscribe/free

Usage::

    export LIVETENNIS_API_KEY=...
    python src/predict_upcoming.py            # writes data/testing/upcoming_predictions.csv

Only fixtures where BOTH players resolve to a known player_id are scored;
unresolved fixtures are reported so the miss rate is visible, not silent.
"""
import json
import logging
import os
import unicodedata
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import PATHS, PLAYERS_DIR, RAW_DIR, TESTING_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LIVETENNIS_BASE_URL = "https://api.livetennisapi.com/api/public/v1"
API_KEY_ENV = "LIVETENNIS_API_KEY"

# Fixture surface strings -> per-surface Elo columns in
# data/players/player_surface_elo_ratings.csv (K=32, same as training).
SURFACE_COLUMNS = {
    "hard": "hard_elo",
    "clay": "clay_elo",
    "grass": "grass_elo",
}

OUTPUT_PATH = TESTING_DIR / "upcoming_predictions.csv"


def _normalize_name(name: str) -> str:
    """Accent-insensitive, lowercased full name ("Novak Djokovic" style)."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.lower().split())


def build_name_index() -> Dict[str, int]:
    """Map normalized player names -> Sackmann player_id from the raw CSVs."""
    index: Dict[str, int] = {}
    for raw_csv in sorted(RAW_DIR.glob("atp_matches_*.csv")):
        frame = pd.read_csv(
            raw_csv,
            usecols=["player_id", "player_name", "opponent_id", "opponent_name"],
        )
        for id_col, name_col in (
            ("player_id", "player_name"),
            ("opponent_id", "opponent_name"),
        ):
            for player_id, name in zip(frame[id_col], frame[name_col]):
                if pd.notna(player_id) and isinstance(name, str):
                    index[_normalize_name(name)] = int(player_id)
    logger.info("Name index built: %d distinct player names", len(index))
    return index


def fetch_upcoming_fixtures(limit: int = 50) -> List[dict]:
    """Upcoming ATP fixtures from the Live Tennis API (free keyed tier)."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise SystemExit(
            f"{API_KEY_ENV} is not set. Free keys: "
            "https://livetennisapi.com/subscribe/free"
        )
    query = urllib.parse.urlencode({"tour": "atp", "limit": limit})
    request = urllib.request.Request(
        f"{LIVETENNIS_BASE_URL}/fixtures?{query}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    fixtures = payload.get("data", []) if isinstance(payload, dict) else []
    logger.info("Fetched %d upcoming ATP fixtures", len(fixtures))
    return fixtures


def load_elo_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    surface = pd.read_csv(
        PLAYERS_DIR / "player_surface_elo_ratings.csv"
    ).set_index("player_id")
    global_elo = pd.read_csv(PATHS["elo_ratings"]).set_index("player_id")
    return surface, global_elo


def elo_for(
    player_id: int,
    surface_column: Optional[str],
    surface: pd.DataFrame,
    global_elo: pd.DataFrame,
) -> Optional[float]:
    """Surface-specific Elo when available, otherwise global, otherwise None."""
    if surface_column and player_id in surface.index:
        return float(surface.loc[player_id, surface_column])
    if player_id in global_elo.index:
        return float(global_elo.loc[player_id, "elo"])
    return None


def win_probability(elo_a: float, elo_b: float) -> float:
    """Standard Elo expectation: P(A beats B)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def score_fixtures() -> pd.DataFrame:
    name_index = build_name_index()
    surface_table, global_table = load_elo_tables()
    fixtures = fetch_upcoming_fixtures()

    rows: List[dict] = []
    unresolved = 0
    for fixture in fixtures:
        p1_name = fixture.get("player1_name") or ""
        p2_name = fixture.get("player2_name") or ""
        p1_id = name_index.get(_normalize_name(p1_name))
        p2_id = name_index.get(_normalize_name(p2_name))
        if p1_id is None or p2_id is None:
            unresolved += 1
            continue

        surface_raw = (fixture.get("surface") or "").lower()
        surface_column = SURFACE_COLUMNS.get(surface_raw)
        elo1 = elo_for(p1_id, surface_column, surface_table, global_table)
        elo2 = elo_for(p2_id, surface_column, surface_table, global_table)
        if elo1 is None or elo2 is None:
            unresolved += 1
            continue

        rows.append(
            {
                "event_date": fixture.get("event_date"),
                "tournament": fixture.get("tournament"),
                "round": fixture.get("round"),
                "surface": surface_raw or "unknown",
                "elo_basis": surface_column or "global",
                "player1": p1_name,
                "player2": p2_name,
                "elo1": round(elo1, 1),
                "elo2": round(elo2, 1),
                "p1_win_probability": round(win_probability(elo1, elo2), 4),
            }
        )

    logger.info(
        "Scored %d fixtures (%d unresolved: player or Elo not in the tables)",
        len(rows),
        unresolved,
    )
    return pd.DataFrame(rows)


def main() -> None:
    predictions = score_fixtures()
    if predictions.empty:
        logger.warning("No fixtures could be scored -- nothing written.")
        return
    predictions = predictions.sort_values(
        ["event_date", "tournament"], na_position="last"
    )
    TESTING_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %s", OUTPUT_PATH)
    with pd.option_context("display.max_rows", 50, "display.width", 160):
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
