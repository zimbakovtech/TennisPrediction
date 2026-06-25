"""Central path configuration for the TennisPrediction pipeline.

All paths are anchored to the repository root (the parent of ``src/``) so that
scripts work regardless of the current working directory. Import this module
from anywhere under ``src/`` with ``from config import PATHS`` (``src/`` is on
``sys.path`` when a script under it is executed directly).
"""
from pathlib import Path

# Repository root: this file lives at <root>/src/config.py
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PLAYERS_DIR = DATA_DIR / "players"
TESTING_DIR = DATA_DIR / "testing"

PATHS = {
    "raw": RAW_DIR,
    "processed_matches": PROCESSED_DIR / "all_matches.csv",
    "elo_ratings": PLAYERS_DIR / "player_elo_ratings.csv",
    "wimbledon_test": TESTING_DIR / "wimbledon_2025.csv",
}
