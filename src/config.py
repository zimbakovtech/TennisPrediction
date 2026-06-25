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
MODELS_DIR = DATA_DIR / "models"

# Supported tours. Raw season files are named ``<tour>_matches_<year>.csv`` for
# both (Jeff Sackmann uses the same schema/naming for tennis_atp and tennis_wta),
# so the only thing that varies per tour is the file prefix and the output paths.
TOURS = ("atp", "wta")


def processed_path(tour: str) -> Path:
    """Processed (feature-engineered, mirrored) match file for a tour."""
    return PROCESSED_DIR / f"{tour}_all_matches.csv"


def elo_path(tour: str) -> Path:
    """Final global Elo-ratings dump for a tour."""
    return PLAYERS_DIR / f"{tour}_player_elo_ratings.csv"


def model_path(tour: str) -> Path:
    """Serialized (joblib) trained model for a tour."""
    return MODELS_DIR / f"{tour}_model.joblib"


PATHS = {
    "raw": RAW_DIR,
    # Defaults point at ATP so existing callers keep their behaviour; the
    # per-tour helpers above are the preferred way to address a specific tour.
    "processed_matches": processed_path("atp"),
    "elo_ratings": elo_path("atp"),
    "wimbledon_test": TESTING_DIR / "wimbledon_2025.csv",
}
