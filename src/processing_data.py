import logging
from pathlib import Path
from typing import List

import pandas as pd

from config import PATHS
from functions.duplicate_entries import duplicate_entries
from functions.preprocessing import load_and_preprocess
from feature_engineering.generate_stats import generate_stats
from feature_engineering.calculate_elo import calculate_elo
from feature_engineering.head2head import add_h2h_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Chronological ordering key. tourney_date groups each event; round orders the
# matches within it (early rounds first); match_num breaks remaining ties.
SORT_KEYS = ["tourney_date", "round", "match_num"]

# Rows missing any of these raw fields are unusable and dropped (after the
# time-dependent features have already seen them as history).
KEY_FEATURES = [
    "surface", "player_rank", "opponent_rank",
    "player_rank_points", "opponent_rank_points",
]

# Explicit, ordered final schema. Selecting an allowlist (instead of dropping a
# blocklist) guarantees the training file always matches the inference schema.
FINAL_COLUMNS = [
    "best_of", "match_importance", "rank_diff", "points_diff", "age_diff",
    "win_loss", "ace_diff", "df_diff", "bp_diff", "h2h_diff",
    "player_elo_before", "opponent_elo_before", "surface_elo_diff",
]


def postprocess_and_save(df: pd.DataFrame, output_path: Path) -> None:
    # Drop rows missing key features. Form averages are intentionally left as
    # NaN where a player has no history -- the gradient-boosted trees handle
    # missing values natively, and 0 would be a false "equal form" signal.
    df = df.dropna(subset=KEY_FEATURES)

    # Mirror each match to the opponent's perspective (balanced target).
    df = duplicate_entries(df)

    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing before save: {missing}")

    final_df = df[FINAL_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows x %d cols to %s",
                len(final_df), final_df.shape[1], output_path)


def process_all(data_dir: Path, output_path: Path) -> None:
    files = sorted(data_dir.glob("atp_matches_*.csv"))
    if not files:
        logger.warning("No raw files found in %s", data_dir)
        return

    logger.info("Processing %d files from %s", len(files), data_dir)

    data_frames: List[pd.DataFrame] = []
    for filepath in files:
        try:
            data_frames.append(load_and_preprocess(filepath))
            logger.debug("Loaded and preprocessed %s", filepath)
        except Exception as e:
            logger.error("Error processing %s: %s", filepath, e)

    combined = pd.concat(data_frames, ignore_index=True)

    # Single global chronological sort -- every time-dependent feature below
    # relies on this order.
    combined = combined.sort_values(SORT_KEYS).reset_index(drop=True)

    # Invariant the whole pipeline depends on: pre-mirror, player == winner.
    assert (combined["win_loss"] == 1).all(), "pre-mirror target must be all wins"

    logger.info("Combined %d matches; computing rolling form...", len(combined))
    combined = generate_stats(combined)

    logger.info("Computing head-to-head...")
    combined = add_h2h_stats(combined)

    logger.info("Computing Elo ratings...")
    combined = calculate_elo(combined)

    postprocess_and_save(combined, output_path)


if __name__ == "__main__":
    process_all(PATHS["raw"], PATHS["processed_matches"])
    logger.info("Data processing complete.")
