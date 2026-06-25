import pandas as pd
import numpy as np

# Columns whose player/opponent values must be swapped in the mirrored row.
SWAP_PAIRS = [
    ("player_id", "opponent_id"),
    ("player_elo_before", "opponent_elo_before"),
]

# Difference columns whose sign must flip in the mirrored row.
DIFF_COLS = [
    "rank_diff", "points_diff", "seed_diff", "age_diff", "height_diff",
    "ace_diff", "df_diff", "bp_diff", "h2h_diff", "surface_elo_diff",
]


def duplicate_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror every match to the opponent's perspective for a balanced target.

    Each input row (player beat opponent, ``win_loss == 1``) is paired with a
    mirrored row from the opponent's point of view: player/opponent identifiers
    and elo are swapped, difference features are negated, and the target is
    flipped to 0. Original and mirror are kept ADJACENT (rows 2i and 2i+1) so
    downstream code can treat ``index // 2`` as a stable match id (used by the
    evaluation split to keep a pair together).

    Note: difference columns must be neutral-at-zero and antisymmetric for this
    sign flip to be valid; elo columns are swapped, not negated, because they
    are absolute ratings.
    """
    mirrored = df.copy()

    for col_a, col_b in SWAP_PAIRS:
        if col_a in mirrored.columns and col_b in mirrored.columns:
            mirrored[[col_a, col_b]] = mirrored[[col_b, col_a]]

    for col in DIFF_COLS:
        if col in mirrored.columns:
            mirrored[col] = -mirrored[col]

    mirrored["win_loss"] = 1 - mirrored["win_loss"]

    # Interleave original (even indices) and mirrored (odd indices).
    n = len(df)
    df = df.copy()
    df.index = np.arange(0, 2 * n, 2)
    mirrored.index = np.arange(1, 2 * n, 2)

    combined_df = pd.concat([df, mirrored]).sort_index().reset_index(drop=True)
    return combined_df
