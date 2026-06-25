import pandas as pd
import numpy as np

# Map an output stat name to the (winner_column, loser_column) it comes from.
STAT_SOURCES = {
    "ace": ("w_ace", "l_ace"),
    "df": ("w_df", "l_df"),
    "bp": ("w_bpSaved", "l_bpSaved"),
}


def generate_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling pre-match form averages for player and opponent.

    For every match we compute each competitor's mean over their own previous
    (up to ``window``) matches, strictly excluding the current match
    (``shift(1)``), so there is no leakage. Implemented with groupby-rolling on
    a long-format frame keyed by player id -- O(n) instead of the old O(n^2)
    row-wise loop.

    The input MUST already be in chronological order (handled globally in
    ``processing_data``). Players with no prior matches get NaN, which the
    gradient-boosted trees handle natively.
    """
    df = df.reset_index(drop=True).copy()
    df["_match_order"] = np.arange(len(df))

    # Build a long frame: one record per (match, role), carrying that
    # competitor's own stats for the match (winner -> w_*, loser -> l_*).
    frames = []
    for role, id_col in (("player", "player_id"), ("opponent", "opponent_id")):
        is_winner = role == "player"
        rec = pd.DataFrame({
            "_match_order": df["_match_order"],
            "role": role,
            "pid": df[id_col],
        })
        for stat, (w_col, l_col) in STAT_SOURCES.items():
            rec[stat] = df[w_col] if is_winner else df[l_col]
        frames.append(rec)

    long = pd.concat(frames, ignore_index=True)
    # Stable sort so that within each player rows stay in chronological order.
    long = long.sort_values(["pid", "_match_order"], kind="mergesort")

    grp = long.groupby("pid", sort=False)
    for stat in STAT_SOURCES:
        long[f"{stat}_avg"] = (
            grp[stat]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .round(2)
        )

    # Pivot the rolling averages back onto the original match rows.
    player_avg = long[long["role"] == "player"].set_index("_match_order")
    opp_avg = long[long["role"] == "opponent"].set_index("_match_order")

    df["player_ace_avg"] = df["_match_order"].map(player_avg["ace_avg"])
    df["player_df_avg"] = df["_match_order"].map(player_avg["df_avg"])
    df["player_bpSaved_avg"] = df["_match_order"].map(player_avg["bp_avg"])
    df["opponent_ace_avg"] = df["_match_order"].map(opp_avg["ace_avg"])
    df["opponent_df_avg"] = df["_match_order"].map(opp_avg["df_avg"])
    df["opponent_bpSaved_avg"] = df["_match_order"].map(opp_avg["bp_avg"])

    # Difference features (player - opponent). df_diff is negated because fewer
    # double faults is better.
    df["ace_diff"] = (df["player_ace_avg"] - df["opponent_ace_avg"]).round(5)
    df["df_diff"] = (-(df["player_df_avg"] - df["opponent_df_avg"])).round(5)
    df["bp_diff"] = (df["player_bpSaved_avg"] - df["opponent_bpSaved_avg"]).round(5)

    return df.drop(columns="_match_order")
