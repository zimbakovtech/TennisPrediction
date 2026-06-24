# upload matches_with_elo.csv and combined_raw_matches.csv to Colab before running

import math
from typing import Tuple

import numpy as np
import pandas as pd
from numba import cuda


@cuda.jit
def _rolling_means_kernel(ace, dfv, bp, won, group_start, window,
                          out_ace, out_dfv, out_bp, out_wins):
    idx = cuda.grid(1)
    n = ace.shape[0]
    if idx >= n:
        return

    start = group_start[idx]
    # left bound of window across PREVIOUS rows only (shifted by 1)
    left = idx - window
    if left < start:
        left = start

    sum_ace = 0.0
    sum_df = 0.0
    sum_bp = 0.0
    sum_w = 0.0

    cnt_ace = 0
    cnt_df = 0
    cnt_bp = 0
    cnt_w = 0

    # iterate over up to `window` previous elements in the same player's group
    for j in range(left, idx):
        v = ace[j]
        if not math.isnan(v):
            sum_ace += v
            cnt_ace += 1

        v = dfv[j]
        if not math.isnan(v):
            sum_df += v
            cnt_df += 1

        v = bp[j]
        if not math.isnan(v):
            sum_bp += v
            cnt_bp += 1

        v = won[j]
        if not math.isnan(v):
            sum_w += v
            cnt_w += 1

    # If there were no valid previous values (min_periods=1 semantics), set NaN
    out_ace[idx] = sum_ace / cnt_ace if cnt_ace > 0 else float('nan')
    out_dfv[idx] = sum_df / cnt_df if cnt_df > 0 else float('nan')
    out_bp[idx] = sum_bp / cnt_bp if cnt_bp > 0 else float('nan')
    out_wins[idx] = sum_w if cnt_w > 0 else float('nan')


def _build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    # Winner side
    winner_df = df[["player_id", "w_ace", "w_df", "w_bpSaved"]].copy()
    winner_df.columns = ["player_id", "ace", "df", "bpSaved"]
    winner_df["won"] = 1.0
    winner_df["original_index"] = df.index

    # Loser side
    loser_df = df[["opponent_id", "l_ace", "l_df", "l_bpSaved"]].copy()
    loser_df.columns = ["player_id", "ace", "df", "bpSaved"]
    loser_df["won"] = 0.0
    loser_df["original_index"] = df.index

    # Combine
    long_df = pd.concat([winner_df, loser_df], ignore_index=True)
    return long_df


def _prepare_sorted_arrays(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    # Sort per player then by original index to match group.sort_values('original_index')
    long_sorted = long_df.sort_values(["player_id", "original_index"]).reset_index(drop=True)

    n = len(long_sorted)
    player_vals = long_sorted["player_id"].to_numpy()

    # group_start[i] = start index of the player's segment that contains i
    group_start = np.empty(n, dtype=np.int32)
    start = 0
    group_start[0] = 0
    for i in range(1, n):
        if player_vals[i] != player_vals[i - 1]:
            start = i
        group_start[i] = start

    return long_sorted, group_start


def generate_stats(df: pd.DataFrame, window: int = 10, lookback: int = 1000) -> pd.DataFrame:
    """
    GPU implementation of rolling stats identical to generate_stats_parallel logic,
    using Numba CUDA. Requires a CUDA-enabled environment (e.g., Google Colab GPU).

    The logic mirrors:
    - For each player, order by original_index
    - Use shifted stats (exclude current match) over a rolling window with min_periods=1
    - Compute means for ace/df/bpSaved and sum for recent wins
    - Merge back to original df for player/opponent side, then compute diffs
    """
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available. Enable a GPU runtime (e.g., in Colab: Runtime -> Change runtime type -> GPU).")

    # 1) Build long player-match view
    long_df = _build_long_df(df)

    # 2) Sort and build group metadata
    long_sorted, group_start = _prepare_sorted_arrays(long_df)

    # 3) Prepare numeric arrays for GPU (float64 for numerical stability)
    ace = long_sorted["ace"].to_numpy(dtype=np.float64)
    dfv = long_sorted["df"].to_numpy(dtype=np.float64)
    bp = long_sorted["bpSaved"].to_numpy(dtype=np.float64)
    won = long_sorted["won"].to_numpy(dtype=np.float64)

    # Allocate outputs
    n = ace.shape[0]
    out_ace = np.empty(n, dtype=np.float64)
    out_dfv = np.empty(n, dtype=np.float64)
    out_bp = np.empty(n, dtype=np.float64)
    out_wins = np.empty(n, dtype=np.float64)

    # Copy to device
    d_ace = cuda.to_device(ace)
    d_dfv = cuda.to_device(dfv)
    d_bp = cuda.to_device(bp)
    d_won = cuda.to_device(won)
    d_group_start = cuda.to_device(group_start)

    d_out_ace = cuda.device_array_like(out_ace)
    d_out_dfv = cuda.device_array_like(out_dfv)
    d_out_bp = cuda.device_array_like(out_bp)
    d_out_wins = cuda.device_array_like(out_wins)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    _rolling_means_kernel[blocks, threads_per_block](
        d_ace, d_dfv, d_bp, d_won, d_group_start, int(window),
        d_out_ace, d_out_dfv, d_out_bp, d_out_wins
    )

    # Copy back
    ace_avg = d_out_ace.copy_to_host()
    df_avg = d_out_dfv.copy_to_host()
    bp_avg = d_out_bp.copy_to_host()
    recent_wins = d_out_wins.copy_to_host()

    # Match pandas semantics: round means to 2 decimals; wins: NaN -> 0 then int
    ace_avg = np.around(ace_avg, 2)
    df_avg = np.around(df_avg, 2)
    bp_avg = np.around(bp_avg, 2)

    recent_wins = np.where(np.isnan(recent_wins), 0, recent_wins).astype(np.int64)

    # 4) Rebuild long stats DataFrame in the same schema as CPU version
    stats_long = pd.DataFrame({
        "original_index": long_sorted["original_index"].values,
        "player_id": long_sorted["player_id"].values,
        "ace_avg": ace_avg,
        "df_avg": df_avg,
        "bp_avg": bp_avg,
        "recent_wins": recent_wins,
    })

    # 5) Merge back to the original dataframe exactly like the CPU version
    df_with_index = df.copy()
    df_with_index["match_index"] = df.index

    # Player (winner side)
    df_out = (
        df_with_index.merge(
            stats_long,
            left_on=["player_id", "match_index"],
            right_on=["player_id", "original_index"],
            how="left",
            suffixes=("", "_p"),
        )
        .rename(
            columns={
                "ace_avg": "player_ace",
                "df_avg": "player_df",
                "bp_avg": "player_bp",
                "recent_wins": f"player_wins_last_{window}",
            }
        )
    )

    # Opponent (loser side)
    df_out = (
        df_out.merge(
            stats_long,
            left_on=["opponent_id", "match_index"],
            right_on=["player_id", "original_index"],
            how="left",
            suffixes=("", "_o"),
        )
        .rename(
            columns={
                "ace_avg": "opponent_ace",
                "df_avg": "opponent_df",
                "bp_avg": "opponent_bp",
                "recent_wins": f"opponent_wins_last_{window}",
            }
        )
    )

    # Clean-up columns introduced by merges
    cols_to_drop = [
        "original_index",
        "original_index_p",
        "player_id_p",
        "player_id_o",
        "original_index_o",
        "match_index",
    ]
    df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns])

    # Compute diffs, rounding consistent with CPU version
    df_out["ace_diff"] = (df_out["player_ace"] - df_out["opponent_ace"]).round(5)
    df_out["df_diff"] = -(df_out["player_df"] - df_out["opponent_df"]).round(5)
    df_out["bp_diff"] = (df_out["player_bp"] - df_out["opponent_bp"]).round(5)

    return df_out
