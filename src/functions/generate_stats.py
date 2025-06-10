import pandas as pd


def generate_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    # Columns for which to calculate rolling averages
    stats_cols = [
        "w_ace",
        "l_ace",
        "w_df",
        "l_df",
        "w_bpSaved",
        "l_bpSaved",
    ]

    # Group by player, shift to exclude current match, then compute rolling mean
    rolling_avgs = (
        df
        .groupby("player_id")[stats_cols]
        .apply(
            lambda grp: grp
            .shift(1)  # exclude the current match from its own average
            .rolling(window=window, min_periods=1)
            .mean()
            .round(2)
        )
        .reset_index(level=0, drop=True)
    )

    # Rename columns to indicate they're averages
    rolling_avgs.columns = [f"{col}_avg" for col in rolling_avgs.columns]

    # Concatenate rolling averages alongside the original data
    result_df = pd.concat([df, rolling_avgs], axis=1)

    return result_df
