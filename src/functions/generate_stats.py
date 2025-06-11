import pandas as pd
import numpy as np


def generate_stats(df: pd.DataFrame, window: int = 10, lookback: int = 600) -> pd.DataFrame:
    # Initialize lists for averages
    w_ace_avgs, l_ace_avgs = [], []
    w_df_avgs, l_df_avgs = [], []
    w_bpSaved_avgs, l_bpSaved_avgs = [], []
    # New metrics
    w_1stIn_avgs, l_1stIn_avgs = [], []
    w_1stWon_avgs, l_1stWon_avgs = [], []
    w_2ndWon_avgs, l_2ndWon_avgs = [], []
    w_bpFaced_avgs, l_bpFaced_avgs = [], []
    w_svpt_avgs, l_svpt_avgs = [], []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        start_idx = max(0, idx - lookback)
        prev_df = df.iloc[start_idx:idx]

        player_id = row['player_id']
        opponent_id = row['opponent_id']

        # Player stats
        player_matches = prev_df[(prev_df['player_id'] == player_id) | (prev_df['opponent_id'] == player_id)]
        # Existing metrics
        player_aces = player_matches.apply(
            lambda r: r['w_ace'] if r['player_id'] == player_id else r['l_ace'], axis=1
        ).tail(window)
        player_df = player_matches.apply(
            lambda r: r['w_df'] if r['player_id'] == player_id else r['l_df'], axis=1
        ).tail(window)
        player_bpSaved = player_matches.apply(
            lambda r: r['w_bpSaved'] if r['player_id'] == player_id else r['l_bpSaved'], axis=1
        ).tail(window)
        # New metrics for player
        player_1stIn = player_matches.apply(
            lambda r: r['w_1stIn'] if r['player_id'] == player_id else r['l_1stIn'], axis=1
        ).tail(window)
        player_1stWon = player_matches.apply(
            lambda r: r['w_1stWon'] if r['player_id'] == player_id else r['l_1stWon'], axis=1
        ).tail(window)
        player_2ndWon = player_matches.apply(
            lambda r: r['w_2ndWon'] if r['player_id'] == player_id else r['l_2ndWon'], axis=1
        ).tail(window)
        player_bpFaced = player_matches.apply(
            lambda r: r['w_bpFaced'] if r['player_id'] == player_id else r['l_bpFaced'], axis=1
        ).tail(window)
        player_svpt = player_matches.apply(
            lambda r: r['w_svpt'] if r['player_id'] == player_id else r['l_svpt'], axis=1
        ).tail(window)

        # Append rounded averages or None
        w_ace_avgs.append(round(player_aces.mean(), 2) if not player_aces.empty else None)
        w_df_avgs.append(round(player_df.mean(), 2) if not player_df.empty else None)
        w_bpSaved_avgs.append(round(player_bpSaved.mean(), 2) if not player_bpSaved.empty else None)
        w_1stIn_avgs.append(round(player_1stIn.mean(), 2) if not player_1stIn.empty else None)
        w_1stWon_avgs.append(round(player_1stWon.mean(), 2) if not player_1stWon.empty else None)
        w_2ndWon_avgs.append(round(player_2ndWon.mean(), 2) if not player_2ndWon.empty else None)
        w_bpFaced_avgs.append(round(player_bpFaced.mean(), 2) if not player_bpFaced.empty else None)
        w_svpt_avgs.append(round(player_svpt.mean(), 2) if not player_svpt.empty else None)

        # Opponent stats
        opponent_matches = prev_df[(prev_df['player_id'] == opponent_id) | (prev_df['opponent_id'] == opponent_id)]
        # Existing metrics
        opponent_aces = opponent_matches.apply(
            lambda r: r['w_ace'] if r['player_id'] == opponent_id else r['l_ace'], axis=1
        ).tail(window)
        opponent_df = opponent_matches.apply(
            lambda r: r['w_df'] if r['player_id'] == opponent_id else r['l_df'], axis=1
        ).tail(window)
        opponent_bpSaved = opponent_matches.apply(
            lambda r: r['w_bpSaved'] if r['player_id'] == opponent_id else r['l_bpSaved'], axis=1
        ).tail(window)
        # New metrics for opponent
        opponent_1stIn = opponent_matches.apply(
            lambda r: r['w_1stIn'] if r['player_id'] == opponent_id else r['l_1stIn'], axis=1
        ).tail(window)
        opponent_1stWon = opponent_matches.apply(
            lambda r: r['w_1stWon'] if r['player_id'] == opponent_id else r['l_1stWon'], axis=1
        ).tail(window)
        opponent_2ndWon = opponent_matches.apply(
            lambda r: r['w_2ndWon'] if r['player_id'] == opponent_id else r['l_2ndWon'], axis=1
        ).tail(window)
        opponent_bpFaced = opponent_matches.apply(
            lambda r: r['w_bpFaced'] if r['player_id'] == opponent_id else r['l_bpFaced'], axis=1
        ).tail(window)
        opponent_svpt = opponent_matches.apply(
            lambda r: r['w_svpt'] if r['player_id'] == opponent_id else r['l_svpt'], axis=1
        ).tail(window)

        # Append rounded averages or None
        l_ace_avgs.append(round(opponent_aces.mean(), 2) if not opponent_aces.empty else None)
        l_df_avgs.append(round(opponent_df.mean(), 2) if not opponent_df.empty else None)
        l_bpSaved_avgs.append(round(opponent_bpSaved.mean(), 2) if not opponent_bpSaved.empty else None)
        l_1stIn_avgs.append(round(opponent_1stIn.mean(), 2) if not opponent_1stIn.empty else None)
        l_1stWon_avgs.append(round(opponent_1stWon.mean(), 2) if not opponent_1stWon.empty else None)
        l_2ndWon_avgs.append(round(opponent_2ndWon.mean(), 2) if not opponent_2ndWon.empty else None)
        l_bpFaced_avgs.append(round(opponent_bpFaced.mean(), 2) if not opponent_bpFaced.empty else None)
        l_svpt_avgs.append(round(opponent_svpt.mean(), 2) if not opponent_svpt.empty else None)

    # Assign new columns to DataFrame
    df = df.copy()
    df['w_ace_avg'] = w_ace_avgs
    df['l_ace_avg'] = l_ace_avgs
    df['w_df_avg'] = w_df_avgs
    df['l_df_avg'] = l_df_avgs
    df['w_bpSaved_avg'] = w_bpSaved_avgs
    df['l_bpSaved_avg'] = l_bpSaved_avgs
    df['w_1stIn_avg'] = w_1stIn_avgs
    df['l_1stIn_avg'] = l_1stIn_avgs
    df['w_1stWon_avg'] = w_1stWon_avgs
    df['l_1stWon_avg'] = l_1stWon_avgs
    df['w_2ndWon_avg'] = w_2ndWon_avgs
    df['l_2ndWon_avg'] = l_2ndWon_avgs
    df['w_bpFaced_avg'] = w_bpFaced_avgs
    df['l_bpFaced_avg'] = l_bpFaced_avgs
    df['w_svpt_avg'] = w_svpt_avgs
    df['l_svpt_avg'] = l_svpt_avgs

    # Calculate percentage metrics
    df['w_bpSavedPer'] = (df['w_bpSaved_avg'] / df['w_bpFaced_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    df['l_bpSavedPer'] = (df['l_bpSaved_avg'] / df['l_bpFaced_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    df['w_1stPer'] = (df['w_1stWon_avg'] / df['w_1stIn_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    df['l_1stPer'] = (df['l_1stWon_avg'] / df['l_1stIn_avg']).replace([np.inf, -np.inf], np.nan).round(2)
    df['w_2ndPer'] = (df['w_2ndWon_avg'] / (df['w_svpt_avg'] - df['w_1stIn_avg'])).replace([np.inf, -np.inf], np.nan).round(2)
    df['l_2ndPer'] = (df['l_2ndWon_avg'] / (df['l_svpt_avg'] - df['l_1stIn_avg'])).replace([np.inf, -np.inf], np.nan).round(2)

    return df


# def generate_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
#     # Columns for which to calculate rolling averages
#     stats_cols = [
#         "w_ace",
#         "l_ace",
#         "w_df",
#         "l_df",
#         "w_bpSaved",
#         "l_bpSaved",
#     ]

#     # Group by player, shift to exclude current match, then compute rolling mean
#     rolling_avgs = (
#         df
#         .groupby("player_id")[stats_cols]
#         .apply(
#             lambda grp: grp
#             .shift(1)  # exclude the current match from its own average
#             .rolling(window=window, min_periods=1)
#             .mean()
#             .round(2)
#         )
#         .reset_index(level=0, drop=True)
#     )

#     # Rename columns to indicate they're averages
#     rolling_avgs.columns = [f"{col}_avg" for col in rolling_avgs.columns]

#     # Concatenate rolling averages alongside the original data
#     result_df = pd.concat([df, rolling_avgs], axis=1)

#     return result_df