import pandas as pd
import numpy as np


def generate_stats(df: pd.DataFrame, window: int = 10, lookback: int = 600) -> pd.DataFrame:
    w_ace_avgs, l_ace_avgs = [], []
    w_df_avgs, l_df_avgs = [], []
    w_bpSaved_avgs, l_bpSaved_avgs = [], []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        start_idx = max(0, idx - lookback)
        prev_df = df.iloc[start_idx:idx]

        player_id = row['player_id']
        opponent_id = row['opponent_id']

        # Player stats
        player_matches = prev_df[(prev_df['player_id'] == player_id) | (prev_df['opponent_id'] == player_id)]
        player_aces = player_matches.apply(
            lambda r: r['w_ace'] if r['player_id'] == player_id else r['l_ace'], axis=1
        ).tail(window)
        player_df = player_matches.apply(
            lambda r: r['w_df'] if r['player_id'] == player_id else r['l_df'], axis=1
        ).tail(window)
        player_bpSaved = player_matches.apply(
            lambda r: r['w_bpSaved'] if r['player_id'] == player_id else r['l_bpSaved'], axis=1
        ).tail(window)

        w_ace_avgs.append(round(player_aces.mean(), 2) if not player_aces.empty else None)
        w_df_avgs.append(round(player_df.mean(), 2) if not player_df.empty else None)
        w_bpSaved_avgs.append(round(player_bpSaved.mean(), 2) if not player_bpSaved.empty else None)

        # Opponent stats
        opponent_matches = prev_df[(prev_df['player_id'] == opponent_id) | (prev_df['opponent_id'] == opponent_id)]
        opponent_aces = opponent_matches.apply(
            lambda r: r['w_ace'] if r['player_id'] == opponent_id else r['l_ace'], axis=1
        ).tail(window)
        opponent_df = opponent_matches.apply(
            lambda r: r['w_df'] if r['player_id'] == opponent_id else r['l_df'], axis=1
        ).tail(window)
        opponent_bpSaved = opponent_matches.apply(
            lambda r: r['w_bpSaved'] if r['player_id'] == opponent_id else r['l_bpSaved'], axis=1
        ).tail(window)

        l_ace_avgs.append(round(opponent_aces.mean(), 2) if not opponent_aces.empty else None)
        l_df_avgs.append(round(opponent_df.mean(), 2) if not opponent_df.empty else None)
        l_bpSaved_avgs.append(round(opponent_bpSaved.mean(), 2) if not opponent_bpSaved.empty else None)
        
    df = df.copy()
    df['w_ace_avg'] = w_ace_avgs
    df['l_ace_avg'] = l_ace_avgs
    df['w_df_avg'] = w_df_avgs
    df['l_df_avg'] = l_df_avgs
    df['w_bpSaved_avg'] = w_bpSaved_avgs
    df['l_bpSaved_avg'] = l_bpSaved_avgs
    df['ace_diff'] = (df['w_ace_avg'] - df['l_ace_avg']).round(5)
    df['df_diff'] = -(df['w_df_avg'] - df['l_df_avg']).round(5)
    df['bp_diff'] = (df['w_bpSaved_avg'] - df['l_bpSaved_avg']).round(5)

    return df