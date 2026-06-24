import pandas as pd
import numpy as np


def generate_stats(df: pd.DataFrame, window: int = 10, lookback: int = 1000) -> pd.DataFrame:
    w_ace_avgs, l_ace_avgs = [], []
    w_df_avgs, l_df_avgs = [], []
    w_bpSaved_avgs, l_bpSaved_avgs = [], []
    player_recent_wins, opponent_recent_wins = [], []
    
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
        ).dropna().tail(window)
        player_df = player_matches.apply(
            lambda r: r['w_df'] if r['player_id'] == player_id else r['l_df'], axis=1
        ).dropna().tail(window)
        player_bpSaved = player_matches.apply(
            lambda r: r['w_bpSaved'] if r['player_id'] == player_id else r['l_bpSaved'], axis=1
        ).dropna().tail(window)
        player_wins = player_matches['player_id'].eq(player_id).astype(int).tail(window)

        w_ace_avgs.append(round(player_aces.mean(), 2) if not player_aces.empty else None)
        w_df_avgs.append(round(player_df.mean(), 2) if not player_df.empty else None)
        w_bpSaved_avgs.append(round(player_bpSaved.mean(), 2) if not player_bpSaved.empty else None)
        player_recent_wins.append(int(player_wins.sum()) if not player_wins.empty else 0)

        # Opponent stats
        opponent_matches = prev_df[(prev_df['player_id'] == opponent_id) | (prev_df['opponent_id'] == opponent_id)]
        opponent_aces = opponent_matches.apply(
            lambda r: r['w_ace'] if r['player_id'] == opponent_id else r['l_ace'], axis=1
        ).dropna().tail(window)
        opponent_df = opponent_matches.apply(
            lambda r: r['w_df'] if r['player_id'] == opponent_id else r['l_df'], axis=1
        ).dropna().tail(window)
        opponent_bpSaved = opponent_matches.apply(
            lambda r: r['w_bpSaved'] if r['player_id'] == opponent_id else r['l_bpSaved'], axis=1
        ).dropna().tail(window)
        opponent_wins = opponent_matches['player_id'].eq(opponent_id).astype(int).tail(window)

        l_ace_avgs.append(round(opponent_aces.mean(), 2) if not opponent_aces.empty else None)
        l_df_avgs.append(round(opponent_df.mean(), 2) if not opponent_df.empty else None)
        l_bpSaved_avgs.append(round(opponent_bpSaved.mean(), 2) if not opponent_bpSaved.empty else None)
        opponent_recent_wins.append(int(opponent_wins.sum()) if not opponent_wins.empty else 0)
        
    df = df.copy()
    df['player_ace'] = w_ace_avgs
    df['opponent_ace'] = l_ace_avgs
    df['player_df'] = w_df_avgs
    df['opponent_df'] = l_df_avgs
    df['player_bp'] = w_bpSaved_avgs
    df['opponent_bp'] = l_bpSaved_avgs
    # wins_suffix = f"last_{window}"
    # df[f'player_wins_{wins_suffix}'] = player_recent_wins
    # df[f'opponent_wins_{wins_suffix}'] = opponent_recent_wins
    # df['wins_diff'] = np.array(player_recent_wins) - np.array(opponent_recent_wins)
    df['ace_diff'] = (df['player_ace'] - df['opponent_ace']).round(5)
    df['df_diff'] = -(df['player_df'] - df['opponent_df']).round(5)
    df['bp_diff'] = (df['player_bp'] - df['opponent_bp']).round(5)

    return df