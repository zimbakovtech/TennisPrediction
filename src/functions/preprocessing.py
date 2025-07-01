import pandas as pd
import numpy as np


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    # Load data
    df = pd.read_csv(filepath)

    # --- 1. Encode round labels to numeric ---
    round_map = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7, 'RR': 3, 'BR': 6
    }
    df['round'] = (
        df['round']
        .map(round_map)
        .fillna(0)
        .astype(int)
    )

    # --- 2. Sort matches within each tournament ---
    df['match_group'] = (df['round'] < df['round'].shift(1, fill_value=float('inf'))).cumsum()
    df = (
        df
        .sort_values(['match_group', 'round'])
        .drop(columns='match_group')
        .reset_index(drop=True)
    )

    # --- 3. Encode categorical features ---
    df['surface'] = df['surface'].map({'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3})

    tourney_level_map = {'D': 1, 'A': 2, 'M': 3, 'F': 4, 'O': 5, 'G': 6}
    df['tourney_level'] = (
        df['tourney_level']
        .map(tourney_level_map)
        .fillna(0)
        .astype(int)
    )

    # --- 5. Compute absolute difference features ---
    df['rank_diff'] = -(df['player_rank'] - df['opponent_rank'])
    df['points_diff'] = (np.log1p(np.abs(df['player_rank_points'] - df['opponent_rank_points'])) * np.sign(df['player_rank_points'] - df['opponent_rank_points'])).round(4)
    df['age_diff'] = (df['player_age'] - df['opponent_age']).round(2)
    df['win_loss'] = 1

    return df
