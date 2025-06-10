import pandas as pd
import numpy as np


def duplicate_entries(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Define column pairs to swap between player and opponent
    swap_pairs = [
        ('player_seed', 'opponent_seed'),
        ('player_age', 'opponent_age'),
        ('player_rank', 'opponent_rank'),
        ('player_rank_points', 'opponent_rank_points'),
        # ('log_player_rank', 'log_opponent_rank'),
        ('player_hand', 'opponent_hand'),
        ('player_ht', 'opponent_ht'),
        ('player_id', 'opponent_id'),
        ('w_ace_avg', 'l_ace_avg'),
        ('w_df_avg', 'l_df_avg'),
        ('w_bpSaved_avg', 'l_bpSaved_avg'),
    ]

    # 2. Create a mirrored copy of the DataFrame
    mirrored = df.copy()

    # 3. Swap values for each pair of columns
    for col_a, col_b in swap_pairs:
        mirrored[[col_a, col_b]] = mirrored[[col_b, col_a]]

    # 4. Interleave original and mirrored entries
    n = len(df)
    df.index = np.arange(0, 2 * n, 2)
    mirrored.index = np.arange(1, 2 * n, 2)

    # 5. Concatenate and reset index
    combined_df = pd.concat([df, mirrored]).sort_index().reset_index(drop=True)
    
    return combined_df
