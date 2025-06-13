import pandas as pd
import numpy as np


def duplicate_entries(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Define column pairs to swap between player and opponent
    swap_pairs = [
        ('player_id', 'opponent_id'),
    ]

    # 2. Create a mirrored copy of the DataFrame
    mirrored = df.copy()

    # 3. Swap values for each pair of columns
    for col_a, col_b in swap_pairs:
        mirrored[[col_a, col_b]] = mirrored[[col_b, col_a]]

    # 4. Flip sign for difference columns in mirrored DataFrame
    diff_cols = ['rank_diff', 'points_diff', 'seed_diff', 'age_diff', 'height_diff', 'ace_diff', 'df_diff', 'bp_diff', 'h2h_diff']
    for col in diff_cols:
        if col in mirrored.columns:
            mirrored[col] = -mirrored[col]

    mirrored['win_loss'] = 1 - mirrored['win_loss']

    # 5. Interleave original and mirrored entries
    n = len(df)
    df.index = np.arange(0, 2 * n, 2)
    mirrored.index = np.arange(1, 2 * n, 2)

    # 6. Concatenate and reset index
    combined_df = pd.concat([df, mirrored]).sort_index().reset_index(drop=True)
    
    return combined_df