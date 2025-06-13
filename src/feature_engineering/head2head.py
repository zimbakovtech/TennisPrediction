import pandas as pd
from collections import defaultdict

def add_h2h_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # h2h[player][opponent] = (player's wins vs opponent) - (opponent's wins vs player)
    h2h = defaultdict(lambda: defaultdict(int))
    h2h_diffs = []

    for _, row in df.iterrows():
        player = row['player_id']      # winner
        opponent = row['opponent_id']  # loser

        # h2h diff from player's perspective
        diff = h2h[player][opponent]
        h2h_diffs.append(diff)

        # Update h2h map after this match
        h2h[player][opponent] += 1
        h2h[opponent][player] -= 1

    df['h2h_diff'] = h2h_diffs
    return df
