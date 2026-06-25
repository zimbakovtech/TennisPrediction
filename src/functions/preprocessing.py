import pandas as pd
import numpy as np


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load one raw season file and add encoded/derived columns.

    IMPORTANT INVARIANT: in the raw files the ``player_*`` columns always
    describe the match *winner* and the ``opponent_*`` columns the *loser*
    (the data is pre-oriented upstream). Every difference feature below is
    therefore computed as winner - loser, which encodes the outcome. This is
    only neutralised later by :func:`duplicate_entries`, which mirrors each row
    and flips the signs. Do not derive features directly from winner/loser
    identity without mirroring, or the target will leak.

    Chronological ordering is handled globally in ``processing_data`` (sorting
    by ``tourney_date, round, match_num``), so this function does not sort.
    """
    df = pd.read_csv(filepath)

    # --- 1. Encode round labels to numeric (higher = later round) ---
    round_map = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7, 'RR': 3, 'BR': 6
    }
    df['round'] = df['round'].map(round_map).fillna(0).astype(int)

    # --- 2. Encode categorical features ---
    df['surface'] = df['surface'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})

    tourney_level_map = {'D': 1, 'A': 2, 'M': 3, 'F': 4, 'O': 5, 'G': 6}
    df['tourney_level'] = (
        df['tourney_level'].map(tourney_level_map).fillna(0).astype(int)
    )

    df['match_importance'] = df['tourney_level'] * df['round'] * df['best_of']

    # --- 3. Difference features (winner - loser; mirrored later) ---
    df['rank_diff'] = -(df['player_rank'] - df['opponent_rank'])
    points_delta = df['player_rank_points'] - df['opponent_rank_points']
    df['points_diff'] = (np.log1p(np.abs(points_delta)) * np.sign(points_delta)).round(4)
    df['age_diff'] = (df['player_age'] - df['opponent_age']).round(2)

    # Target: player (the winner) always wins pre-mirroring.
    df['win_loss'] = 1

    return df
