import os
import glob
import pandas as pd
import numpy as np

data_dir = os.path.join(os.getcwd(), 'data', 'raw')

def duplicate_entries(df: pd.DataFrame) -> pd.DataFrame:
    mirrored = df.copy()
    swap_cols = [
        ('player_seed', 'opponent_seed'),
        ('player_age', 'opponent_age'),
        ('player_rank', 'opponent_rank'),
        ('player_rank_points', 'opponent_rank_points'),
        ('log_player_rank', 'log_opponent_rank'),
        ('player_hand', 'opponent_hand'),
        ('player_ht', 'opponent_ht'),
        ('player_id', 'opponent_id')
    ]
    for a, b in swap_cols:
        mirrored[[a, b]] = mirrored[[b, a]]

    n = len(df)
    df.index = np.arange(0, 2*n, 2)
    mirrored.index = np.arange(1, 2*n, 2)
    return pd.concat([df, mirrored]).sort_index().reset_index(drop=True)


def load_and_clean(filepath: str) -> pd.DataFrame:
    # 1. Read the CSV file
    df = pd.read_csv(filepath)

    # Encode round
    df['round'] = (
        df['round']
        .map({'R128':1,'R64':2,'R32':3,'R16':4,'QF':5,'SF':6,'F':7,'RR':3, 'BR': 6})
        .fillna(0).astype(int)
    )

    df['group'] = (df['round'] > df['round'].shift(1)).cumsum()
    df = df.sort_values(by=['group', 'round'], ascending=[True, True])
    df = df.drop('group', axis=1)
    df = df.reset_index(drop=True)

    # 2. Drop unnecessary columns
    drop_cols = [
        'tourney_id', 'tourney_name', 'match_num', 'player_name', 
        'opponent_name', 'player_entry', 'opponent_entry',
        'score', 'player_ioc', 'opponent_ioc', 'minutes', 'tourney_date',
        'w_SvGms','w_bpFaced','l_SvGms','l_bpFaced',
        # serve stats
        'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_svpt', 'l_1stIn', 'l_1stWon',
        'l_2ndWon', 'l_svpt', 'w_ace', 'l_ace', 'w_df', 'l_df', 'w_bpSaved', 'l_bpSaved'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 3. Encode surface columns
    df['surface'] = df['surface'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})
    df['player_hand'] = df['player_hand'].map({'R': 0, 'L': 1})
    df['opponent_hand'] = df['opponent_hand'].map({'R': 0, 'L': 1})

    # 4. Encode tourney level
    df['tourney_level'] = (
        df['tourney_level']
        .map({'D':1,'A':2,'M':3,'F':4,'O':5,'G':6})
        .fillna(0).astype(int)
    )

    # Fill seeds
    for col in ['player_seed', 'opponent_seed']:
        df[col] = df[col].fillna(0).astype(int)

    # Create absolute difference features
    df['rank_diff']   = abs(df['opponent_rank']        - df['player_rank'])
    df['points_diff'] = abs(df['player_rank_points']    - df['opponent_rank_points'])
    df['seed_diff']   = abs(df['opponent_seed']         - df['player_seed'])
    df['age_diff']    = abs(df['player_age']            - df['opponent_age']).round()

    # Log transforms for rank points
    df['log_player_rank']   = np.log1p(df['player_rank_points'])
    df['log_opponent_rank'] = np.log1p(df['opponent_rank_points'])

    # Drop rows missing critical numeric or categorical features
    df.dropna(subset=['surface', 'player_rank', 'opponent_rank', 'player_rank_points', 'opponent_rank_points'], inplace=True)

    # Mirror each match to augment
    return duplicate_entries(df)


def process_all(data_dir: str = data_dir, out_path: str = 'data/processed/all_matches.csv'):
    pattern = os.path.join(data_dir, 'atp_matches_*.csv')
    files = sorted(glob.glob(pattern))
    processed = []
    for f in files:
        print(f"Processing {os.path.basename(f)}...")
        try:
            processed.append(load_and_clean(f))
        except Exception as e:
            print(f"Error on {f}: {e}")
    if processed:
        all_df = pd.concat(processed, ignore_index=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        all_df.to_csv(out_path, index=False)
        print(f"Saved {len(all_df)} rows to {out_path}")
    else:
        print("No files processed.")

if __name__ == "__main__":
    process_all(data_dir)
    print("Data processing complete.")