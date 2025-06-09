import os
import glob
import pandas as pd
import numpy as np

data_dir = os.path.join(os.getcwd(), 'data', 'raw')


def duplicate_entries(df):
    # new function
    mirrored = df.copy()

    swap_cols = [
        ('player_id', 'opponent_id'),
        ('player_seed', 'opponent_seed'),
        ('player_age', 'opponent_age'),
        ('player_rank', 'opponent_rank'),
        ('player_rank_points', 'opponent_rank_points'),
    ]

    for col1, col2 in swap_cols:
        mirrored[[col1, col2]] = mirrored[[col2, col1]]

    # Negate the differentials
    for diff_col in ['rank_diff', 'points_diff', 'seed_diff']:
        mirrored[diff_col] = -mirrored[diff_col]

    n = len(df)
    df_index      = np.arange(n) * 2
    mirrored_index = df_index + 1

    # apply them and concat + sort
    df.index       = df_index
    mirrored.index = mirrored_index

    result = pd.concat([df, mirrored]).sort_index().reset_index(drop=True)
    return result


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    
    # 1. Remove identifiers and text fields
    drop_cols = [
        'tourney_id', 'tourney_name', 'match_num',
        'player_name', 'opponent_name', 'player_entry', 
        'opponent_entry', 'score', 'player_ioc', 'opponent_ioc',
        'player_hand', 'opponent_hand', 'player_ht', 'opponent_ht',
        'best_of', 'minutes', 'tourney_date','w_SvGms','w_bpFaced',
        'l_SvGms','l_bpFaced', 
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2. Encode categorical features
    # Surface encoding
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
    df['surface'] = df['surface'].map(surface_map)

    # 3. Encode tourney level
    tourney_level_map = {'D': 1, 'A': 2, 'M': 3, 'F': 4, 'O': 5, 'G': 6}
    df['tourney_level'] = df['tourney_level'].map(tourney_level_map).fillna(0).astype(int)
    
    # Round encoding (ordinal)
    round_order = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7, 'RR': 3  # Round Robin as R32 equivalent
    }
    df['round'] = df['round'].map(round_order).fillna(0).astype(int)
    
    
    # Clean up serve stats
    serve_cols = ['w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_svpt',
                 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_svpt',
                 'w_ace', 'l_ace', 'w_df', 'l_df', 
                 'w_bpSaved', 'l_bpSaved']
    df = df.drop(columns=[c for c in serve_cols if c in df.columns])

    # Fill missing seeds with 0
    df['player_seed'] = df['player_seed'].fillna(0).astype(int)
    df['opponent_seed'] = df['opponent_seed'].fillna(0).astype(int)
    
    # 5. Process ranking features
    # Ranking differentials
    df['rank_diff'] = df['opponent_rank'] - df['player_rank']
    df['points_diff'] = df['player_rank_points'] - df['opponent_rank_points'] 
    df['seed_diff'] = df['opponent_seed'] - df['player_seed']
    
    # Log transforms
    df['log_player_rank'] = np.log(df['player_rank_points'].fillna(0) + 1)
    df['log_opponent_rank'] = np.log(df['opponent_rank_points'].fillna(0) + 1)

    # Drop rows with missing ranking information
    df = df.dropna(subset=['player_rank', 'player_rank_points', 'opponent_rank', 'opponent_rank_points', 'surface'])
    
    # Fill remaining NaNs
    # for col in df.select_dtypes(include=[np.number]).columns:
    #     df[col] = df[col].fillna(df[col].median() if df[col].dtype != bool else 0)
    
    df = duplicate_entries(df)

    return df

def process_all(data_dir=data_dir, out_path='data/processed/all_matches.csv'):
    pattern = os.path.join(data_dir, 'atp_matches_*.csv')
    files = sorted(glob.glob(pattern))
    dfs = []
    
    for f in files:
        print(f"Processing {os.path.basename(f)}...")
        try:
            df_clean = load_and_clean(f)
            dfs.append(df_clean)
        except Exception as e:
            print(f"Error processing {f}: {str(e)}")
    
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        all_df.to_csv(out_path, index=False)
        print(f"Saved processed data ({len(all_df)} rows) to {out_path}")
    else:
        print("No valid data processed")

if __name__ == '__main__':
    process_all()