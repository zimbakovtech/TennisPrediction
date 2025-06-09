import os
import glob
import pandas as pd
import numpy as np

data_dir = os.path.join(os.getcwd(), 'data', 'raw')

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    
    # 1. Remove identifiers and text fields
    drop_cols = [
        'tourney_id', 'tourney_name', 'match_num',
        'winner_name', 'loser_name', 'winner_entry', 
        'loser_entry', 'score', 'winner_ioc', 'loser_ioc',
        'winner_hand', 'loser_hand', 'winner_ht', 'loser_ht',
        'best_of', 'minutes',
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Malce mi e cudno ova, namesto samo surface da se smeni vo Hard -> H, Clay -> C
    # i ChatGPT i DeepSeek gi mapiraat na kraj vo razlicni boolean vrednosti

    # 2. Encode categorical features
    # Surface encoding
    surface_map = {'Hard': 'H', 'Clay': 'C', 'Grass': 'G'}
    df['surface'] = df['surface'].map(surface_map)
    
    # Round encoding (ordinal)
    round_order = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7, 'RR': 3  # Round Robin as R32 equivalent
    }
    df['round'] = df['round'].map(round_order).fillna(0).astype(int)
    
    
    # 4. Calculate serve/game rates and differentials
    # Winner serve stats
    df['w_1stIn_pct'] = df['w_1stIn'] / df['w_svpt'].replace(0, np.nan)
    df['w_1stWon_pct'] = df['w_1stWon'] / df['w_1stIn'].replace(0, np.nan)
    df['w_2ndWon_pct'] = df['w_2ndWon'] / (df['w_svpt'] - df['w_1stIn']).replace(0, np.nan)
    
    # Loser serve stats
    df['l_1stIn_pct'] = df['l_1stIn'] / df['l_svpt'].replace(0, np.nan)
    df['l_1stWon_pct'] = df['l_1stWon'] / df['l_1stIn'].replace(0, np.nan)
    df['l_2ndWon_pct'] = df['l_2ndWon'] / (df['l_svpt'] - df['l_1stIn']).replace(0, np.nan)
    
    # Differential stats
    df['ace_diff'] = df['w_ace'] - df['l_ace']
    df['df_diff'] = df['w_df'] - df['l_df']
    df['bp_saved_diff'] = df['w_bpSaved'] - df['l_bpSaved']
    
    # Clean up serve stats
    serve_cols = ['w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_svpt',
                 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_svpt',
                 'w_ace', 'l_ace', 'w_df', 'l_df', 
                 'w_bpSaved', 'l_bpSaved']
    df = df.drop(columns=[c for c in serve_cols if c in df.columns])
    
    # 5. Process ranking features
    # Ranking differentials
    df['rank_diff'] = df['loser_rank_points'] - df['winner_rank_points']
    df['seed_diff'] = df['loser_seed'] - df['winner_seed']
    
    # Log transforms
    df['log_winner_rank'] = np.log(df['winner_rank_points'].fillna(0) + 1)
    df['log_loser_rank'] = np.log(df['loser_rank_points'].fillna(0) + 1)
    
    
    # Fill remaining NaNs
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != bool else 0)
    
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