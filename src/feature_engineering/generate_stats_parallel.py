import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def process_player_group(player_id, group, window):
    """
    Helper function to calculate rolling stats for a single player.
    This function will be run in parallel.
    """
    group = group.sort_values('original_index')
    
    shifted_stats = group[['ace', 'df', 'bpSaved', 'won']].shift(1)

    rolling_means = shifted_stats[['ace', 'df', 'bpSaved']].rolling(window=window, min_periods=1).mean()
    rolling_wins = shifted_stats['won'].rolling(window=window, min_periods=1).sum()
    
    result = pd.DataFrame({
        'original_index': group['original_index'],
        'player_id': player_id,
        'ace_avg': rolling_means['ace'].round(2),
        'df_avg': rolling_means['df'].round(2),
        'bp_avg': rolling_means['bpSaved'].round(2),
        'recent_wins': rolling_wins.fillna(0).astype(int)
    })
    
    return result

def generate_stats(df: pd.DataFrame, window: int = 10, lookback: int = 1000) -> pd.DataFrame:
    
    print("Preparing data for parallel processing...")
    
    # Winner side
    winner_df = df[['player_id', 'w_ace', 'w_df', 'w_bpSaved']].copy()
    winner_df.columns = ['player_id', 'ace', 'df', 'bpSaved']
    winner_df['won'] = 1
    winner_df['original_index'] = df.index
    
    # Loser side
    loser_df = df[['opponent_id', 'l_ace', 'l_df', 'l_bpSaved']].copy()
    loser_df.columns = ['player_id', 'ace', 'df', 'bpSaved']
    loser_df['won'] = 0
    loser_df['original_index'] = df.index
    
    # Combine into one long list of player performances
    long_df = pd.concat([winner_df, loser_df], ignore_index=True)
    
    # 2. Group by player
    player_groups = long_df.groupby('player_id')
    
    # 3. Parallel processing
    print(f"Calculating rolling stats for {len(player_groups)} players in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(process_player_group)(pid, group, window) 
        for pid, group in player_groups
    )
    
    # 4. Combine results
    stats_long = pd.concat(results, ignore_index=True)
    
    # 5. Merge back to original dataframe
    print("Merging stats back to main dataframe...")
    
    # Create a temporary column for the index to merge on
    df_with_index = df.copy()
    df_with_index['match_index'] = df.index

    # Merge for player_id (Winner)
    df_out = df_with_index.merge(
        stats_long, 
        left_on=['player_id', 'match_index'], 
        right_on=['player_id', 'original_index'], 
        how='left',
        suffixes=('', '_p')
    ).rename(columns={
        'ace_avg': 'player_ace',
        'df_avg': 'player_df',
        'bp_avg': 'player_bp',
        'recent_wins': f'player_wins_last_{window}'
    })
    
    # Merge for opponent_id (Loser)
    df_out = df_out.merge(
        stats_long, 
        left_on=['opponent_id', 'match_index'], 
        right_on=['player_id', 'original_index'], 
        how='left',
        suffixes=('', '_o')
    ).rename(columns={
        'ace_avg': 'opponent_ace',
        'df_avg': 'opponent_df',
        'bp_avg': 'opponent_bp',
        'recent_wins': f'opponent_wins_last_{window}'
    })
    
    # Clean up merge columns
    cols_to_drop = ['original_index', 'original_index_p', 'player_id_p', 'player_id_o', 'original_index_o', 'match_index']
    # Filter to only drop columns that actually exist
    df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns])
    
    # Calculate diffs
    df_out['ace_diff'] = (df_out['player_ace'] - df_out['opponent_ace']).round(5)
    df_out['df_diff'] = -(df_out['player_df'] - df_out['opponent_df']).round(5)
    df_out['bp_diff'] = (df_out['player_bp'] - df_out['opponent_bp']).round(5)

    return df_out
