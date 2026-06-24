import logging
from pathlib import Path
from typing import List
import pandas as pd
from functions.duplicate_entries import duplicate_entries
from functions.preprocessing import load_and_preprocess
from feature_engineering.generate_stats import generate_stats
from feature_engineering.generate_stats_parallel import generate_stats as generate_stats_parallel
from feature_engineering.calculate_elo import calculate_elo
from feature_engineering.head2head import add_h2h_stats
import numpy as np


# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DROP_COLUMNS = [
    'tourney_id', 'tourney_name', 'match_num', 'player_name', 'opponent_name',
    'player_entry', 'opponent_entry', 'score', 'player_ioc', 'opponent_ioc', 'opponent_ht',
    'player_seed', 'opponent_seed', 'draw_size',
    'minutes', 'player_hand', 'opponent_hand', 'player_ht', 
    
    'w_SvGms', 'w_bpFaced', 'l_SvGms', 'l_bpFaced', 'w_1stIn', 'w_1stWon', 
    'w_2ndWon', 'w_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_ace_avg',
    'w_ace', 'l_ace', 'w_df', 'l_df', 'l_bpSaved_avg','w_bpSaved_avg',
    'w_bpSaved', 'l_bpSaved', 'w_df_avg', 'l_df_avg','l_svpt', 'w_ace_avg'
]

FILL_COLUMNS = [ 'ace_diff', 'df_diff', 'bp_diff']

KEY_FEATURES = [
    'surface', 'player_rank', 'opponent_rank',
    'player_rank_points', 'opponent_rank_points'
]

def postprocess_and_save(df: pd.DataFrame, output_path: Path) -> None:
    # Remove rows missing key features
    df = df.dropna(subset=KEY_FEATURES)

    # Drop raw serve stats and unused metadata
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Add ELO features
    elo_df = pd.read_csv('matches_with_elo.csv')
    df['player_elo_before'] = elo_df['player_elo_before'].values
    df['opponent_elo_before'] = elo_df['opponent_elo_before'].values
    # df['player_elo_trend'] = elo_df['player_elo_trend'].values
    # df['opponent_elo_trend'] = elo_df['opponent_elo_trend'].values
    hard_diff = elo_df['player_elo_before_hard'].values - elo_df['opponent_elo_before_hard'].values
    clay_diff = elo_df['player_elo_before_clay'].values - elo_df['opponent_elo_before_clay'].values
    grass_diff = elo_df['player_elo_before_grass'].values - elo_df['opponent_elo_before_grass'].values

    df['elo_diff'] = np.select(
        [df['surface'] == 0.0, df['surface'] == 1.0, df['surface'] == 2.0],
        [hard_diff, clay_diff, grass_diff],
        default=0
    )

    # Ensure average columns exist and fill missing values
    for col in FILL_COLUMNS:
        df[col] = df.get(col, 0).fillna(0)

    # Mirror entries to simulate opponent perspective
    duplicated_df = duplicate_entries(df)

    # Add ELO
    # final_elo_df = calculate_elo(duplicated_df)

    # Remove player_id and opponent_id from dataset
    duplicated_df = duplicated_df.drop(columns=['tourney_level', 'round', 'surface', 'tourney_date',
                                                 'player_id', 'opponent_id', 'player_age', 'opponent_age',
                                                 'player_rank', 'player_rank_points', 'opponent_rank', 'opponent_rank_points',
                                                 'player_ace', 'opponent_ace', 'player_df', 'opponent_df', 'player_bp', 'opponent_bp'])

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    duplicated_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(duplicated_df), output_path)


def process_all(data_dir: Path, output_path: Path, combined_raw_path: Path) -> None:
    files = sorted(data_dir.glob('atp_matches_*.csv'))
    if not files:
        logger.warning("No raw files found in %s", data_dir)
        return

    logger.info("Processing %d files from %s", len(files), data_dir)

    # Load and preprocess each file
    data_frames: List[pd.DataFrame] = []
    for filepath in files:
        try:
            df = load_and_preprocess(filepath)
            data_frames.append(df)
            logger.debug("Loaded and preprocessed %s", filepath)
        except Exception as e:
            logger.error("Error processing %s: %s", filepath, e)

    # Combine and compute rolling stats
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df = combined_df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    logger.info(
        "Combined data frame has %d rows, computing rolling averages...",
        len(combined_df)
    )
    combined_df.to_csv(combined_raw_path, index=False)

    averaged_df = generate_stats(combined_df)
    # averaged_df = generate_stats_parallel(combined_df)

    h2h_df = add_h2h_stats(averaged_df)

    # Postprocess and save results
    postprocess_and_save(h2h_df, output_path)


if __name__ == '__main__':
    current_dir = Path.cwd()
    data_directory = current_dir / 'data' / 'raw'
    processed_file = current_dir / 'data' / 'processed' / 'all_matches.csv'
    combined_raw_file = current_dir / 'data' / 'processed' / 'combined_raw_matches.csv'

    process_all(data_directory, processed_file, combined_raw_file)
    logger.info("Data processing complete.")

    # model_input = [
    #         best_of, match_importance, rank_diff, points_diff, age_diff, 
    #         ace_diff, df_diff, bp_diff, h2h_diff, 
    #         player_1_elo, player_2_elo, surface_elo_diff
    #     ]