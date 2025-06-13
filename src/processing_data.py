import logging
from pathlib import Path
from typing import List
import pandas as pd
from functions.duplicate_entries import duplicate_entries
from functions.preprocessing import load_and_preprocess
from feature_engineering.generate_stats import generate_stats
from feature_engineering.calculate_elo import calculate_elo
from feature_engineering.head2head import add_h2h_stats


# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DROP_COLUMNS = [
    'tourney_id', 'tourney_name', 'match_num', 'player_name', 'opponent_name',
    'player_entry', 'opponent_entry', 'score', 'player_ioc', 'opponent_ioc', 'opponent_ht',
    'player_seed', 'opponent_seed', 'player_age', 'opponent_age', 'draw_size',
    'minutes', 'tourney_date', 'player_hand', 'opponent_hand', 'player_rank',
    'player_rank_points', 'opponent_rank_points', 'opponent_rank', 'player_ht', 
    
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

    # Ensure average columns exist and fill missing values
    for col in FILL_COLUMNS:
        df[col] = df.get(col, 0).fillna(0)

    # Mirror entries to simulate opponent perspective
    final_df = duplicate_entries(df)

    # Add ELO
    final_elo_df = calculate_elo(final_df)

    # Remove player_id and opponent_id from dataset
    final_elo_df = final_elo_df.drop(columns=['player_id', 'opponent_id'])

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    final_elo_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(final_elo_df), output_path)


def process_all(data_dir: Path, output_path: Path) -> None:
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
    logger.info(
        "Combined data frame has %d rows, computing rolling averages...",
        len(combined_df)
    )
    averaged_df = generate_stats(combined_df)

    h2h_df = add_h2h_stats(averaged_df)

    # Postprocess and save results
    postprocess_and_save(h2h_df, output_path)


if __name__ == '__main__':
    current_dir = Path.cwd()
    data_directory = current_dir / 'data' / 'raw'
    processed_file = current_dir / 'data' / 'processed' / 'all_matches.csv'

    process_all(data_directory, processed_file)
    logger.info("Data processing complete.")
