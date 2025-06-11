import logging
from pathlib import Path
from typing import List
import pandas as pd
from functions.duplicate_entries import duplicate_entries
from functions.preprocessing import load_and_preprocess
from functions.generate_stats import generate_stats


# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DROP_COLUMNS = [
    'tourney_id', 'tourney_name', 'match_num', 'player_name', 'opponent_name',
    'player_entry', 'opponent_entry', 'score', 'player_ioc', 'opponent_ioc',
    'minutes', 'tourney_date',  'player_rank', 'opponent_rank', 
    'player_rank_points', 'opponent_rank_points', 'draw_size',
    'player_seed', 'opponent_seed', 'player_age', 'opponent_age',
    'player_hand', 'opponent_hand', 'player_ht', 'opponent_ht',
    'player_id', 'opponent_id',
    
    'w_SvGms', 'w_bpFaced', 'l_SvGms', 'l_bpFaced',
    'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_svpt',
    'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_svpt',
    'w_ace', 'l_ace', 'w_df', 'l_df', 'w_bpSaved', 'l_bpSaved',
    'w_1stIn_avg', 'l_1stIn_avg', 'w_1stWon_avg', 'l_1stWon_avg', 
    'w_2ndWon_avg', 'l_2ndWon_avg','w_bpFaced_avg', 'l_bpFaced_avg', 
    'w_svpt_avg', 'l_svpt_avg',
]

AVG_COLUMNS = [
    'w_ace_avg', 'l_ace_avg', 'w_df_avg', 'l_df_avg', 
    'w_bpSaved_avg', 'l_bpSaved_avg',
]

KEY_FEATURES = [
    'surface', 'player_rank', 'opponent_rank',
    'player_rank_points', 'opponent_rank_points',
]


def postprocess_and_save(df: pd.DataFrame, output_path: Path) -> None:
    # Remove rows missing key features
    df = df.dropna(subset=KEY_FEATURES)

    # Drop raw serve stats and unused metadata
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Ensure average columns exist and fill missing values
    for col in AVG_COLUMNS:
        df[col] = df.get(col, 0).fillna(0)

    # Mirror entries to simulate opponent perspective
    final_df = duplicate_entries(df)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    logger.info(
        "Saved %d rows to %s", len(final_df), output_path
    )


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

    # Fill NaN values in the specified columns with 0
    columns_to_fill = [
        'w_1stIn_avg', 'l_1stIn_avg', 'w_1stWon_avg', 'l_1stWon_avg',
        'w_2ndWon_avg', 'l_2ndWon_avg', 'w_bpFaced_avg', 'l_bpFaced_avg',
        'w_svpt_avg', 'l_svpt_avg', 'w_bpSavedPer', 'l_bpSavedPer',
        'w_1stPer', 'l_1stPer', 'w_2ndPer', 'l_2ndPer'
    ]
    for col in columns_to_fill:
        averaged_df[col] = averaged_df.get(col, 0).fillna(0)

    # Postprocess and save results
    postprocess_and_save(averaged_df, output_path)


if __name__ == '__main__':
    current_dir = Path.cwd()
    data_directory = current_dir / 'data' / 'raw'
    processed_file = current_dir / 'data' / 'processed' / 'all_matches.csv'

    process_all(data_directory, processed_file)
    logger.info("Data processing complete.")
