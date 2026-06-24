import sys
import os
import time
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering.generate_stats_parallel import generate_stats as generate_stats_new
from src.feature_engineering.generate_stats import generate_stats as generate_stats_old

def main():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/combined_raw_matches.csv')
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Data file not found. Please check the path.")
        return

    # Use a subset for testing speedup (e.g., 500 rows)
    # The old method is O(N^2) or O(N*lookback), so it's very slow.
    # 500 rows should be enough to see a difference, but maybe too fast for parallel overhead.
    # Let's try 1000 rows.
    # subset_size = 3000
    # df_subset = df.head(subset_size).copy()
    
    # print(f"Testing on first {subset_size} rows...")

    # Measure Old
    start_time = time.time()
    df_old = generate_stats_old(df, window=10, lookback=1000)
    end_time = time.time()
    old_duration = end_time - start_time
    print(f"Old implementation took: {old_duration:.4f} seconds")

    # Measure New
    print("\nRunning NEW implementation (parallel)...")
    start_time = time.time()
    df_new = generate_stats_new(df, window=10, lookback=1000)
    end_time = time.time()
    new_duration = end_time - start_time
    print(f"New implementation took: {new_duration:.4f} seconds")

    # Speedup
    if new_duration > 0:
        print(f"\nSpeedup: {old_duration / new_duration:.2f}x")
    
    # Verification (check a few columns)
    # Note: There might be slight differences due to floating point or sorting stability, 
    # but logic should be similar.
    # Also, the new implementation handles 'min_periods=1' which might differ slightly 
    # from the manual tail(window) if there are fewer than window matches.
    # But generally they should match.
    
    print("\nVerifying results (checking 'ace_diff' column)...")
    # Fill NaNs for comparison
    diff_old = df_old['ace_diff'].fillna(-999)
    diff_new = df_new['ace_diff'].fillna(-999)
    
    matches = np.isclose(diff_old, diff_new, atol=1e-5).sum()
    print(f"Matches: {matches}/{len(df)}")
    
    if matches < len(df):
        print("Warning: Results differ. This might be due to edge cases (start of history) or implementation details.")
        print("First 5 differences:")
        mask = ~np.isclose(diff_old, diff_new, atol=1e-5)
        print(pd.concat([diff_old[mask], diff_new[mask]], axis=1, keys=['Old', 'New']).head())

if __name__ == "__main__":
    main()
