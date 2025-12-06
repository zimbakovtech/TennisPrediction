Native C++ rolling feature computation (pybind11 + OpenMP)

Overview

- Replaces the heavy Python loop in src/feature_engineering/generate_stats.py with a high-performance native module.
- Computes per-player rolling means for aces, double faults, and break points saved using O(1) amortized updates and sliding windows.
- Parallelized across players with OpenMP and processes multiple columns at once for cache locality.

Features

- Exact functional parity with Python implementation (within 1e-6 tolerance):
  - previous-only (exclude current row), last `lookback` rows globally, and last `window` per-player contributions
  - NaNs are skipped when computing means
  - Means rounded to 2 decimals; diffs rounded to 5 decimals
- Robust input validation and logging in the Python wrapper

Build

1. Ensure a C++17 compiler is installed.
   - Windows (MSVC): Install "Desktop development with C++" workload (Build Tools / Visual Studio).
   - Linux/macOS: gcc/clang with OpenMP support.
2. Install Python deps (requires pybind11):
   pip install -r requirements.txt
3. Build in place (creates native_rolling.pyd/.so next to sources):
   python setup_native.py build_ext --inplace

Run

- The data pipeline uses the native module automatically if available:
  python -m src.processing_data

- Output: data/processed/all_matches_from_native.csv

Tuning

- Default window=10, lookback=600. Change in src/processing_data.py or wrapper.
- Threads: auto-detected; override via generate_stats_native(..., threads=8)

Performance

- The module parallelizes by player and uses deques with running sums per stat for O(n) processing.
- Benchmarks are printed in logs: wall time and OpenMP thread count.

Memory

- For N matches and P players, memory is ~O(N) for inputs + O(N) for outputs + small O(P) state.
- A machine with 8GB RAM is sufficient for ATP history (~200K rows). For >1M rows, 16GB is recommended.

Implementation

- Source: src/native/rolling_features.cpp
- Python wrapper: src/functions/native_rolling.py
- Build script: setup_native.py
