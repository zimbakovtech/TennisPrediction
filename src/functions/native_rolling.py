import logging
import time
from typing import Optional
from pathlib import Path
import sys
import importlib

import numpy as np
import pandas as pd

def _try_import_native() -> Optional[object]:
    # Attempt plain import first
    try:
        return importlib.import_module('native_rolling')
    except Exception:
        pass

    # If running script from src/, add project root to sys.path and retry
    try:
        proj_root = Path(__file__).resolve().parents[2]
        if str(proj_root) not in sys.path:
            sys.path.insert(0, str(proj_root))
        return importlib.import_module('native_rolling')
    except Exception:
        pass

    # Also try build output directory
    try:
        build_dir = Path(__file__).resolve().parents[2] / 'build'
        for p in build_dir.rglob('native_rolling*.pyd'):
            sys.path.insert(0, str(p.parent))
            try:
                return importlib.import_module('native_rolling')
            except Exception:
                continue
    except Exception:
        pass
    return None

native_rolling = _try_import_native()
_NATIVE_OK = native_rolling is not None

logger = logging.getLogger(__name__)


def _validate_sorted(df: pd.DataFrame, sort_if_needed: bool) -> pd.DataFrame:
    # The original Python implementation relies on global row order; here we optionally enforce per-player chronological order.
    if 'tourney_date' in df.columns:
        # Check if non-decreasing (global)
        is_sorted = df['tourney_date'].is_monotonic_increasing
        if not is_sorted and sort_if_needed:
            logger.warning("Input not globally sorted by date. Sorting by ['tourney_date'] to ensure determinism.")
            return df.sort_values(['tourney_date']).reset_index(drop=True)
        elif not is_sorted:
            logger.warning("Input not globally sorted by date. Proceeding with current order to preserve parity.")
    else:
        logger.info("No 'tourney_date' column found; assuming current order is chronological.")
    return df


def generate_stats_native(
    df: pd.DataFrame,
    window: int = 10,
    lookback: int = 600,
    sort_if_needed: bool = False,
    threads: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute rolling averages for aces, double faults, and break points saved using a native C++ implementation.

    Produces columns:
      - 'w_ace_avg', 'l_ace_avg', 'w_df_avg', 'l_df_avg', 'w_bpSaved_avg', 'l_bpSaved_avg'
      - 'ace_diff' (w - l), 'df_diff' (-(w - l)), 'bp_diff' (w - l)

    Notes:
      - Matches Python's previous semantics: previous matches only (exclude current), limited to last `lookback` rows and last `window` values per player.
      - NaNs are ignored when computing means (pandas-like behavior).
      - Means are rounded to 2 decimals; diffs to 5 decimals.
    """
    if not _NATIVE_OK:
        raise RuntimeError("native_rolling module is not available. Build it via `python setup_native.py build_ext --inplace`." )

    req_cols = [
        'player_id', 'opponent_id',
        'w_ace', 'l_ace', 'w_df', 'l_df', 'w_bpSaved', 'l_bpSaved'
    ]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _validate_sorted(df, sort_if_needed)

    n = len(df)
    # Ensure correct dtypes (ids as int64, stats as float64)
    player_ids = df['player_id'].astype(np.int64).to_numpy()
    opponent_ids = df['opponent_id'].astype(np.int64).to_numpy()

    # Allow floats; keep as float64
    w_ace = pd.to_numeric(df['w_ace'], errors='coerce').astype(float).to_numpy()
    l_ace = pd.to_numeric(df['l_ace'], errors='coerce').astype(float).to_numpy()
    w_df  = pd.to_numeric(df['w_df'], errors='coerce').astype(float).to_numpy()
    l_df  = pd.to_numeric(df['l_df'], errors='coerce').astype(float).to_numpy()
    w_bp  = pd.to_numeric(df['w_bpSaved'], errors='coerce').astype(float).to_numpy()
    l_bp  = pd.to_numeric(df['l_bpSaved'], errors='coerce').astype(float).to_numpy()

    n_threads = int(threads) if threads is not None else -1

    t0 = time.time()
    cpu0 = time.process_time()
    res = native_rolling.compute_rolling_features(
        player_ids, opponent_ids,
        w_ace, l_ace,
        w_df, l_df,
        w_bp, l_bp,
        int(window), int(lookback), n_threads
    )
    wall = res.pop('wall_time_sec', None)
    omp_threads = res.pop('omp_threads', None)
    t1 = time.time()
    cpu1 = time.process_time()
    wall_py = (t1 - t0)
    cpu_time = max(0.0, cpu1 - cpu0)
    cpu_util = None
    try:
        import os
        cpu_util = (cpu_time / wall_py) / max(1, os.cpu_count())
    except Exception:
        pass

    if cpu_util is not None:
        logger.info(
            "Native rolling: n=%d, window=%d, lookback=%d, threads=%s, wall=%.3fs (py %.3fs), CPU util~%.0f%%",
            n, window, lookback, omp_threads, wall if wall is not None else -1, wall_py, cpu_util * 100.0
        )
    else:
        logger.info(
            "Native rolling: n=%d, window=%d, lookback=%d, threads=%s, wall=%.3fs (py %.3fs)",
            n, window, lookback, omp_threads, wall if wall is not None else -1, wall_py
        )

    # Assign back to DataFrame
    out_df = df.copy()
    for k, arr in res.items():
        out_df[k] = np.asarray(arr)

    return out_df
