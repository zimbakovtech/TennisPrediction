import time
import numpy as np
import pandas as pd

from pathlib import Path

from src.functions.preprocessing import load_and_preprocess
from src.feature_engineering.generate_stats import generate_stats as py_gen
from src.functions.native_rolling import generate_stats_native


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data' / 'raw'
    # Sample a couple of years to keep quick
    files = sorted(data_dir.glob('atp_matches_201[5-6].csv'))[:2]
    if not files:
        print("No sample files found")
        return
    dfs = [load_and_preprocess(str(f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    t0 = time.time()
    py_df = py_gen(df.copy())
    t1 = time.time()
    nt_df = generate_stats_native(df.copy())
    t2 = time.time()

    cols = ['w_ace_avg','l_ace_avg','w_df_avg','l_df_avg','w_bpSaved_avg','l_bpSaved_avg','ace_diff','df_diff','bp_diff']
    ok = True
    for c in cols:
        a = py_df[c].to_numpy()
        b = nt_df[c].to_numpy()
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() == 0:
            print(f"{c}: no comparable values")
            continue
        diffs = np.abs(a[mask] - b[mask])
        max_abs = np.max(diffs)
        print(f"{c}: max abs diff = {max_abs:.8f}")
        if max_abs > 1e-6:
            idx = np.argmax(diffs)
            real_idx = np.where(mask)[0][idx]
            print(f"  first mismatch at row {real_idx}: py={a[mask][idx]} native={b[mask][idx]}")
        if max_abs > 1e-6:
            ok = False
    print(f"Python time: {t1-t0:.3f}s | Native time: {t2-t1:.3f}s")
    print("Parity:", "PASS" if ok else "FAIL")

if __name__ == '__main__':
    main()
