import pandas as pd

from config import PATHS

TARGET = "win_loss"


def load_dataset():
    """Load the processed match dataset.

    Returns ``(X, y, feature_names)`` where ``X``/``y`` are numpy arrays and
    ``feature_names`` is the ordered list of feature columns (everything except
    the target). The file is read once.
    """
    df = pd.read_csv(PATHS["processed_matches"])
    y = df[TARGET].values
    X_df = df.drop(columns=TARGET)
    return X_df.values, y, X_df.columns


# Backwards-compatible helpers (kept so existing callers keep working).
def read_file():
    X, _, feature_names = load_dataset()
    return X, feature_names


def process_y():
    _, y, _ = load_dataset()
    return y
