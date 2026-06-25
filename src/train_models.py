import numpy as np
import pandas as pd

from config import PATHS
from functions.utils import load_dataset, TARGET
from models.xgboost import XGBoost
from model_evaluations import evaluate_model, chronological_pair_split


if __name__ == "__main__":
    X, y, feature_names = load_dataset()

    # Chronological holdout: most recent 30% of matches, pairs kept intact.
    train_idx, test_idx = chronological_pair_split(len(X), test_frac=0.3)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    xgb = XGBoost(X_train, y_train)
    evaluate_model(xgb, X, y, X_test, y_test, X_train, y_train, feature_names=feature_names)

    # --- Wimbledon 2025 inference (demo) ---
    wimbledon = pd.read_csv(PATHS["wimbledon_test"])
    expected = list(feature_names)
    actual = [c for c in wimbledon.columns if c != TARGET]
    if actual != expected:
        raise ValueError(
            "Wimbledon feature schema does not match training data:\n"
            f"  training: {expected}\n  wimbledon: {actual}"
        )

    X_wimbledon = wimbledon[expected].values
    raw_preds = xgb.predict(X_wimbledon)
    preds = [2 if p == 0 else 1 for p in raw_preds]
    print("\n=== Predictions for Wimbledon 2025 ===")
    print(preds)

    if TARGET in wimbledon.columns:
        acc = float(np.mean(raw_preds == wimbledon[TARGET].values))
        print(f"Wimbledon accuracy vs labels: {acc * 100:.2f}%")
