import argparse

import joblib
import numpy as np
import pandas as pd

from config import PATHS, TOURS, model_path
from functions.utils import load_dataset, TARGET
from models.xgboost import XGBoost
from model_evaluations import evaluate_model, chronological_pair_split


def run_wimbledon_demo(xgb, feature_names) -> None:
    """ATP-only sanity check against the hand-built Wimbledon 2025 fixture."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train + evaluate the XGBoost match-winner model.")
    parser.add_argument("--tour", choices=TOURS, default="atp",
                        help="Which tour's processed dataset to train on (default: atp).")
    args = parser.parse_args()

    print(f"=== Training {args.tour.upper()} model ===")
    X, y, feature_names = load_dataset(args.tour)

    # Chronological holdout: most recent 30% of matches, pairs kept intact.
    train_idx, test_idx = chronological_pair_split(len(X), test_frac=0.3)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    xgb = XGBoost(X_train, y_train)
    evaluate_model(xgb, X, y, X_test, y_test, X_train, y_train, feature_names=feature_names)

    # Persist the fitted model for reuse elsewhere. Load with
    # ``joblib.load(...)`` and call ``.predict`` / ``.predict_proba`` on a matrix
    # whose columns are, in order: the feature_names printed above.
    out = model_path(args.tour)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, out)
    print(f"\nSaved {args.tour.upper()} model -> {out}")

    # The Wimbledon fixture is an ATP-only artifact; WTA relies on the
    # chronological split + time-series CV reported above.
    if args.tour == "atp":
        run_wimbledon_demo(xgb, feature_names)
