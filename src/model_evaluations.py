import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, brier_score_loss, log_loss, f1_score,
)
from sklearn.model_selection import TimeSeriesSplit


def chronological_pair_split(n_rows: int, test_frac: float = 0.3):
    """Split row indices chronologically without cutting a mirror pair.

    Rows are ordered as adjacent (original, mirror) pairs in time order, so the
    boundary is rounded down to an even index. The test set is the most recent
    matches -- an honest, leak-free holdout.
    """
    split = int(n_rows * (1 - test_frac))
    if split % 2:                       # never split a pair across the boundary
        split -= 1
    return np.arange(split), np.arange(split, n_rows)


def time_series_pair_cv(n_rows: int, n_splits: int = 8):
    """Expanding-window CV that respects time and keeps mirror pairs together.

    Operates on pair indices (match = rows 2p, 2p+1) so a match's two
    perspectives can never land on opposite sides of a fold (which would leak
    the label). Each fold's test block is strictly later than its train block.
    """
    n_pairs = n_rows // 2
    for tr_pairs, te_pairs in TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_pairs)):
        tr = np.column_stack([2 * tr_pairs, 2 * tr_pairs + 1]).ravel()
        te = np.column_stack([2 * te_pairs, 2 * te_pairs + 1]).ravel()
        yield tr, te


def evaluate_model(model, X, y, X_test, y_test, X_train, y_train, feature_names=None):
    # Sanity-check the mirror-pair invariant: each adjacent pair sums to 1.
    if len(y) % 2 == 0 and not np.array_equal(
        np.asarray(y)[0::2] + np.asarray(y)[1::2], np.ones(len(y) // 2)
    ):
        print("WARNING: rows are not adjacent mirror pairs; CV folds may leak.")

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    if hasattr(model, "predict_proba"):
        test_proba = model.predict_proba(X_test)[:, 1]
    else:
        test_proba = test_pred

    print("\n----- Results -----")
    print(f"Test Accuracy:   {accuracy_score(y_test, test_pred) * 100:.2f}%")
    print(f"Train Accuracy:  {accuracy_score(y_train, train_pred) * 100:.2f}%")
    print(f"Brier Score:     {brier_score_loss(y_test, test_proba):.4f}")
    print(f"Log Loss:        {log_loss(y_test, test_proba):.4f}")
    print(f"Precision:       {precision_score(y_test, test_pred):.4f}")
    print(f"F1 Score:        {f1_score(y_test, test_pred):.4f}")

    if feature_names is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        max_len = max(len(name) for name in feature_names)
        print("\nFeature Importances:")
        for idx in np.argsort(importances)[::-1]:
            pad = " " * (max_len - len(feature_names[idx]))
            print(f"{feature_names[idx]}: {pad}{importances[idx] * 100:.1f}%")

    # Honest, time-respecting, pair-aware cross-validation.
    scores = []
    for tr, te in time_series_pair_cv(len(X), n_splits=8):
        fold = clone(model)
        fold.fit(X[tr], y[tr])
        scores.append(accuracy_score(y[te], fold.predict(X[te])))
    scores = np.array(scores)
    print(f"\nTime-series CV Accuracy: {scores.mean() * 100:.2f}% "
          f"(+/- {scores.std() * 100:.2f}%, {len(scores)} folds)")
