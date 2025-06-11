from sklearn.metrics import accuracy_score, precision_score
import numpy as np


def evaluate_model(model, X_test, y_test, is_keras=False):
    if is_keras:
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        prec = precision_score(y_test, y_pred)
    else:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)

    print(f"\n=== Results ===")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Precision: {prec * 100:.2f}%")


def print_feature_importances(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    print("\nFeature Importances (sorted):")
    for idx in sorted_indices:
        print(f"{feature_names[idx]} (column {idx}): {importances[idx]:.4f}")