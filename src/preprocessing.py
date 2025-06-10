import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


def process_y():
    # balanced 1/0 labels, length = 15710 * 2 = 31420
    return np.array([1, 0] * 15710)


def read_file():
    # fast CSV → numpy array
    df = pd.read_csv("data/processed/all_matches.csv")
    return df.values


def tuned_classifiers(X_train, y_train):
    results = {}

    # --- Random Forest (CPU only) ---
    rf = RandomForestClassifier(random_state=42, n_jobs=2)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 12],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
    }
    print("Tuning Random Forest (RandomizedSearchCV, n_iter=5, cv=3)…")
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        n_jobs=2,
        random_state=42,
        verbose=1
    )
    rf_search.fit(X_train, y_train)
    results['RandomForest'] = rf_search

    # --- XGBoost (CPU implementation) ---
    xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=2,
        tree_method='hist'     # hist = fast CPU algorithm
    )
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [12, 12],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 1.0],
    }
    print("Tuning XGBoost (RandomizedSearchCV, n_iter=5, cv=3)…")
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_params,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        n_jobs=2,
        random_state=42,
        verbose=1
    )
    xgb_search.fit(X_train, y_train)
    results['XGBoost'] = xgb_search

    return results


def evaluate_models(models, X_test, y_test):
    for name, search in models.items():
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n=== {name} Results ===")
        print(f"Best Params: {search.best_params_}")
        print(f"Test Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    # Load data
    X = read_file()
    y = process_y()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    # Hyperparameter tuning
    tuned = tuned_classifiers(X_train, y_train)

    # Evaluation
    evaluate_models(tuned, X_test, y_test)

    # Quick CV summary
    print("\nCross-validation scores for each best estimator:")
    for name, search in tuned.items():
        scores = cross_val_score(
            search.best_estimator_,
            X, y,
            cv=3,
            scoring='accuracy',
            n_jobs=2
        )
        print(f"{name}: Mean={np.mean(scores)*100:.1f}%, Std={np.std(scores)*100:.1f}%")

    # # --- Random Forest ---
    # classifier = RandomForestClassifier(n_estimators=200, criterion='gini', random_state=42)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print(f"RANDOM FOREST FEATURE IMPORTANCES: {classifier.feature_importances_}")
    # print(f"Accuracy single RF: {acc}")

