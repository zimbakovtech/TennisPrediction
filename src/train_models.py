import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from functions.utils import process_y, read_file
from xgboost import XGBClassifier


def tuned_classifiers(X_train, y_train):
    results = {}

    # Define XGBoost classifier with CPU-friendly settings
    xgb = XGBClassifier(
        eval_metric='logloss',  # Standard metric for binary classification
        random_state=42,        # For reproducibility
        verbosity=0,            # Silent mode
        n_jobs=2,               # Use 2 CPU cores
        tree_method='hist'      # Fast histogram-based algorithm for CPU
    )

    # Define hyperparameter search space
    xgb_params = {
        'n_estimators': [100, 200],   # Number of trees
        'max_depth': [12, 12],        # Maximum depth of trees
        'learning_rate': [0.05, 0.1], # Step size shrinkage
        'subsample': [0.7, 1.0],      # Fraction of samples used per tree
    }

    # Perform randomized search for hyperparameter tuning
    print("----- Tuning XGBoost -----")
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_params,
        n_iter=5,            # Number of parameter combinations to try
        scoring='accuracy',  # Optimize for accuracy
        cv=3,                # 3-fold cross-validation
        n_jobs=2,            # Parallelize across 2 cores
        random_state=42,     # For reproducibility
        verbose=1            # Show progress
    )
    xgb_search.fit(X_train, y_train)
    results['XGBoost'] = xgb_search

    return results


def evaluate_models(tuned_models, X_test, y_test):
    for name, search in tuned_models.items():
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n=== {name} Results ===")
        print(f"Best Params: {search.best_params_}")
        print(f"Test Accuracy: {acc * 100:.2f}%")
        # print("Classification Report:")
        # print(classification_report(y_test, y_pred))
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))

def print_feature_importances(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    print("\nFeature Importances (sorted):")
    for idx in sorted_indices:
        print(f"{feature_names[idx]} (column {idx}): {importances[idx]:.4f}")


if __name__ == "__main__":
    # Load data
    X, feature_names = read_file()
    y = process_y()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,    # 20% for testing
        random_state=42,  # For reproducibility
        shuffle=False     # Preserve data order
    )

    # Perform hyperparameter tuning
    tuned = tuned_classifiers(X_train, y_train)

    # Evaluate the tuned model
    evaluate_models(tuned, X_test, y_test)

    # Print feature importances
    print_feature_importances(tuned['XGBoost'].best_estimator_, feature_names)

    # Cross-validation summary on full dataset
    for name, search in tuned.items():
        scores = cross_val_score(
            search.best_estimator_,
            X, y,
            cv=3,            # 3-fold CV
            scoring='accuracy',
            n_jobs=2         # Parallelize across 2 cores
        )
        print(f"{name}: Mean={np.mean(scores)*100:.1f}%, Std={np.std(scores)*100:.1f}%")