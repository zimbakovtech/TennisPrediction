import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from functions.utils import process_y, read_file
from xgboost import XGBClassifier


def classifier_XGBoost(X_train, y_train):
    xgb_params = {
        'n_estimators': 260,  
        'max_depth': 7, 
        'learning_rate': 0.01, 
        'subsample': 0.7,  
    }
    xgb = XGBClassifier(
        eval_metric='logloss',  
        random_state=42,
        verbosity=0,  
        n_jobs=2,   
        tree_method='hist',
        xgb_params=xgb_params
    )
    
    print("----- Tuning XGBoost -----")
    xgb.fit(X_train, y_train)
    return xgb


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== Results ===")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test Precision: {precision_score(y_test, y_pred) * 100:.2f}%")


def print_feature_importances(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    print("\nFeature Importances (sorted):")
    for idx in sorted_indices:
        print(f"{feature_names[idx]} (column {idx}): {importances[idx]:.4f}")


if __name__ == "__main__":
    X, feature_names = read_file()
    y = process_y()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  
        random_state=42, 
        shuffle=False    
    )

    xgb = classifier_XGBoost(X_train, y_train)

    evaluate_model(xgb, X_test, y_test)

    print_feature_importances(xgb, feature_names)