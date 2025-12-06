from sklearn.model_selection import train_test_split
from functions.utils import process_y, read_file
from models.xgboost import XGBoost
from model_evaluations import evaluate_model


if __name__ == "__main__":
    X, feature_names = read_file()
    y = process_y()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        shuffle=False
    )
    
    xgb = XGBoost(X_train, y_train)
    # match = [[3, 36, -1, -155, 10, -3.7, -1.6, -2.2, 11, 2087, 2034, 102]]
    # print(xgb.predict(match))
    # print(xgb.predict_proba(match))
    evaluate_model(xgb, X, y, X_test, y_test, X_train, y_train, feature_names=feature_names)
