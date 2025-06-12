from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions.utils import process_y, read_file
from models.xgboost import XGBoost
from models.neural_network import NeuralNetwork
from model_evaluations import evaluate_model, print_feature_importances


if __name__ == "__main__":
    X, feature_names = read_file()
    y = process_y()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb = XGBoost(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, X_train, y_train)
    print_feature_importances(xgb, feature_names)

    nn = NeuralNetwork(X_train_scaled, y_train, X_train_scaled.shape[1])
    evaluate_model(nn, X_test_scaled, y_test, X_train_scaled, y_train, is_keras=True)
    print("\nNeural Network Summary:")
    nn.summary()
