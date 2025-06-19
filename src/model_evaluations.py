from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss, log_loss, classification_report, confusion_matrix
import numpy as np

def evaluate_model(
    model,
    X_test, y_test,
    X_train, y_train,
    is_keras=False,
    feature_names=None
):
    
    if is_keras:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_pred = (model.predict(X_test) > 0.5).astype(int)
        train_pred = (model.predict(X_train) > 0.5).astype(int)
        test_proba = model.predict(X_test).flatten()
        train_proba = model.predict(X_train).flatten()
        test_prec = precision_score(y_test, test_pred)
        train_prec = precision_score(y_train, train_pred)
        test_recall = recall_score(y_test, test_pred)
        train_recall = recall_score(y_train, train_pred)
    else:
        test_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test)[:, 1]
            train_proba = model.predict_proba(X_train)[:, 1]
        else:
            test_proba = test_pred
            train_proba = train_pred
        test_acc = accuracy_score(y_test, test_pred)
        train_acc = accuracy_score(y_train, train_pred)
        test_prec = precision_score(y_test, test_pred)
        train_prec = precision_score(y_train, train_pred)
        test_recall = recall_score(y_test, test_pred)
        train_recall = recall_score(y_train, train_pred)

    print(f"\n=== Results ===")
    print(f"Accuracy (Test):   {test_acc  * 100:.2f}%")
    print(f"Accuracy (Train):  {train_acc * 100:.2f}%")
    print(f"Brier Score (Test):  {brier_score_loss(y_test, test_proba):.4f}")
    print(f"Log Loss (Test):     {log_loss(y_test, test_proba):.4f}")
    print(f"Precision (Test):    {test_prec:.4f}")
    print(f"Precision (Train):   {train_prec:.4f}")
    print(f"Recall (Test):       {test_recall:.4f}")
    print(f"Recall (Train):      {train_recall:.4f}")

    # print("\nConfusion Matrix (Test):")
    # print(confusion_matrix(y_test, test_pred))

    # print("\nDetailed Classification Report (Test):")
    # print(classification_report(y_test, test_pred))

    # If this is a tree-based model and feature names were given, print importances
    if feature_names is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\nFeature Importances:")
        for idx in sorted_idx:
            print(f"{feature_names[idx]}: {importances[idx] * 100:.2f}%")
