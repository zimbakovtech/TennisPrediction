from sklearn.ensemble import RandomForestClassifier

def RandomForest(X_train, y_train):
    rf_params = {
        'n_estimators': 300,
        'max_depth': 7,
        'min_samples_split': 10,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'n_jobs': 2,
        'random_state': 42
    }
    rf = RandomForestClassifier(**rf_params)

    print("\n----- Training Random Forest -----")
    print(f"Random Forest parameters: {rf_params}")
    rf.fit(X_train, y_train)

    return rf