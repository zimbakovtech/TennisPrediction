from sklearn.tree import DecisionTreeClassifier

def DecisionTree(X_train, y_train):
    dt_params = {
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    }
    dt = DecisionTreeClassifier(**dt_params)

    print("\n----- Training Decision Tree -----")
    print(f"Decision Tree parameters: {dt_params}")
    dt.fit(X_train, y_train)

    return dt