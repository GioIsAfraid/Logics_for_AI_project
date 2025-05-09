from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, X_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,            # Limit to depth for overfitting avoidance
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
