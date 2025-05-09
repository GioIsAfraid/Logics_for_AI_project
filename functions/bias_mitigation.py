from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def mitigate_bias(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test):
    """
    Apply bias mitigation using ExponentiatedGradient with Demographic Parity constraint.

    Parameters:
        X_train, y_train : training features and labels
        X_test, y_test : test features and labels
        sensitive_train : sensitive features for training (e.g., sex)
        sensitive_test : sensitive features for test

    Returns:
        y_pred_mitigated : predictions from the mitigated model
        mitigator : trained mitigation model
        y_test_pred_mitigated : predictions on X_test to be used for fairness analysis
    """
    base_estimator = LogisticRegression(solver='liblinear')
    mitigator = ExponentiatedGradient(
        base_estimator,
        constraints=DemographicParity(),
    )

    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred_mitigated = mitigator.predict(X_test)

    print("\nMitigated Model Evaluation:")
    print(classification_report(y_test, y_pred_mitigated))
    print("Mitigated Accuracy:", accuracy_score(y_test, y_pred_mitigated))

    return y_pred_mitigated, mitigator, y_pred_mitigated
