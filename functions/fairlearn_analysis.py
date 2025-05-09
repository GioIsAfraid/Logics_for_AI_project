from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score

def analyze_fairness(y_true, y_pred, sensitive_feature, label="Sensitive Feature"):
    print(f"Fairlearn Analysis for: {label}\n")

    metric_frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    print("Performance per group:\n")
    print(metric_frame.by_group, "\n")

    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    print(f"Demographic Parity Difference: {dp_diff:.4f}\n")

    disparity = metric_frame.difference()
    per_group = metric_frame.by_group.to_dict()
    
    return {
        "demographic_disparity": disparity,
        "performance_by_group": per_group
    }
