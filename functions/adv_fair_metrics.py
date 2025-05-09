from fairlearn.metrics import MetricFrame, equalized_odds_difference, true_positive_rate

def advanced_fairness_metrics(y_true, y_pred, sensitive_feature, label="Sensitive Feature"):
    print(f"Advanced Fairness Metrics for: {label}\n")

    tpr_frame = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    eq_odds_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )

    print(f"True Positive Rate by group:\n{tpr_frame.by_group}\n")
    print(f"Equalized Odds Difference: {eq_odds_diff:.4f}\n")

    tpr_by_group = tpr_frame.by_group.to_dict()
    equalized_odds = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)

    return {
        "tpr_by_group": tpr_by_group,
        "equalized_odds_difference": equalized_odds
    }
