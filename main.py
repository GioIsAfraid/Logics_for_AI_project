from functions.preprocessing import load_data, preprocess_data
from functions.training import train_model
from functions.evaluation import evaluate_model
from functions.fairlearn_analysis import analyze_fairness
from functions.adv_fair_metrics import advanced_fairness_metrics
from functions.bias_injection import introduce_bias
from functions.bias_mitigation import mitigate_bias
from functions.plotting import plot_fairness_results


def main():

    # Loading Dataset and preprocessing
    print("Loading dataset and preprocessing...\n")
    data_set = load_data('data/student-mat.csv')
    X_train, X_test, y_train, y_test, original_test, original = preprocess_data(df=data_set)
    
    # Training model
    print("Training model...\n")
    model, y_pred = train_model(X_train, y_train, X_test)

    # Evaluation of train and test accuracy
    print("Evaluation...\n")
    y_train_pred = model.predict(X_train)
    eval_results = evaluate_model(y_test, y_pred, y_train, y_train_pred)

    # Fairlearn analysis, with sensitive feature "sex"
    sensitive_feature = original_test['sex']
    fairness_results = analyze_fairness(y_test, y_pred, sensitive_feature, label="Sex")
    adv_metrics = advanced_fairness_metrics(y_test, y_pred, sensitive_feature, label="Sex")
    all_metrics = {**eval_results, **fairness_results, **adv_metrics}

    # Plotting results
    plot_fairness_results(all_metrics, title="Base Model Fairness", filename="base_model_fairness.png")
    
    print("Injecting artificial bias: dropping half of female students\n")
    introduce_bias(df=original, bias_type="drop", group_col="sex", group_value="F", flip_prob=0.5, drop_frac=0.5, save_path="data/biased_student_mat.csv")
   
    # Loading biased Dataset and preprocessing
    print("Loading biased dataset and preprocessing...\n")
    data_set = load_data('data/biased_student_mat.csv')
    X_train_biased, X_test_biased, y_train_biased, y_test_biased, original_test_biased, original_biased = preprocess_data(df=data_set)
    original_train_biased = original_biased.iloc[X_train_biased.index]

    # Training biased model
    print("Training Biased model...\n")
    model_biased, y_pred_biased = train_model(X_train_biased, y_train_biased, X_test_biased)

    # Evaluation of biased model
    print("Evaluation...\n")
    y_train_pred_biased = model_biased.predict(X_train_biased)
    eval_results_biased = evaluate_model(y_test_biased, y_pred_biased, y_train_biased, y_train_pred_biased)

    # Fairness analysis for biased model
    sensitive_feature_biased = original_test_biased['sex']
    fairness_results_biased = analyze_fairness(y_test_biased, y_pred_biased, sensitive_feature_biased, label="Sex")
    adv_metrics_biased = advanced_fairness_metrics(y_test_biased, y_pred_biased, sensitive_feature_biased, label="Sex")
    all_metrics_biased = {**eval_results_biased, **fairness_results_biased, **adv_metrics_biased}

    # Plotting results of bias
    plot_fairness_results(all_metrics_biased, title="Biased Model Fairness", filename="biased_model_fairness.png")

    # Bias mitigation
    print("Applying bias mitigation...\n")
    y_test_pred_mitigated, mitigator, _ = mitigate_bias(
        X_train_biased,
        y_train_biased,
        X_test_biased,
        y_test_biased,
        sensitive_train=original_train_biased["sex"],
        sensitive_test=original_test_biased["sex"]
    )

    # Evaluation after mitigation
    print("Bias mitigation evaluation...\n")
    y_train_pred_mitigated = mitigator.predict(X_train_biased)
    eval_results_mitigated = evaluate_model(y_test_biased, y_test_pred_mitigated, y_train_biased, y_train_pred_mitigated)

    # Fairness analysis for mitigated model
    fairness_results_mitigated = analyze_fairness(y_test_biased, y_test_pred_mitigated, original_test_biased["sex"], label="Sex")
    adv_metrics_mitigated = advanced_fairness_metrics(y_test_biased, y_test_pred_mitigated, original_test_biased["sex"], label="Sex")
    all_metrics_mitigated = {**eval_results_mitigated, **fairness_results_mitigated, **adv_metrics_mitigated}

    # Plotting results of mitigation
    plot_fairness_results(all_metrics_mitigated, title="Mitigated Model Fairness", filename="mitigated_model_fairness.png")

if __name__ == "__main__":
    main()
