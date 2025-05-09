from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(y_test, y_pred, y_train=None, y_train_pred=None):

    results ={}
    print("CLASSIFICATION REPORT (TEST):\n")
    print(classification_report(y_test, y_pred))


    test_acc = accuracy_score(y_test, y_pred)
    results['test_accuracy'] = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy:  {test_acc:.3f}\n")

    if y_train is not None and y_train_pred is not None:
        train_acc = accuracy_score(y_train, y_train_pred)
        results['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        print(f"Train Accuracy: {train_acc:.3f}\n")

    return results