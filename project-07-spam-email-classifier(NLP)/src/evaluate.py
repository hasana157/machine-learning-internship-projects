import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import save_json, ensure_dir

def evaluate_model(model_path, X_test, y_test, out_dir="../reports"):
    """
    Evaluate the trained spam classifier, plot confusion matrix, save metrics.
    """
    model = joblib.load(model_path)
    preds = model.predict(X_test)

    # Metrics
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    ensure_dir(out_dir)
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.show()

    # Save metrics
    save_json(f"{out_dir}/metrics.json", report)

    return report