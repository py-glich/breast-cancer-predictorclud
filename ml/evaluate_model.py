"""Evaluate the saved model on the held-out test set."""

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from preprocess import load_dataset, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ml" / "saved_models"
MODEL_PATH = MODEL_DIR / "breast_cancer_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.png"
QUICK_MODEL_PATH = MODEL_DIR / "breast_cancer_quick_pipeline.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python ml/train_model.py"
        )

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    X, y, _ = load_dataset()
    _, X_test, _, y_test = split_dataset(X, y)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    # Class 0 (malignant) is treated as the positive class for this project.
    malignant_probability = probabilities[:, list(pipeline.classes_).index(0)]
    malignant_actual = (y_test == 0).astype(int)
    malignant_predicted = (predictions == 0).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "malignant_precision": float(
            precision_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "malignant_recall": float(
            recall_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "malignant_f1": float(
            f1_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "roc_auc_malignant": float(
            roc_auc_score(malignant_actual, malignant_probability)
        ),
        "test_samples": int(len(y_test)),
        "positive_class": "malignant",
    }

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            zero_division=0,
        )
    )

    print("\n========== PROJECT METRICS ==========")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["malignant", "benign"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=160)
    plt.close()

    if QUICK_MODEL_PATH.exists():
        q = joblib.load(QUICK_MODEL_PATH)
        q_pipe, q_cols = q["pipeline"], q["feature_names"]
        q_pred = q_pipe.predict(X_test[q_cols])
        q_prob = q_pipe.predict_proba(X_test[q_cols])[:, list(q_pipe.classes_).index(0)]
        metrics["quick"] = {
            "accuracy": float(accuracy_score(y_test, q_pred)),
            "malignant_precision": float(precision_score(y_test, q_pred, pos_label=0, zero_division=0)),
            "malignant_recall": float(recall_score(y_test, q_pred, pos_label=0, zero_division=0)),
            "malignant_f1": float(f1_score(y_test, q_pred, pos_label=0, zero_division=0)),
            "roc_auc_malignant": float(roc_auc_score(malignant_actual, q_prob)),
            "test_samples": int(len(y_test)),
            "positive_class": "malignant",
        }
        print(f"Quick model accuracy: {metrics['quick']['accuracy']:.4f}")

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved metrics to: {METRICS_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_PATH}")


if __name__ == "__main__":
    main()
