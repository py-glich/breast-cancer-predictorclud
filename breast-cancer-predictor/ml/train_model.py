"""Train and save a preprocessing + Logistic Regression pipeline."""

from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from preprocess import load_dataset, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ml" / "saved_models"
MODEL_PATH = MODEL_DIR / "breast_cancer_pipeline.joblib"
QUICK_MODEL_PATH = MODEL_DIR / "breast_cancer_quick_pipeline.joblib"


def build_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    X, y, dataset = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Scaling is fitted only on training data because it is inside the pipeline.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "target_names": {0: "malignant", 1: "benign"},
        "dataset_name": "Wisconsin Breast Cancer Diagnostic",
        "random_state": 42,
    }
    joblib.dump(artifact, MODEL_PATH)

    # Quick model: only the 10 "mean" measurements, so fewer values are needed.
    mean_cols = [c for c in X.columns if c.startswith("mean ")]
    quick = build_pipeline()
    quick.fit(X_train[mean_cols], y_train)
    quick_accuracy = quick.score(X_test[mean_cols], y_test)
    joblib.dump(
        {
            "pipeline": quick,
            "feature_names": mean_cols,
            "target_names": {0: "malignant", 1: "benign"},
            "dataset_name": "Wisconsin Breast Cancer Diagnostic (mean features)",
            "random_state": 42,
        },
        QUICK_MODEL_PATH,
    )
    print(f"Quick model (10 mean features) test accuracy: {quick_accuracy:.4f}")

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy:  {test_accuracy:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
