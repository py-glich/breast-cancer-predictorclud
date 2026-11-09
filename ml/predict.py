"""Load the saved model and make one prediction from 30 named features."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "ml" / "saved_models" / "breast_cancer_pipeline.joblib"


QUICK_MODEL_PATH = PROJECT_ROOT / "ml" / "saved_models" / "breast_cancer_quick_pipeline.joblib"


def load_artifact(mode: str = "full"):
    path = QUICK_MODEL_PATH if mode == "quick" else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Trained model is missing. Run: python ml/train_model.py"
        )
    return joblib.load(path)


def predict_one(features: dict, mode: str = "full"):
    artifact = load_artifact(mode)
    expected_names = artifact["feature_names"]

    if not isinstance(features, dict):
        raise ValueError("Features must be a dictionary of feature names and values.")

    received_names = set(features.keys())
    expected_set = set(expected_names)

    missing = sorted(expected_set - received_names)
    extra = sorted(received_names - expected_set)

    if missing or extra:
        details = []
        if missing:
            details.append(f"Missing features: {', '.join(missing)}")
        if extra:
            details.append(f"Unexpected features: {', '.join(extra)}")
        raise ValueError("; ".join(details))

    try:
        ordered_values = [float(features[name]) for name in expected_names]
    except (TypeError, ValueError):
        raise ValueError("Every feature value must be a valid number.")

    if not np.isfinite(ordered_values).all():
        raise ValueError("Feature values must be finite numbers.")

    input_frame = pd.DataFrame([ordered_values], columns=expected_names)
    pipeline = artifact["pipeline"]

    predicted_class = int(pipeline.predict(input_frame)[0])
    probabilities = pipeline.predict_proba(input_frame)[0]
    class_probabilities = {
        str(int(class_id)): float(probability)
        for class_id, probability in zip(pipeline.classes_, probabilities)
    }

    diagnosis = artifact["target_names"][predicted_class]

    return {
        "mode": mode,
        "class_id": predicted_class,
        "prediction": diagnosis,
        "probabilities": {
            "malignant": class_probabilities["0"],
            "benign": class_probabilities["1"],
        },
        "notice": (
            "Educational model output only. This is not a medical diagnosis "
            "and must not be used for clinical decisions."
        ),
    }


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    def load_dataset():
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target, data

    X, _, _ = load_dataset()
    example_features = X.iloc[0].to_dict()
    print(predict_one(example_features))
