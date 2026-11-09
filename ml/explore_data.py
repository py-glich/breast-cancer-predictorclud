"""Stage 1: explore the Wisconsin Breast Cancer Diagnostic dataset."""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from preprocess import load_dataset


def main():
    X, y, dataset = load_dataset()

    # Combine features and labels for inspection only.
    df = X.copy()
    df["diagnosis"] = y

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Full table shape: {df.shape}")

    print("\n========== FEATURE NAMES ==========")
    for number, feature_name in enumerate(X.columns, start=1):
        print(f"{number:02d}. {feature_name}")

    print("\n========== TARGET ENCODING ==========")
    print("0 = malignant")
    print("1 = benign")
    print("Target names:", list(dataset.target_names))

    print("\n========== FIRST FIVE ROWS ==========")
    print(df.head().to_string())

    print("\n========== DATA TYPES AND NON-NULL COUNTS ==========")
    df.info()

    print("\n========== MISSING VALUES PER COLUMN ==========")
    print(df.isnull().sum().to_string())
    print("Total missing values:", int(df.isnull().sum().sum()))

    print("\n========== DUPLICATE ROWS ==========")
    print("Duplicate rows:", int(df.duplicated().sum()))

    print("\n========== CLASS DISTRIBUTION ==========")
    counts = y.value_counts().sort_index()
    print(f"Malignant (0): {int(counts.get(0, 0))}")
    print(f"Benign (1):    {int(counts.get(1, 0))}")

    print("\n========== NUMERICAL SUMMARY ==========")
    print(X.describe().round(3).to_string())


if __name__ == "__main__":
    main()
