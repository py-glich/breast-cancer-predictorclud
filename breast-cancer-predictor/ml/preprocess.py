"""Shared dataset loading and train/test splitting utilities."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset():
    """Return features (X), target (y), and the original sklearn dataset."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y, dataset


def split_dataset(X, y):
    """Create reproducible, stratified training and testing sets."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
