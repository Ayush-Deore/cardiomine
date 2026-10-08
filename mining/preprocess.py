"""
Mining — Preprocessing
Prepares the cleaned dataframe for ML pipelines (classification & clustering).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "age", "sex", "cp", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "restecg", "max_heart_rate",
    "exang", "st_depression",
]

CLUSTERING_FEATURES = [
    "age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression",
]


def load_clean_data(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "heart_disease_clean.csv"
        )
    df = pd.read_csv(filepath)
    logger.info(f"Loaded processed data: {df.shape}")
    return df


def prepare_classification_data(df: pd.DataFrame):
    """
    Prepare feature matrix X and target y for classification.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Use raw numeric columns (already cleaned)
    available = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Classification features: {available}")

    X = df[available].copy()
    y = df["target"].copy()

    # Fill any residual NaN
    X.fillna(X.median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        f"Train size: {len(X_train)} | Test size: {len(X_test)} | "
        f"Positive rate: {y_train.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test, available


def prepare_clustering_data(df: pd.DataFrame):
    """
    Prepare and scale features for K-Means clustering.

    Returns
    -------
    X_scaled : np.ndarray
    scaler   : fitted StandardScaler
    feature_names : list[str]
    """
    available = [c for c in CLUSTERING_FEATURES if c in df.columns]
    logger.info(f"Clustering features: {available}")

    X = df[available].copy()
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, available
