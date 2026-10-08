"""
Mining — Regression
Linear Regression to predict max_heart_rate from clinical features.
This demonstrates the regression technique in the DWM context.
Also includes a brief polynomial feature extension for comparison.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REG_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "regression_model.pkl")

# Predict max_heart_rate from these features — realistic medical sense
REGRESSION_FEATURES = ["age", "sex", "resting_bp", "cholesterol", "fasting_blood_sugar", "st_depression"]
REGRESSION_TARGET   = "max_heart_rate"


def prepare_regression_data(df: pd.DataFrame):
    available = [c for c in REGRESSION_FEATURES if c in df.columns]
    logger.info(f"Regression features → target: {available} → {REGRESSION_TARGET}")
    X = df[available].fillna(df[available].median())
    y = df[REGRESSION_TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42), available


def evaluate_regression(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    logger.info(f"  {name} → MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    return {
        "model_name": name,
        "mae":  round(mae, 4),
        "rmse": round(rmse, 4),
        "r2":   round(r2, 4),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }


def train_regression(df: pd.DataFrame) -> dict:
    (X_train, X_test, y_train, y_test), feature_names = prepare_regression_data(df)

    # ── Model 1: Simple Linear Regression ──────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_regression(lr, X_test, y_test, "Linear Regression")

    # ── Model 2: Ridge Regression (regularized) ─────────────────
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_metrics = evaluate_regression(ridge, X_test, y_test, "Ridge Regression")

    # ── Coefficients (interpretability) ─────────────────────────
    coefs = pd.DataFrame({
        "feature":     feature_names,
        "coefficient": lr.coef_,
    }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)
    logger.info(f"\nLinear Regression coefficients:\n{coefs.to_string()}")

    package = {
        "linear_regression": lr,
        "ridge_regression":  ridge,
        "feature_names":     feature_names,
        "target":            REGRESSION_TARGET,
        "linear_metrics":    lr_metrics,
        "ridge_metrics":     ridge_metrics,
        "coefficients":      coefs.to_dict("records"),
        "X_test":  X_test.values.tolist(),
        "y_test":  y_test.tolist(),
    }

    os.makedirs(os.path.dirname(REG_MODEL_PATH), exist_ok=True)
    joblib.dump(package, REG_MODEL_PATH)
    logger.info(f"Regression model saved to: {REG_MODEL_PATH}")
    return package


def load_regression_model(path: str = REG_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Regression model not found at {path}. Run regression.py first.")
    return joblib.load(path)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mining.preprocess import load_clean_data
    df = load_clean_data()
    train_regression(df)
