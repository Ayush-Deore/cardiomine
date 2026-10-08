"""
Mining — Classification
Trains and evaluates four classifiers:
  1. Random Forest
  2. Logistic Regression
  3. J48 (Decision Tree with entropy criterion — Weka C4.5 equivalent)
  4. Naive Bayes (GaussianNB)

Saves all models + metrics to models/risk_model.pkl
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.pkl")


# ─────────────────────────────────────────────
# Individual model trainers
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_j48(X_train, y_train, random_state: int = 42):
    """
    J48 — Decision Tree with information-gain (entropy) splitting criterion.
    J48 is the Weka implementation name for the C4.5 algorithm.
    sklearn's DecisionTreeClassifier(criterion='entropy') is the equivalent.
    """
    logger.info("Training J48 Decision Tree (entropy criterion = C4.5 equivalent)...")
    model = DecisionTreeClassifier(
        criterion="entropy",       # ← information gain, matches J48/C4.5
        max_depth=5,               # prune to avoid overfitting on 303 records
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """
    Gaussian Naive Bayes — assumes features are Gaussian-distributed per class.
    Suitable for continuous clinical features (age, cholesterol, BP, etc.).
    """
    logger.info("Training Gaussian Naive Bayes...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name":       model_name,
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_test": y_test.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }

    logger.info(f"\n{'─'*40}")
    logger.info(f"  {model_name}")
    logger.info(f"  Accuracy={metrics['accuracy']}  F1={metrics['f1']}  ROC-AUC={metrics['roc_auc']}")
    logger.info(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

    return metrics


# ─────────────────────────────────────────────
# Feature importance helpers
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    importances = model.feature_importances_
    return pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)


def get_j48_text(model, feature_names: list) -> str:
    """Return a human-readable text representation of the decision tree."""
    return export_text(model, feature_names=feature_names, max_depth=5)


def get_logistic_coefficients(model, feature_names: list) -> pd.DataFrame:
    """Return logistic regression coefficients (log-odds) sorted by magnitude."""
    return pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)


def get_nb_class_stats(model, feature_names: list) -> dict:
    """Return Naive Bayes class mean and variance per feature."""
    classes = ["No Disease", "Heart Disease"]
    return {
        cls: pd.DataFrame({
            "feature":  feature_names,
            "mean":     model.theta_[i].round(3).tolist(),
            "variance": model.var_[i].round(3).tolist(),
        }).to_dict("records")
        for i, cls in enumerate(classes)
    }


# ─────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────

def save_model(rf, lr, j48, nb, feature_names, all_metrics, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    package = {
        # ── primary model for risk prediction ──
        "model":         rf,
        "feature_names": feature_names,
        "metrics":       all_metrics["random_forest"],

        # ── all classifiers ──
        "classifiers": {
            "random_forest":       rf,
            "logistic_regression": lr,
            "j48":                 j48,
            "naive_bayes":         nb,
        },
        "all_metrics": all_metrics,
    }
    joblib.dump(package, path)
    logger.info(f"All models saved → {path}")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run classification.py first.")
    return joblib.load(path)


# ─────────────────────────────────────────────
# Risk prediction (uses Random Forest)
# ─────────────────────────────────────────────

def predict_risk(patient_data: dict, path: str = MODEL_PATH) -> dict:
    package = load_model(path)
    model = package["model"]
    feature_names = package["feature_names"]

    x = np.array([[patient_data.get(f, 0) for f in feature_names]])
    prob = model.predict_proba(x)[0][1]

    if prob < 0.35:
        category = "Low Risk"
    elif prob < 0.65:
        category = "Moderate Risk"
    else:
        category = "High Risk"

    return {"probability": round(prob, 4), "risk_category": category}


# ─────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mining.preprocess import load_clean_data, prepare_classification_data

    df = load_clean_data()
    X_train, X_test, y_train, y_test, feature_names = prepare_classification_data(df)

    rf  = train_random_forest(X_train, y_train)
    lr  = train_logistic_regression(X_train, y_train)
    j48 = train_j48(X_train, y_train)
    nb  = train_naive_bayes(X_train, y_train)

    all_metrics = {
        "random_forest":       evaluate_model(rf,  X_test, y_test, "Random Forest"),
        "logistic_regression": evaluate_model(lr,  X_test, y_test, "Logistic Regression"),
        "j48":                 evaluate_model(j48, X_test, y_test, "J48 Decision Tree"),
        "naive_bayes":         evaluate_model(nb,  X_test, y_test, "Naive Bayes"),
    }

    # Attach extras
    fi = get_feature_importance(rf, feature_names)
    all_metrics["random_forest"]["feature_importance"] = fi.to_dict("records")
    all_metrics["j48"]["tree_text"] = get_j48_text(j48, feature_names)
    all_metrics["j48"]["feature_importance"] = get_feature_importance(j48, feature_names).to_dict("records")
    all_metrics["logistic_regression"]["coefficients"] = get_logistic_coefficients(lr, feature_names).to_dict("records")
    all_metrics["naive_bayes"]["class_stats"] = get_nb_class_stats(nb, feature_names)

    save_model(rf, lr, j48, nb, feature_names, all_metrics)

    print("\n── Summary ──────────────────────────────")
    for name, m in all_metrics.items():
        print(f"  {m['model_name']:25s} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
