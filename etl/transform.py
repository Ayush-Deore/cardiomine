"""
ETL Step 2 — Transform
Cleans, validates, and enriches the raw dataset.
Produces both the cleaned base dataframe and the dimension/fact table records.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Categorization helpers
# ─────────────────────────────────────────────

def age_group(age: float) -> str:
    if age < 30:
        return "Young"
    elif age < 50:
        return "Adult"
    elif age < 65:
        return "Middle Age"
    else:
        return "Senior"


def cholesterol_category(chol: float) -> str:
    if chol < 200:
        return "Normal"
    elif chol < 240:
        return "Borderline"
    else:
        return "High"


def bp_category(bp: float) -> str:
    """Uses standard JNC7 thresholds."""
    if bp < 120:
        return "Normal"
    elif bp < 130:
        return "Elevated"
    elif bp < 140:
        return "Stage 1 Hypertension"
    else:
        return "Stage 2 Hypertension"


def heart_rate_category(hr: float) -> str:
    if hr < 100:
        return "Low"
    elif hr < 150:
        return "Normal"
    else:
        return "High"


def blood_sugar_category(fbs: int) -> str:
    return "High (>120 mg/dl)" if fbs == 1 else "Normal (≤120 mg/dl)"


CHEST_PAIN_MAP = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-Anginal Pain",
    4: "Asymptomatic",
}

RESTECG_MAP = {
    0: "Normal",
    1: "ST-T Abnormality",
    2: "Left Ventricular Hypertrophy",
}

SEX_MAP = {0: "Female", 1: "Male"}


# ─────────────────────────────────────────────
# Main transform function
# ─────────────────────────────────────────────

def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned and fully annotated dataframe ready for warehouse loading.
    """
    df = raw_df.copy()

    logger.info("Starting transformation...")

    # 1. Drop fully duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Duplicates removed: {before - len(df)}")

    # 2. Coerce numeric types
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "target"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Handle missing values
    # ca and thal have a small number of missing values in UCI Cleveland
    if "ca" in df.columns and not df["ca"].isnull().all():
        df["ca"] = df["ca"].fillna(df["ca"].median())
    
    if "thal" in df.columns:
        mode_val = df["thal"].mode()
        if len(mode_val) > 0:
            df["thal"] = df["thal"].fillna(mode_val[0])

    # Flag & drop any remaining rows with critical nulls
    critical_cols = ["age", "sex", "chol", "trestbps", "thalach", "target"]
    flagged = df[df[critical_cols].isnull().any(axis=1)]
    if len(flagged) > 0:
        logger.warning(f"Dropping {len(flagged)} rows with critical missing values.")
    df.dropna(subset=critical_cols, inplace=True)

    # 4. Binarize target (0 = no disease, 1 = disease present)
    df["target"] = (df["target"] > 0).astype(int)

    # 5. Derived / enriched categorical attributes
    df["age_group"] = df["age"].apply(age_group)
    df["cholesterol_category"] = df["chol"].apply(cholesterol_category)
    df["bp_category"] = df["trestbps"].apply(bp_category)
    df["heart_rate_category"] = df["thalach"].apply(heart_rate_category)
    df["blood_sugar_category"] = df["fbs"].apply(blood_sugar_category)

    # 6. Human-readable labels
    df["gender"] = df["sex"].map(SEX_MAP).fillna("Unknown")
    df["chest_pain_type"] = df["cp"].map(CHEST_PAIN_MAP).fillna("Unknown")
    df["resting_ecg"] = df["restecg"].map(RESTECG_MAP).fillna("Unknown")
    df["exercise_angina"] = df["exang"].map({0: "No", 1: "Yes"})
    df["disease_status"] = df["target"].map({0: "No Disease", 1: "Heart Disease"})

    # 7. Rename raw columns to warehouse-friendly names
    df.rename(
        columns={
            "trestbps": "resting_bp",
            "chol": "cholesterol",
            "thalach": "max_heart_rate",
            "oldpeak": "st_depression",
            "fbs": "fasting_blood_sugar",
        },
        inplace=True,
    )

    # 8. Assign surrogate patient_id
    df.insert(0, "patient_id", range(1, len(df) + 1))

    logger.info(f"Transformation complete. {len(df)} clean records ready.")
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from etl.extract import extract
    raw = extract()
    clean = transform(raw)
    print(clean.head())
    print(clean.columns.tolist())
