"""
ETL Step 3 — Load
Loads the cleaned/transformed dataframe into the MySQL data warehouse.
Populates dimension tables and the fact table following the Star Schema.
"""

import pandas as pd
import sqlalchemy
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DB connection — edit these for your MySQL setup
# ─────────────────────────────────────────────
DB_USER = os.environ.get("DWM_DB_USER", "root")
DB_PASS = os.environ.get("DWM_DB_PASS", "")
DB_HOST = os.environ.get("DWM_DB_HOST", "localhost")
DB_PORT = os.environ.get("DWM_DB_PORT", "3306")
DB_NAME = os.environ.get("DWM_DB_NAME", "cardiomine")


def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = sqlalchemy.create_engine(url, echo=False)
    return engine


# ─────────────────────────────────────────────
# Dimension builders
# ─────────────────────────────────────────────

def build_dim_patient(df: pd.DataFrame) -> pd.DataFrame:
    return df[["patient_id", "gender", "age_group"]].copy()


def build_dim_medical(df: pd.DataFrame) -> pd.DataFrame:
    medical_cols = [
        "patient_id",
        "chest_pain_type",
        "resting_ecg",
        "exercise_angina",
        "blood_sugar_category",
        "cholesterol_category",
        "bp_category",
    ]
    dim = df[medical_cols].copy()
    dim = dim.reset_index(drop=True)
    dim.insert(0, "medical_id", dim.index + 1)
    return dim


def build_dim_disease(df: pd.DataFrame) -> pd.DataFrame:
    unique_diseases = (
        df[["disease_status", "target"]]
        .drop_duplicates()
        .rename(columns={"target": "disease_flag"})
    )
    unique_diseases = unique_diseases.reset_index(drop=True)
    unique_diseases.insert(0, "disease_id", unique_diseases.index + 1)
    unique_diseases["disease_name"] = "Heart Disease"
    return unique_diseases[["disease_id", "disease_name", "disease_status", "disease_flag"]]


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since the UCI dataset has no real dates, we generate a synthetic date
    dimension anchored to 2020-01-01 for demonstration purposes.
    This makes OLAP drill-down/roll-up meaningful.
    """
    import datetime
    base_date = datetime.date(2020, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(len(df))]
    dim_date = pd.DataFrame(
        {
            "date_id": range(1, len(df) + 1),
            "date": [str(d) for d in dates],
            "day": [d.day for d in dates],
            "month": [d.month for d in dates],
            "quarter": [(d.month - 1) // 3 + 1 for d in dates],
            "year": [d.year for d in dates],
        }
    )
    return dim_date


def build_fact_table(df: pd.DataFrame, dim_medical: pd.DataFrame, dim_disease: pd.DataFrame) -> pd.DataFrame:
    """Build the central fact table."""
    # Map patient_id → medical_id
    patient_to_medical = dim_medical.set_index("patient_id")["medical_id"].to_dict()

    # Map disease_status → disease_id
    status_to_disease = dim_disease.set_index("disease_status")["disease_id"].to_dict()

    fact = df[
        [
            "patient_id",
            "age",
            "resting_bp",
            "cholesterol",
            "max_heart_rate",
            "st_depression",
        ]
    ].copy()

    fact["date_id"] = range(1, len(fact) + 1)
    fact["medical_id"] = fact["patient_id"].map(patient_to_medical)
    fact["disease_id"] = df["disease_status"].map(status_to_disease)
    fact["risk_probability"] = 0.0  # Updated after mining step

    fact.insert(0, "analysis_id", range(1, len(fact) + 1))

    return fact[
        [
            "analysis_id",
            "patient_id",
            "date_id",
            "medical_id",
            "disease_id",
            "age",
            "resting_bp",
            "cholesterol",
            "max_heart_rate",
            "st_depression",
            "risk_probability",
        ]
    ]


# ─────────────────────────────────────────────
# Main load function
# ─────────────────────────────────────────────

def load(df: pd.DataFrame, engine=None):
    """
    Load transformed data into MySQL warehouse.
    Creates tables if they don't exist (schema.sql should be run first).

    Parameters
    ----------
    df : pd.DataFrame
        Output of transform()
    engine : sqlalchemy Engine, optional
        If None, creates one using environment variables.
    """
    if engine is None:
        engine = get_engine()

    logger.info("Building dimension and fact tables...")

    dim_patient = build_dim_patient(df)
    dim_medical = build_dim_medical(df)
    dim_disease = build_dim_disease(df)
    dim_date = build_dim_date(df)
    fact = build_fact_table(df, dim_medical, dim_disease)

    tables = {
        "dim_patient": dim_patient,
        "dim_medical": dim_medical,
        "dim_disease": dim_disease,
        "dim_date": dim_date,
        "fact_patient_analysis": fact,
    }

    with engine.begin() as conn:
        for table_name, table_df in tables.items():
            rows = len(table_df)
            table_df.to_sql(table_name, con=conn, if_exists="replace", index=False)
            logger.info(f"  Loaded {rows} rows → {table_name}")

    logger.info("✅ Warehouse load complete.")
    return tables


def save_processed(df: pd.DataFrame, output_dir: str = None):
    """Save cleaned dataframe to CSV for offline use (no MySQL needed)."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "heart_disease_clean.csv")
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to: {path}")
    return path


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from etl.extract import extract
    from etl.transform import transform

    raw = extract()
    clean = transform(raw)
    save_processed(clean)

    try:
        load(clean)
    except Exception as e:
        logger.error(f"MySQL load failed (is MySQL running?): {e}")
        logger.info("Processed CSV was saved successfully — continue without MySQL.")
