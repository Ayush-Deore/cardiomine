"""
ETL Step 1 — Extract
Reads the raw CSV and returns a pandas DataFrame.
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "heart_disease.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]


def extract(filepath: str = RAW_PATH) -> pd.DataFrame:
    """
    Extract raw data from CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with original column names.
    """
    logger.info(f"Extracting data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Run data/raw/download_dataset.py first."
        )

    df = pd.read_csv(filepath, names=COLUMNS, na_values=["", " ", "?"])

    logger.info(f"Extracted {len(df)} records, {df.shape[1]} columns.")
    logger.info(f"Missing values per column:\n{df.isnull().sum()}")

    return df


if __name__ == "__main__":
    df = extract()
    print(df.head())
    print(df.dtypes)
