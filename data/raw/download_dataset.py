"""
Script to download the UCI Heart Disease (Cleveland) dataset.
Run this once to populate data/raw/heart_disease.csv
"""

import urllib.request
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT = os.path.join(os.path.dirname(__file__), "heart_disease.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

def download():
    print(f"Downloading UCI Cleveland Heart Disease dataset from:\n  {URL}")
    raw_data = urllib.request.urlopen(URL).read().decode("utf-8")

    # Replace ? with empty string (missing values marker in UCI dataset)
    cleaned = raw_data.replace("?", "")

    # Write header + data
    with open(OUTPUT, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        f.write(cleaned)

    print(f"Saved to: {OUTPUT}")

if __name__ == "__main__":
    download()
