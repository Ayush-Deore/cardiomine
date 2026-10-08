# CardioMine

**Patient Analysis & Heart Disease Risk Mining System**  
*Academic Mini Project — Data Warehousing & Data Mining*

---

## Overview

CardioMine is a full DWM (Data Warehousing & Mining) pipeline built on the **UCI Heart Disease (Cleveland) dataset**. It demonstrates:

- ETL pipeline with data cleaning and transformation
- Star Schema data warehouse (MySQL)
- OLAP analytics (Roll-up, Drill-down, Slice, Dice)
- Classification with Random Forest (+ Logistic Regression baseline)
- K-Means clustering for patient segmentation
- 5-page interactive Streamlit dashboard

---

## Project Structure

```
cardiomine/
├── data/
│   ├── raw/
│   │   ├── download_dataset.py   ← Download UCI dataset
│   │   └── heart_disease.csv     ← Raw data (after download)
│   └── processed/
│       └── heart_disease_clean.csv
│
├── etl/
│   ├── extract.py       ← Read raw CSV
│   ├── transform.py     ← Clean + enrich data
│   └── load.py          ← Load to MySQL + save processed CSV
│
├── warehouse/
│   ├── schema.sql       ← Star schema DDL
│   └── olap_queries.sql ← OLAP demonstration queries
│
├── mining/
│   ├── preprocess.py    ← Feature prep for ML
│   ├── classification.py← Random Forest + Logistic Regression
│   ├── clustering.py    ← K-Means + Elbow Analysis
│   └── evaluation.py   ← (supplementary metrics)
│
├── models/
│   ├── risk_model.pkl   ← Trained RF classifier
│   └── cluster_model.pkl← Trained K-Means model
│
├── dashboard/
│   └── app.py           ← 5-page Streamlit dashboard
│
├── run_pipeline.py      ← End-to-end runner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python data/raw/download_dataset.py
```

### 3. (Optional) Set up MySQL

```bash
# Edit DWM_DB_USER / DWM_DB_PASS environment variables or edit etl/load.py
mysql -u root -p < warehouse/schema.sql
```

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

This will:
- Extract → clean → transform the data
- Save `data/processed/heart_disease_clean.csv`
- Load into MySQL (if available)
- Train Random Forest classifier
- Train K-Means clustering model
- Save `.pkl` model files

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease (Cleveland subset)
- **Records:** 303 patient records
- **Features:** 13 clinical attributes
- **Target:** Binary (0 = No Disease, 1 = Disease Present)

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Data processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model persistence | Joblib |
| Database | MySQL (Star Schema) |
| ORM / connector | SQLAlchemy + PyMySQL |
| Dashboard | Streamlit |
| Visualizations | Plotly |

---

## OLAP Operations

| Operation | Description |
|-----------|-------------|
| Roll-up | Patient → Age Group → Population |
| Drill-down | Year → Quarter → Month → Patient |
| Slice | Filter on Gender = Female |
| Dice | Age Group = Middle Age AND Cholesterol = High |

---

## Disclaimer

This project is for **academic purposes only**. Risk prediction outputs do not constitute medical diagnoses. Always consult a qualified physician.
